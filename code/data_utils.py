"""
author: Florian Krach
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import copy
import json, time, sys
from torch.utils.data import Dataset, DataLoader
import torch
import socket
import pandas as pd
import matplotlib
import tqdm
from joblib import Parallel, delayed
from absl import app
from absl import flags
import scipy.special as scispe
import scipy.stats as stats

import config

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# GLOBAL VARIABLES
FLAGS = flags.FLAGS

flags.DEFINE_string("data_gen_dict", None,
                    "name of the dict to use for data generation")
flags.DEFINE_integer("DATA_NB_JOBS", 4,
                     "nb of parallel jobs for data generation")

data_path = config.data_path
training_data_path = config.training_data_path

CHAT_ID = config.CHAT_ID

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = config.SendBotMessage()


# ==============================================================================
# FUNCTIONS
makedirs = config.makedirs

def get_dataset_overview(training_data_path=training_data_path):
    data_overview = '{}dataset_overview.csv'.format(training_data_path)
    columns = ['id', 'description']
    makedirs(training_data_path)
    if not os.path.exists(data_overview):
        df_overview = pd.DataFrame(
            data=None, columns=columns)
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview, columns


# ==============================================================================
# DATA CLASSES
class SDEIncrements(Dataset):
    def __init__(self, load=True, training_data_path=training_data_path,
                 verbose=0, idx=None, init_distribution=None,
                 **hyperparams):
        """
        :param verbose: int,
        :param load: bool, whether to load existing dataset if exists
        :param idx: array of ints, the indices to use
        :param init_distribution: dict, the distribution of the initial values,
            keywords: name, params
        :param hyperparams:
            - S0: array of floats, the initial stock prices
            - dt: float, the time step
            - r: float, the interest rate
            - nb_steps: int, number of steps of size dt
            - nb_samples: int, number of total sample paths
            - seed: int, seed for sampling
        """
        self.hyperparams = hyperparams
        self.init_distribution = init_distribution
        self.init_vals = None
        if init_distribution is not None:
            self.S0 = np.infty
            self.hyperparams["init_distribution"] = init_distribution
        else:
            self.S0 = hyperparams["S0"]
        self.dt = hyperparams["dt"]
        self.r = hyperparams["r"]
        self.nb_steps = hyperparams["nb_steps"]
        self.maturity = self.nb_steps*self.dt
        self.dimension = np.size(self.S0)
        self.nb_samples = hyperparams["nb_samples"]
        self.verbose = verbose
        self.training_data_path = training_data_path
        self.idx = idx
        if idx is None:
            self.idx = np.arange(self.nb_samples)
        if "seed" in hyperparams:
            self.seed = hyperparams["seed"]
            np.random.seed(self.seed)

        if load:
            df_overview, _, _ = get_dataset_overview(self.training_data_path)
            desc = json.dumps(hyperparams, sort_keys=True)
            df_overview = df_overview.loc[df_overview["description"]==desc]
        if len(df_overview) > 0:
            data_id, descr = df_overview.values[-1]
            hyperparams = json.loads(descr)
            self.hyperparams = hyperparams
            self.data_id = data_id
            self.load_dataset()
        else:
            load = False

        if not load:
            if self.verbose > 0:
                print("generate dataset ...")
            self.increments = self.generate_stochastic_increments()
            self.save_dataset()

        self.increments = self.increments[self.idx]

    def generate_stochastic_increments(self, *args, **kwargs):
        random_numbers = np.random.normal(
            0, 1, (self.nb_samples, self.dimension, self.nb_steps))
        dW = random_numbers * np.sqrt(self.dt)
        if self.init_distribution is not None:
            dist = eval("stats."+self.init_distribution["name"])
            self.init_vals = dist.rvs(
                **self.init_distribution["params"], size=self.nb_samples)
        return dW

    def save_dataset(self):
        df_overview, data_overview, columns = get_dataset_overview(
            self.training_data_path)
        data_id = int(time.time())
        self.data_id = data_id
        path = '{}{}/'.format(self.training_data_path, data_id)
        if os.path.exists(path):
            print('id already exists -> abort')
            raise ValueError
        desc = json.dumps(self.hyperparams, sort_keys=True)
        df_app = pd.DataFrame(data=[[data_id, desc]], columns=columns)
        df_overview = pd.concat([df_overview, df_app], ignore_index=True)
        makedirs(path)
        with open('{}data.npy'.format(path), 'wb') as f:
            np.save(f, self.increments)
            if self.init_distribution is not None:
                np.save(f, self.init_vals)
        df_overview.to_csv(data_overview)
        if self.verbose > 0:
            print("generated & saved dataset with id={}!".format(self.data_id))

    def load_dataset(self):
        path = '{}{}/'.format(training_data_path, int(self.data_id))
        with open('{}data.npy'.format(path), 'rb') as f:
            self.increments = np.load(f)
            if self.init_distribution is not None:
                self.init_vals = np.load(f)
        if self.verbose > 0:
            print("loaded dataset with id={}!".format(self.data_id))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        if self.init_distribution is not None:
            return {"idx": idx, "increments": self.increments[idx],
                    "init_vals": self.init_vals[idx]}
        return {"idx": idx, "increments": self.increments[idx]}


def custom_collate_fn(batch):
    """
    function used in torch.DataLoader to construct the custom training batch out
    of the batch of data (non-standard transformations can be applied here)
    :param batch: the input batch (as returned by StockModelData)
    :return: a batch (as dict) as needed for the training in train.train
    """
    idx = np.concatenate([b['idx'] for b in batch], axis=0)
    increments = torch.tensor(
        np.concatenate([b["increments"] for b in batch], axis=0),
        dtype=torch.float32)
    if "init_vals" in batch[0]:
        init_vals = torch.tensor(
            np.concatenate([b["init_vals"] for b in batch], axis=0),
            dtype=torch.float32)
        res = {'idx': idx, 'increments': increments, "init_vals": init_vals}
    else:
        # shape of path batches: [batch_size, dimension, time_steps]
        res = {'idx': idx, 'increments': increments}
    return res


def main(arg):
    """
    function to run data generation with flags from command line
    """
    del arg
    data_gen_dict = eval("config."+FLAGS.data_gen_dict)
    t = time.time()
    datP = SDEIncrements(**data_gen_dict, load=True, verbose=1)
    print("\ntime for loading/generation: {}".format(time.time() - t))


# ==============================================================================
if __name__ == '__main__':
    app.run(main)

