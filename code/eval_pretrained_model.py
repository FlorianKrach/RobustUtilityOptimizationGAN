"""
author: Florian Krach

code to evaluate pretrained models against noisy sigmas
"""


# ==============================================================================
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy.stats
import os, sys
import pandas as pd
import json
import time
import socket
import matplotlib
import matplotlib.colors
import gc
import warnings
from joblib import Parallel, delayed
import tracemalloc

import data_utils
import models
import config
import extras

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = config.SendBotMessage()


# ==============================================================================
USE_GPU = False
ANOMALY_DETECTION = False
DATA_NB_JOBS = 4

# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    SEND = False
    NB_CPUS = 1
else:
    SERVER = True
    SEND = True
    NB_CPUS = 2
    matplotlib.use('Agg')
print(socket.gethostname())
print('SERVER={}'.format(SERVER))

import matplotlib.pyplot as plt
import matplotlib.colors


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = config.saved_models_path
flagfile = config.flagfile



# ==============================================================================
# Functions
makedirs = data_utils.makedirs


def plot_model_paths(
        path_idx, t, S, X, r, mus, Sigmas, pis,
        plot_path, model_id, which, postfix="", discount=False,
        analytic_pi=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fnames = []
    if Sigmas is not None:
        dsq = Sigmas.shape[2]
        d = int(np.sqrt(dsq))
    if discount:
        D = np.exp(r * t)
    else:
        D = 1.

    for i in path_idx:
        fig, axs = plt.subplots(2, 2)

        # plot X and S
        axs[0, 0].plot(t, np.exp(r * t)/D, label="$S^0$")
        for j in range(S.shape[2]):
            axs[0, 0].plot(t, S[i, :, j]/D, label="$S^{" + f"{j + 1}" + "}$")
        axs[0, 0].plot(t, X[i, :, 0]/D, label="PV")
        axs[0, 0].set_title("Stocks and PV")
        axs[0, 0].legend()

        # plot mus
        axs[0, 1].plot(
            t[:-1], np.array([r for i in t[:-1]]), label="$r$")
        if mus is not None:
            for j in range(mus.shape[2]):
                axs[0, 1].plot(
                    t[:-1], mus[i, :, j], label="$\mu^{" + f"{j + 1}" + "}$")
        axs[0, 1].set_title("Drift")
        axs[0, 1].legend()

        # plot sigmas
        if Sigmas is not None:
            for j in range(dsq):
                jr = j // d
                jc = j % d
                if jc >= jr:
                    axs[1, 0].plot(
                        t[:-1], Sigmas[i, :, j],
                        label="$\Sigma^{" + f"{jr + 1},{jc + 1}" + "}$")
            axs[1, 0].set_title("Covariance")
            axs[1, 0].legend()
            axs[1, 0].set_xlabel("t")

        # plot pis
        axs[1, 1].plot(
            t[:-1], 1 - np.sum(pis[i, :, :], axis=1), label="$\pi^0$",
            color=colors[0])
        for j in range(pis.shape[2]):
            axs[1, 1].plot(
                t[:-1], pis[i, :, j],
                label="$\pi^{" + f"{j + 1}" + "}$", color=colors[j + 1])
        if analytic_pi is not None:
            axs[1, 1].plot(
                t[:-1], 1 - np.sum(analytic_pi[:, :], axis=1), color=colors[0],
                label="Analytic $\pi^0$", linestyle=":")
            for j in range(pis.shape[2]):
                axs[1, 1].plot(
                    t[:-1], analytic_pi[:, j],
                    label="Analytic $\pi^{" + f"{j + 1}" + "}$", linestyle=":",
                    color=colors[j + 1])
        axs[1, 1].set_title("Trading Strategy")
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("t")

        fname = "{}paths_id{}_{}_path{}_{}.pdf".format(
            plot_path, model_id, which, i, postfix)
        fnames.append(fname)
        # files_to_send.append(fname)
        plt.tight_layout()
        fig.savefig(fname)
        plt.close()

    return fnames


def plot_hist(values, plot_path, model_id, which, postfix=""):

    fnames = []

    fig = plt.figure()
    plt.hist(values, density=True, bins=100)
    plt.title("Final PV (vs. baseline measure) - mean={:.3f}, std={:.3f}".format(
        np.mean(values), np.std(values)))

    fname = "{}histogram_id{}_{}_{}.pdf".format(
        plot_path, model_id, which, postfix)
    fnames.append(fname)
    plt.tight_layout()
    fig.savefig(fname)
    plt.close()

    return fnames



def errorbarplot(ax, t, mean, yerr, label, color, std_color_alpha=0.3,
                 type="fill"):
    """
    :param ax: axes object
    :param t: time/x-axis values
    :param mean: mean values y-axis
    :param yerr: error y-axis
    :param label:
    :param color:
    :param std_color_alpha: float, the alpha of the color
    :param type: one of {"fill", "bar"}
    :return:
    """

    if type == "bar":
        ax.errorbar(t, mean, yerr=yerr, label=label, color=color)
    else:
        std_color = list(matplotlib.colors.to_rgb(color)) + [std_color_alpha]
        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - yerr, mean + yerr, color=std_color)


def plot_avg_model_path(
        t, S, X, r, mus, Sigmas, pis,
        plot_path, model_id, which, postfix="",
        errorbar_type="fill", std_color_alpha=0.3, discount=False):
    fig, axs = plt.subplots(2, 2)
    fnames = []
    if Sigmas is not None:
        dsq = Sigmas.shape[2]
        d = int(np.sqrt(dsq))
    if discount:
        D = np.exp(r * t)
    else:
        D = 1.

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']

    # plot X and S
    axs[0, 0].plot(t, np.exp(r * t)/D, label="$S^0$", color=colors[0])
    for j in range(S.shape[2]):
        errorbarplot(
            axs[0, 0], t, np.mean(S[:, :, j], axis=0)/D,
            yerr=np.std(S[:, :, j], axis=0)/D,
            label="$S^{" + f"{j + 1}" + "}$",
            color=colors[j+1], std_color_alpha=std_color_alpha,
            type=errorbar_type)
    errorbarplot(
        axs[0, 0], t, np.mean(X[:, :, 0], axis=0)/D,
        yerr=np.std(X[:, :, 0], axis=0)/D, label="PV",
        color=colors[S.shape[2]+1], std_color_alpha=std_color_alpha,
        type=errorbar_type)
    axs[0, 0].set_title("Stocks and PV")
    axs[0, 0].legend()

    # plot mus
    axs[0, 1].plot(
        t[:-1], np.array([r for i in t[:-1]]), label="$r$", color=colors[0])
    if mus is not None:
        for j in range(mus.shape[2]):
            errorbarplot(
                axs[0, 1], t[:-1], np.mean(mus[:, :, j], axis=0),
                yerr=np.std(mus[:, :, j], axis=0),
                label="$\mu^{" + f"{j + 1}" + "}$",
                color=colors[j+1], std_color_alpha=std_color_alpha,
                type=errorbar_type)
    axs[0, 1].set_title("Drift")
    axs[0, 1].legend()

    # plot sigmas
    if Sigmas is not None:
        count = 0
        for j in range(dsq):
            jr = j // d
            jc = j % d
            if jc >= jr:
                errorbarplot(
                    axs[1, 0], t[:-1], np.mean(Sigmas[:, :, j], axis=0),
                    yerr=np.std(Sigmas[:, :, j], axis=0),
                    label="$\Sigma^{" + f"{jr + 1},{jc + 1}" + "}$",
                    color=colors[count], std_color_alpha=std_color_alpha,
                    type=errorbar_type)
                count += 1
        axs[1, 0].set_title("Covariance")
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("t")

    # plot pis
    errorbarplot(
        axs[1, 1], t[:-1], np.mean(1 - np.sum(pis[:, :, :], axis=2), axis=0),
        yerr=np.std(1 - np.sum(pis[:, :, :], axis=2), axis=0),
        label="$\pi^0$", color=colors[0], std_color_alpha=std_color_alpha,
        type=errorbar_type)
    for j in range(pis.shape[2]):
        errorbarplot(
            axs[1, 1], t[:-1], np.mean(pis[:, :, j], axis=0),
            yerr=np.std(pis[:, :, j], axis=0),
            label="$\pi^{" + f"{j + 1}" + "}$",
            color=colors[j+1], std_color_alpha=std_color_alpha,
            type=errorbar_type)
    axs[1, 1].set_title("Trading Strategy")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("t")

    fname = "{}paths_id{}_{}_average_{}.pdf".format(
        plot_path, model_id, which, postfix)
    fnames.append(fname)
    plt.tight_layout()
    fig.savefig(fname)
    plt.close()

    return fnames


def plot_strategy_vs_refstrategy(
        path_idx, S, X, pis, X_ref, pis_ref, pis_ref_ntc, Delta_pis_ref, t,
        plot_path, model_id, plot_ref_nb_stocks_NN=False):
    """
    :param path_idx: list of int, the indices of the paths to plot
    :param S: np.array, shape (nb_paths, nb_steps, dimension), the stock paths
    :param X: np.array, shape (nb_paths, nb_steps), the PV paths
    :param pis: np.array, shape (nb_paths, nb_steps, dimension), the model's
            trading strategy
    :param X_ref: np.array, shape (nb_paths, nb_steps), the PV paths of the
            reference strategy
    :param pis_ref: np.array, shape (nb_paths, nb_steps, dimension), the
            reference strategy
    :param pis_ref_ntc: np.array, shape (nb_paths, nb_steps, dimension), the
            reference strategy without transaction costs
    :param Delta_pis_ref: np.array, shape (nb_paths, nb_steps, dimension), the
            halfwidth of the no trade zone of the reference strategy
    :param t: np.array, shape (nb_steps,), the time points
    :param plot_path: str, the path to save the plots
    :param model_id: int, the id of the model
    :param plot_ref_nb_stocks_NN: bool, whether to plot the number of stocks
            of the reference strategy with the current wealth of the NN model
    :return:
    """

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fnames = []
    for i in path_idx:
        fig, axs = plt.subplots(2, 1, sharex=True)

        # plot proportional trading strategies
        errorbarplot(
            axs[0], t[:-1], pis_ref_ntc[i, :, 0], yerr=Delta_pis_ref[i, :, 0],
            label="$\\pi^{\\operatorname{ref}}_{\\operatorname{ntc}} \\pm "
                  "c^{1/3} \\Delta "
                  "\\pi^{\\operatorname{ref}}_{\\operatorname{ntc}}$",
            color=colors[0])
        axs[0].plot(
            t[:-1], pis_ref[i, :, 0], label="$\\pi^{\\operatorname{ref}}$",
            color=colors[1])
        axs[0].plot(
            t[:-1], pis[i, :, 0], label="$\\pi^{\\operatorname{NN}}$",
            color=colors[2])
        axs[0].set_ylabel("$\\pi$")
        axs[0].set_title("Proportional trading strategy")

        # plot absolute trading strategies
        nb_stocks_model = pis[i, :, 0] * X[i, :-1, 0] / S[i, :-1, 0]
        nb_stocks_ref = pis_ref[i, :, 0] * X_ref[i, :-1, 0] / S[i, :-1, 0]
        nb_stocks_ref_ntc = \
            pis_ref_ntc[i, :, 0] * X_ref[i, :-1, 0] / S[i, :-1, 0]
        nb_stocks_Delta_ref = \
            Delta_pis_ref[i, :, 0] * X_ref[i, :-1, 0] / S[i, :-1, 0]

        # get ref nb stocks with no trading costs given that current wealth is X
        nb_stocks_ref_ntc_X = \
            pis_ref_ntc[i, :, 0] * X[i, :-1, 0] / S[i, :-1, 0]
        nb_stocks_Delta_ref_X = \
            Delta_pis_ref[i, :, 0] * X[i, :-1, 0] / S[i, :-1, 0]
        errorbarplot(
            axs[1], t[:-1], nb_stocks_ref_ntc, yerr=nb_stocks_Delta_ref,
            color=colors[0], label=None)
        if plot_ref_nb_stocks_NN:
            errorbarplot(
                axs[1], t[:-1], nb_stocks_ref_ntc_X, yerr=nb_stocks_Delta_ref_X,
                color="black", label="ref. #stocks & no-trade with NN wealth")
        axs[1].plot(t[:-1], nb_stocks_ref, color=colors[1])
        axs[1].plot(t[:-1], nb_stocks_model, color=colors[2])
        axs[1].set_ylabel("# risky stocks")
        axs[1].set_title("Absolute trading strategy")
        axs[1].set_xlabel("t")

        axs[0].legend(
            bbox_to_anchor=(1.04, -0.1), loc="center left", borderaxespad=0)
        fname = "{}id{}_strategy_vs_refstrategy_path{}.pdf".format(
            plot_path, model_id, i)
        fnames.append(fname)
        fig.savefig(fname, bbox_inches="tight")

    return fnames


def load_model(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None,
        model_id=None, seed=364, load_best=False,
        saved_models_path=saved_models_path,):
    global ANOMALY_DETECTION, USE_GPU, SEND, N_CPUS, N_DATASET_WORKERS
    if anomaly_detection is not None:
        ANOMALY_DETECTION = anomaly_detection
    if use_gpu is not None:
        USE_GPU = use_gpu
    if send is not None:
        SEND = send
    if nb_cpus is not None:
        NB_CPUS = nb_cpus
    if n_dataset_workers is not None:
        N_DATASET_WORKERS = n_dataset_workers
    initial_print = "model-id: {}\n".format(model_id)

    if ANOMALY_DETECTION:
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)

    # set number of CPUs
    print('nb CPUs: {}'.format(NB_CPUS))
    torch.set_num_threads(NB_CPUS)

    # get the device for torch
    if USE_GPU and torch.cuda.is_available():
        gpu_num = 0
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        initial_print += '\nusing GPU'
    else:
        device = torch.device("cpu")
        initial_print += '\nusing CPU'

    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    model_overview_file_name = '{}model_overview.csv'.format(
        saved_models_path)

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # load model desc
    df_overview = pd.read_csv(model_overview_file_name, index_col=0)
    desc = (df_overview['description'].loc[
        df_overview['id'] == model_id]).values[0]
    params_dict = json.loads(desc)

    # get datasets and dataloaders
    data_dict = params_dict["data_dict"]
    if isinstance(data_dict, str):
        data_dict = eval("config.{}".format(data_dict))
    test_size = params_dict["test_size"]
    if "options" in params_dict:
        options = params_dict["options"]
    else:
        options = params_dict
    eval_on_train = False
    if "eval_on_train" in options:
        eval_on_train = options["eval_on_train"]
    train_idx, val_idx = train_test_split(
        np.arange(data_dict["nb_samples"]), test_size=test_size,
        random_state=seed)
    data_train = data_utils.SDEIncrements(
        load=True, verbose=1, idx=train_idx, **data_dict)
    data_val = data_utils.SDEIncrements(
        load=True, verbose=1, idx=val_idx, **data_dict)
    if eval_on_train:
        data_val = data_train
    dl_val = DataLoader(
        dataset=data_val, collate_fn=data_utils.custom_collate_fn,
        shuffle=False, batch_size=len(data_val),
        num_workers=N_DATASET_WORKERS)

    dimension = data_train.dimension
    dt = data_train.dt
    r = data_train.r
    S0 = data_train.S0
    nb_steps = data_train.nb_steps

    # get model and optimizer
    opt_steps_D_G = params_dict["opt_steps_D_G"]
    penalty_scaling_factor = params_dict["penalty_scaling_factor"]
    utility_func = params_dict["utility_func"]
    penalty_func = params_dict["penalty_func"]
    penalty_function_ref_value = params_dict["penalty_function_ref_value"]
    gen_dict = params_dict["gen_dict"]
    disc_dict = params_dict["disc_dict"]
    learning_rate_D = params_dict["learning_rate_D"]
    beta1_D = params_dict["beta1_D"]
    beta2_D = params_dict["beta2_D"]
    lr_scheduler_D = params_dict["lr_scheduler_D"] if "lr_scheduler_D" in \
                                                      params_dict else None
    learning_rate_G = params_dict["learning_rate_G"]
    lr_scheduler_G = params_dict["lr_scheduler_G"] if "lr_scheduler_G" in \
                                                      params_dict else None
    beta1_G = params_dict["beta1_G"]
    beta2_G = params_dict["beta2_G"]
    path_wise_penalty = None
    if "path_wise_penalty" in params_dict:
        path_wise_penalty = params_dict["path_wise_penalty"]
    penalty_scaling_factor_drift = None
    penalty_func_drift = None
    penalty_function_ref_value_drift = None
    if "penalty_scaling_factor_drift" in options:
        penalty_scaling_factor_drift = \
            options["penalty_scaling_factor_drift"]
    if "penalty_func_drift" in options:
        penalty_func_drift = options["penalty_func_drift"]
    if "penalty_function_ref_value_drift" in options:
        penalty_function_ref_value_drift = \
            options["penalty_function_ref_value_drift"]
    initial_wealth = 1
    if "initial_wealth" in options:
        initial_wealth = options["initial_wealth"]
    use_penalty_for_gen = False
    if "use_penalty_for_gen" in options:
        use_penalty_for_gen = options["use_penalty_for_gen"]
    use_log_return = False
    if "use_log_return" in options:
        use_log_return = options["use_log_return"]
    use_opponents_output = False
    if "use_opponents_output" in options:
        use_opponents_output = options["use_opponents_output"]
    if opt_steps_D_G is None:
        opt_steps_D_G = [1, 1]
    trans_cost_base = 0.
    trans_cost_perc = 0.
    if "trans_cost_perc" in options:
        trans_cost_perc = options["trans_cost_perc"]
    if "trans_cost_base" in options:
        trans_cost_base = options["trans_cost_base"]
    numerically_stabilised_training = False
    if 'numerically_stabilised_training' in options:
        numerically_stabilised_training = \
            options["numerically_stabilised_training"]
    input_increments = False
    if 'input_increments' in options:
        input_increments = options['input_increments']
    use_general_SDE = False
    if 'use_general_SDE' in options:
        use_general_SDE = options['use_general_SDE']
    weight_decay = 0
    if 'weight_decay' in options:
        weight_decay = options['weight_decay']
    train_readout_only = False
    if "train_readout_only" in options:
        train_readout_only = options["train_readout_only"]
    ref_strategy = None
    if "ref_strategy" in options:
        ref_strategy = eval(options["ref_strategy"])

    model = models.MinMaxModel(
        dimension=dimension, scaling_coeff=penalty_scaling_factor,
        utility_func=utility_func, penalty_func=penalty_func,
        penalty_function_ref_value=penalty_function_ref_value,
        gen_dict=gen_dict, disc_dict=disc_dict,
        dt=dt, r=r, S0=S0, nb_steps=nb_steps,
        initial_wealth=initial_wealth,
        use_penalty_for_gen=use_penalty_for_gen,
        use_log_return=use_log_return,
        use_opponents_output=use_opponents_output,
        trans_cost_base=trans_cost_base, trans_cost_perc=trans_cost_perc,
        penalty_func_drift=penalty_func_drift,
        penalty_function_ref_value_drift=penalty_function_ref_value_drift,
        scaling_coeff_drift=penalty_scaling_factor_drift,
        path_wise_penalty=path_wise_penalty,
        numerically_stabilised_training=numerically_stabilised_training,
        input_increments=input_increments, use_general_SDE=use_general_SDE)
    scheduler_D, scheduler_G = None, None
    optimizer_D = None
    if model.discriminator is not None:
        optimizer_D = torch.optim.Adam(
            model.discriminator.parameters(), lr=learning_rate_D,
            betas=(beta1_D, beta2_D))
        if lr_scheduler_D:
            scheduler_D = torch.optim.lr_scheduler.StepLR(
                optimizer_D, step_size=lr_scheduler_D["step"],
                gamma=lr_scheduler_D["gamma"])
    if train_readout_only:
        optimizer_G = torch.optim.Adam(
            model.generator.readout.parameters(), lr=learning_rate_G,
            betas=(beta1_G, beta2_G), weight_decay=weight_decay)
    else:
        optimizer_G = torch.optim.Adam(
            model.generator.parameters(), lr=learning_rate_G,
            betas=(beta1_G, beta2_G), weight_decay=weight_decay)
    if lr_scheduler_G:
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            optimizer_G, step_size=lr_scheduler_G["step"],
            gamma=lr_scheduler_G["gamma"])
    try:
        if load_best:
            models.get_ckpt_model(
                model_path_save_best, model, [optimizer_G, optimizer_D],
                [scheduler_G, scheduler_D], device
            )
        else:
            models.get_ckpt_model(
                model_path_save_last, model, [optimizer_G, optimizer_D],
                [scheduler_G, scheduler_D], device
            )
    except Exception as e:
        print("There was a problem loading the model:\n", e)
        return None

    analytic_sig, analytic_mu, analytic_pi = model.analytic_sol

    return model, dl_val, model_path, penalty_function_ref_value, dimension, \
           r, dt, disc_dict, nb_steps, penalty_function_ref_value_drift, \
           utility_func, path_wise_penalty, analytic_pi, data_train, \
           ref_strategy



def evaluate(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None,
        model_id=None, seed=364, load_best=False,
        evaluate_vs_analytic=False,
        nb_evaluations=1, noise_std=1., noise_std_drift=None,
        noise_type="const", noisy_eval=True,
        saved_models_path=saved_models_path,
        load_saved_eval=True, plot_gen_disc_paths=None,
        plot_noisy_eval_paths=None, discount=False,
        plot_strategy_refstrategy=None, plot_ref_nb_stocks_NN=False,
        **kwargs
):
    """
    :param anomaly_detection:
    :param n_dataset_workers:
    :param use_gpu:
    :param nb_cpus:
    :param send:
    :param model_id:
    :param seed:
    :param load_best:
    :param evaluate_vs_analytic: bool, whether to evaluate the model vs the
            analytic optimal market or against noisy market. default: False
    :param nb_evaluations:
    :param noise_std:
    :param noise_std_drift:
    :param noise_type:
    :param saved_models_path:
    :param noisy_eval: bool, whether to do the noisy evaluation of the model
    :param plot_gen_disc_paths: None or list of int, if not None: plots
            evolution of the indicated batch samples of S,X,mu,sigma,pi
    :param plot_noisy_eval_paths: None or list of int, if not None: plots
            evolution of the indicated batch samples (under indicated noisy
            params) of S,X,mu,sigma,pi; in particular, mu and Sigma are chosen
            randomly from given noise type
    :param load_saved_eval:
    :param discount: bool, whether to discount the stock vals and PV
    :param plot_strategy_refstrategy: None or list of int, if not None: plots
            the strategy of the model vs the reference strategy for the
            indicated batch samples
    :param plot_ref_nb_stocks_NN: bool, whether to plot the number of stocks
            of the reference strategy with the current wealth of the NN model
            in case of plot_strategy_refstrategy
    :param kwargs:

    :return:
    if the model has a ref_strategy and no analytic_pi, then the results of the
    ref_strategy are also saved in the place of analytic_exp_util_w_ref array
    and in analytic_pi_min_exp_util_with_noisy_par in the eval file.
    """
    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    which = "best" if load_best else "last"
    eval_file = "{}evaluation-{}-{}-{}_id-{}.{}.csv".format(
        model_path, nb_evaluations, noise_std, noise_std_drift, model_id, which)

    # load the model
    load = load_model(
        anomaly_detection=anomaly_detection,
        n_dataset_workers=n_dataset_workers, use_gpu=use_gpu,
        nb_cpus=nb_cpus, send=send,
        model_id=model_id, seed=seed, load_best=load_best,
        saved_models_path=saved_models_path, )
    if load is None:
        return [model_id, noise_std, noise_std_drift, ] + [None] * 13
    model, dl_val, model_path, penalty_function_ref_value, dimension, r, \
    dt, disc_dict, nb_steps, penalty_function_ref_value_drift, \
    utility_func, path_wise_penalty, analytic_pi, data_train, \
    ref_strategy = load

    if os.path.exists(eval_file) and load_saved_eval and noisy_eval:
        df = pd.read_csv(eval_file, index_col=0)
        exp_util_w_ref_arr = df['exp_util_w_ref']
        analytic_exp_util_w_ref_arr = df['analytic_exp_util_w_ref']
        oracle_exp_util_w_ref_arr = df['oracle_exp_util_w_ref']
    elif noisy_eval or evaluate_vs_analytic \
            or plot_noisy_eval_paths is not None \
            or plot_strategy_refstrategy is not None:
        np.random.seed(seed)
        exp_util_w_ref_arr = []
        analytic_exp_util_w_ref_arr = []
        oracle_exp_util_w_ref_arr = []
        S = []
        X = []
        mus = []
        Sigmas = []
        pis = []
        X_ref = []
        pis_ref = []
        Delta_pis_ref = []
        pis_ref_ntc = []
        seeds = np.random.choice(1000000, size=nb_evaluations, replace=False)
        for i in tqdm.tqdm(range(nb_evaluations)):
            # get reference solution
            if evaluate_vs_analytic:
                analytic_sig, analytic_mu, analytic_pi = model.analytic_sol
                ref_sig = analytic_sig.reshape((1,-1)).repeat(
                    repeats=nb_steps, axis=0)
                ref_mu = analytic_mu.reshape((1,-1)).repeat(
                    repeats=nb_steps, axis=0)
                ref_pi = analytic_pi.reshape((1,-1)).repeat(
                    repeats=nb_steps, axis=0)
                oracle_pi = None
            else:
                ref_sig, oracle_pi, ref_mu = extras.get_oracle_solution(
                    penalty_function_ref_value=penalty_function_ref_value,
                    dimension=dimension, r=r, mu=disc_dict["drift"],
                    noise_std=noise_std, noise_std_drift=noise_std_drift,
                    noise_seed=seeds[i], noise_type=noise_type, nb_steps=nb_steps,
                    penalty_function_ref_value_drift=penalty_function_ref_value_drift,
                    utility_function=utility_func,
                    path_wise_penalty=path_wise_penalty)
                ref_pi = None

            with torch.no_grad():
                model.eval()  # set model in evaluation mode
                count = 0
                expected_util_with_ref = 0
                analytic_expected_util_with_ref = 0
                oracle_expected_util_with_ref = 0
                penalty_vs_ref = 0
                penalty_drift_vs_ref = 0
                for j, b in enumerate(dl_val):
                    dWs = b["increments"]
                    exp_util2_, penalty2_, penalty_drift2_, pwps_, exp_util_a, \
                    exp_util_o, _S, _X, _mus, _Sigmas, _pis = model.evaluate(
                        dWs, ref_params=ref_sig, analytic_pi=analytic_pi,
                        oracle_pi=oracle_pi, ref_params_drift=ref_mu,
                        return_paths=True)
                    if analytic_pi is None and ref_strategy is not None:
                        # safe the exp util of the reference strategy in the
                        #   analytic_exp_util_w_ref_arr
                        res = model.evaluate(
                            dWs, ref_params=ref_sig, analytic_pi=None,
                            oracle_pi=None, ref_params_drift=ref_mu,
                            return_paths=True, ref_strategy=ref_strategy,
                            return_Delta_pi=bool(plot_strategy_refstrategy))
                        if plot_strategy_refstrategy is not None:
                            exp_util_a, _, _, _, _, _, _, _X_ref, _, _, \
                            _pis_ref, _pis_ntc, _Delta_pis_ref = res
                        else:
                            exp_util_a = res[0]
                    if j == 0 and plot_noisy_eval_paths is not None \
                            or plot_strategy_refstrategy is not None:
                        bs = dWs.shape[0]
                        idx = i % bs
                        if nb_evaluations == 1:
                            S.append(_S)
                            X.append(_X)
                            mus.append(_mus)
                            Sigmas.append(_Sigmas)
                            pis.append(_pis)
                            if plot_strategy_refstrategy is not None:
                                X_ref.append(_X_ref)
                                pis_ref.append(_pis_ref)
                                Delta_pis_ref.append(_Delta_pis_ref)
                                pis_ref_ntc.append(_pis_ntc)
                        else:
                            S.append(_S[idx:idx+1])
                            X.append(_X[idx:idx+1])
                            mus.append(_mus[idx:idx+1])
                            Sigmas.append(_Sigmas[idx:idx+1])
                            pis.append(_pis[idx:idx+1])
                            if plot_strategy_refstrategy is not None:
                                X_ref.append(_X_ref[idx:idx+1])
                                pis_ref.append(_pis_ref[idx:idx+1])
                                Delta_pis_ref.append(_Delta_pis_ref[idx:idx+1])
                                pis_ref_ntc.append(_pis_ntc[idx:idx+1])
                    expected_util_with_ref += exp_util2_.detach().numpy()
                    if exp_util_a is not None:
                        analytic_expected_util_with_ref += \
                            exp_util_a.detach().numpy()
                    if exp_util_o is not None:
                        oracle_expected_util_with_ref += \
                            exp_util_o.detach().numpy()
                    penalty_vs_ref += penalty2_.detach().numpy()
                    penalty_drift_vs_ref += penalty_drift2_.detach().numpy()
                    count += 1
                expected_util_with_ref /= count
                analytic_expected_util_with_ref /= count
                oracle_expected_util_with_ref /= count
                exp_util_w_ref_arr.append(expected_util_with_ref)
                analytic_exp_util_w_ref_arr.append(analytic_expected_util_with_ref)
                oracle_exp_util_w_ref_arr.append(oracle_expected_util_with_ref)
        df = pd.DataFrame(data={
            'exp_util_w_ref': exp_util_w_ref_arr,
            'analytic_exp_util_w_ref': analytic_exp_util_w_ref_arr,
            'oracle_exp_util_w_ref': oracle_exp_util_w_ref_arr})
        if load_saved_eval:
            df.to_csv(eval_file)

        # --------- plotting of noisy (or analytic) eval paths ---------
        if plot_noisy_eval_paths is not None:
            S = torch.cat(S, dim=0).detach().numpy()
            X = torch.cat(X, dim=0).detach().numpy()
            mus = torch.cat(mus, dim=0).detach().numpy()
            Sigmas = torch.cat(Sigmas, dim=0).detach().numpy()
            pis = torch.cat(pis, dim=0).detach().numpy()
            t = np.linspace(0, dt * nb_steps, nb_steps + 1)

            plot_path = "{}plots/".format(model_path)
            makedirs(plot_path)

            # plot individual paths of model agent vs noisy market
            postfix = "noisyeval" if not evaluate_vs_analytic \
                else "analyticeval"
            files_to_send = plot_model_paths(
                plot_noisy_eval_paths, t, S, X, r, mus, Sigmas, pis, plot_path,
                model_id, which, postfix=postfix, discount=discount,
                analytic_pi=ref_pi)

            # average plots
            files_to_send += plot_avg_model_path(
                t, S, X, r, mus, Sigmas, pis,
                plot_path, model_id, which, postfix=postfix,
                discount=discount)

            if send:
                SBM.send_notification(
                    text=None,
                    files=files_to_send,
                    chat_id=CHAT_ID)
                time.sleep(2)

        # --------- plotting of strategy vs ref_strategy ---------
        if plot_strategy_refstrategy is not None:
            if plot_noisy_eval_paths is None:
                S = torch.cat(S, dim=0).detach().numpy()
                X = torch.cat(X, dim=0).detach().numpy()
                pis = torch.cat(pis, dim=0).detach().numpy()
            X_ref = torch.cat(X_ref, dim=0).detach().numpy()
            pis_ref = torch.cat(pis_ref, dim=0).detach().numpy()
            Delta_pis_ref = torch.cat(Delta_pis_ref, dim=0).detach().numpy()
            pis_ref_ntc = torch.cat(pis_ref_ntc, dim=0).detach().numpy()
            t = np.linspace(0, dt * nb_steps, nb_steps + 1)

            plot_path = "{}plots/".format(model_path)
            makedirs(plot_path)

            files_to_send = plot_strategy_vs_refstrategy(
                plot_strategy_refstrategy, S, X, pis,
                X_ref, pis_ref, pis_ref_ntc,
                Delta_pis_ref, t, plot_path, model_id,
                plot_ref_nb_stocks_NN=plot_ref_nb_stocks_NN)

            if send:
                SBM.send_notification(
                    text=None,
                    files=files_to_send,
                    chat_id=CHAT_ID)
                time.sleep(2)

    if noisy_eval:
        which_nan = np.isnan(exp_util_w_ref_arr)
        amount_nans = np.sum(which_nan)
        if amount_nans > 0:
            print("model_id: {}, amount NANs in evaluation: {}".format(
                model_id, amount_nans))
        mean_exp_util_w_ref = np.nanmean(exp_util_w_ref_arr)
        median_exp_util_w_ref = np.nanmedian(exp_util_w_ref_arr)
        std_exp_util_w_ref = np.nanstd(exp_util_w_ref_arr)
        CI_diff = std_exp_util_w_ref/np.sqrt(len(exp_util_w_ref_arr)-amount_nans)*\
                  scipy.stats.norm.ppf(0.975)
        min_exp_util_w_ref = np.min(exp_util_w_ref_arr)

        which_nan = np.isnan(analytic_exp_util_w_ref_arr)
        amount_nans = np.sum(which_nan)
        a_mean_exp_util_w_ref = np.nanmean(analytic_exp_util_w_ref_arr)
        a_median_exp_util_w_ref = np.nanmedian(analytic_exp_util_w_ref_arr)
        a_std_exp_util_w_ref = np.nanstd(analytic_exp_util_w_ref_arr)
        a_CI_diff = a_std_exp_util_w_ref/np.sqrt(
            len(analytic_exp_util_w_ref_arr) - amount_nans) * \
                  scipy.stats.norm.ppf(0.975)
        a_min_exp_util_w_ref = np.min(analytic_exp_util_w_ref_arr)

        which_nan = np.isnan(oracle_exp_util_w_ref_arr)
        amount_nans = np.sum(which_nan)
        o_mean_exp_util_w_ref = np.nanmean(oracle_exp_util_w_ref_arr)
        o_median_exp_util_w_ref = np.nanmedian(oracle_exp_util_w_ref_arr)
        o_std_exp_util_w_ref = np.nanstd(oracle_exp_util_w_ref_arr)
        o_CI_diff = o_std_exp_util_w_ref/np.sqrt(
            len(oracle_exp_util_w_ref_arr)-amount_nans) * \
                  scipy.stats.norm.ppf(0.975)
        o_min_exp_util_w_ref = np.min(oracle_exp_util_w_ref_arr)
    else:
        min_exp_util_w_ref, mean_exp_util_w_ref, median_exp_util_w_ref, \
        std_exp_util_w_ref, CI_diff, \
        a_min_exp_util_w_ref, a_mean_exp_util_w_ref, a_median_exp_util_w_ref, \
        a_std_exp_util_w_ref, a_CI_diff, \
        o_min_exp_util_w_ref, o_mean_exp_util_w_ref, o_median_exp_util_w_ref, \
        o_std_exp_util_w_ref, o_CI_diff = [None]*15

    # ----------- plotting of model paths (gen and disc) --------------
    if plot_gen_disc_paths is not None:
        files_to_send = []
        plot_path = "{}plots/".format(model_path)
        makedirs(plot_path)

        # get the paths of the model (agent & critic) on eval samples
        S = []
        X = []
        mus = []
        Sigmas = []
        pis = []
        for i, b in enumerate(dl_val):
            dWs = b["increments"]
            _, _, _, _, _S, _X, _mus, _Sigmas, _pis = model(
                dWs, which="both", return_paths=True)
            S.append(_S)
            X.append(_X)
            mus.append(_mus)
            Sigmas.append(_Sigmas)
            pis.append(_pis)
        S = torch.cat(S, dim=0).detach().numpy()
        X = torch.cat(X, dim=0).detach().numpy()
        mus = torch.cat(mus, dim=0).detach().numpy()
        Sigmas = torch.cat(Sigmas, dim=0).detach().numpy()
        pis = torch.cat(pis, dim=0).detach().numpy()
        t = np.linspace(0, dt * nb_steps, nb_steps + 1)

        # plot individual paths of model (agent & critic)
        files_to_send += plot_model_paths(
            plot_gen_disc_paths, t, S, X, r, mus, Sigmas, pis, plot_path,
            model_id, which, postfix="modeleval", discount=discount)

        # average plots
        files_to_send += plot_avg_model_path(
            t, S, X, r, mus, Sigmas, pis,
            plot_path, model_id, which, postfix="modeleval", discount=discount)

        if send:
            SBM.send_notification(
                text=None,
                files=files_to_send,
                chat_id=CHAT_ID)
            time.sleep(2)

    return model_id, noise_std, noise_std_drift, \
           min_exp_util_w_ref, mean_exp_util_w_ref, median_exp_util_w_ref, \
           std_exp_util_w_ref, CI_diff, \
           a_min_exp_util_w_ref, a_mean_exp_util_w_ref, a_median_exp_util_w_ref, \
           a_std_exp_util_w_ref, a_CI_diff, \
           o_min_exp_util_w_ref, o_mean_exp_util_w_ref, o_median_exp_util_w_ref, \
           o_std_exp_util_w_ref, o_CI_diff


def evaluate_baseline(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None,
        model_id=None, seed=364, load_best=False,
        saved_models_path=saved_models_path,
        plot_eval_paths=None, discount=False,
        **kwargs
):
    """
    :param anomaly_detection:
    :param n_dataset_workers:
    :param use_gpu:
    :param nb_cpus:
    :param send:
    :param model_id:
    :param seed:
    :param load_best:
    :param saved_models_path:
    :param plot_eval_paths: None or list of int, if not None: plots
            evolution of the indicated batch samples (under baseline measure
            params) of S,X,mu,sigma,pi; in particular, mu_0 and Sigma_0 are used
    :param discount: bool, whether to discount stock vals and PV
    :param kwargs:
    :return:
    """
    # load the model
    load = load_model(
        anomaly_detection=anomaly_detection,
        n_dataset_workers=n_dataset_workers, use_gpu=use_gpu,
        nb_cpus=nb_cpus, send=send,
        model_id=model_id, seed=seed, load_best=load_best,
        saved_models_path=saved_models_path, )
    if load is None:
        return 0
    model, dl_val, model_path, penalty_function_ref_value, dimension, r, \
    dt, disc_dict, nb_steps, penalty_function_ref_value_drift, \
    utility_func, path_wise_penalty, analytic_pi, data_train, \
    ref_strategy = load
    which = "best" if load_best else "last"

    np.random.seed(seed)
    S = []
    X = []
    mus = []
    Sigmas = []
    pis = []

    # get reference solution
    ref_sig, oracle_pi, ref_mu = extras.get_oracle_solution(
        penalty_function_ref_value=penalty_function_ref_value,
        dimension=dimension, r=r, mu=disc_dict["drift"],
        noise_std=0, noise_std_drift=0,
        noise_seed=0, noise_type="const", nb_steps=nb_steps,
        penalty_function_ref_value_drift=penalty_function_ref_value_drift,
        utility_function=utility_func,
        path_wise_penalty=path_wise_penalty)

    with torch.no_grad():
        model.eval()  # set model in evaluation mode
        count = 0
        expected_util_with_ref = 0
        analytic_expected_util_with_ref = 0
        oracle_expected_util_with_ref = 0
        penalty_vs_ref = 0
        penalty_drift_vs_ref = 0
        for j, b in enumerate(dl_val):
            dWs = b["increments"]
            exp_util2_, penalty2_, penalty_drift2_, pwps_, exp_util_a, \
            exp_util_o, _S, _X, _mus, _Sigmas, _pis = model.evaluate(
                dWs, ref_params=ref_sig, analytic_pi=analytic_pi,
                oracle_pi=oracle_pi, ref_params_drift=ref_mu,
                return_paths=True)
            S.append(_S)
            X.append(_X)
            mus.append(_mus)
            Sigmas.append(_Sigmas)
            pis.append(_pis)
            expected_util_with_ref += exp_util2_.detach().numpy()
            if exp_util_a is not None:
                analytic_expected_util_with_ref += \
                    exp_util_a.detach().numpy()
            if exp_util_o is not None:
                oracle_expected_util_with_ref += \
                    exp_util_o.detach().numpy()
            penalty_vs_ref += penalty2_.detach().numpy()
            penalty_drift_vs_ref += penalty_drift2_.detach().numpy()
            count += 1
        expected_util_with_ref /= count
        analytic_expected_util_with_ref /= count
        oracle_expected_util_with_ref /= count

    plot_path = "{}plots/".format(model_path)
    makedirs(plot_path)
    files_to_send = []

    S = torch.cat(S, dim=0).detach().numpy()
    X = torch.cat(X, dim=0).detach().numpy()
    mus = torch.cat(mus, dim=0).detach().numpy()
    Sigmas = torch.cat(Sigmas, dim=0).detach().numpy()
    pis = torch.cat(pis, dim=0).detach().numpy()
    t = np.linspace(0, dt * nb_steps, nb_steps + 1)
    # --------- plotting of noisy eval paths ---------
    if plot_eval_paths is not None:
        # plot individual paths of model agent vs noisy market
        files_to_send += plot_model_paths(
            plot_eval_paths, t, S, X, r, mus, Sigmas, pis, plot_path,
            model_id, which, postfix="baselineeval", discount=discount)

        # average plots
        files_to_send += plot_avg_model_path(
            t, S, X, r, mus, Sigmas, pis,
            plot_path, model_id, which, postfix="baselineeval",
            discount=discount)

    # --------- plotting of final PV dist ---------
    if discount:
        D = np.exp(r*t[-1])
    else:
        D = 1.
    files_to_send += plot_hist(
        X[:, -1, 0]/D, plot_path, model_id, which, postfix="baselineeval")

    if send:
        SBM.send_notification(
            text=None,
            files=files_to_send,
            chat_id=CHAT_ID)
        time.sleep(5)

    return model_id, expected_util_with_ref


def evaluate_garch(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None,
        model_id=None, seed=364, load_best=False,
        saved_models_path=saved_models_path,
        discount=False,
        eval_model_dict=None, load_saved_eval=True,
        nb_evaluations=1, nb_samples_per_param=1,
        plot_garch_eval_paths=None,
        **kwargs
):
    """
    :param anomaly_detection:
    :param n_dataset_workers:
    :param use_gpu:
    :param nb_cpus:
    :param send:
    :param model_id:
    :param seed:
    :param load_best:
    :param saved_models_path:
    :param discount:
    :param eval_model_dict:
    :param load_saved_eval:
    :param nb_evaluations:
    :param nb_samples_per_param:
    :param plot_garch_eval_paths:
    :param kwargs:
    :return:
    """
    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    which = "best" if load_best else "last"
    eval_id = eval_model_dict["eval_id"]
    eval_file = "{}evaluation-{}-{}_id-{}.{}.csv".format(
        model_path, nb_evaluations, eval_id, model_id, which)

    # load the model
    load = load_model(
        anomaly_detection=anomaly_detection,
        n_dataset_workers=n_dataset_workers, use_gpu=use_gpu,
        nb_cpus=nb_cpus, send=send,
        model_id=model_id, seed=seed, load_best=load_best,
        saved_models_path=saved_models_path, )
    if load is None:
        return model_id, None
    model, dl_val, model_path, penalty_function_ref_value, dimension, r, \
    dt, disc_dict, nb_steps, penalty_function_ref_value_drift, \
    utility_func, path_wise_penalty, analytic_pi, data_train, \
    ref_strategy = load
    which = "best" if load_best else "last"

    if os.path.exists(eval_file) and load_saved_eval:
        df = pd.read_csv(eval_file, index_col=0)
        exp_util_w_garch_arr = df['exp_util_w_Garch']
    else:
        # get eval model
        eval_model = extras.get_eval_model(
            name=eval_model_dict["name"], params=eval_model_dict["params"],
            data_train=data_train, fit_seq_len=eval_model_dict["fit_seq_len"],
            S0=model.S0, dt=dt, nb_steps=nb_steps,
            penalty_function_ref_value=penalty_function_ref_value,
            penalty_function_ref_value_drift=penalty_function_ref_value_drift)

        np.random.seed(seed)
        exp_util_w_garch_arr = []
        S = []
        X = []
        pis = []
        seeds = np.random.choice(1000000, size=nb_evaluations, replace=False)

        with torch.no_grad():
            model.eval()
            for i in tqdm.tqdm(range(nb_evaluations)):
                exp_util, _X, _S, _pis = model.evaluate_vs_garch(
                    eval_model=eval_model, eval_model_dict=eval_model_dict,
                    nb_rand_params=1, nb_samples_per_param=nb_samples_per_param,
                    seed=seeds[i], return_paths=True)
                if plot_garch_eval_paths is not None:
                    idx = i % nb_samples_per_param
                    S.append(_S[idx:idx + 1])
                    X.append(_X[idx:idx + 1])
                    pis.append(_pis[idx:idx + 1])

                del _S, _X, _pis
                exp_util_w_garch_arr.append(exp_util)

        df = pd.DataFrame(data={'exp_util_w_Garch': exp_util_w_garch_arr})
        if load_saved_eval:
            df.to_csv(eval_file)

        # --------- plotting of noisy eval paths ---------
        if plot_garch_eval_paths is not None:
            S = np.concatenate(S, axis=0)
            X = np.concatenate(X, axis=0)
            pis = np.concatenate(pis, axis=0)
            t = np.linspace(0, dt * nb_steps, nb_steps + 1)

            plot_path = "{}plots/".format(model_path)
            makedirs(plot_path)

            # plot individual paths of model agent vs noisy market
            files_to_send = plot_model_paths(
                plot_garch_eval_paths, t, S, X, r,
                mus=None, Sigmas=None, pis=pis,
                plot_path=plot_path, model_id=model_id, which=which,
                postfix="garcheval", discount=discount)

            # average plots
            files_to_send += plot_avg_model_path(
                t, S, X, r, mus=None, Sigmas=None, pis=pis,
                plot_path=plot_path, model_id=model_id, which=which,
                postfix="garcheval", discount=discount)

            if send:
                SBM.send_notification(
                    text=None,
                    files=files_to_send,
                    chat_id=CHAT_ID)
                time.sleep(2)

    which_nan = np.isnan(exp_util_w_garch_arr)
    amount_nans = np.sum(which_nan)
    if amount_nans > 0:
        print("model_id: {}, amount NANs in evaluation: {}".format(
            model_id, amount_nans))
    # mean_exp_util_w_ref = np.nanmean(exp_util_w_garch_arr)
    # median_exp_util_w_ref = np.nanmedian(exp_util_w_garch_arr)
    # std_exp_util_w_ref = np.nanstd(exp_util_w_garch_arr)
    # CI_diff = std_exp_util_w_ref / np.sqrt(
    #     len(exp_util_w_garch_arr) - amount_nans) * \
    #           scipy.stats.norm.ppf(0.975)
    min_exp_util_w_garch = np.min(exp_util_w_garch_arr)

    return model_id, min_exp_util_w_garch


def evaluate_models(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None, nb_jobs=1,
        model_ids=None, seed=364, load_best=False, noise_std=1.,
        noise_std_drift=None, noise_type="const", noisy_eval=True,
        evaluate_vs_analytic=False,
        nb_evaluations=1, saved_models_path=saved_models_path,
        plot_gen_disc_paths=None, plot_noisy_eval_paths=None,
        load_saved_eval=True, discount=False,
        plot_strategy_refstrategy=None,
        plot_ref_nb_stocks_NN=False,
        filename=None,
        **kwargs):
    which = "best" if load_best else "last"
    time_id = int(time.time())
    if filename is None:
        filename = "{}model_evaluation_{}.csv".format(
            saved_models_path, time_id)
    else:
        filename = "{}{}.csv".format(saved_models_path, filename)
    data = []
    if nb_jobs == 1:
        for model_id in model_ids:
            _, _, _, min, mean, median, std, CI_diff, \
            a_min, a_mean, a_median, a_std, a_CI_diff, \
            o_min, o_mean, o_median, o_std, o_CI_diff = evaluate(
                anomaly_detection, n_dataset_workers, use_gpu, nb_cpus, send,
                model_id, seed, load_best, evaluate_vs_analytic,
                nb_evaluations, noise_std,
                noise_std_drift, noise_type, noisy_eval,
                saved_models_path, load_saved_eval,
                plot_gen_disc_paths, plot_noisy_eval_paths, discount,
                plot_strategy_refstrategy, plot_ref_nb_stocks_NN)
            data.append([model_id, noise_std, noise_std_drift,
                         min, mean, median, std, CI_diff,
                         a_min, a_mean, a_median, a_std, a_CI_diff,
                         o_min, o_mean, o_median, o_std, o_CI_diff])
    else:
        data = Parallel(n_jobs=nb_jobs)(delayed(evaluate)(
            anomaly_detection, n_dataset_workers, use_gpu, nb_cpus, send,
            model_id, seed, load_best, evaluate_vs_analytic,
            nb_evaluations, noise_std,
            noise_std_drift, noise_type, noisy_eval,
            saved_models_path, load_saved_eval,
            plot_gen_disc_paths, plot_noisy_eval_paths, discount,
            plot_strategy_refstrategy, plot_ref_nb_stocks_NN)
                                        for model_id in model_ids)
    df_out = pd.DataFrame(
        data=data,
        columns=["model_id", "noise_std", "noise_std_drift",
                 "min_exp_util_with_noisy_par",
                 "mean_exp_util_with_noisy_par",
                 "median_exp_util_with_noisy_par",
                 "std_exp_util_with_noisy_par",
                 "CI-diff_exp_util_with_noisy_par",
                 ###
                 "analytic_pi_min_exp_util_with_noisy_par",
                 "analytic_pi_mean_exp_util_with_noisy_par",
                 "analytic_pi_median_exp_util_with_noisy_par",
                 "analytic_pi_std_exp_util_with_noisy_par",
                 "analytic_pi_CI-diff_exp_util_with_noisy_par",
                 ###
                 "oracle_pi_min_exp_util_with_noisy_par",
                 "oracle_pi_mean_exp_util_with_noisy_par",
                 "oracle_pi_median_exp_util_with_noisy_par",
                 "oracle_pi_std_exp_util_with_noisy_par",
                 "oracle_pi_CI-diff_exp_util_with_noisy_par"])
    df_out.to_csv(filename)

    if send:
        files_to_send = [filename]
        SBM.send_notification(
            text='finished evaluation models:\n\n{}\nnoise-std: {}, '
                 'noise-std-drift: {}, noise_type: {}'
                 'nb-evaluations: {}, which-model: {}'.format(
                model_ids, noise_std, noise_std_drift, noise_type,
                nb_evaluations, which),
            files=files_to_send,
            chat_id=CHAT_ID)


def evaluate_models_function(
        send=None, nb_jobs=1,
        model_ids=None, load_best=False,
        saved_models_path=saved_models_path,
        function=evaluate_baseline,
        col_name="", filename_default="model_evaluation_baseline",
        filename=None,
        **kwargs):
    which = "best" if load_best else "last"
    time_id = int(time.time())
    if filename is None:
        filename = "{}{}_{}.csv".format(
            saved_models_path, filename_default, time_id)
    else:
        filename = "{}{}.csv".format(saved_models_path, filename)
    data = []
    if nb_jobs == 1:
        for model_id in model_ids:
            _, exp_util = function(
                send=send, model_id=model_id, load_best=load_best,
                saved_models_path=saved_models_path, **kwargs)
            data.append([model_id, exp_util])
    else:
        data = Parallel(n_jobs=nb_jobs)(delayed(function)(
            send=send, model_id=model_id, load_best=load_best,
            saved_models_path=saved_models_path, **kwargs)
                                        for model_id in model_ids)
    df_out = pd.DataFrame(
        data=data,
        columns=["model_id", col_name,])
    df_out.to_csv(filename)

    if send:
        files_to_send = [filename]
        SBM.send_notification(
            text='finished evaluation of models:'
                 '\n\n{}, which-model: {}'.format(
                model_ids, which),
            files=files_to_send,
            chat_id=CHAT_ID)


def evaluate_models_baseline(**kwargs):
    return evaluate_models_function(
        **kwargs, function=evaluate_baseline,
        col_name="exp_util_with_baseline_par",
        filename_default="model_evaluation_baseline",)

def evaluate_models_garch(**kwargs):
    return evaluate_models_function(
        **kwargs, function=evaluate_garch,
        col_name="min_exp_util_with_garch",
        filename_default="model_evaluation_garch",)



def eval_switcher(which):
    if which == "baseline":
        return evaluate_models_baseline
    elif which == "garch":
        return evaluate_models_garch
    else:
        return evaluate_models





if __name__ == '__main__':
    pass
