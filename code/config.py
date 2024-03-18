"""
author: Florian Krach
"""

import numpy as np
import torch
import os
import copy
import pandas as pd
from sklearn.model_selection import ParameterGrid
import scipy.stats as stats
from torch.cuda.amp import custom_bwd, custom_fwd
import socket



if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
CHAT_ID = "-601964650"
ERROR_CHAT_ID = "-705442106"

data_path = 'data/'
training_data_path = '{}training_data/'.format(data_path)
saved_models_path = '{}saved_models/'.format(data_path)
flagfile = "{}flagfile.tmp".format(data_path)


# ==============================================================================
# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)


class LeakySigmoid(torch.nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        self.beta = torch.nn.Parameter(
            torch.tensor(1.), requires_grad=trainable)

    def forward(self, input):
        out = 1./(1+torch.exp(-input))
        out1 = torch.exp(-self.beta)/(1+torch.exp(-self.beta))**2 * (
                input - self.beta) + 1./(1+torch.exp(-self.beta))
        out[input > self.beta] = out1[input > self.beta]
        return out


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of
    clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


# ==============================================================================
# DICTS
activation_dict = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'prelu': torch.nn.PReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'leaky_sigmoid': LeakySigmoid,
    'sigmoid': torch.nn.Sigmoid,
}

UTILITY_DICT = {
    "ln": torch.log,
    "log": lambda x: torch.log(x+1e-16),
}

UTILITY_DICT_NUMPY = {
    "ln": np.log,
    "log": lambda x: np.log(x+1e-16),
}


# ==============================================================================
# FUNCTIONS
def get_utility_function(desc):
    if desc in UTILITY_DICT:
        return UTILITY_DICT[desc]
    elif desc.split("-")[0] == "power":
        p = float(desc.split("-")[1])
        assert p != 1
        return lambda input: (torch.pow(input, 1-p)-1)/(1-p)
    elif desc.split("-")[0] == "powerc":
        p = float(desc.split("-")[1])
        assert p != 1
        return lambda input: (torch.pow(input, 1-p))/(1-p)
    else:
        raise ValueError(
            "please provid valid argument for the utility function")


def get_utility_function_numpy(desc):
    if desc in UTILITY_DICT_NUMPY:
        return UTILITY_DICT_NUMPY[desc]
    elif desc.split("-")[0] == "power":
        p = float(desc.split("-")[1])
        assert p != 1
        return lambda input: (np.power(input, 1-p)-1)/(1-p)
    elif desc.split("-")[0] == "powerc":
        p = float(desc.split("-")[1])
        assert p != 1
        return lambda input: (np.power(input, 1-p))/(1-p)
    else:
        raise ValueError(
            "please provid valid argument for the utility function")


def get_penalty_function(desc, ref):
    if desc is None:
        return lambda x, y: torch.zeros(1)

    if desc.split("-")[0] == "norm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        # TODO: replace norm by expression that is everywhere diffable (i.e.
        #   also at 0, where the squareroot has nan-grad)
        return lambda x, y: torch.linalg.norm(x-y, ord=p, dim=(1,2))

    elif desc in ["squarenorm_dim4-fro", "squarenorm-fro"]:
        return lambda x, y: torch.sum(
            torch.sum(torch.sum((x-y)**2, dim=3), dim=2), dim=1)

    elif desc.split("-")[0] == "squarenorm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        # TODO: replace norm by expression that is everywhere diffable (i.e.
        #   also at 0, where the squareroot has nan-grad)
        return lambda x, y: torch.linalg.norm(x-y, ord=p, dim=(1,2))**2

    elif desc.split("-")[0] == "squarenorm_dim4":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        # TODO: replace norm by expression that is everywhere diffable (i.e.
        #   also at 0, where the squareroot has nan-grad)
        return lambda x, y: torch.sum(
            torch.linalg.norm(x-y, ord=p, dim=(2,3))**2, dim=1)

    elif desc.split("-")[0] == "squarenormsquare":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        # TODO: replace norm by expression that is everywhere diffable (i.e.
        #   also at 0, where the squareroot has nan-grad)
        return lambda x, y: torch.linalg.norm(
            torch.matmul(x, torch.transpose(x, 1, 2)) -
            torch.matmul(y, torch.transpose(y, 1, 2)), ord=p, dim=(1,2))**2

    elif desc == "mult-frob":
        y_inv = torch.inverse(torch.matmul(ref, torch.transpose(ref, 0, 1)))
        d = y_inv.shape[0]
        # TODO: replace norm by expression that is everywhere diffable (i.e.
        #   also at 0, where the squareroot has nan-grad)
        return lambda x, y: torch.linalg.norm(
            torch.matmul(torch.matmul(x, torch.transpose(x, 1, 2)), y_inv) -
            torch.eye(d), ord='fro', dim=(1,2))

    elif desc == "square-mult-frob":
        y_inv = torch.inverse(torch.matmul(ref, torch.transpose(ref, 0, 1)))
        d = y_inv.shape[0]
        # TODO: replace norm by expression that is everywhere diffable (i.e.
        #   also at 0, where the squareroot has nan-grad)
        return lambda x, y: torch.linalg.norm(
            torch.matmul(torch.matmul(x, torch.transpose(x, 1, 2)), y_inv) -
            torch.eye(d), ord='fro', dim=(1,2))**2

    else:
        raise ValueError(
            "please provid valid argument for the penalty function")


def get_penalty_function_numpy(desc, ref):
    if desc is None:
        return lambda x, y: 0.

    if desc.split("-")[0] == "norm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: np.linalg.norm(x-y, ord=p, axis=(1,2))

    elif desc in ["squarenorm_dim4-fro", "squarenorm-fro"]:
        return lambda x, y: np.sum(
            np.sum(np.sum((x-y)**2, axis=3), axis=2), axis=1)

    elif desc.split("-")[0] == "squarenorm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: np.linalg.norm(x-y, ord=p, axis=(1,2))**2

    elif desc.split("-")[0] == "squarenormsquare":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: np.linalg.norm(
            np.matmul(x, np.transpose(x, axes=(0,2,1))) -
            np.matmul(y, np.transpose(y, axes=(0,2,1))), ord=p, axis=(1,2))**2

    elif desc == "mult-frob":
        y_inv = np.linalg.inv(np.matmul(ref, ref.transpose()))
        d = y_inv.shape[0]
        return lambda x, y: np.linalg.norm(
            np.matmul(np.matmul(x, np.transpose(x, axes=(0,2,1))), y_inv) -
            np.eye(d), ord='fro', axis=(1,2))

    elif desc == "square-mult-frob":
        y_inv = np.linalg.inv(np.matmul(ref, ref.transpose()))
        d = y_inv.shape[0]
        return lambda x, y: np.linalg.norm(
            np.matmul(np.matmul(x, np.transpose(x, axes=(0,2,1))), y_inv) -
            np.eye(d), ord='fro', axis=(1,2))**2

    else:
        raise ValueError(
            "please provid valid argument for the penalty function")


def get_penalty_function_drift(desc):
    if desc is None:
        return lambda x, y: torch.zeros(1)

    if desc.split("-")[0] == "norm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: torch.linalg.norm(x-y, ord=p, dim=1)

    elif desc.split("-")[0] == "squarenorm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: torch.linalg.norm(x-y, ord=p, dim=1)**2

    elif desc.split("-")[0] == "normp":
        p = desc.split("-")[1]
        try:
            p = float(p)
            return lambda x, y: torch.linalg.norm(x-y, ord=p, dim=1)**p
        except Exception:
            raise ValueError("p has to be a float")

    else:
        raise ValueError(
            "please provid valid argument for the penalty function")


def get_penalty_function_drift_numpy(desc):
    if desc is None:
        return lambda x,y: 0.

    if desc.split("-")[0] == "norm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: np.linalg.norm(x-y, ord=p, axis=1)

    elif desc.split("-")[0] == "squarenorm":
        p = desc.split("-")[1]
        try:
            p = float(p)
        except Exception:
            pass
        return lambda x, y: np.linalg.norm(x-y, ord=p, axis=1)**2

    elif desc.split("-")[0] == "normp":
        p = desc.split("-")[1]
        try:
            p = float(p)
            return lambda x, y: np.linalg.norm(x-y, ord=p, axis=1)**p
        except Exception:
            raise ValueError("p has to be a float")

    else:
        raise ValueError(
            "please provid valid argument for the penalty function")


def get_quad_covar_path(X, *args, **kwargs):
    """
    :param X: torch.tensor, paths with shape [batch-size, time-steps, inner-dim]
    :return: path of the quadratic covariation of X over time
    """
    diffs = torch.diff(X, n=1, dim=1)
    return torch.matmul(diffs.unsqueeze(3), diffs.unsqueeze(2)).cumsum(dim=1)


def get_quad_covar(X, *args, **kwargs):
    """
    :param X: torch.tensor, paths with shape [batch-size, time-steps, inner-dim]
    :return: the quadratic covariation of X
    """
    diffs = torch.diff(X, n=1, dim=1)
    return torch.matmul(diffs.unsqueeze(3), diffs.unsqueeze(2)).sum(
        dim=1, keepdim=True)


def get_quad_covar_log(X, *args, **kwargs):
    """
    :param X: torch.tensor, paths with shape [batch-size, time-steps, inner-dim]
    :return: the quadratic covariation of log(X)
    """
    return get_quad_covar(torch.log(X), *args, **kwargs)


def get_CIR_quad_covar(sigma):
    def CIR_quad_covar(X, dt, *args, **kwargs):
        """"
        :param X: torch.tensor, paths with shape
            [batch-size, time-steps, inner-dim]
        """
        assert X.shape[2] == 1
        return sigma**2 * dt * X[:,:-1].cumsum(dim=1).unsqueeze(3)
    return CIR_quad_covar


def get_mean_rel_return(X, *args, **kwargs):
    """
    :param X: torch.tensor, paths with shape [batch-size, time-steps, inner-dim]
    :param kwargs:
    :return:
    """
    return torch.mean(X[:, -1, :]/X[:, 0, :], dim=0)


def get_path_integral(func, cumulative=False):
    if isinstance(func, str):
        func = eval(func)
    def path_integral(X, dt, T, **kwargs):
        """"
        :param X: torch.tensor, paths with shape
            [batch-size, time-steps, inner-dim]
        """
        if cumulative:
            n = X.shape[1]-1
            cs = func(X[:, :-1]).cumsum(dim=1)
            cs = cs / torch.arange(1,n+1).reshape((1,-1,1)).repeat(
                (cs.shape[0], 1, cs.shape[2]))
            return cs
        return func(X[:, :-1]).sum(dim=1, keepdim=True) * dt / T
    return path_integral


def get_empirical_batch_integral(func, ft_only=False):
    if isinstance(func, str):
        func = eval(func)
    def path_integral(X, dt, T, **kwargs):
        """"
        :param X: torch.tensor, paths with shape
            [batch-size, time-steps, inner-dim]
        """
        if ft_only:
            return func(X[:, -1:]).mean(dim=0, keepdim=True)
        else:
            return func(X[:, 1:]).mean(dim=0, keepdim=True)
    return path_integral


def get_parameter_array(param_dict):
    """
    helper function to get a list of parameter-list with all combinations of
    parameters specified in a parameter-dict

    :param param_dict: dict with parameters
    :return: 2d-array with the parameter combinations
    """
    param_combs_dict_list = list(ParameterGrid(param_dict))
    return param_combs_dict_list


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ==============================================================================
# DATASET DICTS
data_dict1 = {
    "S0": [1], "dt": 1/65., "r": 0.015, "nb_steps": 65, "nb_samples": 200000,
}

data_dict2 = {
    "S0": [1,1], "dt": 1/65., "r": 0.015, "nb_steps": 65, "nb_samples": 200000,
    "seed": 264 #397
}

data_dict3 = {
    "S0": [1,1,1,1,1], "dt": 1/65., "r": 0.015, "nb_steps": 65, "nb_samples": 200000,
}


# ==============================================================================
# TRAINING PARAM DICTS
nn1 = ((50, 'leaky_relu'),)
nn2 = ((5, 'leaky_relu'), (10, 'relu'))

nn3 = ((15, 'leaky_relu'), (15, 'relu'))
nn3_d = ((15, 'leaky_relu'), (15, 'relu'), 'leaky_sigmoid')


ffnn_dict1 = {
    "name": "FFNN",
    "nn_desc": nn1, "dropout_rate": 0.1, "bias": True,
}
ffnn_dict1_d = {
    "name": "FFNN", "drift": "nn", "vola": "nn",
    "nn_desc": nn1, "dropout_rate": 0.1, "bias": True,
}
ffnn_dict2 = {
    "name": "FFNN",
    "nn_desc": nn2, "dropout_rate": 0.2, "bias": True,
}
ffnn_dict3 = {
    "name": "FFNN",
    "nn_desc": nn3, "dropout_rate": 0.2, "bias": True,
}
ffnn_dict3_d = {
    "name": "FFNN", "drift": [0.035], "vola": "nn",
    "nn_desc": nn3_d, "dropout_rate": 0.2, "bias": True,
}

timegrid_nn_dict1 = {
    "name": "timegrid_nn",
    "nn_desc": nn1, "dropout_rate": 0.1, "bias": True,
    "timegrid": list(np.linspace(0,1,66))[1:-1]
}
timegrid_nn_dict1_d = {
    "name": "timegrid_nn", "drift": [0.035], "vola": "nn",
    "nn_desc": nn1, "dropout_rate": 0.1, "bias": True,
    "timegrid": list(np.linspace(0,1,66))[1:-1]
}
timegrid_nn_dict1_d1 = {
    "name": "timegrid_nn", "drift": "nn", "vola": "nn",
    "nn_desc": nn1, "dropout_rate": 0.1, "bias": True,
    "timegrid": list(np.linspace(0,1,66))[1:-1]
}

rnn_dict1 = {
    "name": "RNN",
    "hidden_desc": nn1, "readout_desc": None,
    "hidden_size": 50,
    "dropout_rate": 0.1,
}
rnn_dict1_d = {
    "name": "RNN", "drift": [0.035], "vola": "nn",
    "hidden_desc": nn1, "readout_desc": None,
    "hidden_size": 50,
    "dropout_rate": 0.1,
}
rnn_dict2_d = {
    "name": "RNN", "drift": "ref0", "vola": "nn",
    "hidden_desc": nn1, "readout_desc": None,
    "hidden_size": 50,
    "dropout_rate": 0.1,
}
rnn_dict3_d = {
    "name": "RNN", "drift": "nn", "vola": "nn",
    "hidden_desc": nn1, "readout_desc": None,
    "hidden_size": 50,
    "dropout_rate": 0.1,
}

disc_dict1 = copy.copy(ffnn_dict1)
disc_dict1["drift"] = [0.035,]
disc_dict1["vola"] = "nn"

disc_dict2 = copy.copy(ffnn_dict2)
disc_dict2["drift"] = [0.035,]
disc_dict2["vola"] = "nn"

disc_dict_no_disc = {"name": "fixed"}
disc_dict_no_disc["drift"] = [0.035,]
disc_dict_no_disc["vola"] = "ref0"

disc_dict_no_disc_2 = {"name": "fixed", "drift": "ref0", "vola": "ref0"}


path = saved_models_path
param_dict0 = dict(
    test_size=[0.2],
    data_dict=['data_dict2'],
    epochs=[150],
    learning_rate_D=[1e-4], learning_rate_G=[1e-4],
    lr_scheduler_D=[None], lr_scheduler_G=[None],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[0,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["mult-frob"],
    penalty_function_ref_value=[[[0.25,0], [0, 0.25]]],
    gen_dict=[ffnn_dict1],
    disc_dict=[disc_dict_no_disc],
    saved_models_path=[path],
    use_penalty_for_gen=[False],
)
params_list0 = get_parameter_array(param_dict=param_dict0)


path = saved_models_path
param_dict1 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[1e-3, 1e-4, 1e-5], learning_rate_G=[1e-3, 1e-4, 1e-5],
    lr_scheduler_D=[None], lr_scheduler_G=[None],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [1,5], [1,10], [5,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.1],
    gen_dict=[ffnn_dict1, ffnn_dict2],
    disc_dict=[disc_dict1, disc_dict2],
    saved_models_path=[path],
)
params_list1 = get_parameter_array(param_dict=param_dict1)

TO_dict1 = dict(
    ids_from=1, ids_to=len(params_list1),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs_P", "data_dict"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_dist", "eval_dist",
         "min-eval_dist"),),
    sortby=["min-eval_dist"],
)



param_dict2 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[1e-3, 1e-4, 1e-5], learning_rate_G=[1e-3, 1e-4, 1e-5],
    lr_scheduler_D=[None], lr_scheduler_G=[None],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [1,5], [1,10], [5,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.1],
    gen_dict=[ffnn_dict1, ffnn_dict2],
    disc_dict=[disc_dict1, disc_dict2],
    saved_models_path=[path],
)
params_list2 = get_parameter_array(param_dict=param_dict2)

TO_dict2 = dict(
    ids_from=200, ids_to=len(params_list2),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_dist", "eval_dist",
         "min-eval_dist"),),
    sortby=["min-eval_dist"],
)


# ==============================================================================
# Hanna

# 4 layer NN as in paper
nn4 = ((32, 'leaky_relu'), (32, 'leaky_relu'), (32, 'leaky_relu'), (32, 'leaky_relu'), 'leaky_sigmoid')

# 4 layer NN (dropout 0.2)
ffnn_dict4 = {
    "name": "FFNN",
    "nn_desc": nn4, "dropout_rate": 0.2, "bias": True,
}

disc_dict4 = copy.copy(ffnn_dict4)
disc_dict4["drift"] = [0.035,]
disc_dict4["vola"] = "nn"

param_dict4 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[1e-3, 1e-4, 1e-5], learning_rate_G=[1e-3, 1e-4, 1e-5],
    lr_scheduler_D=[None], lr_scheduler_G=[None],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [1,5], [1,10], [5,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.1],
    gen_dict=[ffnn_dict4],
    disc_dict=[disc_dict4],
    saved_models_path=[path],
    use_penalty_for_gen=[True],

)
params_list4 = get_parameter_array(param_dict=param_dict4)

TO_dict4 = dict(
    ids_from=1, ids_to=len(params_list4),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_dist", "eval_dist",
         "min-eval_dist"),),
    sortby=["min-eval_dist"],
)

# 4 layer time dependent NN (dropout 0.1)

timegrid_nn_dict4 = {
    "name": "timegrid_nn",
    "nn_desc": nn4, "dropout_rate": 0.1, "bias": True,
    "timegrid": list(np.linspace(0,1,66))[1:-1]
}

disc_dict5 = copy.copy(timegrid_nn_dict4)
disc_dict5["drift"] = [0.035,]
disc_dict5["vola"] = "nn"

param_dict5 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[1e-3, 1e-4, 1e-5], learning_rate_G=[1e-3, 1e-4, 1e-5],
    lr_scheduler_D=[None], lr_scheduler_G=[None],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [1,5], [5,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.1],
    gen_dict=[timegrid_nn_dict4],
    disc_dict=[disc_dict5],
    saved_models_path=[path],
    use_penalty_for_gen=[True, False],

)
params_list5 = get_parameter_array(param_dict=param_dict5)

TO_dict5 = dict(
    ids_from=200, ids_to=200+len(params_list5),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_dist", "eval_dist",
         "min-eval_dist"),),
    sortby=["min-eval_dist"],
)

# 4 layer time dependent NN (no dropout)
timegrid_nn_dict5 = {
    "name": "timegrid_nn",
    "nn_desc": nn4, "dropout_rate": 0, "bias": True,
    "timegrid": list(np.linspace(0,1,66))[1:-1]
}

disc_dict6 = copy.copy(timegrid_nn_dict5)
disc_dict6["drift"] = [0.035,]
disc_dict6["vola"] = "nn"

param_dict6 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[1e-3, 1e-4, 1e-5], learning_rate_G=[1e-3, 1e-4, 1e-5],
    lr_scheduler_D=[None], lr_scheduler_G=[None],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [10,1], [5,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.1],
    gen_dict=[timegrid_nn_dict5],
    disc_dict=[disc_dict6],
    saved_models_path=[path],
    use_penalty_for_gen=[True],

)
params_list6 = get_parameter_array(param_dict=param_dict6)

TO_dict6 = dict(
    ids_from=300, ids_to=300+len(params_list6),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("min", "eval_dist", "eval_dist",
         "min-eval_dist"),
        ("last", "ref_expected_utility", "ref_expected_utility", "ref_expected_utility"),
        ("min", "eval_dist", "eval_expected_utility", "eval_expected_utility_at_min_eval_dist"),
        ("min", "eval_dist", "eval_penalty", "eval_penalty_at_min_eval_dist")
        ),
    sortby=["min-eval_dist"],
)



# 4 layer time dependent NN (0.1 dropout) --- MOST LIKE PAPER
param_dict7 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[5e-4], learning_rate_G=[5e-4],
    lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [1,5], [5,1]],
    batch_size=[1000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.25],
    gen_dict=[timegrid_nn_dict4],
    disc_dict=[disc_dict5],
    saved_models_path=[path],
    use_penalty_for_gen=[True],

)
params_list7 = get_parameter_array(param_dict=param_dict7)

TO_dict7 = dict(
    ids_from=400, ids_to=400+len(params_list7),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("min", "eval_dist", "eval_dist", "min-eval_dist"),
        ("last", "ref_expected_utility", "ref_expected_utility", "ref_expected_utility"),
        ("min", "eval_dist", "eval_expected_utility", "eval_expected_utility_at_min_eval_dist"),
        ("min", "eval_dist", "eval_penalty", "eval_penalty_at_min_eval_dist"),
        ("max", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "max-eval_expected_utility_with_ref"),
        ("min", "eval_penalty_vs_ref", "eval_penalty_vs_ref", "min-eval_penalty_vs_ref")
        ),
    sortby=["max-eval_expected_utility_with_ref"],
)


# 4 layer time dependent NN (0.1 dropout) --- MOST LIKE PAPER with different batch sizes
param_dict8 = dict(
    test_size=[0.2],
    data_dict=['data_dict1'],
    epochs=[150],
    learning_rate_D=[5e-4], learning_rate_G=[5e-4],
    lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[1,1], [5,1]],
    batch_size=[100, 1000, 10000],
    penalty_scaling_factor=[10.],
    initial_wealth=[5.],
    utility_func=["log"], penalty_func=["norm-2"],
    penalty_function_ref_value=[0.25],
    gen_dict=[timegrid_nn_dict4],
    disc_dict=[disc_dict5],
    saved_models_path=[path],
    use_penalty_for_gen=[True],
    eval_on_train=[True],

)
params_list8 = get_parameter_array(param_dict=param_dict8)

TO_dict8 = dict(
    ids_from=410, ids_to=410+len(params_list8),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("min", "eval_dist", "eval_dist", "min-eval_dist"),
        ("last", "ref_expected_utility", "ref_expected_utility", "ref_expected_utility"),
        ("min", "eval_dist", "eval_expected_utility", "eval_expected_utility_at_min_eval_dist"),
        ("min", "eval_dist", "eval_penalty", "eval_penalty_at_min_eval_dist"),
        ("max", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "max-eval_expected_utility_with_ref"),
        ("min", "eval_penalty_vs_ref", "eval_penalty_vs_ref", "min-eval_penalty_vs_ref"),
        ),
    sortby=["max-eval_expected_utility_with_ref"],
)


# 1 layer ffNN, rnn (0.1 dropout) linear activation on last layer
params_list9 =[]

for gen, disc in [(ffnn_dict1, disc_dict1), (rnn_dict1, rnn_dict1_d)]:
    param_dict9 = dict(
        test_size=[0.2],
        data_dict=['data_dict1'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[1,1], [5,1]],
        batch_size=[100, 1000],
        penalty_scaling_factor=[10.],
        initial_wealth=[5.],
        utility_func=["log"], penalty_func=["norm-2"],
        penalty_function_ref_value=[0.25],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
    
    )
    params_list9 += get_parameter_array(param_dict=param_dict9)

TO_dict9 = dict(
    ids_from=420, ids_to=420+len(params_list9),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict"),
    vals_metric_extract=(
        ("min", "eval_dist", "eval_dist", "min-eval_dist"),
        ("last", "ref_expected_utility", "ref_expected_utility", "ref_expected_utility"),
        ("last", "ref_penalty", "ref_penalty", "ref_penalty"),
        ("min", "eval_dist", "eval_expected_utility", "eval_expected_utility_at_min_eval_dist"),
        ("min", "eval_dist", "eval_penalty", "eval_penalty_at_min_eval_dist"),
        ("max", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "max-eval_expected_utility_with_ref"),
        ("min", "eval_penalty_vs_ref", "eval_penalty_vs_ref", "min-eval_penalty_vs_ref"),
        ),
    sortby=["max-eval_expected_utility_with_ref"],
)

TO_dict9_all = copy.deepcopy(TO_dict9)
TO_dict9_all["ids_from"] = 400


# MULTIDIM: 1 layer rnn (0.1 dropout) linear activation on last layer
params_list10 =[]

for gen, disc, disc_steps in [(rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1)]:
    param_dict10 = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        initial_wealth=[1.],
        utility_func=["log"], 
        penalty_func=["mult-frob", "square-mult-frob", "norm-2", "squarenorm-2"],
        penalty_function_ref_value=[[[0.25,0], [0, 0.25]]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        noise_seed=[496],
        noise_std=[0.025, 0.15]
    )
    params_list10 += get_parameter_array(param_dict=param_dict10)

TO_dict10 = dict(
    ids_from=450, ids_to=450+len(params_list10),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict", "noise_std", "penalty_func"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility", "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "ref_penalty", "ref_penalty", "ref_penalty"),
        ("max", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "max-eval_expected_utility_with_ref"),
        ("min", "eval_penalty_vs_ref", "eval_penalty_vs_ref", "min-eval_penalty_vs_ref"),
        ),
    sortby=["noise_std", "mean_exp_util_with_ref"],
    model_eval_file="model_evaluation",
)

eval_model_dict10_last = dict(
    model_ids=list(range(450, 450+len(params_list10))),
    load_best=False,
    nb_evaluations=1000,
    saved_models_path=path)



# MULTIDIM: with correlation in sigma_0 and asymmetric cases
params_list11 =[]

for gen, disc, disc_steps in [(rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1)]:
    param_dict11 = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        initial_wealth=[1.],
        utility_func=["log"], 
        penalty_func=["square-mult-frob", "norm-2"],
        penalty_function_ref_value=[[[0.15,0], [0, 0.35]], [[0.15, 0.],[0.315, 0.15256146]], [[0.15, 0.],[-0.315, 0.15256146]], [[0.25, 0.],
       [0.225, 0.10897247]]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        noise_seed=[496],
        noise_std=[0.025, 0.15]
    )
    params_list11 += get_parameter_array(param_dict=param_dict11)



param_dict11_1 = dict(
    test_size=[0.2],
    data_dict=['data_dict2'],
    epochs=[150],
    learning_rate_D=[5e-4], learning_rate_G=[5e-4],
    lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[5, 1], [10, 1]],
    batch_size=[1000],
    penalty_scaling_factor=[1., .1],
    initial_wealth=[1.],
    utility_func=["log"], 
    penalty_func=["square-mult-frob", "norm-2"],
    penalty_function_ref_value=[[[0.15,0], [0, 0.35]], [[0.15, 0.],[0.315, 0.15256146]], [[0.15, 0.],[-0.315, 0.15256146]], [[0.25, 0.],
       [0.225, 0.10897247]]],
    gen_dict=[rnn_dict1],
    disc_dict=[rnn_dict1_d],
    saved_models_path=[path],
    use_penalty_for_gen=[True],
    eval_on_train=[False],
    noise_seed=[496],
    noise_std=[0.025, 0.15]
)
params_list11_1 = get_parameter_array(param_dict=param_dict11_1)



TO_dict11 = dict(
    ids_from=500, ids_to=600+len(params_list11_1),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict", "penalty_function_ref_value",
                         "noise_std", "penalty_func", "penalty_scaling_factor"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility", "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "ref_penalty", "ref_penalty", "ref_penalty"),
        ("last", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "last-eval_expected_utility_with_ref"),
        ("max", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "max-eval_expected_utility_with_ref"),
        ("min", "eval_penalty_vs_ref", "eval_penalty_vs_ref", "min-eval_penalty_vs_ref"),
        ),
    sortby=["noise_std",  "mean_exp_util_with_ref"],
    model_eval_file="model_evaluation",
)

eval_model_dict11_last = dict(
    model_ids=list(range(500, 500+len(params_list11)))+list(range(600,600+len(params_list11_1))),
    load_best=False,
    nb_evaluations=1000,
    saved_models_path=path)


# 5-Dim
params_list12 =[]

for gen, disc, disc_steps in [(rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1)]:
    param_dict12 = dict(
        test_size=[0.2],
        data_dict=['data_dict3'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        initial_wealth=[1.],
        utility_func=["log"], 
        penalty_func=["square-mult-frob", "norm-2"],
        penalty_function_ref_value=[np.diag([0.25]*5).tolist()],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        noise_seed=[496],
        noise_std=[0.025, 0.15]
    )
    params_list12 += get_parameter_array(param_dict=param_dict12)


params_list12_1 =[]

param_dict12_1 = dict(
    test_size=[0.2],
    data_dict=['data_dict3'],
    epochs=[150],
    learning_rate_D=[5e-4], learning_rate_G=[5e-4],
    lr_scheduler_D=[{'gamma':0.2, 'step':100}], lr_scheduler_G=[{'gamma':0.2, 'step':100}],
    beta1_D=[0.9,], beta2_D=[0.999,],
    beta1_G=[0.9,], beta2_G=[0.999,],
    opt_steps_D_G=[[5, 1], [10, 1]],
    batch_size=[1000],
    penalty_scaling_factor=[1., .1, 5, 10],
    initial_wealth=[1.],
    utility_func=["log"], 
    penalty_func=["square-mult-frob", "norm-2"],
    penalty_function_ref_value=[np.diag([0.25]*5).tolist()],
    gen_dict=[rnn_dict1],
    disc_dict=[rnn_dict1_d],
    saved_models_path=[path],
    use_penalty_for_gen=[True],
    eval_on_train=[False],
    noise_seed=[496],
    noise_std=[0.025, 0.15]
)
params_list12_1 += get_parameter_array(param_dict=param_dict12_1)
    
params_list12 += params_list12_1

TO_dict12 = dict(
    ids_from=550, ids_to=550+len(params_list12),
    path=path,
    params_extract_desc=('gen_dict', 'disc_dict', 'learning_rate_D',
                         'learning_rate_G', 'opt_steps_D_G', 'batch_size',
                         "epochs", "data_dict", "penalty_function_ref_value",
                         "noise_std", "penalty_func", "penalty_scaling_factor"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility", "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "ref_penalty", "ref_penalty", "ref_penalty"),
        ("max", "eval_expected_util_with_ref", "eval_expected_util_with_ref", "max-eval_expected_utility_with_ref"),
        ("min", "eval_penalty_vs_ref", "eval_penalty_vs_ref", "min-eval_penalty_vs_ref"),
        ),
    sortby=["noise_std", "mean_exp_util_with_ref"],
    model_eval_file="model_evaluation",
)

eval_model_dict12_last = dict(
    model_ids=list(range(550, 550+len(params_list12))), load_best=False,
    nb_evaluations=1000, saved_models_path=path)



# ==============================================================================
# ========================== NEW EVALUATION METHODS ============================
# ==============================================================================
path_2dim = "{}saved_models_2dim/".format(data_path)

params_list_2dim = []
params_list_2dim_1 = []
for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1),
    (rnn_dict1, rnn_dict1_d, 10)]:
    param_dict_2dim = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", "square-mult-frob",],
        penalty_function_ref_value=[
            [[0.25,0], [0, 0.25]],
            [[0.15,0], [0, 0.35]],
            [[0.15, 0.],[0.315, 0.15256146]],
            [[0.15, 0.],[-0.315, 0.15256146]],
            [[0.25, 0.],[0.225, 0.10897247]]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
    )
    params_list_2dim_1 += get_parameter_array(param_dict=param_dict_2dim)
params_list_2dim += params_list_2dim_1

TO_dict_2dim = dict(
    ids_from=1, ids_to=1+len(params_list_2dim),
    path=path_2dim,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "data_dict", "penalty_func"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_sig",
         "eval_expected_util_with_analytic_sig",
         "max-eval_expected_util_with_analytic_sig"),
        ("min", "eval_penalty_vs_analytic_sig", "eval_penalty_vs_analytic_sig",
         "min-eval_penalty_vs_analytic_sig"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_sig"],
    model_eval_file="model_evaluation",
)

eval_model_dict_2dim_best = dict(
    model_ids=list(range(1, 1+len(params_list_2dim))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.025,
    saved_models_path=path_2dim)

eval_model_dict_2dim_best_1 = dict(
    model_ids=list(range(1, 1+len(params_list_2dim))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    saved_models_path=path_2dim)



# --------------- 5-Dim ----------------
path_5dim = "{}saved_models_5dim/".format(data_path)

params_list_5dim = []
params_list_5dim_1 = []

for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1),
    (rnn_dict1, rnn_dict1_d, 10)]:
    param_dict_5dim = dict(
        test_size=[0.2],
        data_dict=['data_dict3'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.1, 1., 10],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", "square-mult-frob"],
        penalty_function_ref_value=[np.diag([0.25]*5).tolist()],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_5dim],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
    )
    params_list_5dim_1 += get_parameter_array(param_dict=param_dict_5dim)
params_list_5dim += params_list_5dim_1

TO_dict_5dim = dict(
    ids_from=1, ids_to=1+len(params_list_5dim),
    path=path_5dim,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "data_dict", "penalty_func"),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_sig",
         "eval_expected_util_with_analytic_sig",
         "max-eval_expected_util_with_analytic_sig"),
        ("min", "eval_penalty_vs_analytic_sig", "eval_penalty_vs_analytic_sig",
         "min-eval_penalty_vs_analytic_sig"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_sig"],
    model_eval_file="model_evaluation",
)

eval_model_dict_5dim_best = dict(
    model_ids=list(range(1, 1+len(params_list_5dim))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.025,
    saved_models_path=path_5dim)

eval_model_dict_5dim_best_1 = dict(
    model_ids=list(range(1, 1+len(params_list_5dim))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    saved_models_path=path_5dim)


# ======================== TRANSACTION COSTS ===================================
path_2dim_TC = "{}saved_models_2dim_transcost/".format(data_path)

params_list_2dim_TC = []
for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1)]:
    param_dict_2dim_TC = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", "square-mult-frob",],
        penalty_function_ref_value=[
            [[0.25,0], [0, 0.25]],
            [[0.15,0], [0, 0.35]],
            [[0.15, 0.],[0.315, 0.15256146]],
            [[0.15, 0.],[-0.315, 0.15256146]],
            [[0.25, 0.],[0.225, 0.10897247]]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_TC],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        trans_cost_base=[0],
        trans_cost_perc=[0.001, 0.01],
        eval_noise_std=[0.025, 0.15],
        eval_noise_seed=[8979],
    )
    params_list_2dim_TC += get_parameter_array(param_dict=param_dict_2dim_TC)

TO_dict_2dim_TC = dict(
    ids_from=1, ids_to=len(params_list_2dim_TC),
    path=path_2dim_TC,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "data_dict", "penalty_func", "trans_cost_perc", "trans_cost_base",
        'eval_noise_std'),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_sig",
         "eval_expected_util_with_analytic_sig",
         "max-eval_expected_util_with_analytic_sig"),
        ("min", "eval_penalty_vs_analytic_sig", "eval_penalty_vs_analytic_sig",
         "min-eval_penalty_vs_analytic_sig"),
        ("max", "eval_expected_util_with_noisy_sig",
         "eval_expected_util_with_noisy_sig",
         "max-eval_expected_util_with_noisy_sig"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_sig"],
    model_eval_file="model_evaluation",
)

eval_model_dict_2dim_best_TC = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_TC))),
    load_saved_eval=False,
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.025,
    saved_models_path=path_2dim_TC)

eval_model_dict_2dim_best_1_TC = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_TC))),
    load_saved_eval=False,
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    saved_models_path=path_2dim_TC)


# --------------- 5-Dim ----------------
path_5dim_TC = "{}saved_models_5dim_transcost/".format(data_path)

params_list_5dim_TC = []
params_list_5dim_1_TC = []

for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc, 0), (rnn_dict1, rnn_dict1_d, 1)]:
    param_dict_5dim = dict(
        test_size=[0.2],
        data_dict=['data_dict3'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.1, 1., 10.],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", "square-mult-frob",],
        penalty_function_ref_value=[np.diag([0.25]*5).tolist()],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_5dim_TC],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        trans_cost_base=[0],
        trans_cost_perc=[0.001, 0.01],
        eval_noise_std=[0.025, 0.15],
        eval_noise_seed=[8979],
    )
    params_list_5dim_1_TC += get_parameter_array(param_dict=param_dict_5dim)
params_list_5dim_TC += params_list_5dim_1_TC

TO_dict_5dim_TC = dict(
    ids_from=1, ids_to=1+len(params_list_5dim_TC),
    path=path_5dim_TC,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "data_dict", "penalty_func", "trans_cost_perc", "trans_cost_base",
        'eval_noise_std'),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("min", "eval_penalty", "eval_penalty", "min-eval_penalty"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_sig",
         "eval_expected_util_with_analytic_sig",
         "max-eval_expected_util_with_analytic_sig"),
        ("min", "eval_penalty_vs_analytic_sig", "eval_penalty_vs_analytic_sig",
         "min-eval_penalty_vs_analytic_sig"),
        ("max", "eval_expected_util_with_noisy_sig",
         "eval_expected_util_with_noisy_sig",
         "max-eval_expected_util_with_noisy_sig"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_sig"],
    model_eval_file="model_evaluation",
)

eval_model_dict_5dim_best_TC = dict(
    model_ids=list(range(1, 1+len(params_list_5dim_TC))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.025,
    saved_models_path=path_5dim_TC)

eval_model_dict_5dim_best_1_TC = dict(
    model_ids=list(range(1, 1+len(params_list_5dim_TC))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    saved_models_path=path_5dim_TC)


# ====================== training with robust drift ============================
path_2dim_robd = "{}saved_models_2dim_robustdrift/".format(data_path)

params_list_2dim_robd = []
for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc_2, 0), (rnn_dict1, rnn_dict2_d, 1),
    (rnn_dict1, rnn_dict3_d, 1)]:
    param_dict_2dim_robd = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        penalty_scaling_factor_drift=[1.],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", ],
        penalty_func_drift=["squarenorm-2",],
        penalty_function_ref_value=[
            [[0.25,0], [0, 0.25]],
            [[0.15,0], [0, 0.35]],
            [[0.15, 0.],[0.315, 0.15256146]],
            [[0.15, 0.],[-0.315, 0.15256146]],
            [[0.25, 0.],[0.225, 0.10897247]]],
        penalty_function_ref_value_drift=[[0.035, 0.035], [0.035, 0.055]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_robd],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.025, 0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
    )
    params_list_2dim_robd += get_parameter_array(param_dict=param_dict_2dim_robd)

for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc_2, 0), (rnn_dict1, rnn_dict2_d, 1),
    (rnn_dict1, rnn_dict3_d, 1)]:
    param_dict_2dim_robd_transc = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        penalty_scaling_factor_drift=[1.],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", ],
        penalty_func_drift=["squarenorm-2",],
        penalty_function_ref_value=[
            [[0.25,0], [0, 0.25]],
            [[0.15,0], [0, 0.35]],
            [[0.15, 0.],[0.315, 0.15256146]],
            [[0.15, 0.],[-0.315, 0.15256146]],
            [[0.25, 0.],[0.225, 0.10897247]]],
        penalty_function_ref_value_drift=[[0.035, 0.035], [0.035, 0.055]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_robd],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.025, 0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        trans_cost_base=[0],
        trans_cost_perc=[0.001, 0.01],
    )
    params_list_2dim_robd += get_parameter_array(
        param_dict=param_dict_2dim_robd_transc)

TO_dict_2dim_robd = dict(
    ids_from=1, ids_to=len(params_list_2dim_robd),
    path=path_2dim_robd,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_func",
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "penalty_func_drift",
        "penalty_function_ref_value_drift", 'penalty_scaling_factor_drift',
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation",
)

eval_model_dict_2dim_best_robd = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_robd))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.025,
    noise_std_drift=0.02,
    saved_models_path=path_2dim_robd)

eval_model_dict_2dim_best_robd_1 = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_robd))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    noise_std_drift=0.02,
    saved_models_path=path_2dim_robd)


# ------- comparison with non RNN models
path_2dim_robd_modelcomp = "{}saved_models_2dim_robustdrift_modelcomp/".format(
    data_path)

params_list_2dim_robd_modelcomp = []
for gen, disc, disc_steps in [
    (ffnn_dict1, ffnn_dict1_d, 1),
    (timegrid_nn_dict1, timegrid_nn_dict1_d1, 1),
    (rnn_dict1, rnn_dict3_d, 1)]:
    param_dict_2dim_robd = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        penalty_scaling_factor_drift=[1.],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", ],
        penalty_func_drift=["squarenorm-2",],
        penalty_function_ref_value=[
            [[0.15,0], [0, 0.35]],],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_robd_modelcomp],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
    )
    params_list_2dim_robd_modelcomp += get_parameter_array(
        param_dict=param_dict_2dim_robd)

TO_dict_2dim_robd_modelcomp = dict(
    ids_from=1, ids_to=len(params_list_2dim_robd_modelcomp),
    path=path_2dim_robd_modelcomp,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_func",
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "penalty_func_drift",
        "penalty_function_ref_value_drift", 'penalty_scaling_factor_drift',
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
    ),
    sortby=["penalty_func", "max-eval_expected_util_with_analytic_par"],
)

plot_model_dict_2dim_best_robd_modelcomp = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_robd_modelcomp))),
    load_best=True,
    nb_evaluations=1,
    evaluate_vs_analytic=True,
    plot_gen_disc_paths=[0,1,2,3,4],
    plot_noisy_eval_paths=[0,1,2,3,4],
    load_saved_eval=False,
    saved_models_path=path_2dim_robd_modelcomp)




# ================= training with robust drift & other utilities ===============
path_2dim_robd_power_util = "{}saved_models_2dim_robustdrift_powU/".format(data_path)

params_list_2dim_robd_powU = []
for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc_2, 0), (rnn_dict1, rnn_dict2_d, 1),
    (rnn_dict1, rnn_dict3_d, 1)]:
    param_dict_2dim_robd_transc_powU = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[1.],
        penalty_scaling_factor_drift=[1.],
        initial_wealth=[1.],
        utility_func=["power-0.5", "power-2"],
        penalty_func=["squarenormsquare-fro", ],
        penalty_func_drift=["squarenorm-2",],
        penalty_function_ref_value=[
            [[0.25,0], [0, 0.25]],
            [[0.15,0], [0, 0.35]],
            [[0.15, 0.],[0.315, 0.15256146]],
            [[0.15, 0.],[-0.315, 0.15256146]],
            [[0.25, 0.],[0.225, 0.10897247]]],
        penalty_function_ref_value_drift=[[0.035, 0.035], [0.035, 0.055]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_robd_power_util],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.025, 0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        trans_cost_base=[0],
        trans_cost_perc=[0., 0.001, 0.01],
    )
    params_list_2dim_robd_powU += get_parameter_array(
        param_dict=param_dict_2dim_robd_transc_powU)

TO_dict_2dim_robd_powU = dict(
    ids_from=1, ids_to=len(params_list_2dim_robd_powU),
    path=path_2dim_robd_power_util,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_func",
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "penalty_func_drift",
        "penalty_function_ref_value_drift", 'penalty_scaling_factor_drift',
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation",
)



# ============================== lambda grids ==================================
path_2dim_lambdagrid = "{}saved_models_2dim_lambdagrid/".format(data_path)

params_list_2dim_lambdagrid = []
for gen, disc, disc_steps in [
    (rnn_dict1, disc_dict_no_disc_2, 0), (rnn_dict1, rnn_dict2_d, 1),
    (rnn_dict1, rnn_dict3_d, 1)]:
    param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.01, 0.5, 0.1, 1., 10, 100],
        penalty_scaling_factor_drift=[0.01, 0.5, 0.1, 1., 10, 100],
        initial_wealth=[1.],
        utility_func=["power-2"],
        penalty_func=["squarenormsquare-fro", ],
        penalty_func_drift=["squarenorm-2",],
        penalty_function_ref_value=[
            [[0.15,0], [0, 0.35]],],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_lambdagrid],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
    )
    params_list_2dim_lambdagrid += get_parameter_array(
        param_dict=param_dict_2dim_lambdagrid)

TO_dict_2dim_lambdagrid = dict(
    ids_from=1, ids_to=len(params_list_2dim_lambdagrid),
    path=path_2dim_lambdagrid,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_func",
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "penalty_func_drift",
        "penalty_function_ref_value_drift", 'penalty_scaling_factor_drift',
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation",
    plot_penalty_scaling_plots=dict(
        remove_rows={"penalty_scaling_factor_drift": 0.01},)
)

eval_model_dict_2dim_best_lambdagrid = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    noise_std_drift=0.02,
    saved_models_path=path_2dim_lambdagrid)



# ----- comparison of different models (RNN, FFNN, timegrid)
path_2dim_lambdagrid_modelcomp = \
    "{}saved_models_2dim_lambdagrid_modelcomp/".format(data_path)

params_list_2dim_lambdagrid_modelcomp = []
for gen, disc, disc_steps in [
    (ffnn_dict1, ffnn_dict1_d, 1),
    (timegrid_nn_dict1, timegrid_nn_dict1_d1, 1),
    (rnn_dict1, rnn_dict3_d, 1)]:
    param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[disc_steps, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.01, 0.5, 0.1, 1., 10, 100],
        penalty_scaling_factor_drift=[0.01, 0.5, 0.1, 1., 10, 100],
        initial_wealth=[1.],
        utility_func=["log"],
        penalty_func=["squarenormsquare-fro", ],
        penalty_func_drift=["squarenorm-2",],
        penalty_function_ref_value=[
            [[0.15,0], [0, 0.35]],],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_2dim_lambdagrid_modelcomp],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
    )
    params_list_2dim_lambdagrid_modelcomp += get_parameter_array(
        param_dict=param_dict_2dim_lambdagrid)

TO_dict_2dim_lambdagrid_modelcomp = dict(
    ids_from=1, ids_to=len(params_list_2dim_lambdagrid_modelcomp),
    path=path_2dim_lambdagrid_modelcomp,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_func",
        "penalty_function_ref_value", 'penalty_scaling_factor',
        "penalty_func_drift",
        "penalty_function_ref_value_drift", 'penalty_scaling_factor_drift',
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    sortby=["penalty_func", "mean_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation",
    plot_penalty_scaling_plots=dict(
        remove_rows={"penalty_scaling_factor_drift": 0.01},)
)

eval_model_dict_2dim_best_lambdagrid_modelcomp = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_modelcomp))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    noise_std_drift=0.02,
    saved_models_path=path_2dim_lambdagrid_modelcomp)


# ===================== plotting generator and discriminator paths =============
eval_model_dict_2dim_best_lambdagrid_plot = dict(
    model_ids=[83, 36, 54],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.15,
    noise_std_drift=0.02,
    plot_gen_disc_paths=[0,1,2,3,4,],
    load_saved_eval=False,
    saved_models_path=path_2dim_lambdagrid)


eval_model_dict_2dim_best_powU_plot = dict(
    model_ids=[311, 71, 191],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.15,
    noise_std_drift=0.02,
    plot_gen_disc_paths=[0,1,2,3,4,],
    load_saved_eval=False,
    saved_models_path=path_2dim_robd_power_util)


# ==============================================================================
# ============================ path wise penalties =============================
# ==============================================================================
path_2dim_lambdagrid_pwp = "{}saved_models_2dim_lambdagrid_pwp/".format(data_path)

params_list_2dim_lambdagrid_pwp = []

# non-robust
param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[0, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0],
        penalty_scaling_factor_drift=[0],
        initial_wealth=[1.],
        utility_func=["power-2", "power-0.5", "log"],
        penalty_func=[None, ],
        penalty_func_drift=[None,],
        penalty_function_ref_value=[
            [[0.15,0], [0, 0.35]],],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        gen_dict=[rnn_dict1],
        disc_dict=[disc_dict_no_disc_2],
        saved_models_path=[path_2dim_lambdagrid_pwp],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
    )
params_list_2dim_lambdagrid_pwp += get_parameter_array(
    param_dict=param_dict_2dim_lambdagrid)

# only sigma robust
path_wise_ref_val1_sig = np.array([[0.15, 0], [0, 0.35]])
T = data_dict2["nb_steps"]*data_dict2["dt"]
path_wise_ref_val1 = np.matmul(
    path_wise_ref_val1_sig, path_wise_ref_val1_sig.transpose())*T
for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
    param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[[1, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["power-2", "power-0.5", "log"],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[
            [[0.15, 0], [0, 0.35]], ],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        path_wise_penalty=[[
            {"path_functional": "config.get_quad_covar_log",
             "ref_value": path_wise_ref_val1.tolist(),
             "penalty_func": "config.get_penalty_function("
                             "'squarenorm-fro', None)",
             "scaling_factor": l1,
             "is_mean_penalty": False}]],
        gen_dict=[rnn_dict1],
        disc_dict=[rnn_dict2_d],
        saved_models_path=[path_2dim_lambdagrid_pwp],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
    )
    params_list_2dim_lambdagrid_pwp += get_parameter_array(
        param_dict=param_dict_2dim_lambdagrid)

# fully robust
path_wise_ref_val2 = np.exp(np.array([0.035, 0.055]) * T)
for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
    for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
        param_dict_2dim_lambdagrid = dict(
            test_size=[0.2],
            data_dict=['data_dict2'],
            epochs=[150],
            learning_rate_D=[5e-4], learning_rate_G=[5e-4],
            lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
            lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
            beta1_D=[0.9, ], beta2_D=[0.999, ],
            beta1_G=[0.9, ], beta2_G=[0.999, ],
            opt_steps_D_G=[[1, 1]],
            batch_size=[1000],
            penalty_scaling_factor=[0.],
            penalty_scaling_factor_drift=[0.],
            initial_wealth=[1.],
            utility_func=["power-2", "power-0.5", "log"],
            penalty_func=[None, ],
            penalty_func_drift=[None, ],
            penalty_function_ref_value=[
                [[0.15, 0], [0, 0.35]], ],
            penalty_function_ref_value_drift=[[0.035, 0.055]],
            path_wise_penalty=[[
                {"path_functional": "config.get_quad_covar_log",
                 "ref_value": path_wise_ref_val1.tolist(),
                 "penalty_func": "config.get_penalty_function("
                                 "'squarenorm-fro', None)",
                 "scaling_factor": l1,
                 "is_mean_penalty": False},
                {"path_functional": "config.get_mean_rel_return",
                 "ref_value": path_wise_ref_val2.tolist(),
                 "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                 "scaling_factor": l2,
                 "is_mean_penalty": True}]],
            gen_dict=[rnn_dict1],
            disc_dict=[rnn_dict3_d],
            saved_models_path=[path_2dim_lambdagrid_pwp],
            use_penalty_for_gen=[True],
            eval_on_train=[False],
            eval_noise_std=[0.15],
            eval_noise_std_drift=[0.02],
            eval_noise_seed=[8979],
            trans_cost_base=[0],
            trans_cost_perc=[0.01],
        )
        params_list_2dim_lambdagrid_pwp += get_parameter_array(
            param_dict=param_dict_2dim_lambdagrid)

TO_dict_2dim_lambdagrid_pwp = dict(
    ids_from=1, ids_to=len(params_list_2dim_lambdagrid_pwp),
    path=path_2dim_lambdagrid_pwp,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    # sortby=["mean_exp_util_with_noisy_par"],
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"mean_exp_util_with_noisy_par": np.nan,
                     'path_wise_penalty-1-scaling_factor': 0.01},)
)

eval_model_dict_2dim_best_lambdagrid_pwp = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    noise_std_drift=0.02,
    saved_models_path=path_2dim_lambdagrid_pwp)

plot_model_dict_2dim_best_lambdagrid_pwp = dict(
    model_ids=[126, 9, 3, 107, 14, 2, 52, 10, 1],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.15,
    noise_std_drift=0.02,
    plot_gen_disc_paths=[0,1,2,3,4,],
    load_saved_eval=False,
    saved_models_path=path_2dim_lambdagrid_pwp)

# ------------------------------------------------------------------------------
# ---- use non-constant noise (for eval) and min over all different random
#       measures in eval_pretrained_model.py
path_2dim_lambdagrid_pwp1 = "{}saved_models_2dim_lambdagrid_pwp1/".format(data_path)

params_list_2dim_lambdagrid_pwp1 = []

# non-robust
param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[0, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0],
        penalty_scaling_factor_drift=[0],
        initial_wealth=[1.],
        utility_func=["power-2", "power-0.5", "log"],
        penalty_func=[None, ],
        penalty_func_drift=[None,],
        penalty_function_ref_value=[
            [[0.15,0], [0, 0.35]],],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        gen_dict=[rnn_dict1],
        disc_dict=[disc_dict_no_disc_2],
        saved_models_path=[path_2dim_lambdagrid_pwp1],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
    )
params_list_2dim_lambdagrid_pwp1 += get_parameter_array(
    param_dict=param_dict_2dim_lambdagrid)

# only sigma robust
path_wise_ref_val1_sig = np.array([[0.15, 0], [0, 0.35]])
T = data_dict2["nb_steps"]*data_dict2["dt"]
path_wise_ref_val1 = np.matmul(
    path_wise_ref_val1_sig, path_wise_ref_val1_sig.transpose())*T
for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
    param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[[1, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["power-2", "power-0.5", "log"],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[
            [[0.15, 0], [0, 0.35]], ],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        path_wise_penalty=[[
            {"path_functional": "config.get_quad_covar_log",
             "ref_value": path_wise_ref_val1.tolist(),
             "penalty_func": "config.get_penalty_function("
                             "'squarenorm-fro', None)",
             "scaling_factor": l1,
             "is_mean_penalty": False}]],
        gen_dict=[rnn_dict1],
        disc_dict=[rnn_dict2_d],
        saved_models_path=[path_2dim_lambdagrid_pwp1],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
    )
    params_list_2dim_lambdagrid_pwp1 += get_parameter_array(
        param_dict=param_dict_2dim_lambdagrid)

# fully robust
path_wise_ref_val2 = np.exp(np.array([0.035, 0.055]) * T)
for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
    for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
        param_dict_2dim_lambdagrid = dict(
            test_size=[0.2],
            data_dict=['data_dict2'],
            epochs=[150],
            learning_rate_D=[5e-4], learning_rate_G=[5e-4],
            lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
            lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
            beta1_D=[0.9, ], beta2_D=[0.999, ],
            beta1_G=[0.9, ], beta2_G=[0.999, ],
            opt_steps_D_G=[[1, 1]],
            batch_size=[1000],
            penalty_scaling_factor=[0.],
            penalty_scaling_factor_drift=[0.],
            initial_wealth=[1.],
            utility_func=["power-2", "power-0.5", "log"],
            penalty_func=[None, ],
            penalty_func_drift=[None, ],
            penalty_function_ref_value=[
                [[0.15, 0], [0, 0.35]], ],
            penalty_function_ref_value_drift=[[0.035, 0.055]],
            path_wise_penalty=[[
                {"path_functional": "config.get_quad_covar_log",
                 "ref_value": path_wise_ref_val1.tolist(),
                 "penalty_func": "config.get_penalty_function("
                                 "'squarenorm-fro', None)",
                 "scaling_factor": l1,
                 "is_mean_penalty": False},
                {"path_functional": "config.get_mean_rel_return",
                 "ref_value": path_wise_ref_val2.tolist(),
                 "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                 "scaling_factor": l2,
                 "is_mean_penalty": True}]],
            gen_dict=[rnn_dict1],
            disc_dict=[rnn_dict3_d],
            saved_models_path=[path_2dim_lambdagrid_pwp1],
            use_penalty_for_gen=[True],
            eval_on_train=[False],
            eval_noise_std=[0.15],
            eval_noise_std_drift=[0.02],
            eval_noise_seed=[8979],
            eval_noise_type=["cumulative"],
            trans_cost_base=[0],
            trans_cost_perc=[0.01],
        )
        params_list_2dim_lambdagrid_pwp1 += get_parameter_array(
            param_dict=param_dict_2dim_lambdagrid)

TO_dict_2dim_lambdagrid_pwp1 = dict(
    ids_from=1, ids_to=len(params_list_2dim_lambdagrid_pwp1),
    path=path_2dim_lambdagrid_pwp1,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation1.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"mean_exp_util_with_noisy_par": np.nan,
                     'path_wise_penalty-1-scaling_factor': 0.01},
        col_names_dict={
            'path_wise_penalty-0-scaling_factor': '$\\lambda_1$',
            'path_wise_penalty-1-scaling_factor': '$\\lambda_2$',
            'min_exp_util_with_noisy_par': '$M_u(\\pi)$',
        }
    )
)

eval_model_dict_2dim_best_lambdagrid_pwp1_sn = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp1))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="nonconst",
    saved_models_path=path_2dim_lambdagrid_pwp1)

eval_model_dict_2dim_best_lambdagrid_pwp1_cum_sn = dict(
    filename="model_evaluation1",
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp1))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    saved_models_path=path_2dim_lambdagrid_pwp1)


plot_model_dict_2dim_best_lambdagrid_pwp1 = dict(
    model_ids=[86],
    load_best=True,
    nb_evaluations=100,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    plot_gen_disc_paths=[0,1,2,3,4,],
    plot_noisy_eval_paths=[0,1,2,3,4],
    load_saved_eval=False,
    saved_models_path=path_2dim_lambdagrid_pwp1)

plot_model_dict_2dim_best_lambdagrid_pwp1_baseline = dict(
    which_eval="baseline",
    model_ids=[86, 97, 45, 2, 1, 3],
    load_best=True,
    plot_eval_paths=[0,1,2],
    discount=True,
    saved_models_path=path_2dim_lambdagrid_pwp1)



# ----- model comparison (RNN, FFNN, timegrid)
path_2dim_lambdagrid_pwp1_modelcomp = \
    "{}saved_models_2dim_lambdagrid_pwp1_modelcomp/".format(data_path)
params_list_2dim_lambdagrid_pwp1_modelcomp = []
for gen, disc in [
    (ffnn_dict1, ffnn_dict1_d),
    (timegrid_nn_dict1, timegrid_nn_dict1_d1),
    (rnn_dict1, rnn_dict3_d)]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_2dim_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict2'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[[1, 1]],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["log"],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    [[0.15, 0], [0, 0.35]], ],
                penalty_function_ref_value_drift=[[0.035, 0.055]],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_2dim_lambdagrid_pwp1_modelcomp],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[0.15],
                eval_noise_std_drift=[0.02],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[0.01],
            )
            params_list_2dim_lambdagrid_pwp1_modelcomp += get_parameter_array(
                param_dict=param_dict_2dim_lambdagrid)

TO_dict_2dim_lambdagrid_pwp1_modelcomp = dict(
    ids_from=1, ids_to=len(params_list_2dim_lambdagrid_pwp1_modelcomp),
    path=path_2dim_lambdagrid_pwp1_modelcomp,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation1.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"min_exp_util_with_noisy_par": np.nan,
                     # 'path_wise_penalty-1-scaling_factor': 0.01
                     },)
)

eval_model_dict_2dim_best_lambdagrid_pwp1_modelcomp = dict(
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp1_modelcomp))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.15,
    noise_std_drift=0.02,
    noise_type="cumulative",
    saved_models_path=path_2dim_lambdagrid_pwp1_modelcomp)

eval_model_dict_2dim_best_lambdagrid_pwp1_modelcomp_sn = dict(
    filename="model_evaluation1",
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp1_modelcomp))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    saved_models_path=path_2dim_lambdagrid_pwp1_modelcomp)




# ------------------------------------------------------------------------------
# ---- use GARCH (for eval) and min over all different random
#       GARCH measures in eval_pretrained_model.py
path_2dim_lambdagrid_pwp_g = "{}saved_models_2dim_lambdagrid_pwp_garch/".format(
    data_path)

params_list_2dim_lambdagrid_pwp_g = []

# non-robust
param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma':0.2, 'step':100}],
        lr_scheduler_G=[{'gamma':0.2, 'step':100}],
        beta1_D=[0.9,], beta2_D=[0.999,],
        beta1_G=[0.9,], beta2_G=[0.999,],
        opt_steps_D_G=[[0, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0],
        penalty_scaling_factor_drift=[0],
        initial_wealth=[1.],
        utility_func=["power-2", "power-0.5", "log"],
        penalty_func=[None, ],
        penalty_func_drift=[None,],
        penalty_function_ref_value=[
            [[0.15,0], [0, 0.35]],],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        gen_dict=[rnn_dict1],
        disc_dict=[disc_dict_no_disc_2],
        saved_models_path=[path_2dim_lambdagrid_pwp_g],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
        eval_model_dict=[
                {"name": "GARCH",
                 "params": dict(p=1, q=1, dist="normal", mean="Constant"),
                 "noise": dict(params_scaling=1., rho_std=0.01),
                 "nb_samples": 500,
                 "fit_seq_len": 100000}],
    )
params_list_2dim_lambdagrid_pwp_g += get_parameter_array(
    param_dict=param_dict_2dim_lambdagrid)

# only sigma robust
path_wise_ref_val1_sig = np.array([[0.15, 0], [0, 0.35]])
T = data_dict2["nb_steps"]*data_dict2["dt"]
path_wise_ref_val1 = np.matmul(
    path_wise_ref_val1_sig, path_wise_ref_val1_sig.transpose())*T
for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
    param_dict_2dim_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[[1, 1]],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["power-2", "power-0.5", "log"],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[
            [[0.15, 0], [0, 0.35]], ],
        penalty_function_ref_value_drift=[[0.035, 0.055]],
        path_wise_penalty=[[
            {"path_functional": "config.get_quad_covar_log",
             "ref_value": path_wise_ref_val1.tolist(),
             "penalty_func": "config.get_penalty_function("
                             "'squarenorm-fro', None)",
             "scaling_factor": l1,
             "is_mean_penalty": False}]],
        gen_dict=[rnn_dict1],
        disc_dict=[rnn_dict2_d],
        saved_models_path=[path_2dim_lambdagrid_pwp_g],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[0.15],
        eval_noise_std_drift=[0.02],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[0.01],
        eval_model_dict=[
            {"name": "GARCH",
             "params": dict(p=1, q=1, dist="normal", mean="Constant"),
             "noise": dict(params_scaling=1., rho_std=0.01),
             "nb_samples": 500,
             "fit_seq_len": 100000}],
    )
    params_list_2dim_lambdagrid_pwp_g += get_parameter_array(
        param_dict=param_dict_2dim_lambdagrid)

# fully robust
path_wise_ref_val2 = np.exp(np.array([0.035, 0.055]) * T)
for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
    for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
        param_dict_2dim_lambdagrid = dict(
            test_size=[0.2],
            data_dict=['data_dict2'],
            epochs=[150],
            learning_rate_D=[5e-4], learning_rate_G=[5e-4],
            lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
            lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
            beta1_D=[0.9, ], beta2_D=[0.999, ],
            beta1_G=[0.9, ], beta2_G=[0.999, ],
            opt_steps_D_G=[[1, 1]],
            batch_size=[1000],
            penalty_scaling_factor=[0.],
            penalty_scaling_factor_drift=[0.],
            initial_wealth=[1.],
            utility_func=["power-2", "power-0.5", "log"],
            penalty_func=[None, ],
            penalty_func_drift=[None, ],
            penalty_function_ref_value=[
                [[0.15, 0], [0, 0.35]], ],
            penalty_function_ref_value_drift=[[0.035, 0.055]],
            path_wise_penalty=[[
                {"path_functional": "config.get_quad_covar_log",
                 "ref_value": path_wise_ref_val1.tolist(),
                 "penalty_func": "config.get_penalty_function("
                                 "'squarenorm-fro', None)",
                 "scaling_factor": l1,
                 "is_mean_penalty": False},
                {"path_functional": "config.get_mean_rel_return",
                 "ref_value": path_wise_ref_val2.tolist(),
                 "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                 "scaling_factor": l2,
                 "is_mean_penalty": True}]],
            gen_dict=[rnn_dict1],
            disc_dict=[rnn_dict3_d],
            saved_models_path=[path_2dim_lambdagrid_pwp_g],
            use_penalty_for_gen=[True],
            eval_on_train=[False],
            eval_noise_std=[0.15],
            eval_noise_std_drift=[0.02],
            eval_noise_seed=[8979],
            eval_noise_type=["cumulative"],
            trans_cost_base=[0],
            trans_cost_perc=[0.01],
            eval_model_dict=[
                {"name": "GARCH",
                 "params": dict(p=1, q=1, dist="normal", mean="Constant"),
                 "noise": dict(params_scaling=1., rho_std=0.01),
                 "nb_samples": 500,
                 "fit_seq_len": 100000}],
        )
        params_list_2dim_lambdagrid_pwp_g += get_parameter_array(
            param_dict=param_dict_2dim_lambdagrid)


TO_dict_2dim_lambdagrid_pwp_g = dict(
    ids_from=1, ids_to=len(params_list_2dim_lambdagrid_pwp_g),
    path=path_2dim_lambdagrid_pwp_g,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth', "seed",
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "eval_model_dict",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_util_with_GARCH",
         "eval_expected_util_with_GARCH",
         "max-eval_expected_util_with_GARCH"),
    ),
    sortby=["min_exp_util_with_garch"],
    model_eval_file="model_evaluation_garch2.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_garch',
        # remove_rows={"min_exp_util_with_garch": np.nan,
        #              'path_wise_penalty-1-scaling_factor': 0.01},
        col_names_dict={
            'path_wise_penalty-0-scaling_factor': '$\\lambda_1$',
            'path_wise_penalty-1-scaling_factor': '$\\lambda_2$',
            'min_exp_util_with_garch': '$M_u(\\pi)$',
        }
    )
)

eval_model_dict_2dim_best_lambdagrid_pwp_g = dict(
    filename="model_evaluation_garch1",
    which_eval="garch",
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp_g))),
    load_best=True,
    nb_evaluations=1000,
    discount=True,
    eval_model_dict={
        "eval_id": 0,
        "name": "GARCH",
        "params": dict(p=1, q=1, dist="normal", mean="Constant"),
        "noise": dict(params_scaling=0.5, rho_std=0.005),
        "fit_seq_len": 100000},
    load_saved_eval=True,
    nb_samples_per_param=5000,
    plot_garch_eval_paths=None,
    saved_models_path=path_2dim_lambdagrid_pwp_g)

eval_model_dict_2dim_best_lambdagrid_pwp_g_ln = dict(
    filename="model_evaluation_garch2",
    which_eval="garch",
    model_ids=list(range(1, 1+len(params_list_2dim_lambdagrid_pwp_g))),
    load_best=True,
    nb_evaluations=1000,
    discount=True,
    eval_model_dict={
        "eval_id": 2,
        "name": "GARCH",
        "params": dict(p=1, q=1, dist="normal", mean="Constant"),
        "noise": dict(params_scaling=0.75, rho_std=0.0075),
        "fit_seq_len": 100000},
    load_saved_eval=True,
    nb_samples_per_param=5000,
    plot_garch_eval_paths=None,
    saved_models_path=path_2dim_lambdagrid_pwp_g)

plot_model_dict_2dim_best_lambdagrid_pwp_g = dict(
    model_ids=[],
    filename="model_evaluation_garch_plot",
    which_eval="garch",
    load_best=True,
    nb_evaluations=100,
    discount=True,
    eval_model_dict={
        "eval_id": 0,
        "name": "GARCH",
        "params": dict(p=1, q=1, dist="normal", mean="Constant"),
        "noise": dict(params_scaling=0.5, rho_std=0.005),
        "fit_seq_len": 100000},
    load_saved_eval=True,
    nb_samples_per_param=5000,
    plot_garch_eval_paths=None,
    saved_models_path=path_2dim_lambdagrid_pwp_g)


# ==============================================================================
# ----- comparison with reference model for small trans costs
r = 0.015
data_dict0_1 = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 200000,
    "seed": 3940
}
data_dict0_1_test = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 40000,
    "seed": 3941
}
data_dict0_1_test_large = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 100000,
    "seed": 3942
}

mu_S = 0.04
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs

def ref_strategy_BS_STC(
        S, X, t, current_amount, mu=mu_S, sigma=sigma_S,
        gamma=gamma, ptc=ptc, return_Delta_pi=False):
    """
    this assumes that S is 1-dim and following a BS model
    see the paper: "A Primer on Portfolio Choice with Small Transaction Costs"

    args:
    - S: the price of the risky asset
    - X: the wealth
    - t: the current time
    - current_amount: the current amount of the risky asset
    - mu: the drift of the risky asset without r
    - sigma: the volatility of the risky asset
    - gamma: the risk aversion, i.e. the power in the utility function
    - ptc: the proportional transaction costs
    - return_Delta_pi: if True, additionally returns pi (i.e. trading strategy
        without trading costs) and Delta_pi
    """
    current_y = current_amount * S
    current_x = X - current_y
    pi = mu/(gamma*sigma**2)
    Delta_pi = (3/(2*gamma)*pi**2*(1-pi)**2)**(1/3)
    # if in the no trade region, the ref strategy is to keep the same amount of
    #   the risky asset
    ref_strat = current_y/X
    # the upper trade region
    T_upper = current_y/X > pi + Delta_pi*ptc**(1/3)
    # the lower trade region
    T_lower = current_y/X < pi - Delta_pi*ptc**(1/3)
    # if in the trade region, the ref strategy is to trade with boundary
    ref_strat[T_upper] = pi + Delta_pi*ptc**(1/3)
    ref_strat[T_lower] = pi - Delta_pi*ptc**(1/3)

    if return_Delta_pi:
        pi = torch.tensor(pi).reshape(1,1).repeat(len(S), 1)
        Delta_pi = torch.tensor(Delta_pi*ptc**(1/3)).reshape(1,1).repeat(len(S), 1)
        return ref_strat, pi, Delta_pi
    return ref_strat

def get_ref_strategy_BS_STC(mu=mu_S, sigma=sigma_S, gamma=gamma, ptc=ptc):
    return lambda S, X, t, current_amount, return_Delta_pi: ref_strategy_BS_STC(
        S=S, X=X, t=t, current_amount=current_amount, mu=mu, sigma=sigma,
        gamma=gamma, ptc=ptc, return_Delta_pi=return_Delta_pi)


nn1_2 = ((50, 'leaky_relu'),(50, 'leaky_relu'),)
nn1_3 = ((100, 'leaky_relu'),)
nn1_4 = ((100, 'tanh'),)
nn1_5 = ((100, 'relu'),)
nn1_6 = ((300, 'tanh'),)

ffnn_dict1_2 = {
    "name": "FFNN", "nn_desc": nn1_2, "dropout_rate": 0.1, "bias": True,}
ffnn_dict1_3 = {
    "name": "FFNN", "nn_desc": nn1_3, "dropout_rate": 0.1, "bias": True,}
ffnn_dict1_4 = {
    "name": "FFNN", "nn_desc": nn1_4, "dropout_rate": 0.1, "bias": True,}
ffnn_dict1_5 = {
    "name": "FFNN", "nn_desc": nn1_5, "dropout_rate": 0.1, "bias": True,}
ffnn_dict1_6 = {
    "name": "FFNN", "nn_desc": nn1_6, "dropout_rate": 0.1, "bias": True,}



path_lambdagrid_pwp_smalltranscost1 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost1/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost1 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_1'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost1],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
    )
    params_list_lambdagrid_pwp_smalltranscost1 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_1'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost1],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
            )
            params_list_lambdagrid_pwp_smalltranscost1 += get_parameter_array(
                param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost1 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost1),
    path=path_lambdagrid_pwp_smalltranscost1,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation1.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"min_exp_util_with_noisy_par": np.nan,
                     # 'path_wise_penalty-1-scaling_factor': 0.01
                     },)
)

eval_model_dict_lambdagrid_pwp_smalltranscost1 = dict(
    filename="model_evaluation1",
    model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost1))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_1_test",
    saved_models_path=path_lambdagrid_pwp_smalltranscost1)

eval_model_dict_lambdagrid_pwp_smalltranscost1_nonoise = dict(
    filename="model_evaluation2",
    model_ids=[1,2,3,4,5,6,34],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_1_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost1)

plot_model_dict_lambdagrid_pwp_smalltranscost1_1 = dict(
    filename="model_eval_plot",
    model_ids=[4, 34],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_1_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost1)

plot_model_dict_lambdagrid_pwp_smalltranscost1_2 = dict(
    filename="model_eval_plot",
    model_ids=[34],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_1_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost1)

plot_model_dict_lambdagrid_pwp_smalltranscost1_baseline = dict(
    which_eval="baseline",
    model_ids=[4, 34],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_1_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost1)


# --------- larger drift and risk-free rate ------------
r = 0.03
data_dict0_2 = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 200000,
    "seed": 3940
}
data_dict0_2_test = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 40000,
    "seed": 3941
}
data_dict0_2_test_large = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 100000,
    "seed": 3942
}

mu_S = 0.07
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost2 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost2/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost2 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_2'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost2],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364, 365, 366],
    )
    params_list_lambdagrid_pwp_smalltranscost2 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_2'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost2],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
                seed=[364, 365, 366],
            )
            params_list_lambdagrid_pwp_smalltranscost2 += get_parameter_array(
                param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost2 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost2),
    path=path_lambdagrid_pwp_smalltranscost2,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation1.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"min_exp_util_with_noisy_par": np.nan,
                     # 'path_wise_penalty-1-scaling_factor': 0.01
                     },)
)

eval_model_dict_lambdagrid_pwp_smalltranscost2 = dict(
    filename="model_evaluation1",
    model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost2))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_2_test",
    saved_models_path=path_lambdagrid_pwp_smalltranscost2)

eval_model_dict_lambdagrid_pwp_smalltranscost2_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,22))+[106],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_2_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost2)

plot_model_dict_lambdagrid_pwp_smalltranscost2_1 = dict(
    filename="model_eval_plot",
    model_ids=[4, 106],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_2_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost2)

plot_model_dict_lambdagrid_pwp_smalltranscost2_2 = dict(
    filename="model_eval_plot",
    model_ids=[106],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_2_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost2)

plot_model_dict_lambdagrid_pwp_smalltranscost2_baseline = dict(
    which_eval="baseline",
    model_ids=[4, 106],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_2_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost2)


# --------- smaller time horizon with large market------------
r = 0.03
data_dict0_3 = {
    "S0": [1], "dt": 1/650., "r": r, "nb_steps": 65, "nb_samples": 200000,
    "seed": 3940
}
data_dict0_3_test = {
    "S0": [1], "dt": 1/650., "r": r, "nb_steps": 65, "nb_samples": 40000,
    "seed": 3941
}
data_dict0_3_test_large = {
    "S0": [1], "dt": 1/650., "r": r, "nb_steps": 65, "nb_samples": 100000,
    "seed": 3942
}

T = 0.1
mu_S = 0.07
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost3 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost3/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost3 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_3'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost3],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364],
    )
    params_list_lambdagrid_pwp_smalltranscost3 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_3'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost3],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
                seed=[364,],
            )
            params_list_lambdagrid_pwp_smalltranscost3 += get_parameter_array(
                param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost3 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost3),
    path=path_lambdagrid_pwp_smalltranscost3,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation1.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"min_exp_util_with_noisy_par": np.nan,
                     # 'path_wise_penalty-1-scaling_factor': 0.01
                     },)
)

eval_model_dict_lambdagrid_pwp_smalltranscost3 = dict(
    filename="model_evaluation1",
    model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost3))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_3_test",
    saved_models_path=path_lambdagrid_pwp_smalltranscost3)

eval_model_dict_lambdagrid_pwp_smalltranscost3_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,8)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_3_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost3)

plot_model_dict_lambdagrid_pwp_smalltranscost3_1 = dict(
    filename="model_eval_plot",
    model_ids=[7, 43],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_3_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost3)

plot_model_dict_lambdagrid_pwp_smalltranscost3_2 = dict(
    filename="model_eval_plot",
    model_ids=[],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_3_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost3)

plot_model_dict_lambdagrid_pwp_smalltranscost3_baseline = dict(
    which_eval="baseline",
    model_ids=[6, 7, 43],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_3_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost3)


# --------- heavy tailed increments ------------
r = 0.03
data_dict0_4 = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 200000,
    "seed": 3940, "dist": "t", "df": 3.5,
}
data_dict0_4_test = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 40000,
    "seed": 3941, "dist": "t", "df": 3.5,
}
data_dict0_4_test_large = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 100000,
    "seed": 3942, "dist": "t", "df": 3.5,
}

T = 1.
mu_S = 0.07
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost4 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost4/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost4 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_4'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost4],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364,],
    )
    params_list_lambdagrid_pwp_smalltranscost4 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_4'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost4],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
                seed=[364,],
            )
            params_list_lambdagrid_pwp_smalltranscost4 += get_parameter_array(
                param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost4 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost4),
    path=path_lambdagrid_pwp_smalltranscost4,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation2.csv",
)

# eval_model_dict_lambdagrid_pwp_smalltranscost4 = dict(
#     filename="model_evaluation1",
#     model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost4))),
#     load_best=True,
#     nb_evaluations=1000,
#     noise_std=0.075,
#     noise_std_drift=0.01,
#     noise_type="cumulative",
#     test_data_dict="data_dict0_4_test",
#     saved_models_path=path_lambdagrid_pwp_smalltranscost4)

eval_model_dict_lambdagrid_pwp_smalltranscost4_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,8)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_4_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost4)

plot_model_dict_lambdagrid_pwp_smalltranscost4_1 = dict(
    filename="model_eval_plot",
    model_ids=[7],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_4_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost4)

# plot_model_dict_lambdagrid_pwp_smalltranscost4_2 = dict(
#     filename="model_eval_plot",
#     model_ids=[],
#     load_best=True,
#     nb_evaluations=1,
#     noise_std=0.075,
#     noise_std_drift=0.01,
#     noise_type="cumulative",
#     test_data_dict="data_dict0_4_test",
#     plot_strategy_refstrategy=[0,1,2,3,4,],
#     load_saved_eval=False,
#     noisy_eval=False,
#     saved_models_path=path_lambdagrid_pwp_smalltranscost4)

plot_model_dict_lambdagrid_pwp_smalltranscost4_baseline = dict(
    which_eval="baseline",
    model_ids=[7],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_4_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost4)



# --------- heavy tailed increments 2 ------------
r = 0.03
data_dict0_5 = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 200000,
    "seed": 3940, "dist": "t", "df": 20,
}
data_dict0_5_test = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 40000,
    "seed": 3941, "dist": "t", "df": 20,
}
data_dict0_5_test_large = {
    "S0": [1], "dt": 1/65., "r": r, "nb_steps": 65, "nb_samples": 100000,
    "seed": 3942, "dist": "t", "df": 20,
}

T = 1.
mu_S = 0.07
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost5 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost5/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost5 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_5'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost5],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364,],
    )
    params_list_lambdagrid_pwp_smalltranscost5 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_5'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost5],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
                seed=[364,],
            )
            params_list_lambdagrid_pwp_smalltranscost5 += get_parameter_array(
                param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost5 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost5),
    path=path_lambdagrid_pwp_smalltranscost5,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation1.csv",
    plot_penalty_scaling_plots=dict(
        psf_col1='path_wise_penalty-0-scaling_factor',
        psf_col2='path_wise_penalty-1-scaling_factor',
        target_col='min_exp_util_with_noisy_par',
        remove_rows={"min_exp_util_with_noisy_par": np.nan,
                     # 'path_wise_penalty-1-scaling_factor': 0.01
                     },)
)

eval_model_dict_lambdagrid_pwp_smalltranscost5 = dict(
    filename="model_evaluation1",
    model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost5))),
    load_best=True,
    nb_evaluations=1000,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_5_test",
    saved_models_path=path_lambdagrid_pwp_smalltranscost5)

eval_model_dict_lambdagrid_pwp_smalltranscost5_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,8)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_5_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost5)

plot_model_dict_lambdagrid_pwp_smalltranscost5_1 = dict(
    filename="model_eval_plot",
    model_ids=[7, 30],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_5_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost5)

plot_model_dict_lambdagrid_pwp_smalltranscost5_2 = dict(
    filename="model_eval_plot",
    model_ids=[30],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.075,
    noise_std_drift=0.01,
    noise_type="cumulative",
    test_data_dict="data_dict0_5_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost5)

plot_model_dict_lambdagrid_pwp_smalltranscost5_baseline = dict(
    which_eval="baseline",
    model_ids=[7, 30],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_5_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost5)



# --------- larger time horizon with large market------------
r = 0.03
data_dict0_6 = {
    "S0": [1], "dt": 1/130., "r": r, "nb_steps": 260, "nb_samples": 200000,
    "seed": 3940
}
data_dict0_6_test = {
    "S0": [1], "dt": 1/130., "r": r, "nb_steps": 260, "nb_samples": 40000,
    "seed": 3941
}
data_dict0_6_test_large = {
    "S0": [1], "dt": 1/130., "r": r, "nb_steps": 260, "nb_samples": 100000,
    "seed": 3942
}

T = 2
mu_S = 0.07
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost6 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost6/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost6 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_6'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost6],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364],
    )
    params_list_lambdagrid_pwp_smalltranscost6 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_6'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost6],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
                seed=[364,],
            )
            # params_list_lambdagrid_pwp_smalltranscost6 += get_parameter_array(
            #     param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost6 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost6),
    path=path_lambdagrid_pwp_smalltranscost6,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation2.csv",
    # plot_penalty_scaling_plots=dict(
    #     psf_col1='path_wise_penalty-0-scaling_factor',
    #     psf_col2='path_wise_penalty-1-scaling_factor',
    #     target_col='min_exp_util_with_noisy_par',
    #     remove_rows={"min_exp_util_with_noisy_par": np.nan,
    #                  # 'path_wise_penalty-1-scaling_factor': 0.01
    #                  },)
)

# eval_model_dict_lambdagrid_pwp_smalltranscost6 = dict(
#     filename="model_evaluation1",
#     model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost6))),
#     load_best=True,
#     nb_evaluations=1000,
#     noise_std=0.075,
#     noise_std_drift=0.01,
#     noise_type="cumulative",
#     test_data_dict="data_dict0_6_test",
#     saved_models_path=path_lambdagrid_pwp_smalltranscost6)

eval_model_dict_lambdagrid_pwp_smalltranscost6_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,8)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_6_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost6)

plot_model_dict_lambdagrid_pwp_smalltranscost6_1 = dict(
    filename="model_eval_plot",
    model_ids=[2],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_6_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost6)

# plot_model_dict_lambdagrid_pwp_smalltranscost6_2 = dict(
#     filename="model_eval_plot",
#     model_ids=[],
#     load_best=True,
#     nb_evaluations=1,
#     noise_std=0.075,
#     noise_std_drift=0.01,
#     noise_type="cumulative",
#     test_data_dict="data_dict0_6_test",
#     plot_strategy_refstrategy=[0,1,2,3,4,],
#     load_saved_eval=False,
#     noisy_eval=False,
#     saved_models_path=path_lambdagrid_pwp_smalltranscost6)

plot_model_dict_lambdagrid_pwp_smalltranscost6_baseline = dict(
    which_eval="baseline",
    model_ids=[2],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_6_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost6)


# --------- larger time horizon with small market------------
r = 0.015
data_dict0_7 = {
    "S0": [1], "dt": 1/130., "r": r, "nb_steps": 260, "nb_samples": 200000,
    "seed": 3940
}
data_dict0_7_test = {
    "S0": [1], "dt": 1/130., "r": r, "nb_steps": 260, "nb_samples": 40000,
    "seed": 3941
}
data_dict0_7_test_large = {
    "S0": [1], "dt": 1/130., "r": r, "nb_steps": 260, "nb_samples": 100000,
    "seed": 3942
}

T = 2
mu_S = 0.04
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost7 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost7/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost7 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_7'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost7],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364],
    )
    params_list_lambdagrid_pwp_smalltranscost7 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

# fully robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, ffnn_dict1_d, 0.15, 0.02, [1,1]),]:
    for l1 in [0.01, 0.5, 0.1, 1., 10, 100]:
        for l2 in [0.01, 0.5, 0.1, 1., 10, 100]:
            param_dict_lambdagrid = dict(
                test_size=[0.2],
                data_dict=['data_dict0_7'],
                epochs=[150],
                learning_rate_D=[5e-4], learning_rate_G=[5e-4],
                lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
                lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
                beta1_D=[0.9, ], beta2_D=[0.999, ],
                beta1_G=[0.9, ], beta2_G=[0.999, ],
                opt_steps_D_G=[opt_s],
                batch_size=[1000],
                penalty_scaling_factor=[0.],
                penalty_scaling_factor_drift=[0.],
                initial_wealth=[1.],
                utility_func=["powerc-{}".format(gamma)],
                penalty_func=[None, ],
                penalty_func_drift=[None, ],
                penalty_function_ref_value=[
                    sigma, ],
                penalty_function_ref_value_drift=[mu],
                path_wise_penalty=[[
                    {"path_functional": "config.get_quad_covar_log",
                     "ref_value": path_wise_ref_val1_2.tolist(),
                     "penalty_func": "config.get_penalty_function("
                                     "'squarenorm-fro', None)",
                     "scaling_factor": l1,
                     "is_mean_penalty": False},
                    {"path_functional": "config.get_mean_rel_return",
                     "ref_value": path_wise_ref_val2_2.tolist(),
                     "penalty_func": "lambda x,y: torch.linalg.norm(x-y, ord=2)**2",
                     "scaling_factor": l2,
                     "is_mean_penalty": True}]],
                gen_dict=[gen],
                disc_dict=[disc],
                saved_models_path=[path_lambdagrid_pwp_smalltranscost7],
                use_penalty_for_gen=[True],
                eval_on_train=[False],
                eval_noise_std=[e_n],
                eval_noise_std_drift=[e_n_d],
                eval_noise_seed=[8979],
                eval_noise_type=["cumulative"],
                trans_cost_base=[0],
                trans_cost_perc=[ptc],
                ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
                    mu_S, sigma_S, gamma, ptc)],
                seed=[364,],
            )
            # params_list_lambdagrid_pwp_smalltranscost7 += get_parameter_array(
            #     param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost7 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost7),
    path=path_lambdagrid_pwp_smalltranscost7,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation2.csv",
    # plot_penalty_scaling_plots=dict(
    #     psf_col1='path_wise_penalty-0-scaling_factor',
    #     psf_col2='path_wise_penalty-1-scaling_factor',
    #     target_col='min_exp_util_with_noisy_par',
    #     remove_rows={"min_exp_util_with_noisy_par": np.nan,
    #                  # 'path_wise_penalty-1-scaling_factor': 0.01
    #                  },)
)

# eval_model_dict_lambdagrid_pwp_smalltranscost6 = dict(
#     filename="model_evaluation1",
#     model_ids=list(range(1, 1+len(params_list_lambdagrid_pwp_smalltranscost6))),
#     load_best=True,
#     nb_evaluations=1000,
#     noise_std=0.075,
#     noise_std_drift=0.01,
#     noise_type="cumulative",
#     test_data_dict="data_dict0_6_test",
#     saved_models_path=path_lambdagrid_pwp_smalltranscost6)

eval_model_dict_lambdagrid_pwp_smalltranscost7_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,8)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_7_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost7)

plot_model_dict_lambdagrid_pwp_smalltranscost7_1 = dict(
    filename="model_eval_plot",
    model_ids=[7],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_7_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost7)

# plot_model_dict_lambdagrid_pwp_smalltranscost6_2 = dict(
#     filename="model_eval_plot",
#     model_ids=[],
#     load_best=True,
#     nb_evaluations=1,
#     noise_std=0.075,
#     noise_std_drift=0.01,
#     noise_type="cumulative",
#     test_data_dict="data_dict0_6_test",
#     plot_strategy_refstrategy=[0,1,2,3,4,],
#     load_saved_eval=False,
#     noisy_eval=False,
#     saved_models_path=path_lambdagrid_pwp_smalltranscost6)

plot_model_dict_lambdagrid_pwp_smalltranscost7_baseline = dict(
    which_eval="baseline",
    model_ids=[7],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_7_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost7)


# --------- smaller time horizon with small market------------
r = 0.015
data_dict0_8 = {
    "S0": [1], "dt": 1/650., "r": r, "nb_steps": 65, "nb_samples": 200000,
    "seed": 3940
}
data_dict0_8_test = {
    "S0": [1], "dt": 1/650., "r": r, "nb_steps": 65, "nb_samples": 40000,
    "seed": 3941
}
data_dict0_8_test_large = {
    "S0": [1], "dt": 1/650., "r": r, "nb_steps": 65, "nb_samples": 100000,
    "seed": 3942
}

T = 0.1
mu_S = 0.04
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)
ptc = 0.01  # proportional trading costs


path_lambdagrid_pwp_smalltranscost8 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost8/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost8 = []

# non-robust
for gen, disc, e_n, e_n_d, opt_s in [
    (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
    (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
    (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
]:
    param_dict_lambdagrid = dict(
        test_size=[0.2],
        data_dict=['data_dict0_8'],
        epochs=[150],
        learning_rate_D=[5e-4], learning_rate_G=[5e-4],
        lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
        lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
        beta1_D=[0.9, ], beta2_D=[0.999, ],
        beta1_G=[0.9, ], beta2_G=[0.999, ],
        opt_steps_D_G=[opt_s],
        batch_size=[1000],
        penalty_scaling_factor=[0.],
        penalty_scaling_factor_drift=[0.],
        initial_wealth=[1.],
        utility_func=["powerc-{}".format(gamma)],
        penalty_func=[None, ],
        penalty_func_drift=[None, ],
        penalty_function_ref_value=[sigma,],
        penalty_function_ref_value_drift=[mu],
        gen_dict=[gen],
        disc_dict=[disc],
        saved_models_path=[path_lambdagrid_pwp_smalltranscost8],
        use_penalty_for_gen=[True],
        eval_on_train=[False],
        eval_noise_std=[e_n],
        eval_noise_std_drift=[e_n_d],
        eval_noise_seed=[8979],
        eval_noise_type=["cumulative"],
        trans_cost_base=[0],
        trans_cost_perc=[ptc],
        ref_strategy=["config.get_ref_strategy_BS_STC("
                      "mu={}, sigma={}, gamma={}, ptc={})".format(
            mu_S, sigma_S, gamma, ptc)],
        seed=[364],
    )
    params_list_lambdagrid_pwp_smalltranscost8 += get_parameter_array(
        param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost8 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost8),
    path=path_lambdagrid_pwp_smalltranscost8,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation2.csv",
)

eval_model_dict_lambdagrid_pwp_smalltranscost8_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,8)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_8_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost8)

plot_model_dict_lambdagrid_pwp_smalltranscost8_1 = dict(
    filename="model_eval_plot",
    model_ids=[7],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_8_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost8)

plot_model_dict_lambdagrid_pwp_smalltranscost8_baseline = dict(
    which_eval="baseline",
    model_ids=[7],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_8_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost8)


# --------- smaller trading costs with small market------------
r = 0.015
T = 1.
mu_S = 0.04
sigma_S = 0.35
mu = [r+mu_S,]
sigma = [[sigma_S,]]
gamma = 0.5  # power of utility
path_wise_ref_val1_2 = np.array(sigma)**2*T
path_wise_ref_val2_2 = np.exp(np.array(mu)*T)


path_lambdagrid_pwp_smalltranscost9 = \
    "{}saved_models_lambdagrid_pwp_smalltranscost9/".format(data_path)
params_list_lambdagrid_pwp_smalltranscost9 = []

# non-robust
for ptc in [0.001, 0.0001]:
    for gen, disc, e_n, e_n_d, opt_s in [
        (ffnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
        (ffnn_dict1_2, disc_dict_no_disc_2, 0., 0., [0,1]),
        (ffnn_dict1_3, disc_dict_no_disc_2, 0., 0., [0,1]),
        (ffnn_dict1_4, disc_dict_no_disc_2, 0., 0., [0,1]),
        (ffnn_dict1_5, disc_dict_no_disc_2, 0., 0., [0,1]),
        (rnn_dict1, disc_dict_no_disc_2, 0., 0., [0,1]),
        (ffnn_dict1_6, disc_dict_no_disc_2, 0., 0., [0,1]),
    ]:
        param_dict_lambdagrid = dict(
            test_size=[0.2],
            data_dict=['data_dict0_1'],
            epochs=[150],
            learning_rate_D=[5e-4], learning_rate_G=[5e-4],
            lr_scheduler_D=[{'gamma': 0.2, 'step': 100}],
            lr_scheduler_G=[{'gamma': 0.2, 'step': 100}],
            beta1_D=[0.9, ], beta2_D=[0.999, ],
            beta1_G=[0.9, ], beta2_G=[0.999, ],
            opt_steps_D_G=[opt_s],
            batch_size=[1000],
            penalty_scaling_factor=[0.],
            penalty_scaling_factor_drift=[0.],
            initial_wealth=[1.],
            utility_func=["powerc-{}".format(gamma)],
            penalty_func=[None, ],
            penalty_func_drift=[None, ],
            penalty_function_ref_value=[sigma,],
            penalty_function_ref_value_drift=[mu],
            gen_dict=[gen],
            disc_dict=[disc],
            saved_models_path=[path_lambdagrid_pwp_smalltranscost9],
            use_penalty_for_gen=[True],
            eval_on_train=[False],
            eval_noise_std=[e_n],
            eval_noise_std_drift=[e_n_d],
            eval_noise_seed=[8979],
            eval_noise_type=["cumulative"],
            trans_cost_base=[0],
            trans_cost_perc=[ptc],
            ref_strategy=["config.get_ref_strategy_BS_STC("
                          "mu={}, sigma={}, gamma={}, ptc={})".format(
                mu_S, sigma_S, gamma, ptc)],
            seed=[364],
        )
        params_list_lambdagrid_pwp_smalltranscost9 += get_parameter_array(
            param_dict=param_dict_lambdagrid)

TO_dict_lambdagrid_pwp_smalltranscost9 = dict(
    ids_from=1, ids_to=len(params_list_lambdagrid_pwp_smalltranscost9),
    path=path_lambdagrid_pwp_smalltranscost9,
    params_extract_desc=(
        'gen_dict', 'disc_dict', 'learning_rate_D',
        'learning_rate_G', 'opt_steps_D_G', 'batch_size',
        "epochs", 'utility_func', 'initial_wealth',
        "penalty_function_ref_value",
        "penalty_function_ref_value_drift",
        "path_wise_penalty-0-scaling_factor",
        "path_wise_penalty-1-scaling_factor",
        "data_dict", "eval_noise_std", "eval_noise_std_drift",
        "eval_noise_type", "seed",
        "trans_cost_perc", "trans_cost_base",),
    vals_metric_extract=(
        ("max", "eval_expected_utility", "eval_expected_utility",
         "max-eval_expected_utility"),
        ("last", "analytic_sol_penalty", "analytic_sol_penalty",
         "analytic_sol_penalty"),
        ("last", "analytic_sol_expected_utility",
         "analytic_sol_expected_utility", "analytic_sol_expected_utility"),
        ("max", "eval_expected_util_with_analytic_par",
         "eval_expected_util_with_analytic_par",
         "max-eval_expected_util_with_analytic_par"),
        ("max", "eval_expected_util_with_noisy_par",
         "eval_expected_util_with_noisy_par",
         "max-eval_expected_util_with_noisy_par"),
        ("max", "eval_expected_utility_with_ref_strategy_noisy_par",
         "eval_expected_utility_with_ref_strategy_noisy_par",
         "max-eval_expected_utility_with_ref_strategy_noisy_par")
    ),
    sortby=["min_exp_util_with_noisy_par"],
    model_eval_file="model_evaluation2.csv",
)

eval_model_dict_lambdagrid_pwp_smalltranscost9_nonoise = dict(
    filename="model_evaluation2",
    model_ids=list(range(1,15)),
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_1_test_large",
    saved_models_path=path_lambdagrid_pwp_smalltranscost9)

plot_model_dict_lambdagrid_pwp_smalltranscost9_1 = dict(
    filename="model_eval_plot",
    model_ids=[7, 10],
    load_best=True,
    nb_evaluations=1,
    noise_std=0.0,
    noise_std_drift=0.0,
    noise_type="cumulative",
    test_data_dict="data_dict0_1_test",
    plot_strategy_refstrategy=[0,1,2,3,4,],
    plot_ref_nb_stocks_NN=False,
    load_saved_eval=False,
    noisy_eval=False,
    saved_models_path=path_lambdagrid_pwp_smalltranscost9)

plot_model_dict_lambdagrid_pwp_smalltranscost9_baseline = dict(
    which_eval="baseline",
    model_ids=[7, 10],
    load_best=True,
    plot_eval_paths=None,
    discount=True,
    test_data_dict="data_dict0_1_test_large",
    compute_value_at_risk=0.05,
    compute_for_ref_strategy=True,
    saved_models_path=path_lambdagrid_pwp_smalltranscost9)




if __name__ == '__main__':
    pass
