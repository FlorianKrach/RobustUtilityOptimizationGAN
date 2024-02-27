"""
author: Florian Krach
"""

# ==============================================================================
import copy
import gc

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import os, sys, time
import socket
import warnings
import scipy.stats as sstats

import data_utils
import config
import extras



# ==============================================================================
# GLOBAL VARIABLES
activation_dict = config.activation_dict



# ==============================================================================
# FUNCTIONS
def init_weights(m, bias=0.0):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(bias)


def save_checkpoint(model, optimizers, lr_schedulers, path, epoch):
    """
    save a trained torch model and the used optimizer at the given path, s.t.
    training can be resumed at the exact same point
    :param model: a torch model
    :param optimizers: list of torch optimizer
    :param lr_schedulers: list of torch lr_schedulers or Nones
    :param path: str, the path where to save the model
    :param epoch: int, the current epoch
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, 'checkpt.tar')
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    for i, opt in enumerate(optimizers):
        if opt is not None:
            save_dict["optimizer_state_dict_{}".format(i)] = opt.state_dict()
    for i, lrs in enumerate(lr_schedulers):
        if lrs is not None:
            save_dict["lr_scheduler_state_dict_{}".format(i)] = lrs.state_dict()
    torch.save(save_dict, filename)


def get_ckpt_model(ckpt_path, model, optimizers, lr_schedulers, device):
    """
    load a saved torch model and its optimizer, inplace
    :param ckpt_path: str, path where the model is saved
    :param model: torch model instance, of which the weights etc. should be
            reloaded
    :param optimizers: list of torch optimizer, which should be loaded
    :param lr_schedulers: list of torch lr_schedulers or Nones, to load
    :param device: the device to which the model should be loaded
    """
    ckpt_path = os.path.join(ckpt_path, 'checkpt.tar')
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    state_dict = checkpt['model_state_dict']
    for i, opt in enumerate(optimizers):
        if opt is not None:
            opt.load_state_dict(checkpt['optimizer_state_dict_{}'.format(i)])
    for i, lrs in enumerate(lr_schedulers):
        if lrs is not None:
            lrs.load_state_dict(checkpt['lr_scheduler_state_dict_{}'.format(i)])
    model.load_state_dict(state_dict)
    model.epoch = checkpt['epoch']
    model.to(device)


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias, **kwargs):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict activation_dict
            for possible options). If None (or "linear"): use a linear map.
            Possibility for the last element of the list to be a str (instead of
            list), which defines the final activation function (i.e. output
            layer has an activation applied).
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN class
    """
    if nn_desc is None or nn_desc == "linear":
        layers = [torch.nn.Linear(input_size, output_size, bias=bias)]
    elif len(nn_desc) == 2 and nn_desc[0] == "linear" and \
            isinstance(nn_desc[-1], str):
        layers = [torch.nn.Linear(input_size, output_size, bias=bias),
                  activation_dict[nn_desc[-1]]()]
    else:
        final_activ = None
        if isinstance(nn_desc[-1], str):
            final_activ = nn_desc[-1]
            nn_desc = nn_desc[:-1]
        if len(nn_desc) > 0:
            layers = [torch.nn.Linear(input_size, nn_desc[0][0], bias=bias)]
            if len(nn_desc) > 1:
                for i in range(len(nn_desc)-1):
                    layers.append(activation_dict[nn_desc[i][1]]())
                    layers.append(torch.nn.Dropout(p=dropout_rate))
                    layers.append(torch.nn.Linear(nn_desc[i][0], nn_desc[i+1][0],
                                                  bias=bias))
            layers.append(activation_dict[nn_desc[-1][1]]())
            layers.append(torch.nn.Dropout(p=dropout_rate))
            layers.append(torch.nn.Linear(nn_desc[-1][0], output_size, bias=bias))
        else:
            layers = []
        if final_activ is not None:
            layers.append(activation_dict[final_activ]())
    return torch.nn.Sequential(*layers)



def get_ffnn_separate_last(input_size, output_size, nn_desc, dropout_rate, bias,
                           **kwargs):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict activation_dict
            for possible options). If None (or "linear"): use a linear map.
            Possibility for the last element of the list to be a str (instead of
            list), which defines the final activation function (i.e. output
            layer has an activation applied).
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN class
    """
    layers_last = []
    layers = []
    if nn_desc is None or nn_desc == "linear":
        layers_last = [torch.nn.Linear(input_size, output_size, bias=bias)]
    elif len(nn_desc) == 2 and nn_desc[0] == "linear" and \
            isinstance(nn_desc[-1], str):
        layers_last = [torch.nn.Linear(input_size, output_size, bias=bias),
                  activation_dict[nn_desc[-1]]()]
    else:
        final_activ = None
        if isinstance(nn_desc[-1], str):
            final_activ = nn_desc[-1]
            nn_desc = nn_desc[:-1]
        if len(nn_desc) > 0:
            layers = [torch.nn.Linear(input_size, nn_desc[0][0], bias=bias)]

            if len(nn_desc) > 1:
                for i in range(len(nn_desc)-1):
                    layers.append(activation_dict[nn_desc[i][1]]())
                    layers.append(torch.nn.Dropout(p=dropout_rate))
                    layers.append(torch.nn.Linear(nn_desc[i][0], nn_desc[i+1][0],
                                                  bias=bias))
            layers.append(activation_dict[nn_desc[-1][1]]())
            layers.append(torch.nn.Dropout(p=dropout_rate))
            layers_last.append(torch.nn.Linear(nn_desc[-1][0], output_size, bias=bias))
        else:
            layers = []
        if final_activ is not None:
            layers_last.append(activation_dict[final_activ]())
    return torch.nn.Sequential(*layers), torch.nn.Sequential(*layers_last)


# ==============================================================================
# CLASSES
class timegrid_nns(torch.nn.Module):
    """
    within each time interval a different NN is applied. Automatically chooses
    correct network from time input.
    """
    def __init__(self, input_size, output_size, nn_desc, dropout_rate, timegrid,
                 bias=True, **kwargs):
        """
        Args:
            input_size:
            output_size:
            nn_desc:
            dropout_rate:
            bias:
            timegrid: list, each element is a time from which on a new network
                is used, hence there are len of list +1 networks. Do not include
                0 or np.infty in the list. If the last time point is the max
                time that is used, do not include it (since another network
                for times until np.infty is added).
        """
        super().__init__()
        self.timegrid = timegrid + [np.infty]
        self.networks = torch.nn.ModuleList([get_ffnn(
            input_size, output_size, nn_desc, dropout_rate, bias)
            for t in self.timegrid])

    def forward(self, time, input):
        which_net = int(np.argmax(time < np.array(self.timegrid)))
        return self.networks[which_net](input)



class FFNN(torch.nn.Module):
    def __init__(
            self, input_size, output_size, nn_desc, dropout_rate, bias=True,
            **kwargs):
        """
        Args:
            input_size:
            output_size:
            nn_desc:
            dropout_rate:
            bias:

        the NN is split in all but the last layer, called hidden, and the last
        layer, called readout. This allows to train the readout layer alone.
        """
        super().__init__()
        self.hidden, self.readout = get_ffnn_separate_last(
            input_size, output_size, nn_desc, dropout_rate, bias)

    def forward(self, t, input):
        hidden = self.hidden(input)
        readout = self.readout(hidden)
        return readout


class RNN(torch.nn.Module):
    """
    RNN with adjustable cell and readout
    """
    def __init__(
            self, hidden_desc, readout_desc,
            input_size, hidden_size, output_size,
            dropout_rate=0.0, bias=True, return_hidden=False,
            **kwargs
    ):
        """
        Args:
            hidden_desc: list of list, describing RNNcell (NN producing hidden
                latent variable)
            readout_desc: list of list, describing readout NN
            input_size: dim of input
            hidden_size: dim of hidden latent variable
            output_size: dim of output
            dropout_rate:
            bias: bool
            return_hidden: bool
            **kwargs: collects unused inputs
        """
        # raise NotImplementedError
        super().__init__()
        self.return_hidden = return_hidden
        self.hidden = get_ffnn(
            input_size+hidden_size, hidden_size, hidden_desc, dropout_rate, bias
        )
        self.readout = get_ffnn(
            hidden_size, output_size,  readout_desc, dropout_rate, bias
        )
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.h = None

    def reset_h(self):
        self.h = None

    def forward(self, t, input, return_hidden=False):
        """
        :param t: float, current time, unused
        :param input: torch.tensor, shape [batch_size, input_size]
                or if size is given
        :param return_hidden: bool, whether to return the current hidden state
        """
        if self.h is None:
            self.h = torch.zeros((input.shape[0], self.hidden_size))
        x = torch.cat([input, self.h], dim=-1)
        self.h = self.hidden(x)
        output = self.readout(self.h)

        if return_hidden:
            return output, self.h
        else:
            return output


MODEL_DICT = {
    "RNN": RNN,
    "timegrid_nn": timegrid_nns,
    "FFNN": FFNN,
}


class MinMaxModel(torch.nn.Module):
    """
    model to train the investor and the market in adversary 'GAN'-like training
    """
    def __init__(
            self, dimension, scaling_coeff, scaling_coeff_drift,
            penalty_func, penalty_func_drift, utility_func,
            penalty_function_ref_value, penalty_function_ref_value_drift,
            gen_dict, disc_dict, dt, r, S0,
            path_wise_penalty=None,
            initial_wealth=1, use_penalty_for_gen=False, use_log_return=False,
            use_opponents_output=False, trans_cost_perc=0., trans_cost_base=0.,
            nb_steps=None, numerically_stabilised_training=False,
            input_increments=False, use_general_SDE=False,
            train_penalty_threshold=None,
            **options,
    ):
        super().__init__()
        self.dimension = dimension
        self.scaling_coeff = scaling_coeff
        self.scaling_coeff_drift = scaling_coeff_drift
        self.dt = dt
        self.r = r
        self.S0 = S0
        self.nb_steps = nb_steps
        self.T = self.dt*self.nb_steps
        self.X0 = initial_wealth
        self.ref0 = penalty_function_ref_value
        self.ref0_drift = penalty_function_ref_value_drift
        self.trans_cost_perc = trans_cost_perc
        self.trans_cost_base = trans_cost_base
        self.path_wise_penalty = []
        self.numerically_stabilised_training = numerically_stabilised_training
        self.input_increments = input_increments
        self.use_general_SDE = use_general_SDE
        self.train_penalty_threshold = train_penalty_threshold
        if path_wise_penalty is not None:
            for d in path_wise_penalty:
                d_new = {}
                d_new["path_functional"] = eval(d["path_functional"])
                d_new["penalty_func"] = eval(d["penalty_func"])
                if isinstance(d["ref_value"], str):
                    d_new["ref_val_path_functional"] = True
                    d_new["ref_value"] = eval(d["ref_value"])
                else:
                    d_new["ref_val_path_functional"] = False
                    d_new["ref_value"] = torch.tensor(
                        d["ref_value"], dtype=torch.float64)
                d_new["scaling_factor"] = d["scaling_factor"]
                d_new["is_mean_penalty"] = d["is_mean_penalty"]
                self.path_wise_penalty.append(d_new)
        if isinstance(self.ref0, (list, tuple, np.ndarray, float)):
            self.ref0 = torch.tensor(self.ref0, dtype=torch.float32)
        if isinstance(self.ref0_drift, (list, tuple, np.ndarray, float)):
            self.ref0_drift = torch.tensor(self.ref0_drift, dtype=torch.float32)
        self.penalty_func = config.get_penalty_function(penalty_func, self.ref0)
        self.penalty_func_drift = config.get_penalty_function_drift(
            penalty_func_drift)
        self.utility_func = config.get_utility_function(utility_func)
        ref0_np = None
        if self.ref0 is not None:
            ref0_np = self.ref0.detach().numpy()
        self.penalty_func_np = config.get_penalty_function_numpy(
            penalty_func, ref0_np)
        self.penalty_func_drift_np = config.get_penalty_function_drift_numpy(
            penalty_func_drift)
        self.utility_func_np = config.get_utility_function_numpy(utility_func)
        self.use_penalty_for_gen = use_penalty_for_gen
        self.use_log_return = use_log_return
        self.use_opponents_output = use_opponents_output
        if self.use_opponents_output:
            self.current_disc = [torch.zeros(size=())]
        robust_pi = True
        if disc_dict["vola"] in ["ref", "sig0", "ref0"]:
            robust_pi = False
        self.analytic_sol = extras.get_analytical_solution(
            utility_function=utility_func, penalty_function=penalty_func,
            penalty_function_ref_value=penalty_function_ref_value,
            dimension=dimension, r=r, mu=disc_dict["drift"],
            penalty_scaling_coeff=scaling_coeff, robust_pi=robust_pi,
            penalty_function_drift=penalty_func_drift,
            penalty_function_ref_value_drift=penalty_function_ref_value_drift,
            penalty_scaling_coeff_drift=scaling_coeff_drift,
            path_wise_penalty=path_wise_penalty,
            **options)
        sig_, mu_, pi_ = self.analytic_sol
        if sig_ is not None:
            self.penalty_func_for_eval = config.get_penalty_function(
                penalty_func, torch.tensor(sig_, dtype=torch.float32))
        else:
            self.penalty_func_for_eval = lambda x, y: torch.nan

        # generator
        self.generator_name = gen_dict["name"]
        gen_dict["input_size"] = self.dimension+2
        gen_dict["output_size"] = self.dimension
        self.generator = MODEL_DICT[self.generator_name](**gen_dict)

        # discriminator
        self.discriminator_name = disc_dict["name"]
        disc_dict["input_size"] = self.dimension+2
        output_size = 0
        drift_ = disc_dict["drift"]
        vola_ = disc_dict["vola"]
        self.disc_drift = False
        self.disc_vola = False
        if drift_ in ["nn", "NN"]:
            output_size += self.dimension
            self.disc_drift = True
        self.vola_dim = self.dimension**2
        if vola_ in ["nn", "NN"]:
            output_size += self.vola_dim
            self.disc_vola = True
        disc_dict["output_size"] = output_size
        if output_size > 0:
            self.discriminator = MODEL_DICT[self.discriminator_name](**disc_dict)
        else:
            self.discriminator = None
        if drift_ in ["nn", "NN"]:
            def drift(t, S):
                return self.discriminator(t, S)[:, :self.dimension]
        elif isinstance(drift_, tuple) or isinstance(drift_, list) or \
                isinstance(drift_, np.ndarray):
            def drift(t, S):
                return torch.tensor(np.repeat(
                    np.array(drift_).reshape(1,-1), repeats=S.shape[0], axis=0),
                    requires_grad=False, dtype=torch.float32)
        elif drift_ in ["ref", "mu0", "ref0", "ref0_drift"]:
            def drift(t, S):
                return self.ref0_drift.unsqueeze(dim=0).repeat(S.shape[0], 1)
        elif isinstance(drift_, str):
            drift = eval(drift_)
        else:
            raise ValueError(
                'please provide supported value for disc_dict["drift"]')
        self.drift = drift
        if vola_ in ["nn", "NN"]:
            def vola(t, S):
                v = self.discriminator(t, S)[:, -self.vola_dim:]
                if self.dimension > 1:
                    v = v.reshape(
                        (S.shape[0], self.dimension, self.dimension))
                return v
        elif isinstance(vola_, tuple) or isinstance(vola_, list) or \
                isinstance(vola_, np.ndarray):
            def vola(t, S):
                return torch.tensor(np.repeat(np.expand_dims(
                    np.array(vola_), axis=0), repeats=S.shape[0], axis=0),
                    requires_grad=False, dtype=torch.float32)
        elif vola_ in ["ref", "sig0", "ref0"]:
            def vola(t, S):
                sig0 = self.ref0
                if self.dimension > 1:
                    return torch.unsqueeze(sig0, dim=0).repeat(S.shape[0], 1, 1)
                else:
                    return sig0.reshape(1,1).repeat(S.shape[0], 1)
        elif isinstance(vola_, str):
            vola = eval(vola_)
        else:
            raise ValueError(
                'please provide supported value for disc_dict["vola"]')
        self.vola = vola  # vola models the standard deviation
        self.disc_dict = disc_dict
        self.gen_dict = gen_dict

        self.epoch = 1
        self.apply(init_weights)

    def get_input(self, t, S, X, bs):
        ts = torch.tensor(t/self.T).reshape(1,1).repeat(bs,1)
        if self.input_increments:
            if len(S) == 1:
                input = torch.concat(
                    [torch.tanh(S[-1]-S[-1]), torch.tanh(X[-1]-X[-1]), ts],
                    dim=1)
            else:
                input = torch.concat(
                    [torch.tanh(S[-1]-S[-2]), torch.tanh(X[-1]-X[-2]), ts],
                    dim=1)
        else:
            input = torch.concat(
                # TODO: does it make sense to use tanh?
                [torch.tanh(S[-1]), torch.tanh(X[-1]), ts], dim=1)
        return input

    def get_next_S_X(self, current_S, current_X, mu, sigma, pi, dW):
        if self.dimension <= 1:
            if self.use_general_SDE:
                # use euler scheme to get SDE solution approximations
                next_S = current_S + mu * self.dt + sigma * dW
                next_X = current_X + current_X * (self.r + torch.sum(
                    pi * (mu - self.r*current_S), dim=1,
                    keepdim=True)) * self.dt + \
                         current_X * torch.sum(
                    pi * sigma * dW, dim=1,
                    keepdim=True)
            elif not self.use_log_return:
                # use euler scheme to get SDE solution approximations
                next_S = current_S + current_S * mu * self.dt + \
                         current_S * sigma * dW
                next_X = current_X + current_X * (self.r + torch.sum(
                    pi * (mu - self.r), dim=1,
                    keepdim=True)) * self.dt + \
                         current_X * torch.sum(
                    pi * sigma * dW, dim=1,
                    keepdim=True)
            else:
                next_S = current_S * torch.exp(
                    (mu - 1 / 2 * sigma ** 2) * self.dt + sigma * dW)
                next_X = current_X * torch.exp(
                    (self.r + torch.sum(
                        pi * (mu - self.r) - 1 / 2 * pi ** 2 * sigma ** 2,
                        dim=1,
                        keepdim=True)) * self.dt + torch.sum(
                        pi * sigma * dW, dim=1, keepdim=True))
        else:
            if self.use_general_SDE:
                next_S = current_S + mu * self.dt + \
                         torch.matmul(sigma, dW.unsqueeze(2)).squeeze(2)
                next_X = current_X + current_X * (self.r + torch.sum(
                    pi * (mu - self.r * current_S), dim=1,
                    keepdim=True)) * self.dt + \
                         current_X * torch.sum(
                    pi * torch.matmul(sigma, dW.unsqueeze(2)).squeeze(2),
                    dim=1, keepdim=True)
            elif not self.use_log_return:
                # use euler scheme to get SDE solution approximations
                next_S = current_S + current_S * mu * self.dt + \
                         current_S * torch.matmul(
                    sigma, dW.unsqueeze(2)).squeeze(2)
                next_X = current_X + current_X * (self.r + torch.sum(
                    pi * (mu - self.r), dim=1,
                    keepdim=True)) * self.dt + \
                         current_X * torch.sum(
                    pi * torch.matmul(sigma, dW.unsqueeze(2)).squeeze(2),
                    dim=1, keepdim=True)
            else:
                raise ValueError(
                    "multidim case only works with Euler scheme so far!")
        return next_S, next_X

    def get_next_S_X_np(self, current_S, current_X, mu, sigma, pi, dW):
        next_S, next_X = self.get_next_S_X(
            torch.tensor(current_S), torch.tensor(current_X), torch.tensor(mu),
            torch.tensor(sigma), torch.tensor(pi), torch.tensor(dW))
        return next_S.detach().numpy(), next_X.detach().numpy()

    def get_trading_costs(self, current_X, current_S, pi, last_amount):
        if self.use_general_SDE:
            current_amount = current_X.repeat(1, self.dimension) * pi
        else:
            current_amount = current_X.repeat(1, self.dimension) * pi/current_S
        amount_diff = torch.abs(current_amount - last_amount)
        amount_trans = torch.sum(amount_diff > 0, dim=1).unsqueeze(1)
        trade_cost = torch.sum(amount_diff*current_S*self.trans_cost_perc,
                               dim=1).unsqueeze(1) + \
                     amount_trans*self.trans_cost_base

        return current_amount, amount_diff, amount_trans, trade_cost

    def get_trading_costs_np(self, current_X, current_S, pi, last_amount):
        current_amount, amount_diff, amount_trans, trade_cost = \
            self.get_trading_costs(
                torch.tensor(current_X), torch.tensor(current_S),
                torch.tensor(pi), torch.tensor(last_amount))

        return current_amount.detach().numpy(), amount_diff.detach().numpy(), \
               amount_trans.detach().numpy(), trade_cost.detach().numpy()

    def forward(self, dWs, which="gen", return_paths=False, DEBUG=False,
                init_vals=None):
        """
        :param dW: torch.tensor, shape [batch_size, dimensions, nb_steps], the
                stochastic increments to use
        :param which: in {"gen", "disc", "both"}
        :param return_paths: bool, whether to return the computed paths of S, X,
                mu, sigma, pi
        :param DEBUG: int, whether to print debug information
        :param init_vals: np.array or None, initial values for S
        :return: loss
        """
        bs = dWs.shape[0]
        if init_vals is not None:
            current_S = torch.tensor(init_vals, dtype=torch.float32).reshape(
                bs,-1)
        else:
            current_S = torch.tensor(self.S0, dtype=torch.float32).reshape(
                1,-1).repeat(bs, 1)
        current_X = torch.tensor(self.X0, dtype=torch.float32).reshape(
            1,-1).repeat(bs, 1)
        t = 0
        S = [current_S]
        X = [current_X]
        mus = []
        Sigmas = []
        pis = []
        penalty = 0.
        penalty_drift = 0.
        last_amount = torch.zeros_like(current_S)

        if self.generator_name == "RNN":
            self.generator.reset_h()
        if self.discriminator_name == "RNN":
            if self.discriminator is not None:
                self.discriminator.reset_h()

        for i in range(dWs.shape[2]):
            dW = dWs[:,:,i]
            input = self.get_input(t, S, X, bs)
            if self.discriminator is not None:
                disc = self.discriminator(t, input)
            if self.disc_drift:
                mu = disc[:, :self.dimension]
            else:
                mu = self.drift(t, S[-1])
            if self.disc_vola:
                sigma = disc[:, -self.vola_dim:]
                if self.dimension > 1:
                    sigma = torch.reshape(
                        sigma, shape=(bs, self.dimension, self.dimension))
            else:
                sigma = self.vola(t, S[-1])
            pi = self.generator(t, input)

            # trading costs
            current_amount, amount_diff, amount_trans, trade_cost = \
                self.get_trading_costs(current_X, current_S, pi, last_amount)
            last_amount = current_amount

            if return_paths or DEBUG >= 1:
                mus.append(mu)
                if self.dimension <= 1:
                    Sigmas.append(sigma**2)
                else:
                    Sigmas.append(
                        torch.matmul(sigma, torch.transpose(sigma, 1, 2)).reshape(
                            bs, -1))
                pis.append(pi)

            next_S, next_X = self.get_next_S_X(
                current_S, current_X,mu, sigma, pi, dW)
            if self.numerically_stabilised_training:
                # lower bound the stock and portfolio value by small positive
                #   number for numerical stability of the training scheme
                next_S = config.dclamp(next_S, min=1e-16, max=np.infty)
                next_X = config.dclamp(
                    next_X - trade_cost * (1 + self.r * self.dt), min=1e-16,
                    max=np.infty)
            else:
                next_X = torch.maximum(
                    next_X - trade_cost*(1+self.r*self.dt), torch.tensor(0.))
            S.append(next_S)
            X.append(next_X)
            current_S = next_S
            current_X = next_X

            if self.ref0 is not None:
                penalty += self.penalty_func(
                    sigma, self.ref0.unsqueeze(dim=0).repeat(bs, 1, 1))
            if self.ref0_drift is not None:
                _penalty_drift = self.penalty_func_drift(
                    mu, self.ref0_drift.unsqueeze(dim=0).repeat(bs, 1))
                penalty_drift += _penalty_drift

            t += self.dt

            if DEBUG >= 3:
                print("mu min", torch.min(mu))
                print("-"*80)
                print("sigma min", torch.min(sigma))
                print("-"*80)
                print("pi min", torch.min(pi))
                print("-"*80)
                print("S min", torch.min(current_S))
                print("-"*80)
                print("X min", torch.min(current_X))
                print("-"*80)
        if DEBUG >= 2:
            print("S", torch.stack(S).detach().numpy())
            print("-" * 80)
            print("X", torch.stack(X).detach().numpy())
            print("-" * 80)
            print("mus", torch.stack(mus, dim=1).detach().numpy())
            print("-" * 80)
            print("Sigmas", torch.stack(Sigmas, dim=1).detach().numpy())
            print("-" * 80)
            print("pis", torch.stack(pis, dim=1).detach().numpy())
            print("-" * 80)
        if DEBUG >= 1:
            print("-" * 80)
            print("S min", np.min(torch.stack(S).detach().numpy()))
            print("X min", np.min(torch.stack(X).detach().numpy()))
            print("mus min", np.min(torch.stack(mus, dim=1).detach().numpy()))
            print("Sigmas min",
                  np.min(torch.stack(Sigmas, dim=1).detach().numpy()))
            print("pis min", np.min(torch.stack(pis, dim=1).detach().numpy()))
            print("-" * 80)
            print("S max", np.max(torch.stack(S).detach().numpy()))
            print("X max", np.max(torch.stack(X).detach().numpy()))
            print("mus max", np.max(torch.stack(mus, dim=1).detach().numpy()))
            print("Sigmas max",
                  np.max(torch.stack(Sigmas, dim=1).detach().numpy()))
            print("pis max", np.max(torch.stack(pis, dim=1).detach().numpy()))
            print("-" * 80)
            if np.any(np.isnan(torch.stack(X).detach().numpy())):
                raise ValueError("encountered NaNs")

        if isinstance(penalty, float):
            penalty = torch.tensor(penalty)
        penalty = torch.mean(penalty*self.dt*self.scaling_coeff)
        if isinstance(penalty_drift, float):
            penalty_drift = torch.tensor(penalty_drift)
        penalty_drift = torch.mean(
            penalty_drift*self.dt*self.scaling_coeff_drift)
        expected_util = torch.mean(self.utility_func(current_X))

        if DEBUG >= 1:
            print("exp-util", expected_util.detach().numpy())
            print("-" * 80)

        # compute path-wise penalties
        pwps = []
        S = torch.stack(S, dim=1)
        for j, d in enumerate(self.path_wise_penalty):
            val = d["path_functional"](S, dt=self.dt, T=self.nb_steps*self.dt)
            if d["ref_val_path_functional"]:
                refval = d["ref_value"](S, self.dt)
            else:
                refval = d["ref_value"]
                if not d["is_mean_penalty"]:
                    orig_shape = refval.shape
                    # val.shape[0] is batch size, val.shape[1] is time_steps
                    refval = refval.unsqueeze(dim=0).unsqueeze(dim=1).repeat(
                        val.shape[0], val.shape[1], *np.ones_like(orig_shape))
            pen = torch.mean(
                d["penalty_func"](val, refval) * d["scaling_factor"])
            pwps.append(pen)
            if DEBUG >= 1:
                print("pwp-{}".format(j), pen.detach().numpy())
                print("-"*80)
        if len(pwps) > 0:
            pwp = torch.sum(torch.stack(pwps))
        else:
            pwp = torch.tensor(0)

        if self.numerically_stabilised_training:
            # lower bound the stock and portfolio value by small positive
            #   number for numerical stability of the training scheme
            pwp = config.dclamp(pwp, min=-np.infty, max=1e10)

        if which == "gen":
            # generator (investor) should maximize the utility (penalization
            #   doesn't change the problem since it doesn't depend on generator)
            if self.use_penalty_for_gen:
                loss = -expected_util-penalty-penalty_drift-pwp
            else:
                loss = -expected_util
            if DEBUG >= 1:
                print("loss", loss.detach().numpy())
                print("-" * 80)
            return loss

        elif which == "disc":
            # discriminator (market) should minimize the penalized utility
            if self.train_penalty_threshold is not None:
                loss = penalty+penalty_drift+pwp
                if loss < self.train_penalty_threshold:
                    loss = loss + expected_util
            else:
                loss = expected_util+penalty+penalty_drift+pwp
            if DEBUG >= 1:
                print("loss", loss.detach().numpy())
                print("-" * 80)
            return loss
        else:
            if return_paths:
                return expected_util, penalty, penalty_drift, pwps, \
                       S, torch.stack(X, dim=1), \
                       torch.stack(mus, dim=1), torch.stack(Sigmas, dim=1), \
                       torch.stack(pis, dim=1)
            return expected_util, penalty, penalty_drift, pwps

    def get_reference_solution(self, dWs, init_vals=None):
        """use the analytic sigma and pi (and mu) in forward pass"""
        if self.analytic_sol is None or self.analytic_sol[0] is None or \
                len(self.path_wise_penalty) > 0:
            return None, None, None, None, None, None
        sig_, mu_, pi_ = self.analytic_sol
        bs = dWs.shape[0]
        if init_vals is not None:
            current_S = init_vals.reshape(bs, -1)
        else:
            current_S = np.array(self.S0, dtype=float).reshape(1,-1).repeat(bs, axis=0)
        current_X = np.array(self.X0, dtype=float).reshape(1,-1).repeat(bs, axis=0)
        if self.dimension <= 1:
            sigma = np.array(sig_).reshape(1,-1).repeat(bs, axis=0)
        else:
            sigma = np.array(sig_).reshape(
                1, self.dimension, self.dimension).repeat(bs, axis=0)
        if pi_ is not None:
            pi = np.array(pi_).reshape(1,-1).repeat(bs, axis=0)
        else:
            pi = np.nan
        mu = np.array(mu_).reshape(1, -1).repeat(bs, axis=0)

        t = 0
        penalty = 0
        penalty_drift = 0
        last_amount = np.zeros_like(current_S)

        for i in range(dWs.shape[2]):
            dW = dWs[:,:,i]

            # trading costs
            current_amount, amount_diff, amount_trans, trade_cost = \
                self.get_trading_costs_np(current_X, current_S, pi, last_amount)
            last_amount = current_amount

            current_S, current_X = self.get_next_S_X_np(
                current_S, current_X, mu, sigma, pi, dW)
            current_X = np.maximum(
                current_X - trade_cost*(1+self.r*self.dt), 0.)
            if self.ref0 is not None:
                ref_sig = np.repeat(
                    np.expand_dims(self.ref0.detach().numpy(), axis=0),
                    repeats=bs, axis=0)
                penalty += self.penalty_func_np(sigma, ref_sig)
            if self.ref0_drift is not None:
                ref_mu = np.repeat(
                    np.expand_dims(self.ref0_drift.detach().numpy(), axis=0),
                    repeats=bs, axis=0)
                penalty_drift += self.penalty_func_drift_np(mu, ref_mu)
            t += self.dt

        penalty = np.mean(penalty*self.dt*self.scaling_coeff)
        penalty_drift = np.mean(penalty_drift*self.dt*self.scaling_coeff_drift)
        expected_util = np.mean(self.utility_func_np(current_X))

        return expected_util, penalty, penalty_drift, sig_, pi_, mu_

    def evaluate(self, dWs, ref_params=None, ref_params_drift=None,
                 analytic_pi=None, oracle_pi=None,
                 random_sigmas=None, random_mus=None, return_paths=False,
                 DEBUG=False, init_vals=None, ref_strategy=None,
                 return_Delta_pi=False):
        """
        evaluate the model trading strategy against the given ref_params for the
        market and compute difference of discriminator output to
        analytic market params

        ref_params should either be the analytic sigma or the real (noisy) 
        sigma; shape [nb_steps, ...]

        ref_params_drift should be None if drift is fixed, or analytic or real
        (noisy) mu; shape [nb_steps, ...]

        if analytic_pi and oracle_pi are give, also evaluate these trading
        strategies against the given ref_params for sigma (this should then be
        the real (noisy) sigma

        if random_sigmas are given, then use these for the single paths of dW.
        for each path, one sigma has to be given; 
        shape [nb_steps, nb_paths, ...]
        same for random_mus.

        if ref_strategy is given, then use this strategy for the investment.
        ref_strategy should be a function taking S, X, t as input and returning
        the investment strategy pi.
        if return_Delta_pi is True, then also return the Delta_pi for the
        no trade region of the ref_strategy.

        """
        bs = dWs.shape[0]
        if init_vals is not None:
            current_S = torch.tensor(init_vals, dtype=torch.float32).reshape(
                bs,-1)
        else:
            current_S = torch.tensor(self.S0, dtype=torch.float32).reshape(
                1,-1).repeat(bs, 1)
        current_X = torch.tensor(self.X0, dtype=torch.float32).reshape(
            1,-1).repeat(bs, 1)
        c_X_analytic = torch.tensor(self.X0, dtype=torch.float32).reshape(
            1,-1).repeat(bs, 1)
        c_X_oracle = torch.tensor(self.X0, dtype=torch.float32).reshape(
            1,-1).repeat(bs, 1)
        t = 0
        S = [current_S]
        X = [current_X]
        mus = []
        Sigmas = []
        pis = []
        Delta_pis = []
        pis_ntc = []
        X_analytic = [c_X_analytic]
        X_oracle = [c_X_oracle]
        last_amount = torch.zeros_like(current_S)
        last_amount_a = torch.zeros_like(current_S)
        last_amount_o = torch.zeros_like(current_S)

        if analytic_pi is not None:
            analytic_pi = torch.tensor(
                analytic_pi, requires_grad=False,
                dtype=torch.float32).reshape((1, -1)).repeat(bs, 1)
        if oracle_pi is not None:
            oracle_pi = torch.tensor(
                oracle_pi, requires_grad=False,
                dtype=torch.float32).reshape((1, -1)).repeat(bs, 1)

        penalty = 0
        penalty_drift = 0

        if self.generator_name == "RNN":
            self.generator.reset_h()
        if self.discriminator_name == "RNN":
            if self.discriminator is not None:
                self.discriminator.reset_h()

        for i in range(dWs.shape[2]):
            dW = dWs[:,:,i]
            input = self.get_input(t, S, X, bs)
            if self.disc_vola:
                model_sigma = self.vola(t, input)
            else:
                model_sigma = self.vola(t, S[-1])
            if self.disc_drift:
                model_mu = self.drift(t, input)
            else:
                model_mu = self.drift(t, S[-1])

            if random_mus is not None:
                mu = torch.tensor(
                    random_mus[i].reshape((-1, self.dimension)), requires_grad=False,
                    dtype=torch.float32)
            elif ref_params_drift is not None:
                mu = torch.tensor(np.repeat(
                    np.array(ref_params_drift[i]).reshape(1,-1),
                    repeats=bs, axis=0), requires_grad=False,
                    dtype=torch.float32)
            elif not self.disc_drift:
                mu = self.drift(t, S[-1])
            elif ref_strategy is not None:
                # possibility to evaluate ref strategy against trained market
                mu = model_mu
            elif random_mus is None:
                raise ValueError("mu not defined")

            if random_sigmas is not None:
                # overwrite sigma
                if self.dimension <= 1:
                    sigma = torch.tensor(
                        random_sigmas[i].reshape((-1, 1)), requires_grad=False,
                        dtype=torch.float32)
                else:
                    sigma = torch.tensor(
                        random_sigmas[i].reshape(
                            (-1, self.dimension, self.dimension)),
                        requires_grad=False, dtype=torch.float32)
            elif ref_params is not None:
                if self.dimension <= 1:
                    sigma = torch.tensor(np.repeat(
                        np.array(ref_params[i]).reshape((1, -1)),
                        repeats=bs, axis=0), requires_grad=False,
                        dtype=torch.float32)
                else:
                    sigma = torch.tensor(np.repeat(
                        np.array(ref_params[i]).reshape(
                            (1, self.dimension, self.dimension)),
                        repeats=bs, axis=0), requires_grad=False,
                        dtype=torch.float32)
                sig_def = True
            elif not self.disc_vola:
                sigma = self.vola(t, S[-1])
            elif ref_strategy is not None:
                # possibility to evaluate ref strategy against trained market
                sigma = model_sigma
            else:
                raise ValueError("sigma not defined")

            pi = self.generator(t, input)
            if ref_strategy is not None:
                pi = ref_strategy(S[-1], X[-1], t, last_amount,
                                  return_Delta_pi=return_Delta_pi)
                if return_Delta_pi:
                    pi, pi_ntc, Delta_pi = pi
                    Delta_pis.append(Delta_pi)
                    pis_ntc.append(pi_ntc)
            if return_paths:
                mus.append(mu)
                if self.dimension <= 1:
                    Sigmas.append(sigma**2)
                else:
                    Sigmas.append(
                        torch.matmul(sigma, torch.transpose(sigma, 1, 2)).reshape(
                            bs, -1))
                pis.append(pi)

            # trading costs
            current_amount, amount_diff, amount_trans, trans_cost = \
                self.get_trading_costs(current_X, current_S, pi, last_amount)
            last_amount = current_amount
            # print("X:\nstock-amount-diff: {}, trans-cost: {}, X-before: {}, X-after: {}".format(
            #     amount_diff, trans_cost, xbefore, current_X))
            # print("trans-cost-perc: {}, trans-cost-base: {}".format(self.trans_cost_perc, self.trans_cost_base))
            if analytic_pi is not None:
                current_amount_a, amount_diff_a, amount_trans_a, \
                trans_cost_a = self.get_trading_costs(
                    c_X_analytic, current_S, analytic_pi, last_amount_a)
                last_amount_a = current_amount_a
                # print("X-analytic:\nstock-amount-diff: {}, trans-cost: {}, X-before: {}, X-after: {}".format(
                #     amount_diff_a, trans_cost_a, xbefore, c_X_analytic))
            if oracle_pi is not None:
                current_amount_o, amount_diff_o, amount_trans_o, \
                trans_cost_o = self.get_trading_costs(
                    c_X_oracle, current_S, oracle_pi, last_amount_o)
                last_amount_o = current_amount_o

            next_S, next_X = self.get_next_S_X(
                current_S, current_X, mu, sigma, pi, dW)
            if analytic_pi is not None:
                _, next_X_a = self.get_next_S_X(
                    current_S, c_X_analytic, mu, sigma, analytic_pi, dW)
            if oracle_pi is not None:
                _, next_X_o = self.get_next_S_X(
                    current_S, c_X_oracle, mu, sigma, oracle_pi, dW)
            next_X = torch.maximum(
                torch.tensor(0.), next_X - trans_cost*(1+self.r*self.dt))
            S.append(next_S)
            X.append(next_X)
            current_S = next_S
            current_X = next_X
            if analytic_pi is not None:
                next_X_a = torch.maximum(
                    torch.tensor(0.),
                    next_X_a - trans_cost_a*(1+self.r*self.dt))
                X_analytic.append(next_X_a)
                c_X_analytic = next_X_a
            if oracle_pi is not None:
                next_X_o = torch.maximum(
                    torch.tensor(0.),
                    next_X_o - trans_cost_o*(1+self.r*self.dt))
                X_oracle.append(next_X_o)
                c_X_oracle = next_X_o
            if ref_params is not None:
                penalty += self.penalty_func_for_eval(
                    model_sigma.reshape((bs, self.dimension, self.dimension)),
                    torch.tensor(ref_params[i], dtype=torch.float32).unsqueeze(
                        0).repeat(bs, 1, 1))
            if ref_params_drift is not None:
                penalty_drift += self.penalty_func_drift(
                    model_mu, torch.tensor(
                        ref_params_drift[i], dtype=torch.float32).unsqueeze(
                        0).repeat(bs, 1))
            t += self.dt

        if ref_params is not None and not isinstance(penalty, float):
            penalty = torch.mean(penalty*self.dt*self.scaling_coeff)
        else:
            penalty = torch.tensor(0.)
        if ref_params_drift is not None:
            penalty_drift = torch.mean(
                penalty_drift*self.dt*self.scaling_coeff_drift)
        expected_util = torch.mean(self.utility_func(current_X))
        expected_util_a = torch.mean(self.utility_func(c_X_analytic))
        expected_util_o = torch.mean(self.utility_func(c_X_oracle))

        # compute path-wise penalties (only for model strategy, since analytic
        #   and oracle are not computed if path_wise_penalty is not None)
        pwps = []
        S = torch.stack(S, dim=1)
        for j, d in enumerate(self.path_wise_penalty):
            val = d["path_functional"](S, dt=self.dt, T=self.nb_steps * self.dt)
            if d["ref_val_path_functional"]:
                refval = d["ref_value"](S, self.dt)
            else:
                refval = d["ref_value"]
                if not d["is_mean_penalty"]:
                    orig_shape = refval.shape
                    # val.shape[0] is batch size, val.shape[1] is time_steps
                    refval = refval.unsqueeze(dim=0).unsqueeze(dim=1).repeat(
                        val.shape[0], val.shape[1], *np.ones_like(orig_shape))
            pen = torch.mean(
                d["penalty_func"](val, refval) * d["scaling_factor"])
            pwps.append(pen)
            if DEBUG >= 1:
                print("pwp-{}".format(j), pen.detach().numpy())
                print("-" * 80)
        if len(pwps) > 0:
            pwp = torch.sum(torch.stack(pwps))
        else:
            pwp = torch.tensor(0)

        if analytic_pi is None:
            expected_util_a = None
        if oracle_pi is None:
            expected_util_o = None
        if return_paths:
            if return_Delta_pi:
                return expected_util, penalty, penalty_drift, pwps, \
                       expected_util_a, expected_util_o, S, \
                       torch.stack(X, dim=1), torch.stack(mus, dim=1), \
                       torch.stack(Sigmas, dim=1), torch.stack(pis, dim=1), \
                       torch.stack(pis_ntc, dim=1), torch.stack(Delta_pis, dim=1)
            return expected_util, penalty, penalty_drift, pwps, \
                   expected_util_a, expected_util_o, S, \
                   torch.stack(X, dim=1), torch.stack(mus, dim=1), \
                   torch.stack(Sigmas, dim=1), torch.stack(pis, dim=1)
        return expected_util, penalty, penalty_drift, pwps, expected_util_a, \
               expected_util_o

    def evaluate_vs_garch(
            self, eval_model, eval_model_dict,
            nb_rand_params, nb_samples_per_param, seed=None,
            return_paths=False):
        """
        evaluate the trained strategy vs the fitted GARCH model for the market

        :param eval_model:
        :param eval_model_dict:
        :param nb_rand_params:
        :param nb_samples_per_param:
        :param seed:
        :param return_paths:
        :return:
        """
        if seed is not None:
            np.random.seed(seed)

        paths = []
        for i in range(nb_rand_params):
            rho0 = eval_model.rho

            # get noisy params
            params = eval_model.sample_params(
                scaling=eval_model_dict["noise"]["params_scaling"])

            # get noisy rho
            rho = rho0
            d = eval_model.d
            for i in range(1, d):
                noise_diag = eval_model_dict["noise"]["rho_std"] * \
                             np.random.normal(loc=0, scale=1, size=d-i)
                add = np.diag(noise_diag, k=i) + np.diag(noise_diag, k=-i)
                rho += add
            rho = np.minimum(np.maximum(rho, -1.), 1.)

            paths.append(eval_model.sample_paths(
                params=params, rho=rho, nb_paths=nb_samples_per_param))
        stockpaths = np.concatenate(paths, axis=0)
        # print("stock paths shape:", stockpaths.shape)

        t = 0
        bs = stockpaths.shape[0]
        current_S = torch.tensor(stockpaths[:, 0, :], dtype=torch.float32).detach()
        current_X = torch.tensor(self.X0, dtype=torch.float32).reshape(
            1, -1).repeat(bs, 1).detach()
        X = [current_X]
        S = [current_S]
        pis = []
        last_amount = torch.zeros_like(current_S).detach()

        if self.generator_name == "RNN":
            self.generator.reset_h()

        for i in range(1, stockpaths.shape[1]):
            input = self.get_input(t, S, X, bs)
            pi = self.generator(t, input)
            if return_paths:
                pis.append(pi)

            # trading costs
            current_amount, amount_diff, amount_trans, trans_cost = \
                self.get_trading_costs(current_X, current_S, pi, last_amount)
            last_amount = current_amount

            # update S and X
            next_S = torch.tensor(stockpaths[:, i, :], dtype=torch.float32)
            if self.use_general_SDE:
                next_X = (torch.sum(current_amount * next_S, dim=1,
                                    keepdim=True) +
                          (1. - torch.sum(pi*current_S, dim=1,
                                          keepdim=True)) * current_X *
                          np.exp(self.r * self.dt))
            else:
                next_X = (torch.sum(current_amount*next_S, dim=1, keepdim=True)+
                         (1.-torch.sum(pi, dim=1, keepdim=True))*current_X*
                         np.exp(self.r*self.dt))
            next_X = torch.maximum(
                torch.tensor(0.), next_X - trans_cost*np.exp(self.r*self.dt))
            X.append(next_X)
            S.append(next_S)
            current_S = next_S
            current_X = next_X
            t += self.dt
        expected_util = torch.mean(self.utility_func(current_X)).detach().numpy()

        if return_paths:
            return expected_util, torch.stack(X, dim=1).detach().numpy(), \
                   stockpaths, \
                   torch.stack(pis, dim=1).detach().numpy()
        return expected_util



if __name__ == '__main__':
    pass

