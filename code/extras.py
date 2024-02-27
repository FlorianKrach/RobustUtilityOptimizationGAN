"""
author: Florian Krach
"""
import copy

# ==============================================================================
import tqdm
import numpy as np
import os, sys
import pandas as pd
import json
import time
import socket
import matplotlib
import matplotlib.colors
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mticker
from torch.backends import cudnn
import gc
import ast
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve
import scipy.stats as sstats
import itertools

import arch

import config, data_utils

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = config.SendBotMessage()

# ==============================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True
print(socket.gethostname())
SEND = False
if SERVER:
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = config.saved_models_path


# ==============================================================================
# Functions
def average_duplicated_runs(
        df, cols_to_average=("eval_MSE_min_Q", "ImplVolErr_min"),
        skipna=True, function="median"
):
    """
    Args:
        df: pd.DataFrame
        cols_to_average: list of strings, the col names that are averaged
        skipna: bool, whether to skip nans
        function: str, the pandas method used to average as str, one of
            {"mean", "median", "min"}

    Returns: pd.DataFrame where the columns were averaged

    """
    desc = "description"

    unique_descriptions = list(set(df[desc].values))

    for d in unique_descriptions:
        df_ = df.loc[df[desc] == d]
        ind = df_.index
        if len(ind) > 1:
            for c in cols_to_average:
                df.loc[ind[0], c] = eval(
                    "df_[c].{}(skipna={})".format(function, skipna))
            df.drop(index=ind[1:], inplace=True)

    return df


def penalty_scaling_plots(
        df, params_extract_desc=None,
        path=saved_models_path, remove_rows=None,
        psf_col1='penalty_scaling_factor',
        psf_col2='penalty_scaling_factor_drift',
        target_col = 'mean_exp_util_with_noisy_par',
        col_names_dict=None):
    """
    function to plot surface plots of the achieved values for different lambda
    values (i.e. scaling factors for vola and drift penalty)
    :param df: pd.DataFrame, as outputted by get_training_overview()
    :param params_extract_desc: same as for get_training_overview()
    :param path: same as for get_training_overview()
    :param remove_rows: None or dict, if dict: for each (key, val) pair in dict
        remove all rows of the df where the column <key> has value <val>
    :param psf_col1: str, the name of the column for the first penalty scaling
        factor
    :param psf_col2: str, the name of the column for the second penalty scaling
        factor
    :param target_col: str, the name of the column to plot
    :param col_names_dict: None or dict, if not None, the keys are the column
        names of the df, the values are the names used in the plot
    :return:
    """
    if col_names_dict is None:
        col_names_dict = {psf_col1: psf_col1, psf_col2: psf_col2,
                          target_col: target_col}

    if remove_rows is not None:
        for k, v in remove_rows.items():
            df.drop(index=df.loc[df[k]==v].index, inplace=True)

    columns = list(params_extract_desc)
    p1 = psf_col1
    p2 = psf_col2

    assert p1 in columns
    assert p2 in columns
    columns.remove(p1)
    columns.remove(p2)

    cols = []
    vals = []
    for c in columns:
        df = df.astype({c: str})
        v = list(set(df[c].values.tolist()))
        if len(v) > 1:
            cols.append(c)
            vals.append(v)

    for v_comb in itertools.product(*vals):
        _df = copy.copy(df)
        for c,v in zip(cols, v_comb):
            _df = _df.loc[_df[c] == v]
        if len(_df) == 0:
            continue
        _df.sort_values(axis=0, by=[p2, p1], inplace=True)
        p1_vals = sorted(list(set(_df[p1].values.tolist())))
        p2_vals = sorted(list(set(_df[p2].values.tolist())))
        filename = "{}penalty-scaling-surfplot-{}.pdf".format(
            path, _df["utility_func"].values[0])
        plotted = False
        if len(p1_vals) > 1 and len(p2_vals) > 1:
            X, Y = np.meshgrid(np.log10(p1_vals), np.log10(p2_vals))
            Z = _df[target_col].values.reshape(X.shape)
            Z = np.maximum(Z, -1)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # surf = ax.plot_surface(
            #     X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
            print(Z)
            max_ind = np.unravel_index(np.nanargmax(Z, axis=None), Z.shape)
            ax.scatter(X[max_ind], Y[max_ind], Z[max_ind], marker="o",
                       color="red")

            def log_tick_formatter(val, pos=None):
                return f"$10^{{{val:g}}}$"

            plt.title("max_val = {}".format(np.nanmax(Z)))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(log_tick_formatter))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(log_tick_formatter))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_xlabel(col_names_dict[p1])
            ax.set_ylabel(col_names_dict[p2])
            ax.set_zlabel(col_names_dict[target_col])
            plt.savefig(filename, bbox_inches="tight")
            plotted = True
        else:
            which = None
            if len(p1_vals) > 1:
                which = p1_vals
                label = p1
            elif len(p2_vals) > 1:
                which = p2_vals
                label = p2

            if which is not None:
                plt.figure()
                Z = _df[target_col].values
                Z = np.maximum(Z, -1)
                plt.plot(p1_vals, Z, )
                max_ind = np.nanargmax(Z, axis=None)
                plt.scatter(p1_vals[max_ind], Z[max_ind], marker="o",
                            color="red")
                plt.title("max_val = {}".format(np.nanmax(Z)))
                plt.xscale("log")
                plt.xlabel(col_names_dict[label])
                plt.ylabel(col_names_dict[target_col])
                plt.savefig(filename, bbox_inches="tight")
                plotted = True

        time.sleep(5)
        if SEND and plotted:
            files_to_send = [filename]
            desc = "{}\n{}".format(cols, v_comb)
            SBM.send_notification(
                text=desc,
                text_for_files='penalty scaling surface plot',
                files=files_to_send,
                chat_id=CHAT_ID)





def get_training_overview(
        path=saved_models_path, ids_from=None,
        ids_to=None, save_file=None,
        params_extract_desc=None,
        vals_metric_extract=None,
        sortby="eval_expected_utility",
        average=False,
        average_func="median",
        cols_to_average=None,
        avergae_skipna=True,
        model_eval_file=None,
        plot_penalty_scaling_plots=None,
):
    """
    function to get the important metrics and hyper-params for each model in the
    models_overview.csv file

    Args:
        path: str, where the saved models are
        ids_from: None or int, which model ids to consider start point
        ids_to: None or int, which model ids to consider end point
        params_extract_desc: list of str, names of params to extract from the
            model description dict, special:
                - network_size: gets size of first layer of enc network
                - activation_function_x: gets the activation function of layer x
                    of enc network
                - path_wise_penalty-x-y: extract from x-th path_wise_penalty the
                    value of key y. x has to be convertable to int.
        vals_metric_extract: None or list of list with 4 string elements:
                0. "min" or "max" or "last" or "average"
                1. col_name where to look for min/max (validation), or where to
                    get last value or average
                2. if 0. is min/max: col_name where to find value in epoch where
                                     1. is min/max (test)
                   if 0. is last/average: not used
                3. name for this output column in overview file
        save_file: str or None
        sortby: str or None, sort the output df by this column (ascending)
        average: bool, whether to apply average to table
        average_func: see average_duplicated_runs
        avergae_skipna: see average_duplicated_runs
        cols_to_average: see average_duplicated_runs
        model_eval_file: None or str, the filename where the evaluation of all
            models is saved to append to the training_overview
        plot_penalty_scaling_plots: None or dict, whether to run function
            penalty_scaling_plots()

    Returns:
    """
    filename = "{}model_overview.csv".format(path)
    df = pd.read_csv(filename, index_col=0)
    if ids_from:
        df = df.loc[df["id"] >= ids_from]
    if ids_to:
        df = df.loc[df["id"] <= ids_to]

    # extract wanted information
    for param in params_extract_desc:
        df[param] = None

    if vals_metric_extract:
        for l in vals_metric_extract:
            df[l[3]] = None

    for i in df.index:
        desc = df.loc[i, "description"]
        param_dict = json.loads(desc)

        values = []
        for param in params_extract_desc:
            try:
                if param == 'network_size':
                    v = param_dict["enc_nn"][0][0]
                elif 'activation_function' in param:
                    numb = int(param.split('_')[-1])
                    v = param_dict["enc_nn"][numb - 1][1]
                elif "path_wise_penalty" in param:
                    _, num, key = param.split("-")
                    num = int(num)
                    v = param_dict["path_wise_penalty"][num][key]
                else:
                    v = param_dict[param]
                values.append(v)
            except Exception:
                values.append(None)
        df.loc[i, params_extract_desc] = values

        id = df.loc[i, "id"]
        file_n1 = "{}id-{}/metric_id-{}.csv".format(path, id, id)
        a = []
        if vals_metric_extract:
            a.append([vals_metric_extract, file_n1])

        for vals_extract, fn in a:
            try:
                df_metric = pd.read_csv(fn, index_col=0)
                for l in vals_extract:
                    if l[0] == 'max':
                        f = np.nanmax
                    elif l[0] == 'min':
                        f = np.nanmin

                    if l[0] in ['min', 'max']:
                        try:
                            ind = (df_metric.loc[df_metric[l[1]] ==
                                                 f(df_metric[l[1]])]).index[0]
                            df.loc[i, l[3]] = df_metric.loc[ind, l[2]]
                        except Exception:
                            pass
                    elif l[0] == 'last':
                        df.loc[i, l[3]] = df_metric[l[1]].values[-1]
                    elif l[0] == 'average':
                        df.loc[i, l[3]] = np.nanmean(df_metric[l[1]])
                    elif l[0] == 'median':
                        df.loc[i, l[3]] = np.nanmedian(df_metric[l[1]])
            except FileNotFoundError:
                pass

    if model_eval_file:
        try:
            dirs = os.listdir(path)
            fs = sorted([i for i in dirs if i.startswith(model_eval_file)])
            model_eval_file = fs[-1]
            model_eval_filename = "{}{}".format(path, model_eval_file)
            df_model_eval = pd.read_csv(model_eval_filename, index_col=0)
            df = pd.merge(left=df, right=df_model_eval, how="left", left_on="id",
                          right_on="model_id")
        except Exception as e:
            pass

    if average:
        df = average_duplicated_runs(
            df, cols_to_average=cols_to_average, skipna=avergae_skipna,
            function=average_func)

    if sortby:
        df.sort_values(axis=0, ascending=True, by=sortby, inplace=True)

    # save
    if save_file is not False:
        if save_file is None:
            save_file = "{}training_overview-ids-{}-{}.csv".format(
                path, ids_from, ids_to)
        df.to_csv(save_file)

    if SEND and save_file is not False:
        files_to_send = [save_file]
        SBM.send_notification(
            text=None,
            text_for_files='training overview',
            files=files_to_send,
            chat_id=CHAT_ID)

    if plot_penalty_scaling_plots is not None:
        penalty_scaling_plots(
            df, params_extract_desc=params_extract_desc, path=path,
            **plot_penalty_scaling_plots)

    return df


def get_analytical_solution(
        utility_function, penalty_function, penalty_function_ref_value,
        dimension, r, mu, penalty_scaling_coeff,
        robust_pi=False, penalty_function_drift=None,
        penalty_function_ref_value_drift=None,
        penalty_scaling_coeff_drift=None,
        path_wise_penalty=None,
        **options):
    """
    :param utility_function:
    :param penalty_function:
    :param penalty_function_ref_value:
    :param dimension: int, number of risky assets
    :param r:
    :param mu:
    :param penalty_scaling_coeff:
    :param robust_pi: bool, whether to compute the optimal robust inv. strategy
            or the non-robust one in the case of dimension > 1
    :param penalty_function_drift: str or None,
    :param penalty_function_ref_value_drift: None or array, the reference value
            for mu, i.e. mu_0, if the drift is trained by discriminator
    :param penalty_scaling_coeff_drift: None or float
    :param options:
    :return:
    """
    if path_wise_penalty is not None:
        return None, None, None

    if penalty_function_drift is not None:
        if not penalty_function_drift == "squarenorm-2":
            return None, None, None

    if mu in ["ref", "mu0", "ref0", "ref0_drift"]:
        mu = penalty_function_ref_value_drift

    if utility_function == "log" and not isinstance(mu, str) and not robust_pi:
        sig0 = np.array(penalty_function_ref_value)
        Sig0 = np.matmul(sig0, sig0.transpose())
        mu = np.array(mu)
        if len(mu) == 1 and len(mu) < dimension:
            mu = mu.repeat(dimension)
        try:
            Sig_inv = np.linalg.inv(Sig0)
            pi = np.matmul(Sig_inv, mu-r)
        except np.linalg.LinAlgError as e:
            pi = None
        return sig0, mu, pi
    if isinstance(mu, str) and not robust_pi:
        return None, None, None
    if utility_function == "log" and penalty_function == "squarenorm-2":
        if dimension == 1 and not isinstance(mu, str):
            if not isinstance(mu, float):
                mu = mu[0]
            sig0 = penalty_function_ref_value
            c = (mu - r)**2/(2*penalty_scaling_coeff)
            func = lambda s: s**4 - sig0*s**3 - c
            ls = np.linspace(0, 10, 1000001)
            out = func(ls)
            # i = np.argmin(np.abs(out))
            # print("init guess:", ls[i])
            # print(out)
            # sig = fsolve(func=func, x0=ls[i])
            sig = fsolve(func=func, x0=sig0)
            pi = (mu - r)/sig**2
            # print("computed sig: {}, inserted in func: {}".format(sig, func(sig)))
            return sig, mu, pi
        elif dimension == 1:
            mu0 = penalty_function_ref_value_drift
            if not isinstance(mu0, float):
                mu0 = mu0[0]
            sig0 = penalty_function_ref_value
            def func(s):
                c = np.sqrt(2*penalty_scaling_coeff*(s - sig0)/s)
                return mu0 - c/(2*penalty_scaling_coeff_drift) - r - c*s**2
            sig = fsolve(func=func, x0=sig0)
            pi = np.sqrt(2*penalty_scaling_coeff*(sig - sig0)/sig)
            mu = mu0 - pi/(2*penalty_scaling_coeff_drift)
            # print("computed sig: {}, inserted in func: {}".format(sig, func(sig)))
            return sig, mu, pi
    if utility_function == "log" and penalty_function == "squarenormsquare-fro":
        sig0 = np.array(penalty_function_ref_value)
        Sig0 = np.matmul(sig0, sig0.transpose())
        if not isinstance(mu, str):
            l = penalty_scaling_coeff
            mu = np.array(mu)
            func = lambda a: np.matmul(
                np.sum(a*a)/4/l * np.eye(dimension) + Sig0, a) - (mu-r)
            pi = fsolve(func, np.ones((dimension,)))
            Sig = np.matmul(pi.reshape((-1,1)), pi.reshape((1, -1)))/4/l + Sig0
            sig = np.linalg.cholesky(Sig)
            return sig, mu, pi
        else:
            mu0 = np.array(penalty_function_ref_value_drift)
            l0 = penalty_scaling_coeff
            l1 = penalty_scaling_coeff_drift
            def func(mu):
                t1 = (l1**2/l0*np.sum((mu-mu0)*(mu-mu0))*np.eye(dimension)+Sig0)
                return np.matmul(t1*2*l1, (mu0-mu)) - (mu - r)
            mu = fsolve(func, mu0)
            pi = 2*l1*(mu0-mu)
            Sig = np.matmul(pi.reshape((-1,1)), pi.reshape((1, -1)))/4/l0 + Sig0
            sig = np.linalg.cholesky(Sig)
            return sig, mu, pi
    if utility_function == "log" and penalty_function == "square-mult-frob":
        if not isinstance(mu, str):
            l = penalty_scaling_coeff
            sig0 = np.array(penalty_function_ref_value)
            Sig0 = np.matmul(sig0, sig0.transpose())
            mu = np.array(mu)
            def func(a):
                siga = np.matmul(Sig0, a)
                v = np.matmul(Sig0, siga)
                c = np.sum(a.reshape((-1,))*v.reshape((-1,)))/4/l
                return a*c + siga - (mu-r)
            pi = fsolve(func, np.ones((dimension,)))
            vv = np.matmul(np.matmul(pi.reshape((1, -1)), Sig0), Sig0)
            Sig = np.matmul(pi.reshape((-1,1)), vv)/4/l +Sig0
            sig = np.linalg.cholesky(Sig)
            return sig, mu, pi

    return None, None, None


def get_oracle_solution(
        penalty_function_ref_value, dimension, r, mu, noise_std=1.,
        noise_std_drift=None, penalty_function_ref_value_drift=None,
        path_wise_penalty=None,
        utility_function="log",
        noise_seed=495, noise_type="const", nb_steps=1,
        **options):

    if penalty_function_ref_value_drift is None:
        penalty_function_ref_value_drift = mu
    if noise_std_drift is None:
        noise_std_drift = 0

    sig0 = np.array(penalty_function_ref_value)
    mu0 = np.array(penalty_function_ref_value_drift)
    np.random.seed(noise_seed)
    if noise_type == "const":
        # first add noise, then repeat for each timestep
        sig = sig0 + noise_std * np.random.normal(size=sig0.shape)
        Sig = np.matmul(sig, sig.transpose())
        sig = np.repeat(np.expand_dims(sig, axis=0), repeats=nb_steps, axis=0)
        mu = mu0 + noise_std_drift * np.random.normal(size=mu0.shape)
        mu = np.repeat(np.expand_dims(mu, axis=0), repeats=nb_steps, axis=0)
    elif noise_type == "nonconst":
        # first repeat for each timestep, then add noise
        sig0 = np.repeat(np.expand_dims(sig0, axis=0), repeats=nb_steps, axis=0)
        sig = sig0 + noise_std * np.random.normal(size=sig0.shape)
        mu0 = np.repeat(np.expand_dims(mu0, axis=0), repeats=nb_steps, axis=0)
        mu = mu0 + noise_std_drift * np.random.normal(size=mu0.shape)
    elif noise_type == "cumulative":
        # first repeat for each timestep, then add cumulative noise
        sig0 = np.repeat(np.expand_dims(sig0, axis=0), repeats=nb_steps, axis=0)
        sig = sig0 + np.cumsum(
            noise_std/np.sqrt(nb_steps) * np.random.normal(size=sig0.shape),
            axis=0)
        mu0 = np.repeat(np.expand_dims(mu0, axis=0), repeats=nb_steps, axis=0)
        mu = mu0 + np.cumsum(
            noise_std_drift/np.sqrt(nb_steps) *np.random.normal(size=mu0.shape),
            axis=0)
    else:
        raise ValueError("noise_type {} is not supported".format(noise_type))
    if utility_function == "log" and path_wise_penalty is None and \
            noise_type == "const":
        try:
            Sig_inv = np.linalg.inv(Sig)
            pi = np.matmul(Sig_inv, mu[0]-r)
        except np.linalg.LinAlgError as e:
            pi = None
    else:
        pi = None
    # shapes of sig and mu: [nb_steps, ...]
    return sig, pi, mu


class Garch():
    def __init__(self, p=1, q=1, dist="normal", mean="Constant", S0=None,
                 nb_steps=None, logreturns=None):
        """
        we use log-returns since the log of an ito-diffusion has gaussian noise

        for each dimension, a single garch(p,q) model is fit


        ------------------------------------------------------------------
        GARCH model for normal-dist:
        r_t = mu_t + eps_t
        eps_t = sig_t * e_t
        sig_t**2 = omega + alpha * eps_{t-1}**2 + beta * sig_{t-1}**2
        e_t ~ N(0,1)
        ------------------------------------------------------------------
        for forcasting returns:
        first get standardized residuals e_t = eps_t/sig_t
        compute the correlation between coord. as rho = corr(e_t) (shape: dxd)
        forecast sigs and mus for wanted amount of steps
        for each forcasting step t:
            compute covariance matrix S_t = diag(sigs_t) @ rho @ diag(sigs_t)
                and its root via cholesky decomp (after transforming to pos def
                matrix via eigendecomp and replacing all eigenvals <=0 by eps)
            sample vector of iid e_t for each coord.
            compute correlated sampled residuals: eps_t = (S_t)^(1/2) e_t
            compute sampled returns: r_t = mu_t + eps_t
            rescale them!
        ------------------------------------------------------------------

        :param p: int, order of garch terms
        :param q: int, order of arch terms
        :param dist: str, one of {"normal", "t"}
        :param mean: str, one of {"Constant", "AR"}
        :param S0: array, starting value for samples
        :param nb_steps: int, nb_steps for samples
        :param logreturns: np.array, shape [dim, time_steps]
        """
        self.garch_models = []
        self.d = logreturns.shape[0]
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist
        self.S0 = np.array(S0)
        self.nb_steps = nb_steps
        self.scales = []
        for i in range(self.d):
            fit = arch.arch_model(
                y=logreturns[i], mean=mean, dist=dist, vol="GARCH", p=p, q=q,
                rescale=False).fit(update_freq=0)
            # print(fit)  # TODO: delete again
            self.garch_models.append(fit)
            self.scales.append(fit.scale)
            print(fit.params)
        self.scales = np.array(self.scales)
        # print("scales:", self.scales)
        self.rho = self.compute_corr()
        self.params = np.array([gm.params.values for gm in self.garch_models])
        self.params_stds = np.array(
            [gm.std_err.values for gm in self.garch_models])

    def compute_corr(self):
        """
        compute the correlation matrix between the standardised residuals of the
        different GARCH models for each dimension.
        """
        e = []
        for i in range(self.d):
            fit = self.garch_models[i]
            e.append(fit.std_resid)
        return np.corrcoef(np.array(e))

    def sample_params(self, scaling=1.):
        params0 = self.params
        params_stds = self.params_stds
        params = params0 + params_stds * scaling * \
                 np.random.normal(loc=0, scale=1, size=params0.shape)
        # constraints for stationarity (see: https://www.sciencedirect.com/
        #   science/article/pii/0304407692900672?ref=cra_js_challenge&fr=RR-1)
        for i in range(params.shape[0]):
            params[i, 1:] = np.maximum(params[i, 1:], 0.)
            l = np.linalg.norm(params[i, 2:], ord=1)
            if l >= 1:
                params[i, 2:] /= (l + 1e-15)
        return params

    def sample_logreturns(self, nb_steps, params=None, rho=None, nb_samples=1):
        """
        sample one log_return path of the GARCH model. possibility to use
        different params and rho than the fitted ones.

        :param nb_steps:
        :param params:
        :param rho:
        :return:
        """
        # forcaste sigs and mus
        sigs = np.zeros((nb_steps, self.d, self.d))
        mus = np.zeros((nb_steps, self.d))
        for i in range(self.d):
            fit = self.garch_models[i]
            if params is None:
                _params = fit.params
            else:
                _params = params[i]
            fc = fit.forecast(horizon=nb_steps, reindex=False, params=_params)
            sigs[:, i, i] = np.sqrt(fc.variance.values)
            mus[:, i] = fc.mean

        # compute S_t and its root
        if rho is None:
            rho = self.rho
        S = sigs @ rho @ sigs
        # get pos. def. approx of S
        w, v = np.linalg.eig(S)
        D = np.zeros_like(v)
        for i in range(D.shape[0]):
            D[i] = np.diag(np.maximum(w[i], 1e-16))
        S_pd = v @ D @ np.linalg.inv(v)
        S_sr = np.linalg.cholesky(S_pd)

        # sample iid noise
        if self.dist == "t":
            e_sample = np.zeros((nb_samples*nb_steps, self.d,1))
            for i in range(self.d):
                fit = self.garch_models[i]
                nu = fit.params["nu"]
                e_sample[:, i, 0] = fit.model.distribution.simulate(nu)(
                    nb_samples*nb_steps)
            e_sample = e_sample.reshape((nb_samples, nb_steps, self.d, 1))
        else:  # normal
            e_sample = np.random.normal(size=(nb_samples, nb_steps,self.d,1))

        eps_sample = S_sr @ e_sample
        r_sample = (mus.reshape((1, nb_steps, -1)).repeat(nb_samples, axis=0) +
                    eps_sample.squeeze(axis=3)) * \
                   self.scales.reshape((1,-1)).repeat(
                       nb_samples*nb_steps, axis=0).reshape(
                       (nb_samples, nb_steps, -1))
        del S, S_pd, S_sr, e_sample, sigs
        gc.collect()
        return r_sample

    def sample_paths(self, params, rho, nb_paths=1, nb_steps=None, S0=None):
        """
        sample multiple paths (not log returns) via the GARCH model. possibility
        to use different params and rho than the fitted ones.

        :param params:
        :param rho:
        :param nb_paths:
        :param nb_steps:
        :param S0:
        :return: np.array, shape: [nb_paths, nb_steps+1, dimension], the
            stockpath samples
        """
        if S0 is None:
            S0 = self.S0
        if nb_steps is None:
            nb_steps = self.nb_steps
        paths = np.zeros(shape=(nb_paths, nb_steps+1, self.d))
        # paths with shape: [nb_paths, nb_steps, dimension]
        paths[:, 0, :] = S0.reshape(1,-1).repeat(axis=0, repeats=nb_paths)
        r_samples = self.sample_logreturns(
            nb_steps=nb_steps, params=params, rho=rho, nb_samples=nb_paths)
        paths[:, 1:, :] = np.exp(r_samples)
        # for i in range(nb_paths):
        #     r_samples = self.sample_logreturns(
        #         nb_steps=nb_steps, params=params, rho=rho)
        #     paths[i, 1:, :] = np.exp(r_samples)
        #     # print("r-samples:")
        #     # print(r_samples)
        #     # print("paths:")
        #     # print(paths)
        del r_samples
        gc.collect()
        paths = np.cumprod(paths, axis=1)
        return paths


EVAL_MODELS_DICT = {
    "GARCH": Garch
}


def get_eval_model(
        name, params, log_return_data=None, data_train=None, fit_seq_len=1000,
        S0=None, dt=None, nb_steps=None,
        penalty_function_ref_value=None, penalty_function_ref_value_drift=None):

    # get log return data
    if log_return_data is None:
        dim = data_train.dimension
        dWs = data_train.increments.transpose(1,2,0).reshape(
            dim, -1)
        l = min(fit_seq_len, dWs.shape[1])
        dWs = dWs[:, :l]

        current_S = np.array(S0, dtype=float).reshape(1,-1)
        if dim <= 1:
            sigma = np.array(penalty_function_ref_value).reshape(1,-1)
        else:
            sigma = np.array(penalty_function_ref_value).reshape(1, dim, dim)
        mu = np.array(penalty_function_ref_value_drift).reshape(1, -1)

        S = [current_S]
        for i in range(l):
            dW = dWs[:, i].reshape(1, -1)
            if dim <= 1:
                # use euler scheme to get SDE solution approximations
                current_S = S[-1] + S[-1]*mu*dt + S[-1]*sigma*dW
            else:
                # use euler scheme to get SDE solution approximations
                current_S = S[-1] + S[-1]*mu*dt + S[-1]*np.squeeze(
                    np.matmul(sigma, np.expand_dims(dW, axis=2)))
            S.append(current_S)

        S = np.stack(S, axis=2).squeeze(axis=0)  # shape: [dim, steps]
        log_return_data = np.log(S[:, 1:]/S[:, :-1])

    return EVAL_MODELS_DICT[name](
        **params, S0=S0, nb_steps=nb_steps, logreturns=log_return_data)



def plot_utility_functions(
        utility_functs, range=(1,3), nb_points=1000, path="data/plots/",):
    """
    :param utility_functs: list of str, the utility functions to plot
    :param range: tuple, the range of the x-axis
    :param nb_points: int, number of points to plot
    :param options:
    :return:
    """
    x = np.linspace(range[0], range[1], nb_points)
    for u in utility_functs:
        if u == "log":
            y = np.log(x)
        elif "power" in u:
            p = float(u.split("-")[-1])
            y = (x**(1-p)-1)/(1-p)
        else:
            raise ValueError("utility function {} not supported".format(u))
        plt.plot(x, y, label=u)
    plt.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}utility_functions.pdf".format(path), bbox_inches='tight')








if __name__ == '__main__':
    plot_utility_functions(["power-0.5", "log",  "power-2"], range=(0.5,3))
    pass




