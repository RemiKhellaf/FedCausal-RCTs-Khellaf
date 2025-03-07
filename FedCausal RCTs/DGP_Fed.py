import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import statsmodels.api as sm
from scipy.stats import multivariate_normal, binom
import scipy.stats as scistats
from utils import basic_utils
import inspect
import copy
from tqdm import tqdm
import time


def columns_list(letter, n):
    if n == 1:
        return [letter]
    else:
        string_list = []
        for i in range(1, n + 1):
            string_list.append(f"{letter}{i}")
        return string_list


def generate_multivariate_normal(
    covariates_name: str or list,
    sample_size,
    mean_vector: int or np.array = 0,
    cov_matrix=1,
    fixed_design=False,
) -> pd.core.frame.DataFrame:
    if fixed_design:
        return pd.DataFrame(
            multivariate_normal(mean_vector, cov_matrix).rvs(
                sample_size, random_state=42
            ),
            columns=covariates_name,
        )
    else:
        return pd.DataFrame(
            multivariate_normal(mean_vector, cov_matrix).rvs(sample_size),
            columns=covariates_name,
        )


def create_matrix_with_sigma(Sigma, add_H_cols, with_intercept, K):
    d = Sigma.shape[0]  
    if not add_H_cols:
        # Create A matrix with 1 on the top left
        A = np.zeros((d + 1, d + 1))
        A[0, 0] = 1

        # Insert Sigma on the bottom right
        A[1 : d + 1, 1 : d + 1] = Sigma

    # Add ones to the right of Sigma
    elif add_H_cols and not with_intercept:
        A = np.zeros((d + K, d + K))
        A[0:d, 0:d] = Sigma
        A[d:, d:] = np.eye(K)

    elif add_H_cols and with_intercept:
        A = np.zeros((d + K + 1, d + K + 1))
        A[0, 0] = 1
        A[1 : d + 1, 1 : d + 1] = Sigma
        A[d + 1 :, d + 1 :] = np.eye(K)

    return A


def individual_treatment_effect(df, treatment):
    if df[treatment] == 1:
        return df["Y"] - df["Y_cf*"]
    else:
        return df["Y_cf*"] - df["Y"]


def param_mse(list_estimated_params, true_param):
    return np.mean(
        [
            (np.array(param) - np.array(true_param)) ** 2
            for param in list_estimated_params
        ]
    )

def mu(X: np.array, beta: np.array):
    return np.dot(beta, X.T)

def model_noise(data: pd.core.frame.DataFrame, sigma2: float):
    return np.random.normal(0, sigma2, data.shape[0])


def generate_Y(
    data: pd.core.frame.DataFrame,
    beta_1: np.array,
    beta_0: np.array,
    covariates_name: list,
    treatment_name: str,
    h_k: float = 0,
    noise: str = "model_noise",
    Y_is_linear=True,
):
    if Y_is_linear == True:
        return (
            data.apply(
                lambda x: (
                    x[covariates_name] @ beta_1 + h_k
                    if x[treatment_name] == 1
                    else x[covariates_name] @ beta_0 + h_k
                ),
                axis=1,
            )
            + data[noise]
        )
    elif Y_is_linear == "polynomial":
        polyn_coeff_treated = np.array([j / 2 for j in range(len(beta_1))])
        polyn_coeff_control = np.array([j / 2 - 1 / 2 for j in range(len(beta_0))])
        return (
            data.apply(
                lambda x: (
                    np.sum(
                        [
                            (
                                polyn_coeff_treated[i] * (x[covariates_name][i] ** 2)
                                if i < len(covariates_name) / 2
                                else polyn_coeff_treated[i] * x[covariates_name][i]
                            )
                            for i in range(len(covariates_name))
                        ]
                    )
                    + h_k
                    + x[covariates_name][-1] * x[covariates_name][-2]
                    if x[treatment_name] == 1
                    else np.sum(
                        [
                            (
                                polyn_coeff_control[i] * (x[covariates_name][i] ** 2)
                                if i < len(covariates_name) / 2
                                else polyn_coeff_control[i] * x[covariates_name][i]
                            )
                            for i in range(len(covariates_name))
                        ]
                    )
                    + h_k
                    + x[covariates_name][0] * x[covariates_name][-1]
                ),
                axis=1,
            )
            + data[noise]
        )
    elif Y_is_linear == "sinus":
        return (
            data.apply(
                lambda x: (
                    np.sin(x[covariates_name][0])
                    + np.cos(x[covariates_name][1]) / (1 + x[covariates_name][2])
                    + h_k
                    if x[treatment_name] == 1
                    else np.cos(np.sum(x[covariates_name]))
                    + np.sin(x[covariates_name][0]) / (1 + x[covariates_name][1])
                    + h_k
                ),
                axis=1,
            )
            + data[noise]
        )


### Parameters
def compute_beta(df: pd.core.frame.DataFrame, covariate_names: list):
    X = df[covariate_names]
    Y = df["Y"]
    beta = sm.OLS(Y, X).fit().params
    return np.array(beta)


def estimate_sigma2(data: pd.core.frame.DataFrame, beta: list, covariate_names: list):
    return np.sum((data["Y"] - np.dot(data[covariate_names], beta)) ** 2) / (
        data.shape[0] - 1
    )


def gradient_beta(
    data: pd.core.frame.DataFrame,
    beta: list,
    covariate_names: list,
):
    X = np.array(data[covariate_names])
    Y = np.array(data["Y"]).reshape(-1, 1)
    n_k = data.shape[0]
    beta = np.array(beta).reshape(-1, 1)
    error = Y - np.dot(X, beta)
    gradient = -2 / n_k * np.sum(np.dot(X.T, error), axis=1)
    return gradient


def estimate_hessian_beta(
    data: pd.core.frame.DataFrame,
    beta: list,
    covariate_names: list,
    sigma2,
    known_sigma2: bool = False,
):
    return (
        1
        / estimate_sigma2(data, beta, covariate_names)
        * np.dot(data[covariate_names].T, data[covariate_names])
        if not known_sigma2
        else 1 / sigma2 * np.dot(data[covariate_names].T, data[covariate_names])
    )  # H(\beta) = X^T X / \sigma^2


def logistic_function(x):
    return np.array(1 / (1 + np.exp(-x)))


### Estimators of the parameters
def estimate_beta_MLE(
    data: pd.core.frame.DataFrame, covariate_names: list,
):
    df = data.copy()
    X = df[covariate_names]
    Y = df["Y"]
    beta = sm.OLS(Y, X).fit().params
    return np.array(beta)


def compute_g_formula(df, beta1, beta0, covariate_names, return_ite=False):
    g_formula = mu(df[covariate_names], beta1) - mu(df[covariate_names], beta0)
    return np.mean(g_formula) if not return_ite else g_formula


def hessian_weighting(
    hessian_matrices_list: list, client_list: list, params_to_weight: list
):
    # convert the elements of the list to np.array
    hessian_matrices_list = [
        np.array(hessian_matrix) for hessian_matrix in hessian_matrices_list
    ]
    params_to_weight = [np.array(param) for param in params_to_weight]
    return np.dot(
        np.linalg.inv(np.sum(hessian_matrices_list, axis=0)),
        np.sum(
            [
                np.dot(hessian_matrix, param)
                for hessian_matrix, param in zip(
                    hessian_matrices_list, params_to_weight
                )
            ],
            axis=0,
        ),
    )


### Federated SGD
def fedavg_sgd_client_update(
    df,
    covariate_names,
    theta,
    beta1_or_beta0,
    batch_size,
    learning_rate,
):
    # shuffle and split the data into batches
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
    if beta1_or_beta0 == "beta1":
        df_w = df[df["W"] == 1]
    elif beta1_or_beta0 == "beta0":
        df_w = df[df["W"] == 0]
    # split the data into batches if batchsize is not None
    if batch_size is not None:
        df_w_batches = [
            df_w.iloc[i : i + batch_size] for i in range(0, len(df_w), batch_size)
        ]
        # iterate over the batches
        for batch in df_w_batches:
            # compute the gradient
            if beta1_or_beta0 == "beta1" or beta1_or_beta0 == "beta0":
                gradient = gradient_beta(
                    batch,
                    theta,
                    covariate_names,
                )
            # update the parameters
            theta = theta - learning_rate * gradient
        return theta
    else:
        # compute the gradient
        if beta1_or_beta0 == "beta1" or beta1_or_beta0 == "beta0":
            gradient = gradient_beta(df_w, theta, covariate_names)
        # update the parameters
        theta = theta - learning_rate * gradient
        return theta


def fedavg_sgd_server_side(
    df_dict,
    covariate_names,
    max_rounds,
    learning_rate,
    beta1_or_beta0,
    batch_size,
    print_global_parameters=True,
):
    # initialize the global parameters
    global_parameters = np.zeros(len(covariate_names))
    # initialize the local parameters
    local_parameters = {client: [] for client in df_dict.keys()}
    # iterate over the rounds
    for round in range(max_rounds):
        fedavg_weights = []
        # iterate over the clients
        for client in df_dict.keys():
            # compute the local parameters ie take one step
            local_parameters[client] = fedavg_sgd_client_update(
                df_dict[client],
                covariate_names,
                global_parameters,
                beta1_or_beta0,
                batch_size,
                learning_rate,
            )
            # compute the weights
            if beta1_or_beta0 == "beta1":
                fedavg_weights.append(
                    df_dict[client][df_dict[client]["W"] == 1].shape[0]
                )
            elif beta1_or_beta0 == "beta0":
                fedavg_weights.append(
                    df_dict[client][df_dict[client]["W"] == 0].shape[0]
                )
            else:
                fedavg_weights.append(df_dict[client].shape[0])
        # average the local parameters
        global_parameters = np.average(
            [local_parameters[client] for client in df_dict.keys()],
            axis=0,
            weights=fedavg_weights,
        )
    print(global_parameters) if print_global_parameters else None
    return global_parameters


class DGP_Fed:
    def __init__(
        self,
        fixed_design: bool,
        dim_x: int,
        mean_covariates: np.array,
        cov_covariates: np.array,
        beta_1: np.array,
        beta_0: np.array,
        tau_k_var: float,
        Y_is_linear: bool = True,  # WIP
        sigma2: float = 0.3,
        sample_size: int = 1000,
        p_k: float = 0.5,
        h_k=0,
        random_intercept: bool = False,
        show_hidden_variables: bool = False,  # show counterfactuals, propensity scores, individual treatment effect and model noise
    ):
        self.fixed_design = fixed_design
        self.dim_x = dim_x
        self.mean_covariates = mean_covariates
        self.cov_covariates = cov_covariates
        self.Y_is_linear = Y_is_linear
        self.tau_k_var = tau_k_var
        self.sample_size = sample_size
        self.beta_1 = beta_1
        self.beta_0 = beta_0
        self.show_hidden_variables = show_hidden_variables
        self.sigma2 = sigma2
        self.p_k = p_k  # probability of being treated in RCT
        self.X_cols = ["intercept"] + columns_list("X", self.dim_x)
        self.h_k = h_k
        self.random_intercept = random_intercept

        # First generate the data with the treatment and the model noise

        pretreatment_df = pd.DataFrame()
        pretreatment_df["intercept"] = [1] * self.sample_size
        pretreatment_df[columns_list("X", self.dim_x)] = generate_multivariate_normal(
            covariates_name=columns_list("X", self.dim_x),
            sample_size=self.sample_size,
            mean_vector=self.mean_covariates,
            cov_matrix=self.cov_covariates,
            fixed_design=self.fixed_design,
        )
        pretreatment_df["W"] = np.random.binomial(1, self.p_k, self.sample_size)
        self.df = pretreatment_df
        self.df["model_noise*"] = model_noise(self.df, self.sigma2)
        # Then add the outcome
        if random_intercept:
            temp_tau = self.beta_1[0] - self.beta_0[0]
            self.beta_1[0] = np.random.normal(self.beta_1[0], 1)
            self.beta_0[0] = self.beta_1[0] - temp_tau
        self.df["Y"] = generate_Y(
            data=self.df,
            beta_1=self.beta_1,
            beta_0=self.beta_0,
            covariates_name=self.X_cols,
            treatment_name="W",
            h_k=self.h_k,
            noise="model_noise*",
            Y_is_linear=self.Y_is_linear,
        )
        self.df.drop(columns=["model_noise*"], inplace=True)

        if self.show_hidden_variables:
            self.df["W_cf*"] = 1 - self.df["W"]
            self.df["model_noise*"] = model_noise(
                self.df, sigma2=self.sigma2
            )  # model noise for the counterfactual does not have to be the same as the noise on the observed outcome so we generate it again
            self.df["Y_cf*"] = generate_Y(
                data=self.df,
                beta_1=self.beta_1,
                beta_0=self.beta_0,
                covariates_name=self.X_cols,
                treatment_name="W_cf*",
                noise="model_noise*",
            )
            self.df["propensity_score"] = self.p_k
            self.df["ITE*"] = self.df.apply(
                lambda x: individual_treatment_effect(x, "W"), axis=1
            )


class Simulations_Fed:
    def __init__(
        self,
        client_params_dict: dict,
        estimator: str = None,
        n_simulations: int = 1000,
        fixed_design: bool = False,
        known_sigma2: bool = False,
        estimate_Sigma: bool = False,
        estimate_sigma2: bool = False,
        estime_norm_beta1_minus_beta0: bool = False,
    ):
        self.n_simulations = n_simulations
        self.client_params_dict = client_params_dict
        self.clients_list = list(client_params_dict.keys())
        self.estimators_dict = {
            "g_formula": basic_utils.g_formula,
        }
        self.estimator = (
            self.estimators_dict[estimator] if estimator is not None else None
        )
        self.dim_x = client_params_dict["client1"]["dim_x"] # all clients need to have the same number of covariates
        self.fixed_design = fixed_design
        self.known_sigma2 = known_sigma2
        self.estimate_Sigma = estimate_Sigma
        self.estimate_sigma2 = estimate_sigma2
        self.estime_norm_beta1_minus_beta0 = estime_norm_beta1_minus_beta0

    def create_data(self, other_params: dict = None):
        if other_params is None:
            client_dataframes = {}
            for client in self.clients_list:
                client_dataframes[client] = DGP_Fed(
                    self.fixed_design, **self.client_params_dict[client]
                ).df
                client_dataframes[client]["client"] = client
                client_dataframes[client]["intercept"] = [1] * client_dataframes[
                    client
                ].shape[0]
            return client_dataframes
        else:
            client_dataframes = {}
            for client in self.clients_list:
                client_dataframes[client] = DGP_Fed(
                    self.fixed_design, **other_params[client]
                ).df
                client_dataframes[client]["client"] = client
                client_dataframes[client]["intercept"] = [1] * client_dataframes[
                    client
                ].shape[0]
            return client_dataframes

    def combine_data(self, dict_data: dict = None):
        if dict_data is None:
            dict_data = self.create_data()
        else:
            dict_data = dict_data
        df = pd.concat(dict_data.values())
        return df

    def beta_estimates(self):
        beta_dict = {client_name: [] for client_name in self.clients_list}
        X_cols = columns_list("X", self.dim_x)
        for _ in range(self.n_simulations):
            data_dict = self.create_data()
            for client in data_dict.keys():
                beta_dict[client].append(compute_beta(data_dict[client], X_cols))
        return beta_dict

    def hessian_estimates(self):
        hessian_dict = {client_name: [] for client_name in self.clients_list}
        X_cols = ["intercept"] + columns_list("X", self.dim_x)
        for _ in range(self.n_simulations):
            data_dict = self.create_data()
            for client in data_dict.keys():
                hessian_dict[client].append(
                    estimate_hessian_beta(
                        data_dict[client],
                        compute_beta(data_dict[client], X_cols),
                        X_cols,
                        sigma2=self.client_params_dict[client]["sigma2"],
                        known_sigma2=self.known_sigma2,
                    )
                )
        return hessian_dict

    def estimate_tau(
        self,
        clip_or_trim: str = "trim",
        estimator_family: list = [
            "g_formula"
        ],  # "g_formula", "aipw", "ipw", "IV", "dm"
        estimators_to_compute: list = [
            "SGD_fed",
            "1S_SW",
            "1S_IVW (est. sigma2)",
            "1S_IVW (true sigma2)",
            "meta_SW",
            "meta_IVW",
            "diff_in_means",
        ],
        aggregations_to_perform: list = ["SW_agg", "IVW_agg"],
        ivw_with_hat_var: bool = False,
        # estimate_var_residuals=False,
        params_for_federation=None or dict,
        # random_intercept=False,
        print_global_parameters=False,
        add_H_cols: bool = False,
        intercepts_shared_for_shot: bool = True,
        generate_test_set: bool = False,
        compute_hessians=False,
        estimate_Sigma=False,
        return_thetas=False,
        cross_device_scenario=False,
        estimate_sigma_pk_Sigma=True,
        pass_cond1_error=True,
    ):
        """Put params_for_federation to None if you don't want to federate the parameters using SGD"""
        ### Estimate Beta 
        # Initiating the dictionaries that will contain the estimations of the parameters and their related hessians
        federation_parameters = params_for_federation 
        dict_beta1 = {
            "OLS": {
                client_name: []
                for client_name in self.clients_list
                + [
                    "total_data",
                    "inverse_variance_weighted_est_sigma",
                    "inverse_variance_weighted_true_sigma",
                    "sample_size_weighted",
                    (
                        "inverse_variance_weighted_est_sigma_noC"
                        if not intercepts_shared_for_shot
                        else ""
                    ),
                    (
                        "inverse_variance_weighted_true_sigma_noC"
                        if not intercepts_shared_for_shot
                        else ""
                    ),
                ]
            },
            "SGD": [],
            "SGD_e_weighted": [],
        }
        dict_beta0 = {
            "OLS": {
                client_name: []
                for client_name in self.clients_list
                + [
                    "total_data",
                    "inverse_variance_weighted_est_sigma",
                    "inverse_variance_weighted_true_sigma",
                    "sample_size_weighted",
                    (
                        "inverse_variance_weighted_est_sigma_noC"
                        if not intercepts_shared_for_shot
                        else ""
                    ),
                    (
                        "inverse_variance_weighted_true_sigma_noC"
                        if not intercepts_shared_for_shot
                        else ""
                    ),
                ]
            },
            "SGD": [],
            "SGD_e_weighted": [],
        }

        dict_hessian_beta1 = {
            "OLS": {
                client_name: [] for client_name in self.clients_list + ["total_data"]
            },
        }
        dict_hessian_beta0 = {
            "OLS": {
                client_name: [] for client_name in self.clients_list + ["total_data"]
            },
        }

        # if estimate_var_residuals == True:
        dict_sigma2 = {
            "sigma2_1": {
                client_name: [] for client_name in self.clients_list + ["total_data"]
            },
            "sigma2_0": {
                client_name: [] for client_name in self.clients_list + ["total_data"]
            },
        }

        # Initiating the estimators dictionaries
        dict_gformula = {
            "local": {client_name: [] for client_name in self.clients_list},
            "SGD_fed": (
                {client_name: [] for client_name in self.clients_list}
                if "SGD_fed" in estimators_to_compute
                else {}
            ),
            "1S_IVW (est. sigma2)": (
                {client_name: [] for client_name in self.clients_list}
                if "1S_IVW (est. sigma2)" in estimators_to_compute
                else {}
            ),
            "1S_IVW (true sigma2)": (
                {client_name: [] for client_name in self.clients_list}
                if "1S_IVW (true sigma2)" in estimators_to_compute
                else {}
            ),
            "1S_SW": (
                {client_name: [] for client_name in self.clients_list}
                if "1S_SW" in estimators_to_compute
                else {}
            ),
            "pool": {
                client_name: [] for client_name in self.clients_list + ["total_data"]
            },
            "meta_SW": {"total_data": []} if "meta_SW" in estimators_to_compute else {},
            "meta_IVW": (
                {"total_data": []} if "meta_IVW" in estimators_to_compute else {}
            ),
            "1S_IVW (est. sigma2) - SW_agg": (
                {"total_data": []}
                if "SW_agg" in aggregations_to_perform
                and "1S_IVW (est. sigma2)" in estimators_to_compute
                else {}
            ),
            "1S_IVW (est. sigma2) - IVW_agg": (
                {"total_data": []}
                if "IVW_agg" in aggregations_to_perform
                and "1S_IVW (est. sigma2)" in estimators_to_compute
                else {}
            ),
            "1S_IVW (true sigma2) - SW_agg": (
                {"total_data": []}
                if "SW_agg" in aggregations_to_perform
                and "1S_IVW (true sigma2)" in estimators_to_compute
                else {}
            ),
            "1S_IVW (true sigma2) - IVW_agg": (
                {"total_data": []}
                if "IVW_agg" in aggregations_to_perform
                and "1S_IVW (true sigma2)" in estimators_to_compute
                else {}
            ),
            "1S_SW - SW_agg": (
                {"total_data": []}
                if "SW_agg" in aggregations_to_perform
                and "1S_SW" in estimators_to_compute
                else {}
            ),
            "1S_SW - IVW_agg": (
                {"total_data": []}
                if "IVW_agg" in aggregations_to_perform
                and "1S_SW" in estimators_to_compute
                else {}
            ),
            "SGD_fed - SW_agg": (
                {"total_data": []}
                if "SW_agg" in aggregations_to_perform
                and "SGD_fed" in estimators_to_compute
                else {}
            ),
            "SGD_fed - IVW_agg": (
                {"total_data": []}
                if "IVW_agg" in aggregations_to_perform
                and "SGD_fed" in estimators_to_compute
                else {}
            ),
        }

        dict_diff_in_means = {
            "local": {client_name: [] for client_name in self.clients_list},
            "meta_SW": {"total_data": []} if "meta_SW" in estimators_to_compute else {},
            "pool": {"total_data": []},
        }

        dict_gformula_ipw = (
            copy.deepcopy(dict_gformula) if "g_formula_ipw" in estimator_family else {}
        )
        dict_aipw = copy.deepcopy(dict_gformula) if "aipw" in estimator_family else {}
        dict_ipw = copy.deepcopy(dict_gformula) if "ipw" in estimator_family else {}

        list_datasets = []
        list_datasets_test = []

        # Compute the estimations
        X_cols = ["intercept"] + columns_list("X", self.dim_x)

        A = create_matrix_with_sigma(
            self.client_params_dict["client1"]["cov_covariates"],
            add_H_cols=False,
            with_intercept=True,
            K=len(self.clients_list),
        )
        A_w_H = create_matrix_with_sigma(
            self.client_params_dict["client1"]["cov_covariates"],
            add_H_cols=True,
            with_intercept=True,
            K=len(self.clients_list),
        )

        if add_H_cols == True:
            dict_beta1_w_H_cols = copy.deepcopy(dict_beta1)
            dict_beta0_w_H_cols = copy.deepcopy(dict_beta0)
            dict_hessian_beta1_w_H_cols = copy.deepcopy(dict_hessian_beta1)
            dict_hessian_beta0_w_H_cols = copy.deepcopy(dict_hessian_beta0)
        initial_client_dict = copy.deepcopy(self.client_params_dict)

        for sim in tqdm(range(self.n_simulations), desc="Simulations"):
            _ = -1
            # Make the data
            data_dict = self.create_data()
            # data_dict_test = self.create_data() if generate_test_set else None
            if add_H_cols == False:
                pooled_data = self.combine_data(data_dict)
                list_datasets.append(pooled_data) if generate_test_set else None
            else:
                data_dict_w_H = data_dict.copy()
                for client in self.clients_list:
                    data_dict_w_H[client]["H"] = client
                pooled_data_H = self.combine_data(data_dict_w_H)
                pooled_data = pd.get_dummies(
                    pooled_data_H, columns=["H"], dtype=int, drop_first=True
                )
                pooled_data["H_client1"] = 0
                for client in self.clients_list:
                    data_dict_w_H[client] = pooled_data[pooled_data["client"] == client]
                X_and_H_cols = (
                    ["intercept"]
                    + columns_list("X", self.dim_x)
                    + columns_list("H_client", len(self.clients_list))
                )

            skip_iteration = False
            if pass_cond1_error:
                for client in self.clients_list:
                    df_k = pooled_data[pooled_data["client"] == client].copy()
                    df_k_1 = df_k[df_k["W"] == 1].copy()
                    df_k_0 = df_k[df_k["W"] == 0].copy()
                    if len(df_k_1) < len(X_cols) or len(df_k_0) < len(X_cols):
                        skip_iteration = True
                        print(
                            f"Iteration skipped because '{client}' has "
                            f"{len(df_k_1)} treated or {len(df_k_0)} control, which are less than required "
                            f"({len(X_cols)})."
                        )
                        break

                if skip_iteration:
                    continue

            for client in self.clients_list:

                # if there is no treated or control observations in the data, print a warning and force that there is one treated and one control observation for this
                if not cross_device_scenario:
                    if data_dict[client][data_dict[client]["W"] == 1].shape[0] == 0:
                        print(
                            f"Warning: no treated observation in the data for client {client}. Forcing one treated observation."
                        )
                        data_dict[client].iloc[
                            0, data_dict[client].columns.get_loc("W")
                        ] = 1
                    if data_dict[client][data_dict[client]["W"] == 0].shape[0] == 0:
                        print(
                            f"Warning: no control observation in the data for client {client}. Forcing one control observation."
                        )
                        data_dict[client].iloc[
                            0, data_dict[client].columns.get_loc("W")
                        ] = 0

                ## Then estimate the betas
                if add_H_cols == True:
                    dict_beta1_w_H_cols["OLS"][client].append(
                        estimate_beta_MLE(
                            data_dict_w_H[client][data_dict_w_H[client]["W"] == 1],
                            X_and_H_cols,
                        )
                    )
                    dict_beta0_w_H_cols["OLS"][client].append(
                        estimate_beta_MLE(
                            data_dict_w_H[client][data_dict_w_H[client]["W"] == 0],
                            X_and_H_cols,
                        )
                    )

                    ## Now the hessians
                    if compute_hessians:
                        dict_hessian_beta1_w_H_cols["OLS"][client].append(
                            estimate_hessian_beta(
                                data_dict_w_H[client][data_dict_w_H[client]["W"] == 1],
                                dict_beta1_w_H_cols["OLS"][client][_],
                                X_and_H_cols,
                                sigma2=self.client_params_dict[client]["sigma2"],
                                known_sigma2=self.known_sigma2,
                            )
                        )
                        dict_hessian_beta0_w_H_cols["OLS"][client].append(
                            estimate_hessian_beta(
                                data_dict_w_H[client][data_dict_w_H[client]["W"] == 0],
                                dict_beta0_w_H_cols["OLS"][client][_],
                                X_and_H_cols,
                                sigma2=self.client_params_dict[client]["sigma2"],
                                known_sigma2=self.known_sigma2,
                            )
                        )
                if not cross_device_scenario:
                    dict_beta1["OLS"][client].append(
                        estimate_beta_MLE(
                            data_dict[client][data_dict[client]["W"] == 1],
                            X_cols,
                        )
                    )
                    dict_beta0["OLS"][client].append(
                        estimate_beta_MLE(
                            data_dict[client][data_dict[client]["W"] == 0],
                            X_cols,
                        )
                    )
                    ## Now the hessians
                    if compute_hessians:
                        dict_hessian_beta1["OLS"][client].append(
                            estimate_hessian_beta(
                                (
                                    data_dict[client][data_dict[client]["W"] == 1]
                                    if not add_H_cols
                                    else data_dict_w_H[client][
                                        data_dict_w_H[client]["W"] == 1
                                    ]
                                ),
                                dict_beta1["OLS"][client][_],
                                X_cols,
                                sigma2=self.client_params_dict[client]["sigma2"],
                                known_sigma2=self.known_sigma2,
                            )
                        )
                        dict_hessian_beta0["OLS"][client].append(
                            estimate_hessian_beta(
                                (
                                    data_dict[client][data_dict[client]["W"] == 0]
                                    if not add_H_cols
                                    else data_dict_w_H[client][
                                        data_dict_w_H[client]["W"] == 0
                                    ]
                                ),
                                dict_beta0["OLS"][client][_],
                                X_cols,
                                sigma2=self.client_params_dict[client]["sigma2"],
                                known_sigma2=self.known_sigma2,
                            )
                        )
                    ## And the associated sigma2 (variance of the residuals)
                    # if estimate_var_residuals == True:
                    # if add_H_cols:
                    #     dict_sigma2_w_H["sigma2_1"][client].append(
                    #         estimate_sigma2(
                    #             data_dict_w_H[client][data_dict_w_H[client]["W"] == 1],
                    #             dict_beta1["OLS"][client][_],
                    #             X_and_H_cols,
                    #         )
                    #     )
                    #     dict_sigma2_w_H["sigma2_0"][client].append(
                    #         estimate_sigma2(
                    #             data_dict_w_H[client][data_dict_w_H[client]["W"] == 0],
                    #             dict_beta0["OLS"][client][_],
                    #             X_and_H_cols,
                    #         )
                    #     )

                    (
                        dict_sigma2["sigma2_1"][client].append(
                            estimate_sigma2(
                                data_dict[client][data_dict[client]["W"] == 1],
                                dict_beta1["OLS"][client][_],
                                X_cols,
                            )
                        )
                    )
                    (
                        dict_sigma2["sigma2_0"][client].append(
                            estimate_sigma2(
                                (data_dict[client][data_dict[client]["W"] == 0]),
                                dict_beta0["OLS"][client][_],
                                X_cols,
                            )
                        )
                    )

            ## Finally, hessian weight the parameters
            def inverse_var_weighting(
                dict_sigma2,
                data_dict,
                covariates_name,
                dict_parameters_to_weight,
                treatment_arm,
                estimate_var_residuals,
                shared_intercepts=True,
            ):
                if treatment_arm == 1:
                    sigma_est = "sigma2_1"
                else:
                    sigma_est = "sigma2_0"
                if shared_intercepts:
                    a = [
                        dict_parameters_to_weight[client][_]
                        for client in self.clients_list
                    ]
                else:
                    a = [
                        dict_parameters_to_weight[client][_][1:]
                        for client in self.clients_list
                    ]
                if estimate_var_residuals:
                    weights = [
                        (
                            1
                            / dict_sigma2[sigma_est][client][_]
                            * data_dict[client][
                                data_dict[client]["W"] == treatment_arm
                            ][covariates_name].T
                            @ data_dict[client][
                                data_dict[client]["W"] == treatment_arm
                            ][covariates_name]
                        )
                        for client in self.clients_list
                    ]
                else:
                    weights = [
                        (
                            1
                            / self.client_params_dict[client]["sigma2"]
                            * data_dict[client][
                                data_dict[client]["W"] == treatment_arm
                            ][covariates_name].T
                            @ data_dict[client][
                                data_dict[client]["W"] == treatment_arm
                            ][covariates_name]
                        )
                        for client in self.clients_list
                    ]
                # Normalize the weights
                normed_weights = []
                for weight in weights:
                    try:
                        normed_weights.append(
                            np.linalg.inv(np.sum(weights, axis=0)) @ weight
                        )
                    except:
                        normed_weights.append(np.eye(len(weight)) @ weight)
                        print("Error in normed_weights")
                # Average weight each parameter
                return np.sum(
                    np.array([normed_weights[i] @ a[i] for i in range(len(a))]), axis=0
                )

            if any("1S_IVW" in item for item in estimators_to_compute):
                if "g_formula" in estimator_family or "aipw" in estimator_family:
                    dict_beta1["OLS"]["inverse_variance_weighted_est_sigma"].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols,
                            estimate_var_residuals=True,
                            dict_parameters_to_weight=dict_beta1["OLS"],
                            treatment_arm=1,
                        )
                    )
                    dict_beta0["OLS"]["inverse_variance_weighted_est_sigma"].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols,
                            estimate_var_residuals=True,
                            dict_parameters_to_weight=dict_beta0["OLS"],
                            treatment_arm=0,
                        )
                    )
                    dict_beta1["OLS"]["inverse_variance_weighted_true_sigma"].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols,
                            estimate_var_residuals=False,
                            dict_parameters_to_weight=dict_beta1["OLS"],
                            treatment_arm=1,
                        )
                    )
                    dict_beta0["OLS"]["inverse_variance_weighted_true_sigma"].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols,
                            estimate_var_residuals=False,
                            dict_parameters_to_weight=dict_beta0["OLS"],
                            treatment_arm=0,
                        )
                    )

                if not intercepts_shared_for_shot:
                    dict_beta1["OLS"]["inverse_variance_weighted_est_sigma_noC"].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols[1:],
                            estimate_var_residuals=True,
                            dict_parameters_to_weight=dict_beta1["OLS"],
                            treatment_arm=1,
                            shared_intercepts=False,
                        )
                    )
                    dict_beta0["OLS"]["inverse_variance_weighted_est_sigma_noC"].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols[1:],
                            estimate_var_residuals=True,
                            dict_parameters_to_weight=dict_beta0["OLS"],
                            treatment_arm=0,
                            shared_intercepts=False,
                        )
                    )
                    dict_beta1["OLS"][
                        "inverse_variance_weighted_true_sigma_noC"
                    ].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols[1:],
                            estimate_var_residuals=False,
                            dict_parameters_to_weight=dict_beta1["OLS"],
                            treatment_arm=1,
                            shared_intercepts=False,
                        )
                    )
                    dict_beta0["OLS"][
                        "inverse_variance_weighted_true_sigma_noC"
                    ].append(
                        inverse_var_weighting(
                            dict_sigma2=dict_sigma2,
                            data_dict=data_dict,
                            covariates_name=X_cols[1:],
                            estimate_var_residuals=False,
                            dict_parameters_to_weight=dict_beta0["OLS"],
                            treatment_arm=0,
                            shared_intercepts=False,
                        )
                    )
            # And sample size weight the parameters too
            if any("1S_SW" in item for item in estimators_to_compute):
                dict_beta1["OLS"]["sample_size_weighted"].append(
                    np.array(
                        np.average(
                            [
                                dict_beta1["OLS"][client][_]
                                for client in self.clients_list
                            ],
                            weights=[
                                len(
                                    data_dict[client][data_dict[client]["W"] == 1]
                                )  # if not add_H_cols else len(data_dict_w_H[client][data_dict_w_H[client]["W"] == 1])
                                for client in self.clients_list
                            ],
                            axis=0,
                        )
                    )
                )
                (
                    dict_beta1_w_H_cols["OLS"]["sample_size_weighted"].append(
                        np.array(
                            np.average(
                                [
                                    dict_beta1_w_H_cols["OLS"][client][_]
                                    for client in self.clients_list
                                ],
                                weights=[
                                    len(
                                        data_dict_w_H[client][
                                            data_dict_w_H[client]["W"] == 1
                                        ]
                                    )
                                    for client in self.clients_list
                                ],
                                axis=0,
                            )
                        )
                    )
                    if add_H_cols
                    else None
                )
                dict_beta0["OLS"]["sample_size_weighted"].append(
                    np.array(
                        np.average(
                            [
                                dict_beta0["OLS"][client][_]
                                for client in self.clients_list
                            ],
                            weights=[
                                len(
                                    data_dict[client][data_dict[client]["W"] == 0]
                                )  # if not add_H_cols else len(data_dict_w_H[client][data_dict_w_H[client]["W"] == 0])
                                for client in self.clients_list
                            ],
                            axis=0,
                        )
                    )
                )
                (
                    dict_beta0_w_H_cols["OLS"]["sample_size_weighted"].append(
                        np.array(
                            np.average(
                                [
                                    dict_beta0_w_H_cols["OLS"][client][_]
                                    for client in self.clients_list
                                ],
                                weights=[
                                    len(
                                        data_dict_w_H[client][
                                            data_dict_w_H[client]["W"] == 0
                                        ]
                                    )
                                    for client in self.clients_list
                                ],
                                axis=0,
                            )
                        )
                    )
                    if add_H_cols
                    else None
                )

            # Estimate parameters beta1, beta0 on the pooled data
            dict_beta1["OLS"]["total_data"].append(
                estimate_beta_MLE(
                    pooled_data[pooled_data["W"] == 1],
                    X_cols if not add_H_cols else X_and_H_cols,
                )
            )
            dict_beta0["OLS"]["total_data"].append(
                estimate_beta_MLE(
                    pooled_data[pooled_data["W"] == 0],
                    X_cols if not add_H_cols else X_and_H_cols,
                )
            )
            # Estimate sigma2 using the pooled data
            # if estimate_var_residuals == True:
            dict_sigma2["sigma2_1"]["total_data"].append(
                estimate_sigma2(
                    pooled_data[pooled_data["W"] == 1],
                    dict_beta1["OLS"]["total_data"][_],
                    X_cols if not add_H_cols else X_and_H_cols,
                )
            )
            dict_sigma2["sigma2_0"]["total_data"].append(
                estimate_sigma2(
                    pooled_data[pooled_data["W"] == 0],
                    dict_beta0["OLS"]["total_data"][_],
                    X_cols if not add_H_cols else X_and_H_cols,
                )
            )
            ## Perform SGD in a fed fashion if params_for_federation is not None
            if "SGD_fed" in estimators_to_compute:
                dict_beta1["SGD"].append(
                    fedavg_sgd_server_side(
                        data_dict if not add_H_cols else data_dict_w_H,
                        X_cols if not add_H_cols else X_and_H_cols,
                        beta1_or_beta0="beta1",
                        **federation_parameters,
                        print_global_parameters=print_global_parameters,
                    )
                )
                dict_beta0["SGD"].append(
                    fedavg_sgd_server_side(
                        data_dict if not add_H_cols else data_dict_w_H,
                        X_cols if not add_H_cols else X_and_H_cols,
                        beta1_or_beta0="beta0",
                        **federation_parameters,
                        print_global_parameters=print_global_parameters,
                    )
                )

            for client in self.clients_list:
                if "g_formula" in estimator_family:
                    dict_gformula["local"][client].append(
                        compute_g_formula(
                            (
                                data_dict[client]
                            ),
                            (dict_beta1["OLS"][client][_]),
                            (dict_beta0["OLS"][client][_]),
                            X_cols,
                        )
                    )
                if "diff_in_means" in estimator_family:
                    dict_diff_in_means["local"][client].append(
                        np.mean(data_dict[client][data_dict[client]["W"] == 1]["Y"])
                        - np.mean(data_dict[client][data_dict[client]["W"] == 0]["Y"])
                    )

                ### tau_hat_fed_k the fed estimations
                #### First with HW federation
                if "1S_IVW (est. sigma2)" in estimators_to_compute:
                    if "g_formula" in estimator_family:
                        dict_gformula["1S_IVW (est. sigma2)"][client].append(
                            compute_g_formula(
                                data_dict[client],
                                (
                                    dict_beta1["OLS"][
                                        "inverse_variance_weighted_est_sigma"
                                    ][_]
                                    if intercepts_shared_for_shot
                                    else np.concatenate(
                                        (
                                            dict_beta1["OLS"][client][_][0],
                                            dict_beta1["OLS"][
                                                "inverse_variance_weighted_est_sigma_noC"
                                            ][_],
                                        ),
                                        axis=None,
                                    )
                                ),
                                (
                                    dict_beta0["OLS"][
                                        "inverse_variance_weighted_est_sigma"
                                    ][_]
                                    if intercepts_shared_for_shot
                                    else np.concatenate(
                                        (
                                            dict_beta0["OLS"][client][_][0],
                                            dict_beta0["OLS"][
                                                "inverse_variance_weighted_est_sigma_noC"
                                            ][_],
                                        ),
                                        axis=None,
                                    )
                                ),
                                X_cols,
                            )
                        )
                if "1S_IVW (true sigma2)" in estimators_to_compute:
                    if "g_formula" in estimator_family:
                        dict_gformula["1S_IVW (true sigma2)"][client].append(
                            compute_g_formula(
                                data_dict[client],
                                (
                                    dict_beta1["OLS"][
                                        "inverse_variance_weighted_true_sigma"
                                    ][_]
                                    if intercepts_shared_for_shot
                                    else np.concatenate(
                                        (
                                            dict_beta1["OLS"][client][_][0],
                                            dict_beta1["OLS"][
                                                "inverse_variance_weighted_true_sigma_noC"
                                            ][_],
                                        ),
                                        axis=None,
                                    )
                                ),
                                (
                                    dict_beta0["OLS"][
                                        "inverse_variance_weighted_true_sigma"
                                    ][_]
                                    if intercepts_shared_for_shot
                                    else np.concatenate(
                                        (
                                            dict_beta0["OLS"][client][_][0],
                                            dict_beta0["OLS"][
                                                "inverse_variance_weighted_true_sigma_noC"
                                            ][_],
                                        ),
                                        axis=None,
                                    )
                                ),
                                X_cols, 
                            )
                        )
                #### Now with SW federation
                if "1S_SW" in estimators_to_compute:
                    if "g_formula" in estimator_family:
                        dict_gformula["1S_SW"][client].append(
                            compute_g_formula(data_dict[client],
                                (
                                    dict_beta1["OLS"]["sample_size_weighted"][_]
                                    if intercepts_shared_for_shot
                                    else np.concatenate(
                                        (
                                            dict_beta1["OLS"][client][_][0],
                                            dict_beta1["OLS"]["sample_size_weighted"][
                                                _
                                            ][1:],
                                        ),
                                        axis=None,
                                    )
                                ),
                                (
                                    dict_beta0["OLS"]["sample_size_weighted"][_]
                                    if intercepts_shared_for_shot
                                    else np.concatenate(
                                        (
                                            dict_beta0["OLS"][client][_][0],
                                            dict_beta0["OLS"]["sample_size_weighted"][
                                                _
                                            ][1:],
                                        ),
                                        axis=None,
                                    )
                                ),
                                X_cols,  # if not add_H_cols else X_and_H_cols,
                            )
                        )

                # Now with SGD federation
                if "SGD_fed" in estimators_to_compute:
                    if "g_formula" in estimator_family:
                        dict_gformula["SGD_fed"][client].append(
                                compute_g_formula(
                                    (
                                        data_dict[client]
                                        if not add_H_cols
                                        else data_dict_w_H[client]
                                    ),
                                    dict_beta1["SGD"][_],
                                    dict_beta0["SGD"][_],
                                    X_cols if not add_H_cols else X_and_H_cols,
                                )
                            )
                # Apply the global estimations to the local data
                if "g_formula" in estimator_family:
                    dict_gformula["pool"][client].append(
                        compute_g_formula(
                            data_dict[client] if not add_H_cols else data_dict_w_H[client],
                            dict_beta1["OLS"]["total_data"][_],
                            dict_beta0["OLS"]["total_data"][_],
                            X_cols if not add_H_cols else X_and_H_cols,
                        )
                    )

            # Estimate global tau hat on total data which is the baseline to beat
            if "g_formula" in estimator_family:
                dict_gformula["pool"]["total_data"].append(
                    compute_g_formula(
                        (pooled_data),
                        dict_beta1["OLS"]["total_data"][_],
                        dict_beta0["OLS"]["total_data"][_],
                        X_cols if not add_H_cols else X_and_H_cols,
                    )
                )
            if "diff_in_means" in estimator_family:
                dict_diff_in_means["pool"]["total_data"].append(
                    np.mean(pooled_data[pooled_data["W"] == 1]["Y"])
                    - np.mean(pooled_data[pooled_data["W"] == 0]["Y"])
                )

            # Meta analysis estimates
            ## Sample size weighting the local estimations
            if "meta_SW" in estimators_to_compute:
                if "g_formula" in estimator_family:
                    dict_gformula["meta_SW"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["local"][client][_]
                                for client in self.clients_list
                            ],
                            weights=[
                                data_dict[client].shape[0]
                                for client in self.clients_list
                            ],
                        )
                    )

            ## Inverse variance weighting the local estimations
            def inv_variance_tau_hat_k(
                hat_tau,
                data_dict,
                client,
                beta_1,
                beta_0,
                X_cols,
                ivw_with_hat_var=False,
                estimate_sigma_pk_Sigma=True,
            ):
                if ivw_with_hat_var:
                    hat_var_tau_k = np.var(
                        np.array(
                            compute_g_formula(
                                (
                                    data_dict[client]
                                ),
                                beta_1,
                                beta_0,
                                X_cols,
                                return_ite=True,
                            )
                        )
                    )
                    return 1 / hat_var_tau_k
                else:
                    if hat_tau == "fed":
                        data_client = (
                            data_dict[client]
                        )
                        X_cols_no_H = columns_list("X", self.dim_x)
                        if estimate_Sigma == True:
                            Sigma = (
                                1
                                / (len(data_client) - 1)
                                * np.dot(
                                    data_client[X_cols_no_H].T, data_client[X_cols_no_H]
                                )
                            )
                        else:
                            Sigma = self.client_params_dict[client]["cov_covariates"]

                        avar_tau_hat_k = self.client_params_dict[client][
                            "sigma2"
                        ] / len(data_client) * (
                            np.sum(
                                [len(data_dict[client]) for client in self.clients_list]
                            )
                            / np.sum(
                                [
                                    len(data_dict[client][data_dict[client]["W"] == 1])
                                    for client in self.clients_list
                                ]
                            )
                            + np.sum(
                                [len(data_dict[client]) for client in self.clients_list]
                            )
                            / np.sum(
                                [
                                    len(data_dict[client][data_dict[client]["W"] == 0])
                                    for client in self.clients_list
                                ]
                            )
                        ) + 1 / len(
                            data_client
                        ) * np.array(
                            [
                                beta1 - beta0
                                for beta1, beta0 in zip(
                                    self.client_params_dict[client]["beta_1"][1:],
                                    self.client_params_dict[client]["beta_0"][1:],
                                )
                            ]
                        ).T @ Sigma @ np.array(
                            [
                                beta1 - beta0
                                for beta1, beta0 in zip(
                                    self.client_params_dict[client]["beta_1"][1:],
                                    self.client_params_dict[client]["beta_0"][1:],
                                )
                            ]
                        )
                    elif hat_tau == "meta":
                        data_client = (
                            data_dict[client]
                        )
                        if estimate_sigma_pk_Sigma == True:
                            mu = np.array(np.mean(data_client[X_cols[1:]], axis=0))
                            hat_Sigma = (
                                1
                                / (len(data_client) - 1)
                                * np.dot(
                                    data_client[X_cols[1:]].T, data_client[X_cols[1:]]
                                )
                            ) - np.outer(mu, mu)
                            # Calculate residuals for each group and combine
                            residuals_treated = (
                                data_client[data_client["W"] == 1]["Y"]
                                - data_client[data_client["W"] == 1][X_cols] @ beta_1
                            )
                            residuals_untreated = (
                                data_client[data_client["W"] == 0]["Y"]
                                - data_client[data_client["W"] == 0][X_cols] @ beta_0
                            )
                            sum_of_squared_residuals = (
                                residuals_treated**2
                            ).sum() + (residuals_untreated**2).sum()
                            degrees_of_freedom = len(data_client) - (len(X_cols) + 1)
                            hat_sigma2 = sum_of_squared_residuals / degrees_of_freedom

                            hat_pk = len(data_client[data_client["W"] == 1]) / len(
                                data_client
                            )

                            avar_tau_hat_k = hat_sigma2 / len(data_client) * (
                                1 / hat_pk + 1 / (1 - hat_pk)
                            ) + 1 / len(data_client) * np.array(
                                [
                                    beta1 - beta0
                                    for beta1, beta0 in zip(
                                        beta_1[1:],
                                        beta_0[1:],
                                    )
                                ]
                            ).T @ hat_Sigma @ np.array(  # ).T @ (Sigma - np.outer(mu, mu) if estimate_Sigma else Sigma) @ np.array(
                                [
                                    beta1 - beta0
                                    for beta1, beta0 in zip(
                                        beta_1[1:],
                                        beta_0[1:],
                                    )
                                ]
                            )
                        else:
                            Sigma = self.client_params_dict[client]["cov_covariates"]
                            A = create_matrix_with_sigma(
                                self.client_params_dict[client]["cov_covariates"],
                                add_H_cols=add_H_cols,
                                with_intercept=True,
                                K=len(self.clients_list),
                            )
                            mu = np.array(
                                self.client_params_dict[client]["mean_covariates"]
                            )

                            avar_tau_hat_k = self.client_params_dict[client][
                                "sigma2"
                            ] * (
                                1 / len(data_client[data_client["W"] == 1])
                                + 1 / len(data_client[data_client["W"] == 0])
                            ) + 1 / len(
                                data_client
                            ) * np.array(
                                [
                                    beta1 - beta0
                                    for beta1, beta0 in zip(
                                        self.client_params_dict[client]["beta_1"][1:],
                                        self.client_params_dict[client]["beta_0"][1:],
                                    )
                                ]
                            ).T @ (
                                Sigma - np.outer(mu, mu) if estimate_Sigma else Sigma
                            ) @ np.array(
                                [
                                    beta1 - beta0
                                    for beta1, beta0 in zip(
                                        self.client_params_dict[client]["beta_1"][1:],
                                        self.client_params_dict[client]["beta_0"][1:],
                                    )
                                ]
                            )
                return 1 / avar_tau_hat_k

            if "meta_IVW" in estimators_to_compute:
                if "g_formula" in estimator_family:
                    dict_gformula["meta_IVW"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["local"][client][_]
                                for client in self.clients_list
                            ],
                            weights=[
                                inv_variance_tau_hat_k(
                                    "meta",
                                    data_dict,
                                    client,
                                    (
                                        dict_beta1["OLS"][client][_]
                                    ),
                                    (
                                        dict_beta0["OLS"][client][_]
                                    ),
                                    X_cols,
                                    ivw_with_hat_var=ivw_with_hat_var,
                                    estimate_sigma_pk_Sigma=estimate_sigma_pk_Sigma,
                                )
                                for client in self.clients_list
                            ],
                        )
                    )

            # Estimate agg estimates
            ## Sample size weighting the fed estimations
            if (
                "SGD_fed" in estimators_to_compute
                and "SW_agg" in aggregations_to_perform
            ):
                n_ks = [data_dict[client].shape[0] for client in self.clients_list]
                ss_weights = [n_k / np.sum(n_ks) for n_k in n_ks]
                if "g_formula" in estimator_family:
                    dict_gformula["SGD_fed - SW_agg"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["SGD_fed"][client][_]
                                for client in self.clients_list
                            ],
                            weights=ss_weights,
                        )
                    )

            if (
                "1S_IVW (est. sigma2)" in estimators_to_compute
                and "SW_agg" in aggregations_to_perform
            ):
                if "g_formula" in estimator_family:
                    dict_gformula["1S_IVW (est. sigma2) - SW_agg"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["1S_IVW (est. sigma2)"][client][_]
                                for client in self.clients_list
                            ],
                            weights=ss_weights,
                        )
                    )
            if (
                "1S_IVW (true sigma2)" in estimators_to_compute
                and "SW_agg" in aggregations_to_perform
            ):
                if "g_formula" in estimator_family:
                    dict_gformula["1S_IVW (true sigma2) - SW_agg"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["1S_IVW (true sigma2)"][client][_]
                                for client in self.clients_list
                            ],
                            weights=ss_weights,
                        )
                    )
            if "1S_SW" in estimators_to_compute and "SW_agg" in aggregations_to_perform:
                ### SW fed parameters, SW agg estimations
                if "g_formula" in estimator_family:
                    dict_gformula["1S_SW - SW_agg"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["1S_SW"][client][_]
                                for client in self.clients_list
                            ],
                            weights=ss_weights,
                        )
                    )
            ## Inverse variance weighting the fed estimations

            if (
                "1S_SW" in estimators_to_compute
                and "IVW_agg" in aggregations_to_perform
            ):
                if "g_formula" in estimator_family:
                    dict_gformula["1S_SW - IVW_agg"]["total_data"].append(
                        np.average(
                            [
                                dict_gformula["1S_SW"][client][_]
                                for client in self.clients_list
                            ],
                            weights=inv_variance_tau_hat_k(
                                "fed",
                                (
                                    data_dict
                                ),
                                (
                                    dict_beta1["OLS"]["sample_size_weighted"]
                                ),
                                (
                                    dict_beta0["OLS"]["sample_size_weighted"]
                                ),
                                X_cols=X_cols,
                                ivw_with_hat_var=ivw_with_hat_var,
                            ),
                        )
                    )
            # IVW fed parameters, IVW agg estimations
            if (
                "1S_IVW (est. sigma2)" in estimators_to_compute
                and "IVW_agg" in aggregations_to_perform
            ):
                dict_gformula["1S_IVW (est. sigma2) - IVW_agg"]["total_data"].append(
                    np.average(
                        [
                            dict_gformula["1S_IVW (est. sigma2)"][client][_]
                            for client in self.clients_list
                        ],
                        weights=inv_variance_tau_hat_k(
                            "fed",
                            (
                                data_dict
                            ),
                            (
                                dict_beta1["OLS"]["inverse_variance_weighted_est_sigma"]
                            ),
                            (
                                dict_beta0["OLS"]["inverse_variance_weighted_est_sigma"]
                            ),
                            X_cols,
                            ivw_with_hat_var=ivw_with_hat_var,
                        ),
                    )
                )

            if (
                "1S_IVW (true sigma2)" in estimators_to_compute
                and "IVW_agg" in aggregations_to_perform
            ):
                dict_gformula["1S_IVW (true sigma2) - IVW_agg"]["total_data"].append(
                    np.average(
                        [
                            dict_gformula["1S_IVW (true sigma2)"][client][_]
                            for client in self.clients_list
                        ],
                        weights=inv_variance_tau_hat_k(
                            "fed",
                            (
                                data_dict
                            ),
                            (
                                dict_beta1["OLS"][
                                    "inverse_variance_weighted_true_sigma"
                                ]
                            ),
                            (
                                dict_beta0["OLS"][
                                    "inverse_variance_weighted_true_sigma"
                                ]
                            ),
                            X_cols,
                            ivw_with_hat_var=ivw_with_hat_var,
                        ),
                    )
                )
            # SGD fed parameters, IVW agg estimations
            if (
                "SGD_fed" in estimators_to_compute
                and "IVW_agg" in aggregations_to_perform
            ):
                dict_gformula["SGD_fed - IVW_agg"]["total_data"].append(
                    np.average(
                        [
                            dict_gformula["SGD_fed"][client][_]
                            for client in self.clients_list
                        ],
                        weights=inv_variance_tau_hat_k(
                            "fed",
                            (
                                data_dict
                            ),
                            (
                                dict_beta1["SGD"]
                                if not add_H_cols
                                else dict_beta1_w_H_cols["SGD"]
                            ),  
                            (
                                dict_beta0["SGD"]
                                if not add_H_cols
                                else dict_beta1_w_H_cols["SGD"]
                            ),  
                            X_cols if not add_H_cols else X_and_H_cols,
                            ivw_with_hat_var=ivw_with_hat_var,
                        ),
                    )
                )

        dict_estimations = {
            "g_formula": dict_gformula if "g_formula" in estimator_family else None,
            "diff_in_means": dict_diff_in_means,
        }

        dict_parameters = {
            "beta1": dict_beta1,
            "beta0": dict_beta0,
            "sigma2": dict_sigma2,
        }
        if return_thetas:
            return dict_parameters, dict_estimations
        else:
            return dict_estimations


# ------------------------------------ #
class Plotter:
    def __init__(
        self,
        client_params_dict: dict,
    ):
        self.client_params_dict = client_params_dict
        self.clients_list = list(client_params_dict.keys())
        self.dim_x = client_params_dict["client1"]["dim_x"]

    def plot_estimator(
        self,
        dict_estimations: dict,  # for local ate without federation
        estimator: str,  # choose between "aipw", "ipw", "g_formula", "g_formula_ipw", "diff_in_means"
        scenario_name: str,  # name of the scenario,
        loc_legend: str = "upper left",
        y_window: tuple = None,
        hide_legend: bool = False,
        figsize: tuple = (15, 10),
        show_locals=True,
        set_dashed_line_to=None,
        hide_axis_legend=False,
        colors_for_poster=False,
        intercepts_shared_for_legend=True,
        save_pdf=None,
    ):
        df_tau_hat = pd.DataFrame(columns=["client", "tau_hat", "estimator", "method"])

        methods_local = ["local"] if show_locals else []
        if estimator not in ["diff_in_means"]:
            if bool(dict_estimations[estimator]["1S_IVW (est. sigma2)"]):
                methods_local.append("1S_IVW (est. sigma2)")
            if bool(dict_estimations[estimator]["1S_IVW (true sigma2)"]):
                methods_local.append("1S_IVW (true sigma2)")
            if bool(
                dict_estimations[estimator]["1S_SW"]
            ):  # and dict_estimations[estimator]["1S_SW"] is not None:
                methods_local.append("1S_SW")
            if bool(
                dict_estimations[estimator]["SGD_fed"]
            ):  # and dict_estimations[est]["SGD_fed"] is not None:
                methods_local.append("SGD_fed")
            methods_local.append("pool")
        else:
            methods_local.append("diff_in_means")

        for client in self.clients_list + ["total_data"]:
            # Treat the clients boxplots first, which contain the local estimations tau_hat_k(beta_k, D_k), the fed estimation tau_hat_k(beta_fed, D_k) and the global estimation on local datasets tau_hat_k(beta_global, D_k)
            if client in self.clients_list:
                if show_locals:
                    for method in methods_local:
                        df_tau_hat = pd.concat(
                            [
                                df_tau_hat,
                                pd.DataFrame(
                                    {
                                        "client": [client],
                                        "tau_hat": [
                                            dict_estimations[estimator][method][client]
                                        ],
                                        "estimator": [estimator],
                                        "method": method,
                                    }
                                ),
                            ]
                        )
            # Treat the total data boxplots, which contain the global estimations tau_hat(beta_global, D), the two meta analysis methods and the two agregation methods
            else:
                methods_total = ["pool"]
                if estimator != "diff_in_means":
                    if dict_estimations[estimator][
                        "meta_SW"
                    ]:  # and dict_estimations[est]["meta_SW"] is not None:
                        methods_total.append("meta_SW")
                    if dict_estimations[estimator][
                        "meta_IVW"
                    ]:  # and dict_estimations[est]["meta_IVW"] is not None:
                        methods_total.append("meta_IVW")
                    if dict_estimations[estimator][
                        "1S_IVW (est. sigma2) - SW_agg"
                    ]:  # and dict_estimations[est]["1S_IVW - SW_agg"] is not None:
                        methods_total.append("1S_IVW (est. sigma2) - SW_agg")
                    if dict_estimations[estimator][
                        "1S_IVW (true sigma2) - SW_agg"
                    ]:  # and dict_estimations[est]["1S_IVW - SW_agg"] is not None:
                        methods_total.append("1S_IVW (true sigma2) - SW_agg")
                    if dict_estimations[estimator][
                        "1S_SW - SW_agg"
                    ]:  # and dict_estimations[estimator]["1S_SW - SW_agg"] is not None:
                        methods_total.append("1S_SW - SW_agg")
                    if dict_estimations[estimator][
                        "1S_IVW (est. sigma2) - IVW_agg"
                    ]:  # and dict_estimations[estimator]["1S_IVW - IVW_agg"] is not None:
                        methods_total.append("1S_IVW (est. sigma2) - IVW_agg")
                        # if intercepts_shared_for_legend:
                        #     methods_total.append("1S_IVW ($\hat\sigma_k^2$) - IVW_agg")
                        # else:
                        #     methods_total.append(r"IVW°_fed ($\hat\sigma_k^2$) - IVW_agg")
                    if dict_estimations[estimator][
                        "1S_IVW (true sigma2) - IVW_agg"
                    ]:  # and dict_estimations[estimator]["1S_IVW - IVW_agg"] is not None:
                        methods_total.append("1S_IVW (true sigma2) - IVW_agg")
                        # if intercepts_shared_for_legend:
                        #     methods_total.append(r"1S_IVW ($\sigma_k^2$) - IVW_agg")
                        # else:
                        #     methods_total.append(r"IVW°_fed ($\sigma_k^2$) - IVW_agg")
                    if dict_estimations[estimator][
                        "1S_SW - IVW_agg"
                    ]:  # and dict_estimations[estimator]["1S_SW - IVW_agg"] is not None:
                        methods_total.append("1S_SW - IVW_agg")
                    if dict_estimations[estimator][
                        "SGD_fed - SW_agg"
                    ]:  # and dict_estimations[estimator]["SGD_fed - SW_agg"] is not None:
                        methods_total.append("SGD_fed - SW_agg")
                    if dict_estimations[estimator][
                        "SGD_fed - IVW_agg"
                    ]:  # and dict_estimations[est]["SGD_fed - IVW_agg"] is not None:
                        methods_total.append("SGD_fed - IVW_agg")
                else:
                    methods_total.append("meta_SW")
                methods_total.append("pool")

        for method in methods_total:
            df_tau_hat = pd.concat(
                [
                    df_tau_hat,
                    pd.DataFrame(
                        {
                            "client": [client],
                            "tau_hat": [dict_estimations[estimator][method][client]],
                            "estimator": [estimator],
                            "method": method,
                        }
                    ),
                ]
            )
        df_tau_hat_expanded = df_tau_hat.explode("tau_hat").reset_index(drop=True)
        dict_newcols = {
            "1S_IVW (est. sigma2)": (
                "1S_IVW (est. sigma2)"
                if intercepts_shared_for_legend 
                else "IVW°_fed (est. sigma2)"
            ),
            "1S_IVW (true sigma2)": (
                "1S_IVW" if intercepts_shared_for_legend else "IVW°_fed"
            ),
            "1S_IVW (true sigma2) - SW_agg": (
                "1S_IVW - SW_agg"
                if intercepts_shared_for_legend
                else "IVW°_fed - SW_agg"
            ),
            "1S_IVW (est. sigma2) - SW_agg": (
                "1S_IVW - SW_agg"
                if intercepts_shared_for_legend 
                else "IVW°_fed - SW_agg (est. sigma2)"
            ),
            "1S_IVW (true sigma2) - IVW_agg": (
                "1S_IVW - IVW_agg"
                if intercepts_shared_for_legend
                else "IVW°_fed - IVW_agg"
            ),
            "1S_IVW (est. sigma2) - IVW_agg": (
                "1S_IVW - IVW_agg"
                if intercepts_shared_for_legend 
                else "IVW°_fed - IVW_agg (est. sigma2)"
            ),
            "1S_SW": "1S_SW" if intercepts_shared_for_legend else "SW°_fed",
            "1S_SW - SW_agg": (
                "1S_SW - SW_agg" if intercepts_shared_for_legend else "SW°_fed - SW_agg"
            ),
            "1S_SW - IVW_agg": (
                "1S_SW - IVW_agg"
                if intercepts_shared_for_legend
                else "SW°_fed - IVW_agg"
            ),
            "SGD_fed": "SGD_fed" if intercepts_shared_for_legend else "SGD°_fed",
            "SGD_fed - SW_agg": (
                "SGD_fed - SW_agg"
                if intercepts_shared_for_legend
                else "SGD°_fed - SW_agg"
            ),
            "SGD_fed - IVW_agg": (
                "SGD_fed - IVW_agg"
                if intercepts_shared_for_legend
                else "SGD°_fed - IVW_agg"
            ),
        }

        sorter = [
            "local",
            "1S_SW",
            "1S_IVW (est. sigma2)",
            "1S_IVW (true sigma2)",
            "SGD_fed",
            "diff_in_means",
            "meta_SW",
            "meta_IVW",
            "1S_SW - SW_agg",
            "1S_SW - IVW_agg",
            "1S_IVW (est. sigma2) - SW_agg",
            "1S_IVW (est. sigma2) - IVW_agg",
            "1S_IVW (true sigma2) - SW_agg",
            "1S_IVW (true sigma2) - IVW_agg",
            "SGD_fed - SW_agg",
            "SGD_fed - IVW_agg",
            "pool",
        ]

        # Define a custom sorting function for the "method" column
        def custom_sort_method(column):
            return [sorter.index(e) for e in column]

        # Sort the DataFrame by "client" and "method" columns
        df_tau_hat.sort_values(
            by=["client", "method"],
            ascending=[True, True],
            key=lambda x: custom_sort_method(x) if x.name == "method" else x,
            inplace=True,
        )
        methods_used = list(df_tau_hat["method"].unique())

        df_tau_hat_expanded["method"] = (
            df_tau_hat_expanded["method"]
            .map(
                dict_newcols,
            )
            .fillna(df_tau_hat_expanded["method"])
        )

        # Plot the boxplot
        plt.figure(figsize=figsize)

        # Set the style and palette
        sns.set_style("whitegrid")
        if colors_for_poster == False:
            custom_palette = [
                # sns.color_palette("pastel")[0],
                # sns.color_palette("pastel")[1],
                # sns.color_palette("pastel")[2],
                # sns.color_palette("pastel")[3],
                # sns.color_palette("pastel")[4],
                # sns.color_palette("husl", 8)[5],
                # sns.color_palette("hls", 8)[3],
                # sns.color_palette("Set2")[1],
                (
                    sns.color_palette("light:gray", n_colors=10)[3]
                    if "local" in methods_used
                    else None
                ),  # for local,
                (
                    sns.color_palette("light:blue", n_colors=10)[2]
                    if "1S_IVW (est. sigma2)" in methods_used
                    else None
                ),  # for 1S_IVW
                (
                    sns.color_palette("light:purple", n_colors=10)[2]
                    if "1S_IVW (true sigma2)" in methods_used
                    else None
                ),  # for 1S_IVW
                (
                    sns.color_palette("light:green", n_colors=10)[2]
                    if "1S_SW" in methods_used
                    else None
                ),  # 1S_SW
                (
                    sns.color_palette("light:red", n_colors=10)[3]
                    if "SGD_fed" in methods_used
                    else None
                ),  # SGD_fed
                sns.color_palette("pastel")[1],  # pool
                (
                    sns.color_palette("light:yellow", n_colors=10)[4]
                    if "meta_SW" in methods_used
                    else None
                ),  # meta_SW
                (
                    sns.color_palette("light:yellow", n_colors=10)[8]
                    if "meta_IVW" in methods_used
                    else None
                ),  # meta_IVW
                (
                    sns.color_palette("light:blue", n_colors=10)[4]
                    if "1S_IVW (est. sigma2) - SW_agg" in methods_used
                    else None
                ),  # 1S_IVW (est. sigma2) - SW_agg
                (
                    sns.color_palette("light:purple", n_colors=10)[4]
                    if "1S_IVW (true sigma2) - SW_agg" in methods_used
                    else None
                ),  # 1S_IVW (true sigma2) - SW_agg
                (
                    sns.color_palette("light:green", n_colors=10)[4]
                    if "1S_SW - SW_agg" in methods_used
                    else None
                ),  # 1S_SW - SW_agg
                (
                    sns.color_palette("light:blue", n_colors=10)[7]
                    if "1S_IVW (est. sigma2) - IVW_agg" in methods_used
                    else None
                ),  # 1S_IVW - IVW_agg
                (
                    sns.color_palette("light:purple", n_colors=10)[7]
                    if "1S_IVW (true sigma2) - IVW_agg" in methods_used
                    else None
                ),  # 1S_IVW - IVW_agg
                (
                    sns.color_palette("light:green", n_colors=10)[7]
                    if "1S_SW - IVW_agg" in methods_used
                    else None
                ),  # 1S_SW - IVW_agg
                (
                    sns.color_palette("light:red", n_colors=10)[4]
                    if "SGD_fed - SW_agg" in methods_used
                    else None
                ),  # SGD_fed - SW_agg
                (
                    sns.color_palette("light:red", n_colors=10)[7]
                    if "SGD_fed - IVW_agg" in methods_used
                    else None
                ),  # SGD_fed - IVW_agg
                (
                    sns.color_palette("light:orange", n_colors=10)[3]
                    if "diff_in_means" in methods_used
                    else None
                ),  # diff_in_means
            ]
        else:
            custom_palette = [
                # sns.color_palette("pastel")[0],
                # sns.color_palette("pastel")[1],
                # sns.color_palette("pastel")[2],
                # sns.color_palette("pastel")[3],
                # sns.color_palette("pastel")[4],
                # sns.color_palette("husl", 8)[5],
                # sns.color_palette("hls", 8)[3],
                # sns.color_palette("Set2")[1],
                (
                    (193 / 255, 210 / 255, 255 / 255)
                    if "local" in methods_used
                    else None
                ),  # for local BLUE,
                (
                    (143 / 255, 236 / 255, 143 / 255)
                    if "1S_IVW (est. sigma2)" in methods_used
                    else None
                ),  # for 1S_IVW light green
                (
                    (125 / 255, 196 / 255, 125 / 255)
                    if "1S_IVW (true sigma2)" in methods_used
                    else None
                ),  # for 1S_IVW light green darker
                (
                    (179 / 255, 243 / 255, 94 / 255)
                    if "1S_SW" in methods_used
                    else None
                ),  # 1S_SW
                (
                    sns.color_palette("light:red", n_colors=10)[3]
                    if "SGD_fed" in methods_used
                    else None
                ),  # SGD_fed
                (255 / 255, 190 / 255, 129 / 255),  # pool
                (
                    sns.color_palette("light:yellow", n_colors=10)[4]
                    if "meta_SW" in methods_used
                    else None
                ),  # meta_SW
                (
                    sns.color_palette("light:yellow", n_colors=10)[8]
                    if "meta_IVW" in methods_used
                    else None
                ),  # meta_IVW
                (
                    (143 / 255, 236 / 255, 143 / 255)
                    if "1S_IVW (est. sigma2) - SW_agg" in methods_used
                    else None
                ),  # 1S_IVW (est. sigma2) - SW_agg
                (
                    (125 / 255, 196 / 255, 125 / 255)
                    if "1S_IVW (true sigma2) - SW_agg" in methods_used
                    else None
                ),  # 1S_IVW (true sigma2) - SW_agg
                (
                    (179 / 255, 243 / 255, 94 / 255)
                    if "1S_SW - SW_agg" in methods_used
                    else None
                ),  # 1S_SW - SW_agg
                (
                    sns.color_palette("light:blue", n_colors=10)[7]
                    if "1S_IVW (est. sigma2) - IVW_agg" in methods_used
                    else None
                ),  # 1S_IVW - IVW_agg
                (
                    sns.color_palette("light:purple", n_colors=10)[7]
                    if "1S_IVW (true sigma2) - IVW_agg" in methods_used
                    else None
                ),  # 1S_IVW - IVW_agg
                (
                    sns.color_palette("light:green", n_colors=10)[7]
                    if "1S_SW - IVW_agg" in methods_used
                    else None
                ),  # 1S_SW - IVW_agg
                (
                    (255 / 255, 128 / 255, 128 / 255)
                    if "SGD_fed - SW_agg" in methods_used
                    else None
                ),  # SGD_fed - SW_agg
                (
                    sns.color_palette("light:red", n_colors=10)[7]
                    if "SGD_fed - IVW_agg" in methods_used
                    else None
                ),  # SGD_fed - IVW_agg
                (
                    sns.color_palette("light:orange", n_colors=10)[3]
                    if "diff_in_means" in methods_used
                    else None
                ),  # diff_in_means
            ]
        custom_palette = sns.set_palette([x for x in custom_palette if x is not None])
        # Make one plot per estimator
        # Plot the boxplot
        sns.boxplot(
            x="client",
            y="tau_hat",
            data=df_tau_hat_expanded[df_tau_hat_expanded["estimator"] == estimator],
            hue="method",
            palette=custom_palette,
            showfliers=False,
            dodge=True,
            width=0.95,
        )


        if set_dashed_line_to == "expectancy_of_tau":
            expectancy_of_tau = (
                self.client_params_dict["client1"]["beta_1"][0]
                - self.client_params_dict["client1"]["beta_0"][0]
                + np.average(
                    [
                        self.client_params_dict[client]["mean_covariates"]
                        @ (
                            np.array(self.client_params_dict[client]["beta_1"][1:])
                            - np.array(
                                self.client_params_dict[client]["beta_0"][1:]
                            )
                        ).T
                        for client in self.clients_list
                    ],
                    weights=[
                        self.client_params_dict[client]["sample_size"]
                        for client in self.clients_list
                    ],
                )
            )
            print(f"Expectancy of tau: {expectancy_of_tau}")
            plt.axhline(
                y=expectancy_of_tau,
                color="red",  # Use a custom color
                linestyle="--",
                label="True Tau",
            )
        elif set_dashed_line_to == "expectancy_of_y1_y0":
            for client in self.clients_list:
                self.client_params_dict[client]["sample_size"] = 5000
                self.client_params_dict[client]["show_hidden_variables"] = True
            list_ITEs = np.concatenate(
                [
                    DGP_Fed(
                        fixed_design=False,
                        **self.client_params_dict[client],
                    )
                    .df["ITE*"]
                    .values
                    for client in self.clients_list
                ]
            ).tolist()

            expectancy_of_y1_y0 = np.mean(list_ITEs)

            print(f"Expectancy of ITE: {expectancy_of_y1_y0}")
            plt.axhline(
                y=expectancy_of_y1_y0,
                color="red",  # Use a custom color
                linestyle="--",
                label="True Tau",
            )

        plt.title(f"{estimator} for {scenario_name}")
        plt.xlabel("Client")
        plt.ylabel("Estimation")

        plt.legend(loc=loc_legend, ncol=1)
        if hide_legend:
            plt.legend().remove()

        if hide_axis_legend:
            plt.xlabel("")
            plt.ylabel("")
        # fix y axis between -1 and 1
        if y_window is not None:
            plt.ylim(y_window)

        if save_pdf is not None:
            plt.savefig(save_pdf, format="pdf")
        plt.show()

        return df_tau_hat_expanded

