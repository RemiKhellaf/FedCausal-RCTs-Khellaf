import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import statsmodels.api as sm
import scipy.stats as scistats

class basic_utils:
    def __init__(self):
        pass

    # ATE methods
    @staticmethod
    def compute_normal_ci(_tau_hat:float, _se_hat:float) -> tuple:
        _lower_ci = _tau_hat - 1.96 * _se_hat
        _upper_ci = _tau_hat + 1.96 * _se_hat
        return _lower_ci, _upper_ci
    
    @staticmethod
    def compute_quantile_ci(_bootstrap_repetition_l:list) -> tuple:
        return np.quantile(_bootstrap_repetition_l, q=0.025), np.quantile(_bootstrap_repetition_l, q=0.975)
    
    @staticmethod
    def get_outcomes(_data_df: pd.DataFrame) -> tuple:
        treated_idx = _data_df["W"] == 1
        control_idx = _data_df["W"] == 0

        y1 = _data_df.loc[treated_idx, "Y"]
        y0 = _data_df.loc[control_idx, "Y"]

        return y0, y1
    
    # Tests
    @staticmethod
    def t_test(_data_df):
        return scistats.ttest_ind(_data_df.loc[_data_df["W"] == 1, "Y"].reset_index(drop=True), _data_df.loc[_data_df["W"] == 0, "Y"].reset_index(drop=True))
    
    # Difference in means
    @staticmethod
    def difference_in_means(_data_df:pd.DataFrame) -> pd.DataFrame:
        y0, y1 = basic_utils.get_outcomes(_data_df)
        n0, n1 = len(y0), len(y1)
        n = _data_df.shape[0]
        #     check whether everything is ok
        assert n == (n0 + n1)

        tau_hat = y1.mean() - y0.mean()
        se_hat = np.sqrt(np.var(y0)/n0 + np.var(y1)/n1)
        lower_ci, upper_ci = basic_utils.compute_normal_ci(tau_hat, se_hat)

        return pd.DataFrame({"ATE": [tau_hat], "lower_ci": [lower_ci], "upper_ci": [upper_ci]})
    
    @staticmethod
    def difference_in_means_var(_data_df):
        y0, y1 = basic_utils.get_outcomes(_data_df)
        n0, n1 = len(y0), len(y1)
        n = _data_df.shape[0]
        #     check whether everything is ok
        assert n == (n0 + n1)
        return n * (np.var(y0)/n0 + np.var(y1)/n1)
    
    def diff_in_means(df):
        df0 = df[df["W"] == 0]
        df1 = df[df["W"] == 1]
        
        dm_estimate = df1.Y.mean() - df0.Y.mean()
        # std_dm = np.sqrt(np.var(df1["Y"])/len(df1) + np.var(df0["Y"])/len(df0))
        # ci_low = dm_estimate - 1.96 * std_dm
        # ci_high = dm_estimate + 1.96 * std_dm
        #return pd.Series({"dm": dm_estimate,"var": std_dm**2,"ci_low": ci_low,"ci_high": ci_high})
        return dm_estimate

    def e_hat(df, is_parametric:bool=True):
        X_cols = list(filter(lambda x: x.startswith('X'), df.columns))
        if is_parametric:
            propensity_model = LogisticRegression().fit(df[X_cols], df["W"])
        else:
            propensity_model = GRFForestClassifier(min_node_size=len(df)**(1/2)).fit(df[X_cols], df["W"])
        propensity_estimates = propensity_model.predict_proba(df[X_cols])[:, 1]
        return propensity_estimates

    def g_formula(df:pd.DataFrame, Y_is_linear:bool=True) -> np.array:
        # Select dataframe
        df0 = df[df["W"] == 0]
        df1 = df[df["W"] == 1]
        X_cols = list(filter(lambda x: x.startswith('X'), df.columns))

        if Y_is_linear:
            # Control
            control_ols = sm.OLS(df0["Y"], sm.add_constant(df0[X_cols], has_constant='add'))
            control_reg = control_ols.fit()
            mu0_hat = control_reg.predict(sm.add_constant(df[X_cols], has_constant='add'))

            # Treatment
            treatment_ols = sm.OLS(df1["Y"], sm.add_constant(df1[X_cols], has_constant='add'))
            treatment_reg = treatment_ols.fit()
            mu1_hat = treatment_reg.predict(sm.add_constant(df[X_cols], has_constant='add'))
        else:
            # Control
            control_reg = GRFForestRegressor(min_node_size=len(df)**(1/2))
            control_reg.fit(X=df0[X_cols], y=df0["Y"], w=df0["W"]) 
            mu0_hat = control_reg.predict(df[X_cols])

            # Treatment
            treatment_reg = GRFForestRegressor(min_node_size=len(df)**(1/2))
            treatment_reg.fit(X=df1[X_cols], y=df1["Y"], w=df1["W"]) 
            mu1_hat = treatment_reg.predict(df[X_cols])

        g_formula_estimate = (np.array(mu1_hat) - np.array(mu0_hat)).mean()
        # std_gformula = np.sqrt(np.var(np.array(mu1_hat) - np.array(mu0_hat))/len(df))
        # ci_low = g_formula_estimate - 1.96 * std_gformula
        # ci_high = g_formula_estimate + 1.96 * std_gformula
        #return pd.Series({"g_formula": g_formula_estimate,"var": std_gformula**2,"ci_low": ci_low,"ci_high":ci_high})
        return g_formula_estimate

    def ipw(df:pd.DataFrame, propensity_is_parametric:bool=True) -> np.array:
        # Select dataframe
        df["e_hat"] = basic_utils.e_hat(df, is_parametric=propensity_is_parametric)

        gamma = df["W"] * df["Y"] / df["e_hat"] - (1 - df["W"]) * df["Y"] / (1 - df['e_hat'])
            
        ipw_estimate = gamma.mean()
        return ipw_estimate


    def aipw(df:pd.DataFrame, Y_is_linear:bool=True, propensity_is_parametric:bool=True) -> np.array:
        # Select dataframe
        df0 = df[df["W"] == 0]
        df1 = df[df["W"] == 1]
        X_cols = list(filter(lambda x: x.startswith('X'), df.columns))

        if Y_is_linear:
            # Control
            control_ols = sm.OLS(df0["Y"], sm.add_constant(df0[X_cols], has_constant='add'))
            control_reg = control_ols.fit()
            mu0_hat = control_reg.predict(sm.add_constant(df[X_cols], has_constant='add'))

            # Treatment
            treatment_ols = sm.OLS(df1["Y"], sm.add_constant(df1[X_cols], has_constant='add'))
            treatment_reg = treatment_ols.fit()
            mu1_hat = treatment_reg.predict(sm.add_constant(df[X_cols], has_constant='add'))
            
        else:
            # Control
            control_reg = GRFForestRegressor(min_node_size=len(df0)**(1/2))
            control_reg.fit(X=df0[X_cols], y=df0["Y"]) 
            mu0_hat = control_reg.predict(df[X_cols])

            # Treatment
            treatment_reg = GRFForestRegressor(min_node_size=len(df1)**(1/2))
            treatment_reg.fit(X=df1[X_cols], y=df1["Y"]) 
            mu1_hat = treatment_reg.predict(df[X_cols])
            
        gamma = mu1_hat - mu0_hat + df["W"] / basic_utils.e_hat(df, propensity_is_parametric) * (df["Y"] -  mu1_hat) - (1 - df["W"]) / (1 - basic_utils.e_hat(df, propensity_is_parametric)) * (df["Y"] -  mu0_hat)
        aipw_estimate = gamma.mean()
        
        return aipw_estimate

    def wald_linear(df: pd.DataFrame, z_is_binary: bool = True) -> np.array:
        try:
            if z_is_binary:
                # Select dataframe
                dfz0 = df[df["Z"] == 0]
                dfz1 = df[df["Z"] == 1]
                t_mean_diff = dfz1['T'].mean() - dfz0['T'].mean()
                if t_mean_diff != 0:
                    return (dfz1['Y'].mean() - dfz0['Y'].mean()) / t_mean_diff
                else:
                    return 0
            else:
                z_cov = df['T'].cov(df['Z'])
                if z_cov != 0:
                    return df['Y'].cov(df['Z']) / z_cov
                else:
                    return 0
        except:
            pass

    def two_stage_least_square(df: pd.DataFrame, z_is_binary=True,) -> np.array:
        X_cols = list(filter(lambda x: x.startswith('X'), df.columns))
        # First stage
        Z_first = df[['Z'] + X_cols]  # Input feature(s) for the first stage
        T_first = df['T']    # Target variable for the first stage
        reg_first = LinearRegression().fit(Z_first, T_first)
        df['T_hat'] = reg_first.predict(Z_first)

        # Second stage
        T_second = df[['T_hat'] + X_cols]  # Input feature(s) for the second stage
        y_second = df['Y']        # Target variable for the second stage
        reg_second = LinearRegression().fit(T_second, y_second)
        return reg_second.coef_[0]

    def IV(df, z_is_binary=True):
        if z_is_binary:
            dfz1 = df[df['Z']==1]
            dfz0 = df[df['Z']==0]
            if dfz1['T'].mean() - dfz0['T'].mean() == 0:
                return 0
            else:
                return (dfz1['Y'].mean() - dfz0['Y'].mean()) / (dfz1['T'].mean() - dfz0['T'].mean())
        else:
            z_cov = df['T'].cov(df['Z'])
            if z_cov == 0:
                return 0
            else:
                return df['Y'].cov(df['Z']) / z_cov
        
