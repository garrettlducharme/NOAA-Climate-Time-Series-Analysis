import pandas as pd
import pandas.tseries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

def ts_plot(df_dict, metric, state, start):
    """
    Plots the time series for a given state and metric for the dataframe
    dictionary from the provided start date.
    """
    ts = df_dict[metric][state]
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(ts[start:], label = f'{metric}')
    ax.axhline(ts[start:].mean(), label = f'Average {metric}', color='g')
    ax.set_title(f" {state}'s Monthly {metric}")
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{metric}')
    ax.legend(fancybox=True, framealpha=0.4, loc='best', ncol=2)
    plt.tight_layout()
    
def MA_plot(df_dict, metric, state, start, n_months):
    """
    Plots the moving average of a time series over a given number of months
    for the provided state, metric, and start date.
    """
    ts = df_dict[metric][state]
    ts_rolling_mean = ts.rolling(n_months).mean().dropna()
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(ts_rolling_mean[start:], label = f'{round(n_months/12, 2)} Year Rolling {metric}')
    ax.axhline(ts_rolling_mean[start:].mean(), label = f'Overall Average {metric}', color='g')
    ax.set_title(f"Long Term Changes in {state}'s {metric}")
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{metric}')
    ax.legend(fancybox=True, framealpha=0.4, loc='best', ncol=2)
    plt.tight_layout()
    
def plot_state_ts(df_dict, metrics, state, start_date='1940-01-01'):
    
    """
    Plots the provided metrics for a given state and a start date.
    """

    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(24, 9))

    for n, metric in enumerate(metrics):
        x = n//3
        y = n%3
        axs[x, y].plot(df_dict[metric][state][start_date:], label = metric)
        axs[x, y].legend(loc='best')
    
    fig.tight_layout()
    
def plot_state_MA(df_dict, metrics, state, start_date, n_months):
    """
    Plots moving averages for every metric in a given state
    """
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(24, 9))

    for n, metric in enumerate(metrics):
        x = n//3
        y = n%3
        ts = df_dict[metric][state][start_date:].rolling(n_months).mean()
        axs[x, y].plot(ts, label = metric)
        axs[x, y].legend(loc='best')
        
    fig.tight_layout()
    
def differencer(df):
    """
    returns 12 month, 1 month, and 12_1 month differenced data for a dataframe or timeseries.
    """
    df_12 = df.diff(12).dropna()
    df_1 = df.diff(1).dropna()
    df_12_1 = df_12.diff(1).dropna()
    
    return df_12, df_1, df_12_1

def corr_plotter(ts, lags):
    """
    Plots the ACF and PACF for the given dataframe, states, and lags
    """
    fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(12,5), sharex=True)
    
    sm.graphics.tsa.plot_acf(ts, lags=lags, ax=axs1);
    sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=axs2);
    
    plt.tight_layout()
    
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
def calc_d_fuller(df_dict, metric, start_year):
    """
    Runs a Dickey-Fuller test for all state time series within a given dataframe.
    Returns a DataFrame with the Results of the Dickey-Fuller test for each state time 
    series. 
    """
    df = df_dict[metric][start_year:]
    
    test_stats, p_vals, n_lags = {}, {}, {}
    
    for state in df_dict[metric].columns:
        results = adfuller(df_dict[metric][state][start_year:], autolag='AIC')
        test_stats[state] = results[0]
        p_vals[state] = results[1]
        n_lags[state] = results[2]
        
    results_df = pd.DataFrame([test_stats, p_vals, n_lags],
                              index=['test_stat', 'p_value', 'n_lags'])
    
    return results_df.T

def all_auto_arima(df):
    
    order, seasonal_order, AIC, BIC, MSE = {}, {}, {}, {}, {}
    forecasts = pd.DataFrame()
    
    for col in df.columns:
        train = df[col][:int(0.8*len(df[col]))]
        test = df[col][int(0.8*len(df[col])):]
        
        model = auto_arima(train, start_p=0, start_q=0, max_p=5, max_d=1, max_q=5,
                           start_P=0, start_Q=0, max_P=2, max_Q=2, D=1, m=12,
                           seasonal=True)
        model.fit(train)
        forecast = model.predict(n_periods=len(test))
        forecast = pd.Series(forecast, index=test.index, name=test.name)
        
        order[col] = model.order
        seasonal_order[col] = model.seasonal_order
        AIC[col] = model.aic()
        BIC[col] = model.bic()
        MSE[col] = mean_squared_error(test, forecast)
        
        forecasts = forecasts.append(forecast)
        
    forecasts = forecasts.T
    results = pd.DataFrame([order, seasonal_order, AIC, BIC, MSE]).T
    results.columns = ['Order', 'Seasonal_Order', 'AIC', 'BIC', 'MSE']
    
    return [forecasts, results]