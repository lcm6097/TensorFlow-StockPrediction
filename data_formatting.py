#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:54:15 2019

File containg all functions to format the data, and add all the indicators needed

@name: data_formatting.py

@author: Lucas Magalhaes
"""

import math
from sklearn.preprocessing import MinMaxScaler

def add_boulinger_bands(df):
    middle_band = []
    upper_band = []
    lower_band = []

    for i in range(20, df.shape[0]+1):

        lst_num = df['Adj_Close'][(i-20):i].tolist()
        _mid_band = sum(lst_num)/20
        std = math.sqrt(sum([math.pow((x-_mid_band),2) for x in lst_num])/20)
        _upper_band = _mid_band + (2*std)
        _lower_band = _mid_band - (2*std)

        if i == 20:
            middle_band = [_mid_band for x in range(20)]
            upper_band = [_upper_band for x in range(20)]
            lower_band = [_lower_band for x in range(20)]
        else:
            middle_band.append(_mid_band)
            upper_band.append(_upper_band)
            lower_band.append(_lower_band)

    df['Middle_Band'] = middle_band
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    df['Lower_Upper_Diff'] = df['Upper_Band'] - df['Lower_Band']
    df['Upper_Price_Diff'] = df['Upper_Band'] - df['Adj_Close']
    df['Price_Lower_Diff'] = df['Adj_Close'] - df['Lower_Band']

    return df

def add_RSI(df):
    rsi = []

    av_gain = 0
    av_loss = 0

    for i in range(15, df.shape[0]+1):
        lst_num = df['Adj_Close'][(i-15):i].tolist()
        lst_change = [lst_num[x+1] - lst_num[x] for x in range(len(lst_num)-1)]

        if i == 15:
            _av_gains = sum([x for x in lst_change if x > 0])/14
            _av_loss = -1*sum(list(filter(lambda x: x < 0, lst_change)))/14

            if _av_loss == 0:
                _rsi = 100
            else:
                rs = _av_gains/_av_loss
                _rsi = 100 - (100/(1+rs))

            av_gain = _av_gains
            av_loss = _av_loss

            rsi = [_rsi for x in range(15)]

        else:
            curr_gain = lambda x: x if x > 0 else 0
            curr_loss = lambda x: abs(x) if x < 0 else 0

            _av_gains = ((av_gain*13)+curr_gain(lst_change[(len(lst_change)-1)]))/14
            _av_loss = ((av_loss*13)+curr_loss(lst_change[(len(lst_change)-1)]))/14
            
            if _av_loss == 0:
                _rsi = 100
            else:
                rs = _av_gains/_av_loss
                _rsi = 100 - (100/(1+rs))

            av_gain = _av_gains
            av_loss = _av_loss
            rsi.append(_rsi)

    df['RSI'] = rsi
    return df

def add_moving_averages_MACD(df):

    #26 ema line
    two_six_ema = [0 for i in range(25)]
    #12 ema line
    one_two_ema = [0 for i in range(11)]
    #signal line is a 9 day ema line
    signal_ema = [0 for i in range(33)]
    #macd line
    macd_line = [0 for i in range(25)]
    #simple 50 day moving average
    fifty_ma = [0 for i in range(49)]
    #simple 200 day moving average
    two_hondo_ma = [0 for i in range(199)]

    #Turning the price column into list for ease of use
    lst_num = df['Adj_Close'].tolist()

    _12_day_ema = 0
    _26_day_ema = 0
    _9_day_ema = 0
    for i in range(len(lst_num)):
        """
        12 DAY EXPONENTIAL MOVING AVERAGE
        """
        if i >= 11:
            if i == 11:
                _12_day_ema = sum(lst_num[0:i+1])/12
            else:
                c = 2/13
                _12_day_ema = (lst_num[i]-_12_day_ema)*c + _12_day_ema

            one_two_ema.append(_12_day_ema)

        """
        26 DAY EXPONENTIAL MOVING AVERAGE
        """
        if i >= 25:
            if i == 25:
                _26_day_ema = sum(lst_num[0:i+1])/26
            else:
                c = 2/27
                _26_day_ema = (lst_num[i]-_26_day_ema)*c + _26_day_ema

            two_six_ema.append(_26_day_ema)

        """
        MACD LINE
        """
        if i >= 25:
            macd_line.append(one_two_ema[i] - two_six_ema[i])

        """
        9 DAY EXPONENTIAL MOVING AVERAGE - SIGNAL LINE
        """
        if i >= 33:
            if i == 33:
                _9_day_ema = sum(macd_line[i-9:i+1])/9
            else:
                c = 2/10
                _9_day_ema = (macd_line[i]-_9_day_ema)*c + _9_day_ema

            signal_ema.append(_9_day_ema)

        """
        50 DAY SIMPLE MOVING AVERAGE
        """
        if i >= 49:
            _50_day_sma = sum(lst_num[i-49:i+1])/50

            fifty_ma.append(_50_day_sma)           

        """
        200 DAY SIMPLE MOVING AVERAGE
        """
        if i >= 199:
            _200_day_sma = sum(lst_num[i-199:i+1])/200
        
            two_hondo_ma.append(_200_day_sma)

    df['MACD_Line'] = macd_line 
    df['Signal_Line'] = signal_ema
    df['MACD_Diff'] = df['MACD_Line'] - df['Signal_Line']
    df['200_day_SMA'] = two_hondo_ma
    df['50_day_SMA'] = fifty_ma
    return df

def scaler(df, scaler_type, smoothing_size='365'):
    
    """
    Min Max Scaler will scale all features to fit between -1 and 1,
    it uses a sliding window, that will prevent early stock prices
    to be useless since they would get even smaller.
    The default value of the sliding window is 365 (a year)
    """
    if scaler_type == 'min_max_scaler':
        #Turn dataFrame into numpy array
        all_data = df.values
        scaler = MinMaxScaler()
        smoothing_window_size = 365
        for di in range(0,((all_data.shape[0]//smoothing_window_size)*smoothing_window_size), smoothing_window_size):
            scaler.fit(all_data[di:di+smoothing_window_size])
            all_data[di:di+smoothing_window_size, :] = scaler.transform(all_data[di:di+smoothing_window_size, :])

        scaler.fit(all_data[di+smoothing_window_size:])
        all_data[di+smoothing_window_size:, :] = scaler.transform(all_data[di+smoothing_window_size:, :])
        
        return all_data
    """
    Percent change scaler, will scale the:
    #Name, Index
    - Adj_Close
    - 200_day_SMA
    - 50_day_SMA
    - Middle_Band
    - Lower_Band
    - Upper_Band
    
    The rest of the features are small and have low variance
    It will change the values based on percent change from
    previous value
    """    
    if scaler_type == 'percent_change':
        #Turn dataFrame into numpy array
        all_data = df.values
        #list of columns to be scaled
        cols = ['Adj_Close','200_day_SMA','50_day_SMA','Middle_Band',
               'Lower_Band','Upper_Band']
        #Get index of the columns to be scaled
        cols_i = [df.columns.get_loc(i) for i in cols]
        
        #Loop throgh all rows and scale them
        for di in range(1, all_data.shape[0]):
            for col in cols_i:
                all_data[di,col] = (all_data[di,col]-df.iloc[di-1,col])/df.iloc[di-1,col]
            
        return all_data
    
    """
    High Low Day scaler will change the feature values for negative percent
    change to -1 and feature values with positive percent change from previous
    value to 1. This will only work on the features:
    - Adj_Close
    - 200_day_SMA
    - 50_day_SMA
    
    I am also going to drop:
    - Middle_Band
    - Lower_Band
    - Upper_Band
    """    
    if scaler_type == 'high_low_day':
        #Throw away columns not used
        cols_throw = ['Middle_Band','Lower_Band','Upper_Band']
        df = df.drop(cols_throw, 1)
        
        #Columns to be scaled
        cols = ['Adj_Close','200_day_SMA','50_day_SMA']
        cols_i = cols_i = [df.columns.get_loc(i) for i in cols]
        
        #Turn dataFrame into numpy array
        all_data = df.values
        #print(all_data.shape)
        
        chg = 0
        for di in range(1, all_data.shape[0]):
            for col in cols_i:
                chg = (all_data[di,col]-df.iloc[di-1,col])/df.iloc[di-1,col]
                if chg >= 0:
                    all_data[di,col] = 1
                else:
                    all_data[di,col] = -1
    
        return all_data
        
    return None

def create_data_sets(df, scaler_type='min_max_scaler'):
    #Copy dataframe to a new dataframe that will be manipulated
    df_stock = df.copy()
    #drop useless columns, keep all indicators plus Adj CLose and Adj Volume
    df_stock.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Dividend', 'Split', 'Adj_Open', 'Adj_High', 'Adj_Low'], 1, inplace=True)
    #Need to get rid of the first 200 rows so all indicators will have their true values intead of zeros
    df_stock = df_stock.iloc[199:, :]
    
    #print(df_stock.head())
    #print(df_stock.tail())
    
    #Test set will be 20% of data
    test_set_size = math.floor(df_stock.shape[0]*.2)
    #Train set size will be the total set minus test set
    train_set_size = df_stock.shape[0] - test_set_size
    #Need to scale the data
    all_data = scaler(df_stock, scaler_type=scaler_type)
    
    if all_data is not None:
        #Seperate Data between training and testing
        train_data = all_data[:train_set_size, :]
        test_data = all_data[train_set_size:, :]
        
        #Return train and test data
        return train_data, test_data
    else:
        return None


def add_indicators(df):
    """
    Adds all indicators to the Stock value dataframe
    """
    #Flip index and reset it so the top of the DataFrame
    #so most recent value is at the bottom of the DataFrame
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)

    #Add all indicators to the DataFrame
    df = add_moving_averages_MACD(df)
    df = add_boulinger_bands(df)
    df = add_RSI(df)

    #Return DataFrame
    return df 