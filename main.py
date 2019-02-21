#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:55:45 2019

@name: main.py

@author: Lucas Magalhaes
"""
#Library imports
import pandas as pd

#Local imports
import data_formatting
import lstm_cell
import model_class

def main():

    #File path where the CSV files are located
    filepath = "C:/Users/LucasMagalhaes/Documents/Projects/ML/Stock_predixtion/Tests/"

    #Load files to a pandas dataframe
    aapl = pd.read_csv(filepath+'EOD-AAPL.csv')
    msft = pd.read_csv(filepath+'EOD-MSFT.csv')
    #Formats the data
    aapl = data_formatting.add_indicators(aapl)
    msft = data_formatting.add_indicators(msft)
    #Breaks dataset into train and test, 80/20 train/test
    aapl_train, aapl_test = data_formatting.create_data_sets(aapl)
    msft_train, msft_test = data_formatting.create_data_sets(msft)

    #Hyperparamters for the LSTM RNN
    hy_num_unrolling = 10
    hy_batch_size = 5
    hy_lstm_size = [128, 256, 128]
    hy_learning_rate = 0.0001
    hy_learning_rate_decay = 0.75
    hy_init_epoch_decay = 3
    hy_num_epoch = 5

    #Build the RNNs for each of the dataframes
    rnn_aapl = lstm_cell.LSTMStockPredictor("Apple", aapl_train.shape[1], lstm_size=hy_lstm_size, batch_size=hy_batch_size, num_unrollings=hy_num_unrolling, learning_rate=hy_learning_rate, learning_rate_decay=hy_learning_rate_decay, init_epoch_decay=hy_init_epoch_decay)
    rnn_msft = lstm_cell.LSTMStockPredictor("Microsoft", msft_train.shape[1], lstm_size=hy_lstm_size, batch_size=hy_batch_size, num_unrollings=hy_num_unrolling, learning_rate=hy_learning_rate, learning_rate_decay=hy_learning_rate_decay, init_epoch_decay=hy_init_epoch_decay)

    #Creates a stock predictive model class that will train and test the RNN
    modelAapl = model_class.StockPredictiveModel(rnn_aapl, aapl_train, aapl_test, hy_num_epoch)
    modelMsft = model_class.StockPredictiveModel(rnn_msft, msft_train, msft_test, hy_num_epoch)

    #plot the train and testing data
    modelAapl.plotter()
    modelMsft.plotter()

    return None

main()