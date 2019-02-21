"""
This file contains the class definition for the training and prediction of the RNN model as a whole and 
it will hold all the information that is rturned by running the TensorFlow models

It will also be able to plot the data for the training loss and the prediction vs real values of the 
test data and also its losses

@name: Model_class.py

@author: Lucas Magalhaes
"""

import matplotlib.pyplot as plt
import numpy as np

class StockPredictiveModel():

    def __init__(self, graph, train, test, num_epoch):
        #constructor for this class

        #This is the class instance of the TensorFlow model
        self.graph = graph
        #Train and test data
        self.train = train
        self.test = test
        #Number of Epochs to train the model for
        self.num_epoch = num_epoch
        #This will run the training of the model and hold the training loss
        self.train_loss = graph.train(self.train, self.num_epoch)
        #This variables will hold the predictions made by the model
        #and the loss of the predictions
        self.preds, self.test_loss = graph.predict(self.test)

    def plotter(self):
        #Plot all the data from the training and testing of the model

        """
        Plotting the training loss of the model
        """
        plt.figure()
        plt.title("Loss Mean Squared Error - " + self.graph.name)
        plt.plot(np.arange(len(self.train_loss)), self.train_loss)
        plt.ylabel("MSE")
        plt.xlabel("Iteration")
        plt.show()

        """
        Plotting the predictions and the loss of the predictions of the model
        """
        #This fixes the output of the predictions that is returned by the model by
        #rotating the data
        predictions_fix = []
        for i in range(len(self.preds)):
            for z in range(len(self.preds[i])):
                predictions_fix.append(self.preds[i][z][0])

        plt.figure()

        plt.subplot(211)
        plt.title(self.graph.name)
        plt.plot(range(self.test.shape[0]), self.test[:,0], label = 'Actual')
        plt.plot(range(self.graph.batch_size,len(predictions_fix)+self.graph.batch_size), predictions_fix, label="Predicted")
        plt.legend()
        plt.ylabel("US Dollar(Normalized)")
        plt.xlabel("Time Index")

        plt.subplot(212)
        plt.plot(range(0, len(predictions_fix), self.graph.batch_size), self.test_loss, label='Prediction Losses')
        plt.xlabel("Time index")
        plt.ylabel("MSE")
        average_loss = sum(self.test_loss)/(len(self.test_loss))
        print("Average MSE testing: " + str(average_loss))

        plt.show()
        