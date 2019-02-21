# -*- coding: utf-8 -*-
"""
This is the Recurrent Neural Network model class, it will create the tensors,
it will create the train and the predict class methods, using the TensorFlow library

@name: LSTM_Cell.py

@author: Lucas Magalhaes
"""

import tensorflow as tf
import numpy as np

class LSTMStockPredictor():

    def __init__(self, name, data_dim, lstm_size=[128], batch_size=10, num_unrollings=10, learning_rate=0.01, 
                learning_rate_decay=0.99, init_epoch_decay=5):
        
        #name of the stock
        self.name = name
        #numer of hidden units in the LSTM cell
        self.lstm_size = lstm_size
        #number of layers in the LSTM network
        self.num_layers = len(lstm_size)
        #batch size
        self.batch_size = batch_size
        #number of sequential data that will be used to train model
        self.num_unrollings = num_unrollings
        #dimension of data
        self.data_dim = data_dim
        #learning rate
        self.learning_rate = learning_rate
        #Learning rate decay
        self.learning_rate_decay = learning_rate_decay
        #First epoch to start the decay of the learning rate
        self.init_epoch_decay = init_epoch_decay

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

            
    def create_batch_generator(self, x, num_unroll, batch_size):
        #Create empty numpy arrays that will generate the data in the correct format
        unrolled_x = np.zeros(shape=(batch_size, x.shape[1], num_unroll))
        unrolled_y = np.zeros(shape=(batch_size, 1, num_unroll))
        batches_x = np.zeros(shape=(batch_size, x.shape[1], num_unroll))
        #Number of unrolled batches that can be created without stepping over the end
        n_unrolled_batches = len(x)//(batch_size)

        #Cut off ends that do not have enough data samples to make a unrolled batch
        #print(n_unrolled_batches%num_unroll)
        x = x[:((n_unrolled_batches-(n_unrolled_batches%num_unroll))*batch_size)]
    
        #Generate data
        batches_x = np.zeros(shape=(batch_size, x.shape[1], n_unrolled_batches))
        min_x = 0
        max_x = min_x + batch_size
        for i in range((n_unrolled_batches-(n_unrolled_batches%num_unroll))):
            batches_x[:,:,i] = x[min_x:max_x,:]
            min_x = max_x
            max_x += batch_size
        print(batches_x.shape)
        for i in range(batches_x.shape[2]-num_unroll-1):
            unrolled_x = batches_x[:,:,i:i+num_unroll]
            unrolled_y = batches_x[:,0,i+num_unroll+1]
            yield unrolled_x, unrolled_y
        
    def build(self):
        #Define placeholders
        #Placeholder for dropout rate and learning rate
        tf_lr = tf.placeholder(tf.float32, shape=None, name='tf_learning_rate')
        tf_keep_prob = tf.placeholder(tf.float32, shape=None, name='tf_keep_prob')
        print("Learning Rate >>> ",tf_lr)
        print("Keep Prob >>> ",tf_keep_prob)

        #Place holder for input data and target data
        #input of size [batch_size, data_dim, num_unrollings]
        tf_x = tf.placeholder(tf.float32, shape=(self.batch_size, self.data_dim, self.num_unrollings), name='tf_x')
        #target of size [batch_size, 1, num_unrolling]
        tf_y = tf.placeholder(tf.float32, shape=(self.batch_size, 1), name='tf_y')
        print("input tensor >>> ", tf_x)
        print("Target tensor >>> ", tf_y)

        #Create LSTM layers, stacked using MultiRNNCell, and DropoutWrapper for overfitting
        lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.LSTMCell(num_units=n, initializer=tf.contrib.layers.xavier_initializer()), 
                    output_keep_prob=tf_keep_prob
                    ) for n in self.lstm_size])
        print("LSTM CELLS >>> ", lstm_cells)

        #Create inital state
        self.initial_state = lstm_cells.zero_state(self.batch_size, dtype=tf.float32)
        print("Initial State >>> ", self.initial_state)

        #format input data to match the required format from dynamic_rnn
        #[num_unrollings, batch_size, data_dim]
        formatted_input = tf.transpose(tf_x, [0, 2, 1])
        print("Formatted input >>> ", formatted_input)

        #Connect all Nodes to create the RNN
        lstm_output, self.final_state = tf.nn.dynamic_rnn(lstm_cells, formatted_input,
                                                        initial_state=self.initial_state)
        print("LSTM NETWORK OUTPUT >>> ", lstm_output)
        #format output data to fit to prediction layer
        #[num_unrolling, batch_size, last LSTMCell output size] -> [(num_unrolling * batch_size), last LSTMCell output size]
        #formatted_lstm_output = tf.reshape(lstm_output, [self.num_unrollings*self.batch_size, self.lstm_size[-1]])
        #or pick only last output batch
        #formatted_lstm_output = tf.gather(lstm_output, int(lstm_output.get_shape()[0]) - 1, name='last_lstm_output')
        formatted_lstm_output = lstm_output[:, -1]
        print("Formatted LSTM output >>> ", formatted_lstm_output)

        #define variables for prediction layer
        tf_w = tf.Variable(tf.truncated_normal([self.lstm_size[-1], 1]), name="w")
        tf_b = tf.Variable(tf.constant(0.1, shape=[1], name="b"))

        #Calculate the RNN prediction
        predictions = tf.nn.xw_plus_b(formatted_lstm_output, tf_w, tf_b, name='predictions')

        #Define Cost function
        cost = tf.reduce_mean(tf.square(predictions - tf_y), name='train_cost_mse')

        #Define optimizer to use
        optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr)

        #train funtion
        train_op = optimizer.minimize(cost, name='train_op')

    def train(self, train, num_epoch=30):
        #Start TensorFlow session
        with tf.Session(graph=self.g) as sess:
            #Initialize variables
            sess.run(self.init_op)
            #Create learning rate array, which will start at learning_rate and decay exponentially by learning_rate_decay
            lr_array = [self.learning_rate for _ in range(self.init_epoch_decay)] + [(self.learning_rate * (self.learning_rate_decay ** (i+1))) for i in range(num_epoch-self.init_epoch_decay)]
            #Loop through num_epoch
            iteration = 1
            losses = []
            for i in range(num_epoch):
                state = sess.run(self.initial_state)
                
                for batch_x, batch_y in self.create_batch_generator(train, self.num_unrollings, self.batch_size):
                    
                    #Set input values for model placeholders
                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y.reshape(self.batch_size,1),
                            'tf_keep_prob:0': 0.8,
                            'tf_learning_rate:0': lr_array[i],
                            self.initial_state: state}

                    #Run the training for 1 iteration
                    loss, _, state = sess.run(
                            ['train_cost_mse:0', 'train_op', self.final_state],
                            feed_dict=feed)
                    losses.append(loss)
                    #Print info about epoch loss
                    if iteration % 20 == 0:
                        print("Epoch: %d/%d Iteration: %d "
                              "| Train loss: %.5f" % (
                               i + 1, num_epoch,
                               iteration, loss))

                    #Add 1 to iteration counter
                    iteration +=1
                #Save the model after 10 epochs
                if (i+1)%10 == 0:
                    self.saver.save(sess,
                        "model/%s_price_prediction.ckpt" % self.name)
                    self.latest_iter = i

            self.saver.save(sess, "model/%s_price_prediction.ckpt" % self.name)
            self.latest_iter = i
            return losses


    def predict(self, x_data):
        #Create the list that will hold the predictions and errors
        preds = []
        error = []
        #Create TensorFlow session
        with tf.Session(graph=self.g) as sess:
            #Restore model
            #self.saver.restore(sess, tf.train.latest_checkpoint('model/', '%s_price_prediction.ckpt' % self.name))
            self.saver.restore(sess, 'model/%s_price_prediction.ckpt' % self.name)
            #Runs intial state
            test_state = sess.run(self.initial_state)
            #for each batch intialize the state, and then predict values
            for batch_x, batch_y in self.create_batch_generator(x_data, self.num_unrollings, self.batch_size):
                feed = {'tf_x:0': batch_x,
                        'tf_y:0': batch_y.reshape(self.batch_size,1),
                        'tf_keep_prob:0': 0.8,
                        self.initial_state: test_state}

                pred, loss, test_state = sess.run(
                    ['predictions:0','train_cost_mse:0', self.final_state],
                    feed_dict=feed
                )
                preds.append(pred.tolist())
                error.append(loss)
        #Returns the prediction and the loss of each prediction
        return preds, error
    
