# Some code stolen from Danijar
'''
Please fill up the functions calc_cost, calc_optim and _weight_and_bias functions.
'''
import tensorflow as tf
from reader import textDataLoader
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper, LSTMCell
import numpy as np
import pdb

class SequenceLabelling(object):
    def __init__(self, data, target, dropout, num_hidden=128, num_layers=2):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.predictions = self.calc_pred()
        self.error = self.calc_error()
        self.cost = self.calc_cost()
        self.optimize = self.calc_optim()

    def calc_pred(self):
        # Recurrent network.
        cell = LSTMCell(self._num_hidden)
        cell_drop = DropoutWrapper(
            cell, output_keep_prob=self.dropout)
        self.network = MultiRNNCell([cell_drop] * self._num_layers)
        max_length = int(self.target.get_shape()[1])
        output, _ = tf.nn.dynamic_rnn(self.network, self.data, dtype=tf.float32)
        ## What is the functionality of dynamic_rnn ##

        # Softmax layer.
        num_classes = int(self.target.get_shape()[2])
        self.weight, self.bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        predictions = tf.nn.softmax(tf.matmul(output, self.weight) + self.bias)
        predictions = tf.reshape(predictions, [-1, max_length, num_classes])
        return predictions

    def calc_cost(self):
        '''
        Define a Cross Entropy loss between targets and predictions. 
        '''
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.target))

    def calc_optim(self):
        '''
        Write an optimizer for the cost.
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(self.cost)

    def calc_error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.predictions, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        '''
        Define a function that return a weight and bias variable.
        '''
        weight = tf.random_normal([in_size, out_size])
        bias = tf.random_normal([out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def test_rnn(self, inputs, pred_length):
        self.test_inputs = inputs
        length = int(self.test_inputs.get_shape()[1])
        num_classes = int(self.test_inputs.get_shape()[2])
        outputs_onehot = []
        outputs = []
        state = self.network.zero_state(self.test_inputs.get_shape()[0], tf.float32)
        with tf.variable_scope("rnn"):
            for time_step in range(length):
                tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.network(self.test_inputs[:, time_step, :], state)
            for time_step in range(pred_length):
                tf.get_variable_scope().reuse_variables()
                prediction = tf.nn.softmax(tf.matmul(cell_output, self.weight) + self.bias)
                prediction = tf.argmax(prediction, 1)
                outputs.append(tf.squeeze(prediction))
                prediction = tf.one_hot(prediction, num_classes)
                outputs_onehot.append(tf.squeeze(prediction))
                new_input = prediction
                (cell_output, state) = self.network(new_input, state)
        ## What is the difference between the testing graph and training graph that uses dynamic_rnn ##
        return outputs, outputs_onehot


def ord_to_text(ord_list):
#     # L = ''
#     # for o in ord_list:
#         return L
    return ''.join(chr(i) for i in ord_list)

if __name__ == '__main__':
    # tD = textDataLoader('dataset.txt')
    tD = textDataLoader('dataset_small.txt')
    batch_size = 50
    length = 50
    feat_size = tD.nFeats
    num_classes = feat_size

    data = tf.placeholder(tf.float32, [batch_size, length, feat_size])
    target = tf.placeholder(tf.float32, [batch_size, length, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceLabelling(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ## Setting up testing ##
    warmString = 'Happy '
    ## What is the need of the warm string ##
    pred_len = 100
    warm_length = len(warmString)
    inputs = tf.placeholder(tf.float32, [1, warm_length, feat_size])
    inputs_val = np.array([tD.convertHot(warmString)])
    pred_outs = model.test_rnn(inputs, pred_len)
    # Testing!
    P = sess.run(pred_outs, {inputs: inputs_val, dropout: 1.0})
    print('Predicted text is \n{}\n'.format(ord_to_text(P[0])))

    for epochNo in range(1000):
        for itrNO in range(10):
            input_b, output_b = tD.getBatch(batch_size, length)
            # pdb.set_trace()
            _, error = sess.run([model.optimize, model.error], {data: input_b, target: output_b, dropout: 0.5})
            print('{}/{} Error {}'.format(epochNo, itrNO, error))
        # Testing!
        P = sess.run(pred_outs, {inputs: inputs_val, dropout: 1.0})
        print('Predicted text is \n{}\n'.format(ord_to_text(P[0])))
