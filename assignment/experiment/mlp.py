#!python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>
# Modified by Abhishek Ranjan Singh <arsingh3@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',help='Test the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784,50,10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)


def train():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784,50,20,16,10])
    # train the network using SGD
    e = 100
    eval_cost,eval_accuracy,train_cost, train_accuracy  = model.SGD(
        training_data=train_data,
        epochs=e,
        mini_batch_size=50,
        eta=1e-3,
        lmbda = 0.001,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    #Saving the Neuaral Network

    model.save('NeuralNet_new_fc.csv')

    #Saving the errors vs Epcoh data
    SaveCsv = {'Evaluation Cost': eval_cost,
               'Evaluation Accuracy' : eval_accuracy,
               'Train Cost' : train_cost,
               'Train Accuracy' : train_accuracy}
    df = pd.DataFrame(SaveCsv)
    df.to_csv('Error_Details_layers_new_fc' + '.csv')

    #Visualising the error vs epoch
    plt.figure()
    x = [i for i in range(0,e)]
    ax = plt.subplot(1, 2, 1)
    ax.plot(x, train_cost,'r-',label = 'Training Cost')
    ax.plot(x,eval_cost,'b-',label = 'Evaluation Cost')
    plt.title('Error comparision plot')
    #plt.legend()

    ay = plt.subplot(1, 2, 2)
    ay.plot(x, train_accuracy,'r-',label = 'Training accuracy')
    ay.plot(x,eval_accuracy,'b-',label = 'Evaluation accuracy')
    plt.title('Classification comparision plot')
    #plt.legend()
    plt.show()




def test():
    # load train_data, valid_data, test_data
    test_data = load_data()[2]
    #loading the neural Network
    model = network2.load('NeuralNet_new_fc.csv')
    test_cost = model.total_cost(test_data, lmbda=0.0, convert=True)
    test_accuracy = model.accuracy(test_data)
    '''result = []
    for i in range(len(test_data[0])):
        result.append(network2.vectorized_result(np.argmax(model.feedforward(test_data[0][i]))))

    df = pd.DataFrame(list(map(np.ravel, result)))
    df.to_csv('Arsingh3_hw02b_Prediction_Submission.csv')'''

    #result = model.feedforward()
    print('Test Error: {}'.format(test_cost))
    print('Test accuracy: {}'.format(test_accuracy))


if __name__ == '__main__':
    FLAGS = get_args()
    print(type(FLAGS))
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        train()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        test()
