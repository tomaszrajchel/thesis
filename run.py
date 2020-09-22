import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from BDT import prepare_train_data
from BDT import prepare_test_data
from BDT import read_data
from BDT import read_features
from BDT import prepare_train_data_matrix
from BDT import prepare_test_data_matrix
from BDT import read_option
from BDT import read_num_of_trees
from BDT import train_classifier
from BDT import plot_performance_plots
from BDT import CM_analysis
from BDT import plot_correlation_matrix

import logging 

def run():

    logging.basicConfig(filename='output.log', level=logging.INFO)

    input_path = os.path.abspath(__file__)[:-7] + '/input/'
    output_path = os.path.abspath(__file__)[:-7] 

    massage = "   Output path: {}".format(output_path)
    logging.info(massage)

    data = read_data(input_path)
    data_train = prepare_train_data(data)
    data_test = prepare_test_data(data)

    massages = []
    massages.append('   Size of data: {}'.format(data.shape))
    massages.append('   Number of events: {}'.format(data.shape[0]))
    massages.append('   Number of columns: {}'.format(data.shape[1]))
    massages.append('   List of possible features in dataset:')

    for variables in data.columns:
        massages.append("\t" + variables)

    massages.append('   Number of signal events: {}'.format(len(data[data.Label == 's'])))
    massages.append('   Number of background events: {}'.format(len(data[data.Label == 'b'])))
    massages.append('   Fraction signal: {}'.format(len(data[data.Label == 's'])/(float)(len(data[data.Label == 's']) + len(data[data.Label == 'b']))))
    massages.append('   Number of training samples: {}'.format(len(data_train)))
    massages.append('   Number of testing samples: {}'.format(len(data_test)))
    massages.append('   Number of signal events in training set: {}'.format(len(data_train[data_train.Label == 's'])))
    massages.append('   Number of background events in training set: {}'.format(len(data_train[data_train.Label == 'b'])))
    massages.append('   Fraction signal: {}'.format(len(data_train[data_train.Label == 's'])/(float)(len(data_train[data_train.Label == 's']) + len(data_train[data_train.Label == 'b']))))

    feature_names = read_features(data)

    train_data_matrix = prepare_train_data_matrix(data_train, feature_names)
    test_data_matrix = prepare_test_data_matrix(data_test, feature_names)

    option_list = read_option()
    num_trees = read_num_of_trees()
    booster = train_classifier(option_list,train_data_matrix,num_trees)
   
    massages.append('   {}'.format(booster.eval(test_data_matrix)))

    plot_performance_plots(test_data_matrix, booster)

    massages.append('   Confusion matrix: ')

    CM_massages = CM_analysis(test_data_matrix, booster)
    
    for massage in massages:
        logging.info(massage)

    for massage in CM_massages:
        logging.info(massage)

    plot_correlation_matrix(data_train)

    plt.show()

if __name__ == '__main__':
    run()

