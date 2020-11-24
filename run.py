#!/usr/bin/env python3
import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from BDT import (
    prepare_train_data,
    prepare_test_data,
    read_data,
    read_features,
    prepare_train_data_matrix,
    prepare_test_data_matrix,
    read_option,
    read_num_of_trees,
    train_classifier,
    plot_performance_plots,
    CM_analysis,
    plot_correlation_matrix,
)


def run():
    logging.basicConfig(filename="output.log", level=logging.INFO)

    input_path = os.path.abspath(__file__)[:-7] + "/input/"
    output_path = os.path.abspath(__file__)[:-7]

    message = "   Output path: {}".format(output_path)
    logging.info(message)

    data = read_data(input_path)
    data_train = prepare_train_data(data)
    data_test = prepare_test_data(data)

    messages = []
    messages.append("   Size of data: {}".format(data.shape))
    messages.append("   Number of events: {}".format(data.shape[0]))
    messages.append("   Number of columns: {}".format(data.shape[1]))
    messages.append("   List of possible features in dataset:")

    for variables in data.columns:
        messages.append("\t" + variables)

    messages.append("   Number of signal events: {}".format(len(data[data.Label == "s"])))
    messages.append("   Number of background events: {}".format(len(data[data.Label == "b"])))
    messages.append(
        "   Fraction signal: {}".format(
            len(data[data.Label == "s"]) / (float)(len(data[data.Label == "s"]) + len(data[data.Label == "b"]))
        )
    )
    messages.append("   Number of training samples: {}".format(len(data_train)))
    messages.append("   Number of testing samples: {}".format(len(data_test)))
    messages.append("   Number of signal events in training set: {}".format(len(data_train[data_train.Label == "s"])))
    messages.append(
        "   Number of background events in training set: {}".format(len(data_train[data_train.Label == "b"]))
    )
    messages.append(
        "   Fraction signal: {}".format(
            len(data_train[data_train.Label == "s"])
            / (float)(len(data_train[data_train.Label == "s"]) + len(data_train[data_train.Label == "b"]))
        )
    )

    feature_names = read_features(data)

    train_data_matrix = prepare_train_data_matrix(data_train, feature_names)
    test_data_matrix = prepare_test_data_matrix(data_test, feature_names)

    option_list = read_option()
    num_trees = read_num_of_trees()
    booster = train_classifier(option_list, train_data_matrix, num_trees)

    messages.append("   {}".format(booster.eval(test_data_matrix)))

    plot_performance_plots(test_data_matrix, booster)

    messages.append("   Confusion matrix: ")

    CM_messages = CM_analysis(test_data_matrix, booster)

    for message in messages:
        logging.info(message)

    for message in CM_messages:
        logging.info(message)

    plot_correlation_matrix(data_train)

    plt.show()


if __name__ == "__main__":
    run()
