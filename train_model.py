#!/usr/bin/env python3
import sys
import os
import logging
import datetime

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from options import option_list
from BDT import (
    prepare_train_data,
    prepare_test_data,
    read_data,
    read_features,
    prepare_train_data_matrix,
    prepare_test_data_matrix,
    plot_performance_plots,
    CM_analysis,
    plot_correlation_matrix,
)

NUM_TREES = 300
SPLIT_FRACTION = 0.5


def main():
    logging.basicConfig(
        filename="output.log",
        level=logging.INFO,
        format="%(levelname)s:%(name)s:\t%(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info(str(datetime.datetime.now()))

    input_path = os.path.dirname(os.path.abspath(__file__)) + "/input/"

    data = read_data(input_path)
    data_train = prepare_train_data(data, SPLIT_FRACTION)
    data_test = prepare_test_data(data, SPLIT_FRACTION)

    logger.info(f"Size of data: {data.shape}")
    logger.info(f"Number of events: {data.shape[0]}")
    logger.info(f"Number of columns: {data.shape[1]}")
    logger.info("List of possible features in dataset:")

    for variable in data.columns:
        logger.info("\t" + variable)

    logger.info(f"Number of signal events: {len(data[data.Label == 's'])}")
    logger.info(f"Number of background events: {len(data[data.Label == 'b'])}")
    fraction_signal = len(data[data.Label == 's']) / ((float)(len(data[data.Label == 's'])) + len(data[data.Label == 'b']))
    logger.info(f"Fraction signal: {fraction_signal}")
    logger.info(f"Number of training samples: {len(data_train)}")
    logger.info(f"Number of testing samples: {len(data_test)}")
    logger.info(f"Number of signal events in training set: {len(data_train[data_train.Label == 's'])}")
    logger.info(f"Number of background events in training set: {len(data_train[data_train.Label == 'b'])}")
    training_fraction_signal = len(data_train[data_train.Label == 's']) / ((float)(len(data_train[data_train.Label == 's'])) + len(data_train[data_train.Label == 'b']))
    logger.info(f"Training fraction signal: {training_fraction_signal}")

    feature_names = read_features(data)
    train_data_matrix = prepare_train_data_matrix(data_train, feature_names)
    test_data_matrix = prepare_test_data_matrix(data_test, feature_names)

    booster = xgb.train(option_list, train_data_matrix, NUM_TREES)
    booster.save_model("./output/booster.bin")

    logger.info(f"{booster.eval(test_data_matrix)}")

    plot_performance_plots(test_data_matrix, booster)
    CM_analysis(test_data_matrix, booster)
    xgb.plot_importance(booster, grid=False)
    plot_correlation_matrix(data_train)
    plt.show()


if __name__ == "__main__":
    main()
