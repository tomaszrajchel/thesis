import os
import sys
import logging
import statistics as stc
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def read_data(input_path):
    """read input data file"""
    data = pd.read_csv(input_path + "training.csv")
    return data


def prepare_test_data(data, split_fraction):
    sample_size = data.shape
    data["Label"] = data.Label.astype("category")
    data_test = data[int(sample_size[0] * split_fraction) :]
    return data_test


def prepare_train_data(data, split_fraction):
    sample_size = data.shape
    data["Label"] = data.Label.astype("category")
    data_train = data[: int(sample_size[0] * split_fraction)]
    return data_train


def read_features(data):
    # skipping indices: 0-index, 3-label
    # feature_names = ["DeltaR_lab23"] # baseline
    feature_names = [*data.columns[1:3], *data.columns[4:]]
    return feature_names

def prepare_data_matrix(data, feature_names):
    data_matrix = xgb.DMatrix(
        data=data[feature_names], label=data.Label.cat.codes, missing=-999.0, feature_names=feature_names
    )
    return data_matrix


def prepare_train_data_matrix(data_train, feature_names):
    train_data_matrix = xgb.DMatrix(
        data=data_train[feature_names], label=data_train.Label.cat.codes, missing=-999.0, feature_names=feature_names
    )
    return train_data_matrix


def prepare_test_data_matrix(data_test, feature_names):
    test_data_matrix = xgb.DMatrix(
        data=data_test[feature_names], label=data_test.Label.cat.codes, missing=-999.0, feature_names=feature_names
    )
    return test_data_matrix


def plot_performance_plots(test_data_matrix, booster, save=False):
    plt.rcParams.update({"figure.max_open_warning": 0})
    predictions = booster.predict(test_data_matrix)

    # All events
    plt.figure()
    #plt.xscale("log")
    #plt.hist(predictions, bins=np.logspace(np.log10(1e-6),np.log10(1e2), 100), histtype="step", color="darkgreen")
    plt.hist(predictions, bins=100, histtype="step", color="red")
    plt.title("BDT Predictions")
    plt.xlabel("Model Output", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.legend(frameon=False)
    if save:
        plt.savefig("img/bdt_predictions_testdata.png")

    # Signal vs background events
    plt.figure()
    plt.hist(
        predictions[test_data_matrix.get_label().astype(bool)],
        bins=100,
        histtype="step",
        color="green",
        label="signal",
    )
    plt.hist(
        predictions[~(test_data_matrix.get_label().astype(bool))],
        bins=100,
        histtype="step",
        color="lightgray",
        label="background",
    )
    plt.title("BDT Predictions")
    plt.xlabel("Model Output", fontsize=12)
    plt.ylabel("Events", fontsize=12)
    plt.legend(frameon=False)
    if save:
        plt.savefig("img/bdt_labeled_predictions_testdata.png")


def CM_analysis(test_data_matrix, booster, save=False):
    predictions = booster.predict(test_data_matrix)

    # CM
    bins = 101
    cuts = np.linspace(0, 1, bins)

    true_positive = np.zeros(len(cuts))
    true_negative = np.zeros(len(cuts))
    false_positive = np.zeros(len(cuts))
    false_negative = np.zeros(len(cuts))

    precision = np.zeros(len(cuts))
    sensitivity = np.zeros(len(cuts))
    f1score = np.zeros(len(cuts))
    specificity = np.zeros(len(cuts))

    true_positive_rate = np.zeros(len(cuts))
    false_positive_rate = np.zeros(len(cuts))

    # choosing working point
    # - shortest distance to (0,1) point on ROC
    # - ROC 1 (specificity : sensitivity)
    # - ROC 2 (precision : sensitivity)

    best_ROC_ss = 0.0
    best_ROC_ps = 0.0
    best_ROC_distance = 1.0

    for i, cut in enumerate(cuts):
        true_positive[i] = len(np.where(predictions[test_data_matrix.get_label().astype(bool)] > cut)[0])
        true_negative[i] = len(np.where(predictions[~(test_data_matrix.get_label().astype(bool))] < cut)[0])
        false_positive[i] = len(np.where(predictions[~(test_data_matrix.get_label().astype(bool))] > cut)[0])
        false_negative[i] = len(np.where(predictions[test_data_matrix.get_label().astype(bool)] < cut)[0])

        sensitivity[i] = true_positive[i] / (true_positive[i] + false_negative[i])
        specificity[i] = true_negative[i] / (true_negative[i] + false_positive[i])
        true_positive_rate[i] = true_positive[i] / (true_positive[i] + false_negative[i])
        false_positive_rate[i] = false_positive[i] / (false_positive[i] + true_negative[i])

        if (true_positive[i] + false_positive[i]) == 0:
            precision[i] = 1
            f1score[i] = 0
        else:
            precision[i] = true_positive[i] / (true_positive[i] + false_positive[i])
            f1score[i] = stc.harmonic_mean([precision[i], sensitivity[i]])

        ROC_area_ss = specificity[i] * sensitivity[i]
        ROC_area_ps = precision[i] * sensitivity[i]
        ROC_distance = ((0.0-false_positive_rate[i])**2 + (1.0 - true_positive_rate[i])**2 )**0.5

        if ROC_distance < best_ROC_distance:
            best_ROC_distance = ROC_distance
            best_cut_distance = cut
            best_point = (false_positive_rate[i], true_positive_rate[i])

        if ROC_area_ss > best_ROC_ss:
            best_ROC_ss = ROC_area_ss
            best_cut_ss = cut

        if ROC_area_ps > best_ROC_ps:
            best_ROC_ps = ROC_area_ps
            best_cut_ps = cut

    eval_string = booster.eval(test_data_matrix)
    roc_area = float(eval_string[-8:-1])
    cut_list = [0.5, best_cut_distance]

    logger.info(f"Confusion matrix:")
    logger.info(f"CM:  Area of the surface under the ROC curve: {roc_area}")
    logger.info(f"CM:  Area of the surface under the best cut (ROC 1 - (specificity : sensitivity)): {best_ROC_ss}")
    logger.info(f"CM:  Best cut (ROC 1): {best_cut_ss}")
    logger.info(f"CM:  Area of the surface under the best cut (ROC 2 - (precision : sensitivity)): {best_ROC_ps}")
    logger.info(f"CM:  Best cut (ROC 2): {best_cut_ps}")
    logger.info(f"CM:  Distance to the (0,1) point: {best_ROC_distance}")
    logger.info(f"CM:  Best point: {best_point}")
    logger.info(f"CM:  Best cut (dist): {best_cut_distance}")
    logger.info(f"CM:  Best average cut: {stc.mean(cut_list)}")

    plt.figure()
    plt.step(false_positive_rate, true_positive_rate, label="AUC = %0.4f" % roc_area)
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axis('scaled')
    plt.xlim([0,1])
    plt.ylim([0,1])
    if save:
        plt.savefig("img/ROC.png")

    # plt.figure()
    # plt.step(specificity, sensitivity)
    # plt.xlabel("Specificity", fontsize=12)
    # plt.ylabel("Sensitivity", fontsize=12)
    # plt.legend(frameon=False)
    # plt.axis('scaled')
    # plt.xlim([0,1])
    # plt.ylim([0,1])

    # plt.figure()
    # plt.step(precision, sensitivity)
    # plt.xlabel("Precision", fontsize=12)
    # plt.ylabel("Sensitivity", fontsize=12)
    # plt.legend(frameon=False)
    # plt.axis('scaled')
    # plt.xlim([0,1])
    # plt.ylim([0,1])

    # plt.figure()
    # plt.step(precision, specificity)
    # plt.xlabel("Precision", fontsize=12)
    # plt.ylabel("Specificity", fontsize=12)
    # plt.legend(frameon=False)
    # plt.axis('scaled')
    # plt.xlim([0,1])
    # plt.ylim([0,1])

    # Confusion Matrix
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5, left=0.2)
    fig.suptitle('Confusion Matrices', fontsize=16)
    for idx, cut in enumerate(cut_list):
        cut_index = int(cut * (bins-1))
        confusion_matrix = [
            [true_positive[cut_index], false_positive[cut_index]],
            [false_negative[cut_index], true_negative[cut_index]]
        ]
        positive = len(np.where(test_data_matrix.get_label().astype(bool))[0]) # alternative: true_positive[cut_index] + false_negative[cut_index]
        negative = len(np.where(~test_data_matrix.get_label().astype(bool))[0]) # alternative: true_negative[cut_index] + false_positive[cut_index]
        divisors = [[positive, negative], [positive, negative]]
        confusion_matrix_normalized = [[x/y for x,y in zip(m1, m2)] for m1, m2 in zip(confusion_matrix, divisors)]
        accuracy = (true_positive[cut_index] + true_negative[cut_index]) / predictions.size
        logger.info(f"Correctly classified (cut={cut}): {accuracy}")

        # Annotations
        ax[idx][0].annotate(f"cut={cut}", xy=(0, 0.5), xytext=(-ax[idx][0].yaxis.labelpad - 5, 0),
            xycoords=ax[idx][0].yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center'
        )

        # Absolute
        sns.heatmap(confusion_matrix, annot=True, cmap="Greys", fmt="g", xticklabels=["s", "b"], yticklabels=["s", "b"], ax=ax[idx][0])
        ax[idx][0].set(xlabel="Actual class", ylabel="Predicted class")

        # Normalized
        sns.heatmap(confusion_matrix_normalized, annot=True, cmap="Blues", fmt=".2%", xticklabels=["s", "b"], yticklabels=["s", "b"], ax=ax[idx][1])
        ax[idx][1].set(xlabel="Actual class", ylabel="Predicted class")

    if save:
        plt.savefig("img/confusion_matrix.png")


def plot_correlation_matrix(data, save=False):
    fig = plt.figure()
    plt.title(f"Pearson Correlation Coefficients")
    corrMatrix = data.corr()
    sns.heatmap(corrMatrix, annot=True, cmap="vlag", fmt=".2f", vmin=-1.0, vmax=1.0)
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.subplots_adjust(right=1.0, bottom=0.17, top=0.95)
    fig.set_size_inches(14,12)
    if save:
        plt.savefig("img/data_correlation.png")

def plot_training(evals, evals_result, save=False):
    fig = plt.figure()
    values = evals_result[evals[0][1]]["auc"]
    # values = list(map(lambda x: 1-x, evals_result[evals[0][1]]["auc"]))
    plt.plot(range(0,len(values)), values)
    plt.xlim([0,len(values)])
    plt.ylim([0.98,1])
    # plt.yscale("log")
    plt.title(f"Training Performance")
    plt.xlabel("Epoch")
    plt.ylabel("AUC - Test Data")
    plt.grid()
    if save:
        plt.savefig("img/training_plot.png")

def cross_validate(k, params, num_trees, metrics=None):
    # Setup
    input_path = os.path.dirname(os.path.abspath(__file__)) + "/input/"
    data = read_data(input_path)
    data["Label"] = data.Label.astype("category")
    size = data.shape[0]
    feature_names = read_features(data)
    eval_history = []

    # Shuffle
    data = data.sample(frac=1)
    
    for c in range(k):
        # Split
        i1 = int((c/k)*size)
        i2 = int((c+1)/k*size)
        test_data = data[i1:i2]
        train_data = pd.concat([data[0:i1], data[i2:-1]])
        # print(f"i1={i1}\ti2={i2} test_size={test_data.shape}, train_size={train_data.shape}")
        test_data_matrix = prepare_test_data_matrix(test_data, feature_names)
        train_data_matrix = prepare_train_data_matrix(train_data, feature_names)

        # Train
        booster = xgb.train(
            params,
            dtrain=train_data_matrix,
            num_boost_round=num_trees,
        )
        eval_history.append(booster.eval(test_data_matrix))

    # Calculate metrics
    if metrics is None:
        metrics = {metric:[] for metric in params["eval_metric"]}
    for line in eval_history:
        eval_strings = line.split("\t")[1:]
        for eval_string in eval_strings:
            metrics[eval_string[5:-9]].append(float(eval_string[-8:-1]))
    
    return metrics
    

def describe_metrics(metrics):
    mean_metrics = {key:np.mean(val) for key,val in metrics.items()}
    std_metrics = {key:np.std(val) for key,val in metrics.items()}
    rstd_metrics = {key:100*np.std(val)/np.mean(val) for key,val in metrics.items()}

    logger.info(metrics)
    logger.info(f"mean:\t{mean_metrics}")
    logger.info(f"std:\t{std_metrics}")
    logger.info(f"rstd:\t{rstd_metrics}")
