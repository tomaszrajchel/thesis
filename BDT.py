#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import statistics as stc
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sn

SPLIT_FRACTION = 0.5

def read_data(input_path):

    data = pd.read_csv(input_path + 'training.csv')
    return data

def prepare_test_data(data):
    sample_size = data.shape
    data['Label'] = data.Label.astype('category')
    data_test = data[int(sample_size[0] * SPLIT_FRACTION):]
    return data_test

def prepare_train_data(data):
    sample_size = data.shape
    data['Label'] = data.Label.astype('category')
    data_train =  data[:int(sample_size[0] * SPLIT_FRACTION)] 
    return data_train

def read_features(data):
    # skipping 1st column - index and 3rd - label
    feature_names = [*data.columns[1:2], *data.columns[4:data.shape[1]]]
    return feature_names


def prepare_train_data_matrix(data_train, feature_names):
    train_data_matrix = xgb.DMatrix(data=data_train[feature_names],label=data_train.Label.cat.codes,
                        missing=-999.0,feature_names=feature_names)   
    return train_data_matrix 

def prepare_test_data_matrix(data_test, feature_names):
    test_data_matrix = xgb.DMatrix(data=data_test[feature_names],label=data_test.Label.cat.codes,
                        missing=-999.0,feature_names=feature_names)   
    return test_data_matrix 

def read_option():

    options = []
    with open("option.txt") as f:
        for line in f:
            (key, val) = line.split()
            if val.isnumeric():
                if(key == "max_depth"):
                    options.append((key,int(val)))
                else:
                    options.append((key,float(val)))
            else:
                options.append((key,val))

    options.append(('eval_metric', 'logloss'))
    options.append(('eval_metric', 'rmse'))
    options.append(('eval_metric', 'auc'))

    return options

def read_num_of_trees():
    num_trees = 300 
    return num_trees

def train_classifier(param, train_data_matrix, num_trees):

    booster = xgb.train(param, train_data_matrix, num_trees)
    return booster

def plot_performance_plots(test_data_matrix, booster):

    plt.rcParams.update({'figure.max_open_warning': 0})
    predictions = booster.predict(test_data_matrix)

    plt.figure()
    plt.hist(predictions,bins=np.linspace(0,1,50),histtype='step',color='darkgreen',label='All events')
    plt.xlabel('Prediction from BDT',fontsize=12)
    plt.ylabel('Events',fontsize=12)
    plt.legend(frameon=False)

    plt.figure()
    plt.hist(predictions[test_data_matrix.get_label().astype(bool)],bins=np.linspace(0,1,50),
            histtype='step',color='midnightblue',label='signal')
    plt.hist(predictions[~(test_data_matrix.get_label().astype(bool))],bins=np.linspace(0,1,50),
            histtype='step',color='firebrick',label='background')
    plt.xlabel('Prediction from BDT',fontsize=12)
    plt.ylabel('Events',fontsize=12)
    plt.legend(frameon=False)

def CM_analysis(test_data_matrix, booster):

    massages = []

    predictions = booster.predict(test_data_matrix)

    #CM
    bins = 100
    cuts = np.linspace(0,1,bins)
    
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
    # - ROC 1 (specificity : sensitivity)
    # - ROC 2 (precision : sensitivity)
    # - closest distance between pont and precision = specificity

    best_ROC_ss = 0
    best_ROC_ps = 0
    best_ROC_distance = 1

    for i,cut in enumerate(cuts):

        true_positive[i] = len(np.where(predictions[test_data_matrix.get_label().astype(bool)] > cut)[0])
        true_negative[i] = len(np.where(predictions[~(test_data_matrix.get_label().astype(bool))] < cut)[0])
        false_positive[i] = len(np.where(predictions[~(test_data_matrix.get_label().astype(bool))] > cut)[0])
        false_negative[i] = len(np.where(predictions[test_data_matrix.get_label().astype(bool)] < cut)[0])
        
        sensitivity[i] = true_positive[i]/(true_positive[i] + false_negative[i])
        specificity[i] = true_negative[i]/(true_negative[i] + false_positive[i])
        true_positive_rate[i] = true_positive[i]/(true_positive[i] + false_positive[i])
        false_positive_rate[i] = false_positive[i]/(false_positive[i] + false_negative[i])

        if (true_positive[i] + false_positive[i]) == 0:
            precision[i] = 1
            f1score[i] = 0

        else:
            precision[i] = true_positive[i]/(true_positive[i] + false_positive[i])
            f1score[i] = stc.harmonic_mean([precision[i],sensitivity[i]])

        ROC_area_ss = specificity[i]*sensitivity[i]
        ROC_area_ps = precision[i]*sensitivity[i]
        ROC_area_distance = abs(precision[i] - sensitivity[i])
        
        if ROC_area_ss > best_ROC_ss:
            best_ROC_ss = ROC_area_ss
            best_cut_ss = cut
            
        if ROC_area_ps > best_ROC_ps:
            best_ROC_ps = ROC_area_ps
            best_cut_ps = cut
            
        if ROC_area_distance < best_ROC_distance:
            best_ROC_distance = ROC_area_distance
            best_cut_distance = cut
            
    roc_area = float(booster.eval(test_data_matrix)[-8:-1])

    massages.append('   CM :  Area of the surface under the ROC curve: {}'.format(roc_area))
    massages.append('   CM :  Area of the surface under the best cut (ROC 1 - (specificity : sensitivity)): {}'.format(best_ROC_ss))
    massages.append('   CM :  Best cut (ROC 1): {}'.format(best_cut_ss))
    massages.append('   CM :  Area of the surface under the best cut (ROC 2 - (precision : sensitivity)): {}'.format(best_ROC_ps))
    massages.append('   CM :  Best cut (ROC 2): {}'.format(best_cut_ps))
    massages.append('   CM :  Distance to the 1:1 line (precision = specificity): {}'.format(best_ROC_distance))
    massages.append('   CM :  Best cut (dist): {}'.format(best_cut_distance))

    cut_list = [best_cut_ss, best_cut_ps, best_cut_distance]

    massages.append('   CM :  Best average cut: {}'.format(stc.mean(cut_list)))

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(true_positive_rate,false_positive_rate,'o-',color='blueviolet', label = 'AUC = %0.2f' % roc_area)

    plt.figure()
    plt.plot(specificity,sensitivity,'o-',color='blueviolet')
    plt.xlabel('Specificity',fontsize=12)
    plt.ylabel('Sensitivity',fontsize=12)
    plt.legend(frameon=False)

    plt.figure()
    plt.plot(precision,sensitivity,'o-',color='blueviolet')
    plt.xlabel('Precision',fontsize=12)
    plt.ylabel('Sensitivity',fontsize=12)
    plt.legend(frameon=False)

    plt.figure()
    plt.plot(precision,specificity,'o-',color='blueviolet')
    plt.xlabel('Precision',fontsize=12)
    plt.ylabel('Specificity',fontsize=12)
    plt.legend(frameon=False)

    xgb.plot_importance(booster,grid=False)

    return  massages

def plot_correlation_matrix(data_train):
 
    corrMatrix = data_train.corr()
    sn.heatmap(corrMatrix, annot=True, cmap="YlGnBu", fmt=".2g")
    factor = len(data_train[data_train.Label == 's'])/len(data_train[data_train.Label == 'b'])

