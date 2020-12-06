#!/usr/bin/env python3
import sys
import os
import logging

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from BDT import (
    read_data,
    read_features
)

def main():
    data = read_data("./input/")
    feature_names = read_features(data)
    X = data[feature_names]
    
    booster = xgb.Booster()
    booster.load_model("./output/booster.bin")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)    

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    main()
