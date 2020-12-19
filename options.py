"""xgboost train parameters"""

option_list = [
    ("eta", 0.1),
    ("gamma", 0.0),
    ("max_depth", 10),
    ("min_child_weight", 1.0),
    ("max_delta_step", 0.0),
    ("subsample", 0.8),
    ("colsample_bytree", 0.8),
    ("objective", "binary:logistic"),
    ("eval_metric", "error"),
    ("eval_metric", "logloss"),
    ("eval_metric", "rmse"),
    ("eval_metric", "auc"),
]
