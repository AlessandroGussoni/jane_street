import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score

def weighted_rmse_loss(preds, train_data):
    labels = train_data.get_label()
    weights = train_data.get_weight()
    grad = weights * (preds - labels)
    hess = weights
    return grad, hess

def weighted_rmse_eval(preds, train_data):
    labels = train_data.get_label()
    weights = train_data.get_weight()
    weighted_mse = np.average((preds - labels) ** 2, weights=weights)
    return 'w-rmse', np.sqrt(weighted_mse), False

def weighted_r2_score(preds, data):
    labels = data.get_label()
    weights = data.get_weight()
    score = r2_score(labels, np.clip(preds, -5, 5), sample_weight=weights)
    return 'w-r2_score', score, True

def weighted_rmse_objective_xgb(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    grad = weights * (preds - labels)
    hess = weights
    return grad, hess

def weighted_rmse_metric_xgb(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    weighted_mse = np.average((preds - labels) ** 2, weights=weights)
    return 'w-rmse', np.sqrt(weighted_mse)

def weighted_r2_metric_xgb(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    score = r2_score(labels, preds, sample_weight=weights)
    return 'w-r2', score