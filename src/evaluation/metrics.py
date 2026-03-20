from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import numpy as np

def classification_metrics(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse