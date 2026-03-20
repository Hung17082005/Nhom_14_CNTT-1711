import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import f1_score

def simulate_low_labels(y, percent=0.1):
    y_partial = np.copy(y)
    n_labeled = int(len(y) * percent)

    mask = np.ones(len(y), dtype=bool)
    mask[:n_labeled] = False
    np.random.shuffle(mask)

    y_partial[mask] = -1
    return y_partial

def run_label_spreading(X, y, percent=0.1):
    y_partial = simulate_low_labels(y, percent)

    model = LabelSpreading()
    model.fit(X, y_partial)

    y_pred = model.transduction_

    f1 = f1_score(y, y_pred, average="macro")
    return f1