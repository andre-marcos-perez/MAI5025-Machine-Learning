import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

# ------------------------------------------------------------------------------
# -- Variables -----------------------------------------------------------------
# ------------------------------------------------------------------------------

# -- Environment

DATASET = "iris"
#DATASET = "glass"
EXTENSION = ".csv"
DIR = "../dataset/"
DATAPATH = DIR + DATASET + EXTENSION

# -- Train

SEED = 123
FOLDS = 10
SPLIT = 0.3

# -- Hyperparams

TOL = [1e-1, 1e-2, 1e-3, 1e-4]
CLASS_WEIGHT = [None, "balanced"]
MAX_ITER = [100, 1000, 10000]

# ------------------------------------------------------------------------------
# -- Main ----------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ----------------------------------------
# -- Data --------------------------------
# ----------------------------------------

with open(DIR + DATASET + "_names" + EXTENSION, mode="r") as file:
    line = file.readline()
    HEADER = line.split(',')

data = pd.read_csv(DATAPATH, names=HEADER)
features = data.drop(HEADER[-1], axis=1)
target = data.select_dtypes(include=[object])
data = None

cols = list(features.columns)
for col in cols:
    features[col] = (features[col] - features[col].mean()) / features[col].std()

# ----------------------------------------
# -- Train -------------------------------
# ----------------------------------------

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=SPLIT, random_state=SEED)

# Train model

slnn = Perceptron(random_state=SEED)
hyper = {
    "tol":TOL,
    "class_weight":CLASS_WEIGHT,
    "max_iter":MAX_ITER
}

model = GridSearchCV(slnn, param_grid=hyper, n_jobs=-1, iid=True, cv=FOLDS)
model.fit(X_train, y_train.values.ravel())
prediction = model.predict(X_test)

# Performance metrics

metrics = dict.fromkeys(["acc", "pre", "sen", "esp", "f1"], 0)
mcm = multilabel_confusion_matrix(y_test, prediction)
tn = mcm[:, 0, 0]
tp = mcm[:, 1, 1]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]
metrics["acc"] = (tp + tn) / (tp + tn + fp + fn)
metrics["pre"] = tp / (tp + fp)
metrics["sen"] = tp / (tp + fn)
metrics["esp"] = tn / (tn + fp)
metrics["f1"] = 2*tp / (2*tp + fp + fn)
for key in metrics:
    metrics[key] = np.append(metrics[key], np.mean(metrics[key]))
metrics["acc"] = np.append(metrics["acc"], (np.sum(tp) + np.sum(tn)) / (np.sum(tp) + np.sum(tn) + np.sum(fp) + np.sum(fn)))
metrics["pre"] = np.append(metrics["pre"], np.sum(tp) / (np.sum(tp) + np.sum(fp)))
metrics["sen"] = np.append(metrics["sen"], np.sum(tp) / (np.sum(tp) + np.sum(fn)))
metrics["esp"] = np.append(metrics["esp"], np.sum(tn) / (np.sum(tn) + np.sum(fp)))
metrics["f1"] = np.append(metrics["f1"], 2*np.sum(tp) / (2*np.sum(tp) + np.sum(fp) + np.sum(fn)))

# ----------------------------------------
# -- Report ------------------------------
# ----------------------------------------

cols = list(target[HEADER[-1]].unique())
cols.append("Macro Avg")
cols.append("Micro Avg")
report = pd.DataFrame.from_dict(metrics, orient="index", columns=cols)

print("")
print("Dataset: " + DATASET)
print("Model: Perceptron")
print("")
print("# ------------------------")
print("# -- Target Class Count --")
print("# ------------------------")
print("")
print(print(target[HEADER[-1]].value_counts() / len(target.index)))
print("")
print("# ------------------------")
print("# -- Best Params ---------")
print("# ------------------------")
print("")
for key in model.best_params_:
    print(key + ": " + str(model.best_params_[key]))
print("")
print("# ------------------------")
print("# -- Performance ---------")
print("# ------------------------")
print("")
print(report)
