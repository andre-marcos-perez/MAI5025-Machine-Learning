import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix

# ------------------------------------------------------------------------------
# -- Variables -----------------------------------------------------------------
# ------------------------------------------------------------------------------

# -- Environment

DATASET = "iris"
#DATASET = "glass"
EXTENSION = ".csv"
DIR = "../dataset/"
MODEL = "perceptron"

DATAPATH = DIR + DATASET + EXTENSION

# -- Train & Model Hiperparams

TOL = 1e-1
SEED = 123
KFOLDS = 10

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
target = data.loc[:, HEADER[-1]]
data = None

cols = list(features.columns)
for col in cols:
    features[col] = (features[col] - features[col].mean()) / features[col].std()

try:
    plot = sns.pairplot(pd.concat([features, target], axis=1, sort=False), hue=HEADER[-1])
    plot.savefig(DATASET + ".png")
except Exception as e:
    print("-- Unable to plot dataset: " + e)

# ----------------------------------------
# -- Train -------------------------------
# ----------------------------------------

skf = StratifiedKFold(n_splits=KFOLDS)
skf.get_n_splits(features, target)

model = Perceptron(tol=TOL, random_state=SEED)
metrics = dict.fromkeys(["acc", "pre", "sen", "esp", "f1"], 0)

for train_index, test_index in skf.split(features, target):
    model.fit(features.iloc[train_index, :], target.iloc[train_index])
    prediction = model.predict(features.iloc[test_index, :])
    mcm = multilabel_confusion_matrix(target.iloc[test_index], prediction)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    metrics["acc"] = metrics["acc"] + (tp + tn) / (tp + tn + fp + fn)
    metrics["pre"] = metrics["pre"] + tp / (tp + fp)
    metrics["sen"] = metrics["sen"] + tp / (tp + fn)
    metrics["esp"] = metrics["esp"] + tn / (tn + fp)
    metrics["f1"] = metrics["f1"] + 2*tp / (2*tp + fp + fn)

for key in metrics:
    metrics[key] = list(metrics[key] / KFOLDS)
    metrics[key].append(np.mean(metrics[key]))

# ----------------------------------------
# -- Report ------------------------------
# ----------------------------------------

cols = list(target.unique())
cols.append("Average")
report = pd.DataFrame.from_dict(metrics, orient="index", columns=cols)
print(report)
