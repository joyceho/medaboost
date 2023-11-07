# medaboost
Medical Expert Disagreement AdaBoost

This Python package implements an extension of AdaBoost that leverages disagreements 

# Prerequisites

* Python 3
* numpy
* pandas
* pytest
* pymongo (only for experiments)
* scikit-learn
* tqdm

# Example Code
You can use MedaBoost almost the same you use AdaBoost (i.e., you can replace MedaBoost with AdaBoost). The big change is that you can pass in the original annotation labels as an additional keyword, raw_y, at the fit stage.

## Generating synthetic annotations for Iris Dataset
```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle

# load the iris dataset
iris = datasets.load_iris()
# keep only 2 labels and do shuffle
iris.data, iris.target = shuffle(iris.data[0:100, ], iris.target[0:100], random_state=0)
# create 3 worker annotation
y = pd.concat([pd.DataFrame(iris.target)]*3,
              axis=1,
              ignore_index=True)

rng = np.random.default_rng(seed=2021)
# permute the labels to mimic reliability switches
for i, wi in enumerate([80, 120, 100]):
    # set the permutation for worker i
    w_perm = rng.choice(range(iris.target.size), size=wi)
    y[i].iloc[w_perm] = rng.permutation(y[i].iloc[w_perm])
```
## Training MedaBoost

Train MedaBoost using DS as the truth inference and 100 iterations

```python
clf = MedaBoost(truth_inf="DS", ds_iter=100)
# given a 3 worker annotation matrix, infer the annotation
single_y = clf.infer_labels(y)
# fit the classifier using the data
clf.fit(iris.data, single_y, raw_y=y)
```

Train MedaBoost using weighted majority voting as the truth inference method

```python
clf = MedaBoost(truth_inf="WMV")
# given a 3 worker annotation matrix, infer the annotation
single_y = clf.infer_labels(y)
# fit the classifier using the data
clf.fit(iris.data, single_y, raw_y=y)
```

## Predict using MedaBoost
```python
import sklearn.metrics as skm

y_hat = clf.predict_proba(iris.data)
# score the model
clf_auc = skm.roc_auc_score(iris.target, y_hat[:, 1])
```
