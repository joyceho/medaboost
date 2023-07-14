import numpy as np
import numpy.testing as npt
import pandas as pd


from medaboost import MedaBoost, TruthAdaBoost
from sklearn import datasets
from sklearn.utils import shuffle


# load the iris dataset + shuffle
iris = datasets.load_iris()
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=0)

rng = np.random.default_rng(seed=2021)
y = pd.concat([pd.DataFrame(iris.target)]*3,
              axis=1,
              ignore_index=True)

for i, wi in enumerate([80, 120, 100]):
    # set the permutation for worker i
    w_perm = rng.choice(range(iris.target.size), size=wi)
    y[i].iloc[w_perm] = rng.permutation(y[i].iloc[w_perm])


def test_iris_meda():
    # Check consistency on dataset iris.
    classes = np.unique(iris.target)
    clf_samme = prob_samme = None

    clf = MedaBoost()
    clf.fit(iris.data, iris.target, raw_y=y)

    npt.assert_array_equal(classes, clf.ada_model.classes_)
    proba = clf.predict_proba(iris.data)
    npt.assert_equal(proba.shape[1],len(classes))
    npt.assert_equal(clf.ada_model.decision_function(iris.data).shape[1],
                     len(classes))

    score = clf.ada_model.score(iris.data, iris.target)
    print(score)

    # Check we used multiple estimators
    assert len(clf.ada_model.estimators_) > 1


def _test_iris_truth(tinf):
    # Check consistency on dataset iris.
    classes = np.unique(iris.target)
    clf_samme = prob_samme = None

    clf = TruthAdaBoost(truth_inf=tinf)
    clf.fit(iris.data, iris.target, raw_y=y)

    npt.assert_array_equal(classes, clf.ada_model.classes_)
    proba = clf.predict_proba(iris.data)
    npt.assert_equal(proba.shape[1],len(classes))
    npt.assert_equal(clf.ada_model.decision_function(iris.data).shape[1],
                     len(classes))

    score = clf.ada_model.score(iris.data, iris.target)
    print(score)

    # Check we used multiple estimators
    assert len(clf.ada_model.estimators_) > 1


def test_iris_truth_mv():
    _test_iris_truth("MV")

    
def test_iris_truth_wmv():
    _test_iris_truth("WMV")

def test_iris_truth_ds():
    _test_iris_truth("DS")
