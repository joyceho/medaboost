import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from medaboost.truth import MajorityVote, WeightedMajorityVote, DS

MIN_WEIGHT = 1e-20


class TruthBoost(ClassifierMixin, BaseEstimator):
    truth_inf = None
    ada_model = None
    ds_iter = None
    
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None,
                 truth_inf='MV',
                 ds_iter=20):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.truth_inf = truth_inf
        self.ds_iter = ds_iter
        if truth_inf == "WMV":
            self.truth_model = WeightedMajorityVote()
        elif truth_inf == "DS":
            self.truth_model = DS(max_iter=ds_iter)
        elif truth_inf == "MV":
            self.truth_model = MajorityVote()
        else:
            raise ValueError("Invalid truth inference" + truth_inf)

    def infer_labels(self, y):
        y_est = self.truth_model.infer(y)
        return y_est.astype('int')

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.ada_model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def staged_predict_proba(self, X):
        return self.ada_model.staged_predict_proba(X)
    
    def score(self, X, y, sample_weight=None):
        return self.ada_model.score(X, y, sample_weight)

    def get_feats(self):
        return self.ada_model.feature_importances_

    
class TruthAdaBoost(TruthBoost):
    def fit(self, X, y, raw_y=None,
            sample_weight=None,
            sample_type=0,
            loss_func='exponential'):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.ada_model = GradientBoostingClassifier(loss=loss_func,
                                                    n_estimators=self.n_estimators,
                                                    learning_rate=self.learning_rate,
                                                    random_state=self.random_state)
        self.is_fitted_ = True
        tmp = self.ada_model.fit(X, y, sample_weight)
        self.classes_ = self.ada_model.classes_
        self.decision_function = self.ada_model.decision_function
        return tmp



class MedaBoost(TruthBoost):
    def calc_weight(self, X, y, raw_y, sample_type):
        # y is a matrix of values in this case
        M = raw_y.shape[1]
        y_est_mat = np.tile(y, M)
        y_est_mat = y_est_mat.reshape((raw_y.shape[0],
                                       raw_y.shape[1]))
        # count the number of disagrements
        d_y = (y_est_mat != raw_y).sum(axis=1)
        if sample_type == 0:
            sample_weight = d_y.apply(lambda x: max((M-x)/(M+x),
                                                    MIN_WEIGHT))
        elif sample_type == 1:
            sample_weight = d_y.apply(lambda x: max(np.exp(-x),
                                                    MIN_WEIGHT))
        elif sample_type == 2:
            sample_weight = d_y.apply(lambda x: max(np.exp(-x/M),
                                                    MIN_WEIGHT))
        else:
            sample_weight = None
        return sample_weight


    def fit(self, X, y, raw_y=None,
            sample_weight=None,
            sample_type=0,
            loss_func='exponential'):
        X, y = check_X_y(X, y, accept_sparse=True)
        if loss_func == "exponential":
            self.ada_model = AdaBoostClassifier(n_estimators=self.n_estimators,
                                                learning_rate=self.learning_rate,
                                                random_state=self.random_state)
        else:
            self.ada_model = GradientBoostingClassifier(loss=loss_func,
                                                        n_estimators=self.n_estimators,
                                                        learning_rate=self.learning_rate,
                                                        random_state=self.random_state)
        sample_weight = self.calc_weight(X, y, raw_y, sample_type)
        self.is_fitted_ = True
        tmp = self.ada_model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.ada_model.classes_
        self.decision_function = self.ada_model.decision_function
        return tmp
    

def _sample0(x, M):
    return max((M-x)/(M+x),
               MIN_WEIGHT)

def _sample1(x):
    return max(np.exp(-x),
               MIN_WEIGHT)


class WeightMedaBoost(MedaBoost):
    def calc_weight(self, X, y, raw_y, sample_type):
        M = raw_y.shape[1]
        y_est_mat = np.tile(y, M)
        y_est_mat = y_est_mat.reshape((raw_y.shape[0],
                                       raw_y.shape[1]))
        m_weights = self.truth_model.worker_confidence()
        sample_tmp = np.dot(y_est_mat != raw_y, m_weights)
        d_y = np.clip(sample_tmp, 1e-12, 1)
        if sample_type == 0:
            sfunc = np.vectorize(_sample0)
            sample_weight = sfunc(d_y, M)
        elif sample_type == 1:
            sfunc = np.vectorize(_sample1)
            sample_weight = sfunc(d_y)
        else:
            sample_weight = None
        
        return sample_weight
