from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import random

from medaboost.truth import BaseTruth



class DS(BaseTruth):

    def __init__(self, max_iter=20, init_quality=0.7):
        self.max_iter = max_iter
        self.init_quality = init_quality
        self.w_worker = None

    def e_step(self):
        self.e2lpd = np.zeros((self.n_tasks, self.n_labels))
        total_weight = 0
        # for each sample, do something
        for idx in range(self.n_tasks):
            row = self.y_mat[idx, :]
            # set the base probability distribution to default
            lpd = self.l2pd
            # get the workers who did stuff
            worker_mask = np.argwhere(~np.isnan(row)).flatten()
            # get their weights
            for worker in worker_mask:
                lpd = np.multiply(lpd, self.w2cm[:, int(row[worker]), worker])
            # normalize to sum to 1
            self.e2lpd[idx, :] = lpd / lpd.sum(0)
        return


    def m_step_l2pd(self):
        # normalized column sum
        self.l2pd = self.e2lpd.sum(axis=0) / self.n_tasks


    def m_step_w2cm(self):
        self.w2cm = np.zeros((self.n_labels, self.n_labels, self.n_workers))
        for worker in range(self.n_workers):
            worker_answer = self.y_mat[:, worker]
            example_mask = np.argwhere(~np.isnan(worker_answer)).flatten()
            # column sum
            tmp_w2cm = self.e2lpd[example_mask, :].sum(axis=0)
            # then for each example weight it by the original
            tmp_ex = np.multiply(self.e2lpd[example_mask, :], 1.0/tmp_w2cm)
            for label in range(self.n_labels):
                label_mask = np.argwhere(worker_answer[example_mask] == label).flatten()
                self.w2cm[:, label, worker] += (tmp_ex[label_mask, :]).sum(axis=0)
        return


    def infer(self, y):
        if (isinstance(y, pd.DataFrame)):
            # assume it's integer type
            y = y.to_numpy(dtype='int')
        self.y_mat = y
        # get the label_set
        self.n_tasks = y.shape[0]
        self.n_workers = y.shape[1]
        self.w_workers = np.ones(self.n_workers)
        self.n_labels = 2 # hardcode it to be binary for now
        # initalize
        self.l2pd = np.full(self.n_labels, 1/self.n_labels)
        # default confusion matrix
        # row = true label, column = label
        cm = np.full((self.n_labels, self.n_labels),
                     (1-self.init_quality)/(self.n_labels-1))
        np.fill_diagonal(cm, self.init_quality)
        self.w2cm = np.repeat(cm[:, :, np.newaxis],
                              self.n_workers,
                              axis=2)

        for i in range(self.max_iter):
            # E-step
            self.e_step()

            # M-step
            self.m_step_l2pd()
            self.m_step_w2cm()

            # compute the likelihood
            # print self.computelikelihood()
        # set the worker confidence for [1,1]
        for w in range(self.n_workers):
            self.w_workers[w] = self.w2cm[1, 1, w]

        # generate 1 if prob_1 >= 0.5
        return np.where(self.e2lpd[:, 1] >= 0.5, 1, 0)
