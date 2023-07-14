from collections import Counter
import numpy as np
import random

from medaboost.truth import BaseTruth


def compute_weighted_majority(x, w_worker):
    x_count = Counter()
    for i in range(0, len(x)):
        if not np.isnan(x[i]):
            x_count.update({x[i]:w_worker[i]})
    x_count = x_count.most_common()
    max_freq = x_count[0][1]
    for i in range(1, len(x_count)):
        if x_count[i][1] != max_freq:
            x_count = x_count[:i]
            break
    if len(x_count) > 1:
        # randomly assign
        return random.choice(x_count)[0]
    else:
        return x_count[0][0]


class WeightedMajorityVote(BaseTruth):
     def __init__(self, beta=0.5):
         self.beta = beta
         self.w_workers = None

     def infer(self, y, max_iter=10):
        # set all the weights to be 1:
        n_workers = y.shape[1]
        self.w_workers = np.ones(n_workers)
        # renormalize
        self.worker_renorm()
        y_est = y.apply(compute_weighted_majority,
                        w_worker=self.w_workers,
                        axis=1)
        for k in range(max_iter):
            # update the worker
            for worker in range(0, n_workers):
                # get the answer for each worker
                worker_y = y.iloc[:, worker]
                mask = ~np.isnan(worker_y)
                # get the number of disagreements
                n_disagree = (y_est[mask] != worker_y[mask]).sum()
                self.w_workers[worker] = self.beta ** n_disagree * self.w_workers[worker]
            self.worker_renorm()
            y_est = y.apply(compute_weighted_majority,
                            w_worker=self.w_workers,
                            axis=1)
        return y_est


