from collections import Counter
import numpy as np
import random

from medaboost.truth import BaseTruth


def compute_majority(x):
    # drop the missing
    x_count = Counter(x.dropna()).most_common()
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


class MajorityVote(BaseTruth):

    def infer(self, y):
        self.w_workers = np.ones(y.shape[1])
        y_est = y.apply(compute_majority, axis=1)
        return y_est


