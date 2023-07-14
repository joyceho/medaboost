import numpy as np
import numpy.testing as npt
import pandas as pd

from medaboost.truth import WeightedMajorityVote


def test_infer():
	y = pd.DataFrame(np.array([[0, np.nan, 1],
							   [1, 0, 0],
							   [1, 0, 0],
							   [0, 1, 0],
							   [0, 1, 0],
							   [0, 0, 1]]),
                   columns=['w1', 'w2', 'w3'])
	mv = WeightedMajorityVote()
	y_est = mv.infer(y)
	npt.assert_array_equal([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], y_est)
