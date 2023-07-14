import numpy as np
import numpy.testing as npt
import pandas as pd

from medaboost.truth import MajorityVote


def test_infer():
	y = pd.DataFrame(np.array([[0, np.nan, 1],
							   [1, 0, 0],
							   [1, 0, 0],
							   [0, 1, 0],
							   [0, 1, 0],
							   [0, 0, 1]]),
                   columns=['w1', 'w2', 'w3'])
	mv = MajorityVote()
	y_est = mv.infer(y)
	npt.assert_array_equal([0.0, 0.0, 0.0, 0.0, 0.0], y_est[1:])
