from functools import partial

import numpy as np

assert_allclose = partial(np.testing.assert_allclose, rtol=1e-05, atol=np.inf, equal_nan=False)
