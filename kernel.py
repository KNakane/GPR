import numpy as np

class RBF(object):
    def __init__(self, params):
        assert np.shape(params) == (3,)
        self.params = params

    def get_params(self):
        return np.copy(self.params)

    def setting(self, x, y):
        return self.params[0] * np.exp(-((x - y)**2 * self.params[1])) + self.params[2]*(x == y)

    def derivatives(self, x, y):
        sq_diff = (x - y) ** 2
        delta_0 = np.exp(-sq_diff * self.params[1])
        delta_1 = -sq_diff * delta_0 * self.params[0]
        delta_2 = 1. * (x == y)
        return (delta_0, delta_1, delta_2)

    def update(self, updates):
        assert np.shape(updates) == (3,)
        self.params += updates

class Test(object):

    def __init__(self, params):
        assert np.shape(params) == (2,)
        self.params = params

    def get_params(self):
        return np.copy(self.params)

    def setting(self, x, y):
        return self.params[0] * np.exp(-0.5 * self.params[1] * (x - y) ** 2)

    def derivatives(self, x, y):
        sq_diff = (x - y) ** 2
        delta_0 = np.exp(-0.5 * self.params[1] * sq_diff)
        delta_1 = -0.5 * sq_diff * delta_0 * self.params[0]
        return (delta_0, delta_1)

    def update(self, updates):
        assert np.shape(updates) == (2,)
        self.params += updates
