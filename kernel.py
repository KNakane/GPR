import numpy as np

class Gaussian(object):

    def __init__(self, params):
        assert np.shape(params) == (2,)
        self.__params = params

    def get_params(self):
        return np.copy(self.__params)

    def __call__(self, x, y):
        return self.__params[0] * np.exp(-0.5 * self.__params[1] * (x - y) ** 2)

    def derivatives(self, x, y):
        sq_diff = (x - y) ** 2
        delta_0 = np.exp(-0.5 * self.__params[1] * sq_diff)
        delta_1 = -0.5 * sq_diff * delta_0 * self.__params[0]
        return (delta_0, delta_1)

    def update(self, updates):
        assert np.shape(updates) == (2,)
        self.__params += updates

class RBF(object):

    def __init__(self, params):
        assert np.shape(params) == (2,)
        self.__params = params

    def get_params(self):
        return np.copy(self.__params)

    def __call__(self, x, y):
        return self.__params[0] * np.exp(-0.5 * self.__params[1] * (x - y) ** 2)

    def derivatives(self, x, y):
        sq_diff = (x - y) ** 2
        delta_0 = np.exp(-0.5 * self.__params[1] * sq_diff)
        delta_1 = -0.5 * sq_diff * delta_0 * self.__params[0]
        return (delta_0, delta_1)

    def update(self, updates):
        assert np.shape(updates) == (2,)
        self.__params += updates
