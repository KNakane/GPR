import os, sys, re
import numpy as np
import GPy
import pickle
from kernel import *

class Method():
    def __init__(self):
        self.model_name = self.__class__.__name__
    
    def fitting(self, x, y):
        raise Exception('please set fitting function')

    def predict(self, x, restore_dir=None):
        raise Exception('please set fitting function')

    def save_model(self, model_dir):
        with open(model_dir + '/model.pkl', mode='wb') as f:
            pickle.dump(self.model, f)

    def restore_model(self, load_dir):
        assert re.search(self.model_name , load_dir), 'Not model same!! {0} <=> {1}'.format(self.model_name, load_dir)
        with open(load_dir + '/model.pkl', mode='rb') as f:
            model = pickle.load(f)
        return model

class GPR(Method):
    def __init__(self):
        super().__init__()
        self.kernel = GPy.kern.RBF(1)
        self.model = GPy.models.GPRegression

    def fit(self, x, y):
        self.model = self.model(x, y, kernel=self.kernel)
        self.model.optimize()
        return

    def predict(self, x, restore_dir=None):
        if restore_dir is not None:
            self.model = self.restore_model(restore_dir)
        mean, var = self.model.predict(x)
        return [mean, var]


class MyModel(Method):
    def __init__(self, kernel='Gaussian', beta=10.):
        super().__init__()
        
        if kernel == 'RBF':
            self.params = np.array([1.,0.4,0.1])
        elif kernel == 'Gaussian':
            self.params = np.array([1., 1.])
        self.beta = beta
        self.kernel = eval(kernel)(params=self.params)

    def fit(self, x, y, learning_rate=0.1, iteration=1000):
        self.x = x
        self.t = y

        for i in range(iteration):
            params = self.kernel.get_params()
            Gram = self.kernel(*np.meshgrid(x, x))
            self.covariance = Gram + np.identity(len(x)) / self.beta
            self.precision = np.linalg.inv(self.covariance)
            gradients = self.kernel.derivatives(*np.meshgrid(x, x))    
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + ((self.precision @ y).T @ grad @ (self.precision @ y))[0, 0] for grad in gradients])
            self.kernel.update(learning_rate * updates)
            if np.allclose(params, self.kernel.get_params()):
                break
        return

    def predict(self, x, restore_dir=None):
        if restore_dir is not None:
            self.model = self.restore_model(restore_dir)
        gram = self.kernel(*np.meshgrid(x, self.x, indexing='ij'))
        mean = gram.dot(self.precision @ self.t)
        var = self.kernel(x, x) + 1 / self.beta - np.sum(gram.dot(self.precision) * gram, axis=1)
        return [mean, var**2]