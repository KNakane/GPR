import os, sys, re
import numpy as np
import GPy
import pickle

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