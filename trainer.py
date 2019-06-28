import os, sys, time
import numpy as np
from utility import Utility
from collections import OrderedDict
from methods import GPR, GPy

class Trainer():
    def __init__(self, model):
        self.regression_model = eval(model)()
        self.util = Utility(model)
        self.util.initialize()
        self.train_x, self.train_y = self.load(high=0.7, std=0.1)
        self.message = OrderedDict({'model':model})

    def load(self, low=0, high=1., n=10, std=1.):
        def func(x):
            return np.sin(2 * np.pi * x)
        x = np.random.uniform(low, high, n)
        t = func(x) + np.random.normal(scale=std, size=n)
        return x, t

    def train(self):
        start_time = time.time()
        self.regression_model.fit(self.train_x, self.train_y)
        self.message['learning_time(sec)'] = time.time() - start_time
        self.util.logger(self.message)
        return

    def test(self):
        x = np.reshape(np.linspace(np.min(self.train_x), np.max(self.train_y), 100), [-1, 1])
        output = self.regression_model.predict(x) # 1D
        return