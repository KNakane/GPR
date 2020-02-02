import os, sys, time
import numpy as np
from utility import Utility
import matplotlib.pyplot as plt
from collections import OrderedDict
from methods import GPR, MyModel

class Trainer():
    def __init__(self, model):
        self.regression_model = eval(model)()
        self.util = Utility(model)
        self.util.initialize()
        self.train_x, self.train_y = self.load(low=0, high=1, n=1000, std=0.1)
        self.test_x, self.test_y = self.load(low=-2, high=3, n=1000, std=0.1, test=True)
        self.message = OrderedDict({'model':model})

    def func(self, x):
        return np.sin(2 * np.pi * x) + np.cos(np.pi * x)

    def load(self, low=0, high=1., n=10, std=1., test=False):
        self.x = np.sort(np.random.uniform(low, high, n))
        if test == False:
            self.x = self.x[(self.x < 0.3) | (self.x > 0.7)]
        self.t = self.func(self.x) if test else self.func(self.x) + np.random.normal(scale=std, size=self.x.shape[0])
        return self.x[:, np.newaxis], self.t[:, np.newaxis]

    def train(self):
        start_time = time.time()
        self.regression_model.fit(self.train_x, self.train_y)
        self.message['learning_time(sec)'] = time.time() - start_time
        self.util.logger(self.message)
        
        
        mean, std = self.regression_model.predict(self.test_x)
        std = std**0.5

        plt.scatter(self.x, self.t, alpha=0.5, color="blue", label="observation")
        plt.plot(self.test_x[:,0], self.test_y, color="blue", label="sin$(2\pi x)$", markersize=0.5)
        plt.plot(self.test_x[:,0], mean, color="red", label="predict_mean")
        plt.fill_between(self.test_x[:,0], mean[:,0] - std[:,0], mean[:,0] + std[:,0], color="pink", alpha=0.5, label="predict_std")
        plt.legend(loc="lower left")
        plt.savefig(self.util.res_dir + '/train.png')
        return

    def test(self):
        x = np.reshape(np.linspace(np.min(self.train_x), np.max(self.train_y), 100), [-1, 1])
        output = self.regression_model.predict(x) # 1D
        return