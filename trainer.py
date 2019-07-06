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
        self.train_x, self.train_y = self.load(low=0, high=1, n=40, std=0.1)
        self.message = OrderedDict({'model':model})

    def func(self, x):
        return np.sin(2 * np.pi * x)

    def load(self, low=0, high=1., n=10, std=1.):
        self.x = np.random.uniform(low, high, n)
        self.t = self.func(self.x) + np.random.normal(scale=std, size=n)
        return self.x[:, np.newaxis], self.t[:, np.newaxis]

    def train(self):
        start_time = time.time()
        self.regression_model.fit(self.train_x, self.train_y)
        self.message['learning_time(sec)'] = time.time() - start_time
        self.util.logger(self.message)
        
        x_test = np.reshape(np.linspace(np.min(self.train_x), np.max(self.train_y), 100), [-1, 1])
        mean, std = self.regression_model.predict(x_test)
        std = std**0.5

        plt.scatter(self.x, self.t, alpha=0.5, color="blue", label="observation")
        plt.plot(x_test[:,0], self.func(x_test), color="blue", label="sin$(2\pi x)$")
        plt.plot(x_test[:,0], mean, color="red", label="predict_mean")
        plt.fill_between(x_test[:,0], mean[:,0] - std[:,0], mean[:,0] + std[:,0], color="pink", alpha=0.5, label="predict_std")
        plt.legend(loc="lower left")
        plt.show()
        return

    def test(self):
        x = np.reshape(np.linspace(np.min(self.train_x), np.max(self.train_y), 100), [-1, 1])
        output = self.regression_model.predict(x) # 1D
        return