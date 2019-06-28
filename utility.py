# -*- coding: utf-8 -*-
import os, re
import pickle
import numpy as np
import datetime

class Utility():
    def __init__(self, model):
        dt_now = datetime.datetime.now()
        self.res_dir = "results/"+dt_now.strftime("%y%m%d_%H%M%S_{}".format(model))
        self.log_dir = self.res_dir + "/log"
        self.model_path = self.res_dir + "/model"

    def initialize(self):
        if os.path.exists(self.res_dir):
            os.remove(self.res_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

    def logger(self, message):
        with open(self.log_dir + '/log.txt', 'a') as f:
            for key, info in message.items():
                f.write("%s : %s\n"%(key, info))
        return