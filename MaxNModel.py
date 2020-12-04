import numpy as np

class MaxNModel:
    def __init__(self, round_non_negative_int_func, n):
        self.n = int(n)
        self.model_name = "Max_{}_Model".format(self.n)
        self.round_non_negative_int_func = round_non_negative_int_func
        self.max = 0

    def fit(self, data):
        self.max = np.max(data[-self.n:])
        return self

    def predict(self, next_n_prediction):
        return [self.max] * next_n_prediction