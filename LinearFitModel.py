import numpy as np
from scipy import stats

class LinearFitModel:
    def __init__(self, round_non_negative_int_func, latest_n, add_std_factor = 0):
        self.model_name = "LinearFit_{}_{}_Model".format(latest_n, add_std_factor)
        self.round_non_negative_int_func = round_non_negative_int_func
        self.latest_n = latest_n
        self.add_std_factor = add_std_factor

    def fit(self, data):
        self.x = [i for i in range(len(data[-self.latest_n:]))]
        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(self.x, data[-self.latest_n:])
        yhat =  np.array([self.slope * x_n + self.intercept for x_n in self.x])
        self.std_estimate = np.std(yhat-np.array(data[-self.latest_n:]))

    def predict(self, next_n_prediction):
        x_next = [self.x[-1] + 1 + i for i in range(next_n_prediction)]
        pred = np.array([self.slope * x_n + self.intercept for x_n in x_next]) + self.add_std_factor * self.std_estimate
        pred = self.round_non_negative_int_func(pred)
        return pred
