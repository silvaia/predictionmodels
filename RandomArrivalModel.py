from datetime import datetime
import pandas as pd
from scipy.stats import expon
from scipy.stats import exponweib
import numpy as np

class RandomArrivalModel:
    def __init__(self, round_non_negative_int_func, fit_model="Expon"):
        '''

        :param round_non_negative_int_func:
        :param fit_model: "Expon" or "Weibull" or "Sampling"
        '''
        self.model_name = "RandomArrival_{}_Model".format(fit_model)
        self.round_non_negative_int_func = round_non_negative_int_func
        self.spike_std_factor = 2
        self.fit_model = fit_model
        return

    def fit(self, data):
        """
        data is an np array
        :param data:
        :return:
        """
        data = np.array(data)
        nPoints = len(data)
        avg = np.mean(data)
        std = np.std(data)
        spikes = data > ([avg +  self.spike_std_factor *std] * nPoints)
        self.last = data[-1]
        self.params = None
        if any(spikes):
            self.spike_max = max(data[spikes])
            self.spike_avg = np.mean(data[spikes])
            last_nonzero_idx = np.max(np.nonzero(data))
            self.time_since_last_spike = len(data) - 1 - np.max(np.nonzero(spikes))
            interarrivaltime = 0
            spikewidth = 0
            inter_arrival_times = []
            in_spike = False
            has_spiked = False
            spikewidths = []
            for isspike in spikes:
                if not isspike:
                    if in_spike: # was in spike, now not spike
                        spikewidths.append(spikewidth)
                    spikewidth = 0
                    interarrivaltime = interarrivaltime+1
                    in_spike = False
                else:
                    if not in_spike and has_spiked:
                        inter_arrival_times.append(interarrivaltime)
                    interarrivaltime = 0
                    spikewidth = spikewidth + 1
                    in_spike = True
                    has_spiked = True
            if len(inter_arrival_times)>0:
                if self.fit_model == "Weibull":
                    self.params = exponweib.fit(inter_arrival_times, floc=0, f0=1) # a, c, loc, scale
                elif self.fit_model == "Expon":
                    self.params = expon.fit(inter_arrival_times, floc=0)  # returns loc, scale
                else: # self.fit_model == "Sampling":
                    self.params = inter_arrival_times
            self.spike_width_avg = int(np.mean(spikewidths)) if len(spikewidths) > 0 else 1

        return self

    def predict(self, next_n):
        if not self.params:
            pred = [0] * next_n
        elif self.fit_model == "Sampling":
            pred = self.generate_samples(self.params, next_n, self.time_since_last_spike, self.spike_width_avg, self.spike_max)
        elif self.time_since_last_spike== 0:
            return [self.last]*next_n
        elif self.fit_model == "Weibull":
            pred = exponweib.cdf([x for x in range(next_n)], a = self.params[0], c= self.params[1], loc=-self.time_since_last_spike, scale = self.params[3]) * self.spike_avg
        else: # self.fit_model == "Expon":
            pred = expon.cdf([x for x in range(next_n)], -self.time_since_last_spike, self.params[1]) * self.spike_avg

        return self.round_non_negative_int_func(pred)

    def generate_samples(self, inter_arrival_time_samples, next_n, preceeding_zeros, spike_width, spike_height):
        pred = []
        total_len = preceeding_zeros + next_n
        while (len(pred) <= total_len):
            sample = np.random.choice(inter_arrival_time_samples, 1, True)
            pred.extend(np.zeros(sample))
            pred.extend(np.ones(spike_width) * spike_height)

        result_pred = pred[preceeding_zeros:total_len]
        return result_pred
