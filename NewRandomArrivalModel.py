import numpy as np
from scipy.stats import expon
import math


class NewRandomArrivalModel:
    def __init__(self, round_non_negative_int_func, spike_detect_lag=12, spike_detect_std_threshold=2, spike_detect_influence=0.5, ts_len=360,
                 rise_strategy="auto", decline_strategy="exponential", add_std_factor=0, confidence_threshold=0.75, height_limit="average"):
        self.has_spike = False
        self.round_non_negative_int_func = round_non_negative_int_func
        self.spike_detect_lag = spike_detect_lag
        self.spike_detect_std_threshold = spike_detect_std_threshold
        self.spike_detect_influence = spike_detect_influence
        self.ts_len = ts_len
        self.rise_strategy = rise_strategy
        self.decline_strategy = decline_strategy
        self.add_std_factor = add_std_factor
        self.confidence_threshold = confidence_threshold
        self.height_limit = height_limit
        self.model_name = "NewRandomArriaval_%s_%s_%s_(%d_%d_%f)_%d_%f_Model" % (self.rise_strategy, self.decline_strategy, self.height_limit,
                                                                                 self.spike_detect_lag, self.spike_detect_std_threshold, self.spike_detect_influence,
                                                                                 self.ts_len, self.confidence_threshold)

    def fit(self, data):
        self.data = list(data)[-(self.ts_len):]
        if len(data) <= self.spike_detect_lag:
            spike_num = 0
        else:
            self.detect_result = self.spike_detect(
                self.data, self.spike_detect_lag, self.spike_detect_std_threshold, self.spike_detect_influence)
            spike_num = sum(self.detect_result["signals"])

        if spike_num > 2:
            self.has_spike = True
            self.spike = [index for index, signal in enumerate(
                self.detect_result["signals"]) if signal == 1]
            self.valley = self.detect_result["valley"]
            self.rise_length = [spike_index - valley_index for valley_index,
                                spike_index in zip(self.valley, self.spike[1:])]
            self.spike_height = [self.data[spike_index]
                                 for spike_index in self.spike]
            self.decline_length = [valley_index - spike_index for valley_index,
                                   spike_index in zip(self.valley, self.spike[:-1])]
            self.valley_height = [self.data[i] for i in self.valley]

            self.avg_rise_length = sum([rise_len*distance_weight for rise_len, distance_weight in zip(
                self.rise_length, self.spike[1:])]) / sum(self.spike[1:])
            self.avg_spike_height = sum(
                [height*distance_weight for height, distance_weight in zip(self.spike_height, self.spike)]) / sum(self.spike)
            self.avg_decline_length = sum([decline_len*distance_weight for decline_len, distance_weight in zip(
                self.decline_length, self.spike[:-1])]) / sum(self.spike[:-1])
            self.avg_valley_height = sum([val_height*distance_weight for val_height,
                                          distance_weight in zip(self.valley_height, self.valley)]) / sum(self.valley)

            self.rise_k = (self.avg_spike_height) / self.avg_rise_length
            self.rise_alpha = (self.avg_spike_height)**(1/self.avg_rise_length)
            self.spike_interval = [spike_index - valley_index for spike_index,
                                   valley_index in zip(self.spike[1:], self.valley)]
            self.expon_params = expon.fit(self.spike_interval, floc=0)
            self.last_spike_height = self.spike_height[-1]
            self.decline_alpha = (
                self.last_spike_height)**(1/self.avg_decline_length)
            self.decline_k = (self.avg_spike_height) / self.avg_decline_length
        else:
            self.has_spike = False

    def predict(self, next_n_predict):
        if not self.has_spike:
            predictions = [self.data[-1]] * next_n_predict
        else:
            predictions = []
            for diff in range(next_n_predict):
                since_latest_spike = len(self.data) - self.spike[-1] + diff

                if since_latest_spike <= self.avg_decline_length:
                    decline_step = self.avg_decline_length - since_latest_spike
                    if self.decline_strategy == "exponential":
                        pred = self.decline_alpha**decline_step
                    elif self.decline_strategy == "expectation":
                        pred = expon.cdf(0, -decline_step,
                                         self.last_spike_height)
                    elif self.decline_strategy == "linear":
                        pred = self.decline_k * decline_step
                    else:
                        raise Exception(
                            "unknown decline strategy: %s" % self.decline_strategy)

                else:
                    rise_step = since_latest_spike-self.avg_decline_length
                    confidence = expon.cdf(0, -rise_step, self.expon_params[1])
                    if self.height_limit == "average":
                        limit = self.avg_spike_height
                    elif "max_" in self.height_limit:
                        n = int(self.height_limit.split("_")[1])
                        limit = max(self.spike_height[-n:])
                    else:
                        raise Exception("unknown height limit: %s" %
                                        self.height_limit)

                    if math.log(limit) < rise_step*math.log(self.rise_alpha):
                        pred_eia = limit
                    else:
                        pred_eia = self.rise_alpha**rise_step
                    pred_ee = confidence * (self.avg_spike_height)
                    pred_li = min(limit, self.rise_k * rise_step)
                    if self.rise_strategy == "exponential":
                        pred = pred_eia
                    elif self.rise_strategy == "expectation":
                        pred = pred_ee
                    elif self.rise_strategy == "linear":
                        pred = pred_li
                    elif self.rise_strategy == "auto":
                        if confidence < self.confidence_threshold:
                            pred = pred_ee
                        else:
                            pred = max(pred_eia, pred_ee)
                    else:
                        raise Exception("unknown rise strategy: %s" %
                                        self.rise_strategy)

                predictions.append(pred.real)
        return self.round_non_negative_int_func(predictions)

    def spike_detect(self, y, lag, std_threshold, influence):
        """
        Spike detection, the original algorithm can be found here:
            https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887
        Included some additional rules:
            valley: a valley is the smallest value between two spikes. If a valley is lager than last spike, 
                    means the last spike actually halfway up the next one. It should not be considered as a spike.
        """
        signals = np.zeros(len(y))
        all_valley = []
        filtered_y = np.array(y)
        avg_filter = [0] * len(y)
        std_filter = [0] * len(y)
        avg_filter[lag - 1] = np.mean(y[0: lag])
        std_filter[lag - 1] = np.std(y[0: lag])

        last_spike = -1
        for i in range(lag, len(y)):
            is_spike = False
            if abs(y[i] - avg_filter[i-1]) > std_threshold * std_filter[i-1] and y[i] > y[i-1]:
                if last_spike > 0:
                    valley = np.min(y[last_spike: i + 1])
                    if y[i] - valley > avg_filter[i-1]:
                        if valley >= y[last_spike] / 2:
                            signals[last_spike] = 0
                            filtered_y[last_spike] = y[last_spike]
                        else:
                            all_valley.append(
                                y[last_spike: i + 1].index(valley)+last_spike)
                        is_spike = True
                        last_spike = i
                else:
                    is_spike = True
                    last_spike = i

            if is_spike:
                signals[i] = 1
            else:
                signals[i] = 0
            if abs(y[i] - avg_filter[i-1]) > std_threshold * std_filter[i-1]:
                filtered_y[i] = influence * y[i] + \
                    (1 - influence) * filtered_y[i-1]
            else:
                filtered_y[i] = y[i]

            avg_filter[i] = np.mean(filtered_y[(i - lag + 1): i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1): i + 1])
        return {"signals": signals, "avg_filter": avg_filter, "std_filter": std_filter, "valley": all_valley}
