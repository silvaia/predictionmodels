import os
from datetime import timedelta
import datetime
import numpy as np
import json
from scipy import stats
import pandas as pd
import traceback
from functools import partial
from .FbProphetModel import FbProphetModel
from .MaxNModel import MaxNModel
from .LinearFitModel import LinearFitModel
from .RandomArrivalModel import RandomArrivalModel
from .UnobservedComponent import UnobservedComponentModel
from .LstmModel import LstmModel
from .NewRandomArrivalModel import NewRandomArrivalModel
from .AdaptiveMaxNModel import AdaptiveMaxNModel
from .AdaptiveAverageNModel import AdaptiveAverageNModel

import logging
logger = logging.getLogger("sps.schedule")

def log_time_msg(msg):
    """
    For debugging purpose, to log the timestamp and pid info along with the message
    @param msg:
    @return:
    """
    # logger.info("[%s][%s] %s" % (str(os.getpid()), str(datetime.datetime.now()), msg))
    return

def expand(predictions, expand_ratio, expandDistribution):
    dist = [1, 1, 1, 1]
    if expandDistribution:
        try:
            kv = map(lambda x: x.split(':'), expandDistribution)
            kv = dict(map(lambda x: (int(x[0]), int(x[1])), kv))
            for i in range(expand_ratio):
                if i in kv.keys():
                    dist[i] = kv[i]
                else:
                    dist[i] = 0
        except:
            logger.info("Error parsing InHourVmSplit column {}!".format(expandDistribution))

    dist = np.array(dist) / np.sum(dist)
    expanded = []
    for i in predictions:
        splited = np.rint(i * dist)
        splited[0] = i - np.sum(splited[1:])
        splited[0] = splited[0] if splited[0] >= 0 else 0
        expanded.extend(splited)
    return expanded


def trim_start_zeros(historical_data):
    """

    @param historical_data:
    @return:
    """
    historical_data = [float(x) for x in historical_data]
    non_zero_data = historical_data
    # next(iterator, default) returns the next item from the iterator.
    # here the iterator is the index for which the element x is non-zero.
    # if every element is 0, return None
    first_non_zero_idx = next((i for i, x in enumerate(historical_data) if x),
                              None)  # find the index of first non-zero element in the history data
    if first_non_zero_idx:
        non_zero_data = historical_data[first_non_zero_idx:]  # read from non-zero index
    return non_zero_data

def isActiveRecently(historical_data, recent_days, interval_in_mins, active_percentage = 0.15):
    """

    @param historical_data: a list of historical data
    @param recent_days: how many days to check for recent activity
    @param interval_in_mins: the aggregation unit for the numbers in historical data.
    @param active_percentage: simple check if there are at least active_percentage points recently have non-zero deplooyments
    @return: true if there are still active deployments for the VM configuration recently 
    """
    points_per_day = 1440 / interval_in_mins
    lookback_points = int(points_per_day * recent_days)
    # get the length of the indices when there are non-zero value in the lookSback window in history
    # nonzero returns a tuple of arrays, one for each dimension of the input,
    # containing the indices of the non-zero elements in that dimension.
    recent_activity_cnt = len(np.nonzero(historical_data[-lookback_points:])[0])
    hasSignificantActivity = recent_activity_cnt>=lookback_points * active_percentage

    # make an exception if the deployment activity cont is low but volume is quite high
    hasSignificantVolume = sum(historical_data[-lookback_points:]) > recent_days * 100
    return hasSignificantActivity or hasSignificantVolume

def round_non_negative_int(arr):
    return [round(p) if p >0 else 0 for p in arr]

def eval_accuracy(actual, pred, overestimate_cost, model_name):
    hit_count = 0
    over_estimate_count = 0
    under_estimate_count = 0
    request_count = sum(actual)
    accuracy = {}
    for i in range(len(actual)):
        hit_count = hit_count + min(actual[i], pred[i])
        if(pred[i] > actual[i]):
            over_estimate_count = over_estimate_count + pred[i] - actual[i]
        else:
            under_estimate_count = under_estimate_count + actual[i] - pred[i]

    hit_rate = (hit_count+1) / (request_count+1)
    over_estimate_rate = (over_estimate_count +1) / (request_count + 1)
    under_estimate_rate = (under_estimate_count +1) / ( request_count + 1)
    overall_score = hit_rate - overestimate_cost * over_estimate_rate + (1 if hit_rate > 0.35 else 0)

    accuracy["model_name"] = model_name
    accuracy["overall_score"] = overall_score
    accuracy["hit_rate"] = hit_rate
    accuracy["over_estimate_rate"] = over_estimate_rate
    accuracy["under_estimate_rate"] = under_estimate_rate
    accuracy["request_count"] = request_count
    accuracy["hit_count"] = hit_count
    accuracy["over_estimate_count"] = over_estimate_count
    accuracy["under_estimate_count"] = under_estimate_count
    accuracy["prediction"] = pred
    return accuracy

def run_simple_models(short_term_train_data, recent_n_validation, points_per_day, total_points, add_std_factor, is_active, row_type="general"):
    log_time_msg("run simple models")
    model_results = []
    short_term_models_general = [
        AdaptiveAverageNModel(round_non_negative_int_func=round_non_negative_int, evaluation_function=partial(eval_accuracy, overestimate_cost=0.5)),
        LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=total_points),
        LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=recent_n_validation),
        LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=total_points, add_std_factor=add_std_factor),
        LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=recent_n_validation, add_std_factor=add_std_factor),
        AdaptiveMaxNModel(round_non_negative_int_func=round_non_negative_int, evaluation_function=partial(eval_accuracy, overestimate_cost=0.5)),
        MaxNModel(round_non_negative_int_func=round_non_negative_int, n=recent_n_validation),
        MaxNModel(round_non_negative_int_func=round_non_negative_int, n=3 * points_per_day),
        MaxNModel(round_non_negative_int_func=round_non_negative_int, n=total_points),
        RandomArrivalModel(round_non_negative_int_func=round_non_negative_int, fit_model="Expon"),
        RandomArrivalModel(round_non_negative_int_func=round_non_negative_int, fit_model="Sampling"),
        NewRandomArrivalModel(round_non_negative_int_func=round_non_negative_int),
        NewRandomArrivalModel(round_non_negative_int_func=round_non_negative_int, rise_strategy="linear"),
        NewRandomArrivalModel(round_non_negative_int_func=round_non_negative_int, height_limit="max_2")
    ]

    # mitigate overestimate rate in windows case
    short_term_models_for_windows = [
        AdaptiveAverageNModel(round_non_negative_int_func=round_non_negative_int, evaluation_function=partial(eval_accuracy, overestimate_cost=0.5)),
        LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=total_points),
        AdaptiveMaxNModel(round_non_negative_int_func=round_non_negative_int, evaluation_function=partial(eval_accuracy, overestimate_cost=0.5))
    ]

    if row_type == "windows":
        short_term_models = short_term_models_for_windows
    else:
        short_term_models = short_term_models_general

    if is_active:
        short_term_models.extend([
            UnobservedComponentModel(round_non_negative_int_func=round_non_negative_int),
            LstmModel(round_non_negative_int_func=round_non_negative_int, sample_num=20, feature_length_used=10),
            LstmModel(round_non_negative_int_func=round_non_negative_int, sample_num=20, feature_length_used=20),
            LstmModel(round_non_negative_int_func=round_non_negative_int, sample_num=30, feature_length_used=30),
        ])

    for m in short_term_models:
        result = {}
        try:
            log_time_msg("run simple models: fit %s" % (m.model_name))
            m.fit(short_term_train_data)
            pred = m.predict(recent_n_validation)
            if len(pred)!=recent_n_validation:
                Ex = ValueError()
                Ex.strerror = "model {} produced {} points prediction while needs {} points".format(m.model_name, len(pred), recent_n_validation)
                raise Ex
            result["successful"] = True
            result["prediction"] = pred
            result["model"] = m
            result["model_name"] = m.model_name
            result["error"] = None
            model_results.append(result)
            log_time_msg("finish: run simple models: fit %s" % (m.model_name))

        except Exception as ex:
            err_msg = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            result["successful"] = False
            result["prediction"] = None
            result["model"] = m
            result["model_name"] = m.model_name
            result["error"] = err_msg
            model_results.append(result)

    log_time_msg("finish: run simple models")
    return model_results


def processrow(row, end_time, nextKPrediction, expand_ratio, default_add_std_factor, regional_add_std_factor):
    """
    Process one rdd Row, which contains the time series data
    @param row: one instance of spark Row
    @param end_time: the end time of the historical data
    @param nextKPrediction: make prediction for the next k points
    @param expand_ratio: each predicted value will be expanded to K values according to the expand_ratio.
    This allows us to hourly job while output results every 15 minutes for example.
    @param default_add_std_factor: the output will be mean + add_std_factor * std
    @param regional_add_std_factor: a json string contains region:add_std_factor pairs
    @return: a dictionary contains prediction results and allows to convert to spark data frame later.
    """
    log_time_msg("process row")
    ts_data_array = row["TimeSeriesValues"] # an array of long
    region = row["Region"] # string
    availability_zone = row["AvailablityZone"]
    deployment_type = row["DeploymentType"] # string
    process_starttime = row["ProcessedDataStartTime"] # timestamp
    process_endtime = row["ProcessedDataEndTime"] # timestamp
    intervalInMins = row["BinIntervalInMins"] # double
    expandDistribution = row["InHourVmSplit"]
    isPIR = row['IsPIR']
    isSIG = row['IsSIG']
    abcSupportedRatio = row["ABCSupportedRatio"] # double
    err_msg = ""
    add_std_factor = default_add_std_factor

    if regional_add_std_factor:
        try:
            regional_add_std_factor_dic = json.loads(regional_add_std_factor)
            if region in regional_add_std_factor_dic.keys():
                add_std_factor = regional_add_std_factor_dic[region]
        except Exception as ex:
            err_msg = "Exception loading regional std factor setting! Region: {}, setting: {}".format(region, regional_add_std_factor)

    # get none zero history
    non_zero_data = trim_start_zeros(ts_data_array)
    is_active = isActiveRecently(historical_data=ts_data_array, recent_days=7, interval_in_mins=intervalInMins)

    info = {}
    model_score = []
    stderr = [0] * nextKPrediction
    pred = ts_data_array[-1] * nextKPrediction # predict the last demand number by default
    recent_n_validation = get_validation_period(non_zero_data)
    points_per_day = int(1440 / intervalInMins)
    total_points = len(ts_data_array)
    over_estimate_cost = 1/(1+add_std_factor)

    long_term_data = non_zero_data
    long_term_timestamps = [(process_endtime - timedelta(minutes=(intervalInMins * (i + 1)))) for i in
                            reversed(range(len(non_zero_data)))]

    if len(non_zero_data) > recent_n_validation:
        short_term_train_data = non_zero_data[0:-recent_n_validation]
        short_term_validation_data = non_zero_data[-recent_n_validation:]
    else:
        short_term_train_data = non_zero_data
        short_term_validation_data = non_zero_data + [0] * (recent_n_validation - len(non_zero_data))

    # handle windows case
    if "windows" in deployment_type:
        row_type = "windows"
        is_active = False
    else:
        row_type = "general"

    # run simple predictors and pick up a best model to use
    simple_model_pred = run_simple_models(short_term_train_data, recent_n_validation, points_per_day, total_points, add_std_factor, is_active, row_type)
    for m in simple_model_pred:
        if m["successful"]:  # log error message somewhere
            model_score.append((m["model"], eval_accuracy(short_term_validation_data, m["prediction"], over_estimate_cost, m["model"].model_name)))
        else:
            err_msg = err_msg + " " + m["error"]
    try:
        best_model, best_score = sorted(model_score, key=lambda m_s: m_s[1]["overall_score"], reverse=True)[0]
    except Exception as ex:
        err_msg = err_msg + " " + "All simple model failed, need attention here "
        best_score = {}
        best_model = LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=recent_n_validation)
        best_score["overall_score"] = -100

    # use the best model to predict future
    try:
        best_model.fit(long_term_data)
        pred = best_model.predict(nextKPrediction)
        info["fit_method"] = best_model.model_name
    except Exception as ex:
        err_msg = err_msg + " " + ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        lsfit_points = 4
        m = LinearFitModel(round_non_negative_int_func=round_non_negative_int, latest_n=lsfit_points)
        m.fit(non_zero_data[-lsfit_points:])
        pred = m.predict(nextKPrediction)
        info["fit_method"] = "Fallback to {}".format(m.model_name)

    if is_active:
        try:
            # make long term prediction
            long_term_model = FbProphetModel(round_non_negative_int_func = round_non_negative_int , add_std_factor = add_std_factor)
            long_term_model.fit(long_term_timestamps, long_term_data)

            (long_term_pred, short_term_pred) = long_term_model.predict(next_n_prediction = nextKPrediction, past_n_validation= recent_n_validation)
            info["fit_method"] = long_term_model.model_name
            long_term_model_score = eval_accuracy(short_term_validation_data, short_term_pred, over_estimate_cost, long_term_model.model_name)
            model_score.append((long_term_model, long_term_model_score))

            if long_term_model_score["overall_score"] > best_score["overall_score"]:
                info["fit_method"] = long_term_model.model_name
                pred = long_term_pred
            else:
                overwrite_start = 4 # overwrite the rest for long term prediction
                pred[overwrite_start:nextKPrediction] = long_term_pred[overwrite_start:nextKPrediction]
                info["fit_method"] = "{} + {}".format(best_model.model_name, long_term_model.model_name)
        except Exception as ex:
            err_msg = err_msg + " " + ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))

    info["model_scores"] = [ms[1] for ms in model_score]
    info["InHourVmSplit"] = expandDistribution
    expanded_pred = expand(pred, expand_ratio, expandDistribution)
    adjusted_pred = np.array(pred) * abcSupportedRatio
    adjusted_expanded_pred = np.array(expanded_pred) * abcSupportedRatio
    result = {}
    result["Region"] = region
    result['IsPIR'] = isPIR
    result['IsSIG'] = isSIG
    result["AvailabilityZone"] = availability_zone
    result["DeploymentType"] = deployment_type
    result["Ts_ProcessedDataEndTime"] = process_endtime
    result["Ts_HistoryValues"] = ",".join(map(str, map(int,ts_data_array)))
    result["Ts_IntervalInMins"] = int(intervalInMins)
    result["Ts_LatestDemand"] = int(ts_data_array[-1])
    result["Ts_Next_Forecast"] = int(pred[0])
    result["Ts_NextN_Forecast"] = ",".join(map(str, map(int,pred)))
    result["Ts_StdDev_History"] = float(np.std(non_zero_data[-points_per_day:])/expand_ratio)
    result["Ts_SumForecast"] = sum(map(int,pred))
    result["Next_Prediction"] = int(expanded_pred[0])
    result["NextN_Predictions"] = ",".join(map(str, map(int,expanded_pred)))
    result['ABCSupportedRatio'] = abcSupportedRatio
    result['Ts_Next_AdjustedForecast'] = int(adjusted_pred[0])
    result["Ts_NextN_AdjustedForecast"] = ",".join(map(str, map(int, adjusted_pred)))
    result["Next_AdjustedPrediction"] = int(adjusted_expanded_pred[0])
    result["NextN_AdjustedPredictions"] = ",".join(map(str, map(int,adjusted_expanded_pred)))
    result['PredictIntervalStart'] = end_time
    result['PredictIntervalEnd'] = end_time + nextKPrediction * timedelta(minutes=intervalInMins)
    result['PredictIntervalInMins'] = int(intervalInMins / expand_ratio)
    result["ModelName"] = "Hybrid"
    result["ModelVersion"] = "{0:05d}".format(3)
    result["Info"] = json.dumps(info)
    result["ErrorMessage"] = err_msg
    
    log_time_msg("finish: process row")
    return result

def get_validation_period(ts_data):
    """
    Evaluate how long should be the validation data set
    @param ts_data: the time series data
    @return: an integer represents the last k points for validation
    """
    total_sum = sum(ts_data)
    total_points = len(ts_data)
    per_point_sum = total_sum / total_points
    min_points = 5
    max_points = 48
    tolerance = 0.8
    vpts = 0
    # limit the validation points to be at least 5 data points and at most 48 data points
    if total_points <= min_points:
        return min_points

    cumSum = sum(ts_data[-min_points+1:])
    for i in range(min_points, min(max_points, total_points)):
        vpts = i
        cumSum += ts_data[-i]
        threshold = per_point_sum * i * tolerance
        if cumSum >= threshold:
            break

    return vpts
