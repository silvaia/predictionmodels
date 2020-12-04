from LstmModel import LstmModel
from MaxNModel import MaxNModel
from ARIMAModel import ARIMAModel
from AdaptiveNN import AdaptiveNN
from LstmLongModel import LstmLongModel
from LinearFitModel import LinearFitModel
from AdaptiveMaxNModel import AdaptiveMaxNModel
from RandomArrivalModel import RandomArrivalModel
from NewRandomArrivalModel import NewRandomArrivalModel
from AdaptiveAverageNModel import AdaptiveAverageNModel

from scipy import stats
from functools import partial
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import os
import json
import datetime
import traceback
import numpy as np
import pandas as pd
import pmdarima as pmd
import matplotlib.pyplot as plt
import statsmodels


from keras.utils import to_categorical
import matplotlib.pyplot as plt

def plot(data, pred, model_name):
    #plt.title(model_name)
    #plt.figure(figsize = (8,5))
    #ax = plt.gca()
    #plt.plot(data.index, data.values, color = "blue", alpha=0.5, label = 'Actual data')
    plt.plot(data['PreciseTimeStamp'], data['e2EDurationInMilliseconds'], color = "blue", alpha=0.5, label = 'Actual data')
    
    #plt.plot(pred, data[int(0.3*(len(data))):], color = "red", alpha=0.5, label = 'Predicted data')
    plt.plot(pred[:][1], pred[:][0], color = "red", alpha=0.5, label = 'Predicted data')
    
    #plt.plot(data.index, pred, color = "red", alpha=0.5, label = 'Predicted data')
    plt.legend(loc = 'upper right')
    plt.ylabel("StartCount")
    plt.xlabel("TIMESTAMP")
    plt.show()

def round_non_negative_int(arr):
    return [round(p) if p > 0 else 0 for p in arr]

def processrow(row, end_time, nextKPrediction, expand_ratio, default_add_std_factor):
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
    print("processing row: ", row[0])
    ts_data_array = row["PreciseTimeStamp"] # an array of long
    region = row["region"] # string
    num_starts_per_region = row['num_starts_per_region']
    e2EDurationInMilliseconds = row['e2EDurationInMilliseconds']
    process_endtime = timedelta(days=5)
    expandDistribution = ["0:1"]
    abcSupportedRatio = 1

    err_msg = ""
    add_std_factor = default_add_std_factor
    intervalInMins = 15

    # get none zero history
    non_zero_data = trim_start_zeros(ts_data_array)
    non_zero_data = trim_start_zeros(row)
    #is_active = isActiveRecently(historical_data=ts_data_array, recent_days=7, interval_in_mins=intervalInMins)

    info = {}
    model_score = []
    stderr = [0] * nextKPrediction
    pred = ts_data_array.iloc[-1] * nextKPrediction # predict the last demand number by default
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

    # run simple predictors and pick up a best model to use
    simple_model_pred = run_simple_models(short_term_train_data, recent_n_validation, points_per_day, total_points, add_std_factor, "general")

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


    info["model_scores"] = [ms[1] for ms in model_score]
    info["InHourVmSplit"] = expandDistribution
    expanded_pred = expand(pred, expand_ratio, expandDistribution)
    result = {}
    result["Region"]                    = region
    result["Ts_ProcessedDataEndTime"]   = process_endtime
    result["Ts_HistoryValues"]          = ",".join(map(str, map(int,ts_data_array)))
    result["Ts_IntervalInMins"]         = int(intervalInMins)
    result["Ts_LatestDemand"]           = int(ts_data_array.iloc[-1])
    result["Ts_Next_Forecast"]          = int(pred[0])
    result["Ts_NextN_Forecast"]         = ",".join(map(str, map(int,pred)))
    result["Ts_StdDev_History"]         = float(np.std(non_zero_data[-points_per_day:])/expand_ratio)
    result["Ts_SumForecast"]            = sum(map(int,pred))
    result["Next_Prediction"]           = int(expanded_pred[0])
    result["NextN_Predictions"]         = ",".join(map(str, map(int,expanded_pred)))
    result["Info"]                      = json.dumps(info)
    result["ErrorMessage"]              = err_msg
    
    print("finish: process row")
    print("Scores: ", info["model_scores"])
    return result
def run_simple_models(short_term_train_data, recent_n_validation, points_per_day, total_points, add_std_factor, row_type="general"):
    print("run simple models")
    model_results = []
    short_term_models = [
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

    for m in short_term_models:
        result = {}
        try:
            print("run simple models: fit %s" % (m.model_name))
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
            print("finish: run simple models: fit %s" % (m.model_name))

        except Exception as ex:
            err_msg = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            result["successful"] = False
            result["prediction"] = None
            result["model"] = m
            result["model_name"] = m.model_name
            result["error"] = err_msg
            model_results.append(result)

    print("finish: run simple models")
    return model_results
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
            print("Error parsing InHourVmSplit column {}!".format(expandDistribution))

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




data = pd.read_csv('duration_by_region_sub3d01cf6f.csv',
                   #index_col=1, 
                   parse_dates=True, squeeze=True)

le = OneHotEncoder()
data['region'] = le.fit_transform(np.asarray(data['region']).reshape(-1,1))
#data = data.drop('region', axis=1)


aggregation_interval_in_mins    = int(60)
predict_future_in_days          = int(7)
nextNPredictions                = int(1440 / aggregation_interval_in_mins * predict_future_in_days)
expand_ratio                    = 4 # this will split hourly data into 15 minute intervals for predictions
add_std_factor                  = 1.0

date_time_str = '2020-12-03 16:00:00.00'
datetimeobj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

train = data[:int(0.7*(len(data)))]
test = data[int(0.7*(len(data))):]

#train = np.asarray(train).astype('float32')
#test = np.asarray(test).astype('float32')

train_X = train.drop('e2EDurationInMilliseconds', axis=1)
train_Y = train['e2EDurationInMilliseconds']

test_X = test.drop('e2EDurationInMilliseconds', axis=1)
test_Y = test['e2EDurationInMilliseconds']

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_X, test_Y))

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


#model = statsmodels.tsa.statespace.varmax.VARMAX(np.asarray(train), order=(1, 0), measurement_error=False, error_cov_type='unstructured', enforce_stationarity=True, enforce_invertibility=True)

#model = model.fit()
#pred = model.predict()

#acc = eval_accuracy(test, pred, 0.5, "VARMAX")
#print("Accuracy: ", acc)

print(pred[1][:])
print("__________________")
print(pred[0][:])
#plot(data, pred, "LSTMLongModel")

#model = LstmLongModel(round_non_negative_int)
#model = ARIMAModel(round_non_negative_int, (4,1,1), 1)

#result = processrow(data, datetimeobj, nextNPredictions, expand_ratio, add_std_factor)

