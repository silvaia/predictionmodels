import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class LstmLongModel:
    def __init__(self, round_non_negative_int_func, lstm_cells_per_layer_used=100, loss_used='mean_squared_error', optimizer_used='adam', epochs_used=100, batch_size_used=5, random_seed_used=1, sample_fold_used=2):
        self.model_name = 'LSTM_Long_{}_Model'.format(sample_fold_used)
        self.round_non_negative_int_func = round_non_negative_int_func
        self.lstm_cells_per_layer_used = lstm_cells_per_layer_used
        self.loss_used = loss_used
        self.optimizer_used = optimizer_used
        self.epochs_used = epochs_used
        self.batch_size_used = batch_size_used
        self.model = None
        
        self.random_seed_used = random_seed_used
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.data_ori = None
        self.sample_fold_used = max(sample_fold_used, 2)
        self.sample_num = None
        self.feature_length_used = None
        
        return
    
    def fit(self, data):
        import random
        import tensorflow as tf
        from keras import backend
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Activation

        backend.clear_session()
        np.random.seed(self.random_seed_used)
        random.seed(self.random_seed_used)
        tf.random.set_seed(self.random_seed_used)

        self.data_ori = data
        if len(self.data_ori) <= 1:
            self.model = None
            return
        
        self.sample_num = max(len(self.data_ori)//self.sample_fold_used, 1)
        self.feature_length_used = len(self.data_ori) - self.sample_num
        
        if self.feature_length_used <= 0:
            self.sample_num -= 1
            self.feature_length_used = 1
            if self.sample_num <= 0:
                raise Exception('Insufficient data!')
        
        self.data = np.array(self.data_ori)
        self.data = self.data.astype(np.float64)
        self.data = self.scaler.fit_transform(self.data.reshape(-1, 1)).T[0]
        
        x_train, y_train = [], []
        for i in range(0, self.sample_num):
            feature_vec = []
            label_val = self.data[len(self.data) - self.sample_num + i]
            for j in range(0, self.feature_length_used):
                val = self.data[i+j]
                feature_vec.append(val)
            x_train.append(feature_vec)
            y_train.append(label_val)
            
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_cells_per_layer_used, input_shape=(1, self.feature_length_used)))
        self.model.add(Dense(1))
        self.model.compile(loss=self.loss_used, optimizer=self.optimizer_used)
        verbose_used = 0
        self.model.fit(x_train, y_train, epochs=self.epochs_used, batch_size=self.batch_size_used, verbose=verbose_used)
        
        return
        
    
    def predict(self, next_n_prediction):
        pred = []
        if self.model == None:
            if len(self.data_ori) <= 0:
                pred = [np.nan, ] * next_n_prediction
            else:
                pred = [self.data_ori[-1], ] * next_n_prediction
            return pred

        rest_prediction_num = next_n_prediction
        round_num = 0
        while rest_prediction_num > 0:
            x_test = []
            feature_vec = []
            for i in range(0, self.feature_length_used):
                val = self.data[self.sample_num+i+round_num]
                feature_vec.append(val)
            x_test.append(feature_vec)
            
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
            
            predict_test = self.model.predict(x_test)
            
            predict_test_scaled = predict_test
            predict_test_scaled = [item[0] for item in predict_test_scaled]
            predict_test_scaled = np.array(predict_test_scaled)
            predict_test_scaled = predict_test_scaled.astype(np.float64)
            
            
            predict_test = self.scaler.inverse_transform(predict_test)
            predict_test = [item[0] for item in predict_test]
            
            if predict_test[0] >= 0:
                self.data = np.append(self.data, predict_test_scaled)
                pred += predict_test
            # predict must be positive
            else:
                self.data = np.append(self.data, self.scaler.transform([[0]])[0])
                pred += [0]
            
            round_num += 1
            rest_prediction_num -= len(predict_test)
            
        pred = pred[0:next_n_prediction]
        pred_pre = np.array(pred)
        pred_pre = pred_pre.astype(np.float64)
        pred = list(pred_pre)
        pred = self.round_non_negative_int_func(pred)
        return pred