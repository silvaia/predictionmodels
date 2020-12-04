from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa as tsa
import pmdarima as pmd

#vector arima for multivariate data

class ARIMAModel:
    def __init__(self, round_non_negative_int_func, arima_order = (4, 1, 1), add_std_factor = 0):
        self.arima_order = arima_order
        self.add_std_factor = add_std_factor
        self.model_name = "ARIMA_{}_Model".format(self.arima_order)
        self.round_non_negative_int_func = round_non_negative_int_func

    def fit(self, data):
        model = tsa.statespace.varmax.VARMAX(data, order=(2,0), trend='n')
        res = model.fit(maxiter=1000, disp=False)
        print(res.summary())
        #model = pmd.auto_arima(data, order=self.arima_order, start_p=0,d = 1,start_q=0,
        #                      test="adf", supress_warnings = True)
        self.model_fit = model.fit()
        return self

    def predict(self, next_n_prediction):
        return self.model_fit.predict(n_periods = next_n_prediction)
        #return self.model_fit.forecast(steps=next_n_prediction)