import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoTBATS
from mlforecast import MLForecast
from statsmodels.tsa.deterministic import Fourier
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import lightgbm as lgb
from multiprocessing import Process
from mlforecast.utils import PredictionIntervals
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import TFT
import re
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from neuralforecast.models import LSTM

import torch
import gc

gc.collect()
torch.cuda.empty_cache()

TRAINING_SET = './dataset/cleaned/auth_order_train.csv'
TESTING_SET = './dataset/cleaned/auth_order_test.csv'
WINDOW_SIZE = 1680
CATEGORICALS = ['holiday']

class DataLoader:
    def __init__(self, df_1, df_2):
        self.full_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
        self.full_df['ds'] = pd.to_datetime(self.full_df['ds'])
        for col in CATEGORICALS:
            self.full_df[col] = self.full_df[col].astype('category')
    def LoadMSTLA(self):
        prepped = _prep_df(self.full_df, target_feature='y_log', exo = ['holiday']).reset_index(drop=True)
        return prepped

    def LoadSARIMAX(self):
        prepped = _prep_df(self.full_df, target_feature='y_detrended', exo = ['holiday']).reset_index(drop=True)
        exo = _fourier_exo(prepped).reset_index(drop=True)
        
        df_X = pd.concat([prepped, exo], 
                            axis=1)

        return df_X
    
    def LoadTBATS(self):
        prepped = _prep_df(self.full_df).reset_index(drop=True)
        return prepped
    
    def LoadLightGBM(self):
        prepped = _prep_df(self.full_df, target_feature='y_log', exo = ['holiday']).reset_index(drop=True)
        f_exo = _fourier_exo(prepped, [48, 336]).reset_index(drop=True)

        df_X = pd.concat([prepped, f_exo], 
                            axis=1)

        return df_X
    
    def loadXGB(self):
        prepped = _prep_df(self.full_df, target_feature='y_log', exo = ['holiday']).reset_index(drop=True)
        f_exo = _fourier_exo(prepped, [48, 336]).reset_index(drop=True)

        df_X = pd.concat([prepped, f_exo], 
                            axis=1)

        return df_X

    def loadCatBoost(self):
        prepped = _prep_df(self.full_df, target_feature='y_log', exo = ['holiday']).reset_index(drop=True)
        f_exo = _fourier_exo(prepped, [48, 336]).reset_index(drop=True)

        df_X = pd.concat([prepped, f_exo], 
                            axis=1)

        return df_X

    def LoadNHITS(self):
        prepped = _prep_df(self.full_df, target_feature='y_detrended', exo = ['holiday', 'y_log']).reset_index(drop=True)
        f_exo = _fourier_exo(prepped, [48, 336]).reset_index(drop=True)

        df_X = pd.concat([prepped, f_exo], 
                            axis=1)

        return df_X
    
    def LoadLSTM(self):
        prepped = _prep_df(self.full_df, target_feature='y_detrended', exo = ['holiday', 'y_log']).reset_index(drop=True)
        f_exo = _fourier_exo(prepped, [48, 336]).reset_index(drop=True)

        df_X = pd.concat([prepped, f_exo], 
                            axis=1)

        return df_X
    
    def LoadTFT(self):
        prepped = _prep_df(self.full_df, target_feature='y_detrended', exo = ['holiday', 'y_log']).reset_index(drop=True)
        f_exo = _fourier_exo(prepped, [48, 336]).reset_index(drop=True)

        df_X = pd.concat([prepped, f_exo], 
                            axis=1)

        return df_X



def _prep_df(df: pd.DataFrame, target_feature : str = '', exo : list = []) -> pd.DataFrame:
    new_df = df.copy()
    if target_feature:
        new_df['y'] = new_df[target_feature]
    new_df = new_df[['unique_id', 'ds', 'y', *exo]]
    return new_df

def _fourier_exo(df : pd.DataFrame, terms = [336]) -> pd.DataFrame:
    terms_df = []
    for term in terms:
        terms_df.append(Fourier(period=term, order=2).in_sample(df["ds"]).add_prefix("w_"))
    return pd.concat(terms_df, axis=1)

PARAMS = {
    'h' : 1,
    'step_size' : 53,
    'input_size' : WINDOW_SIZE,
    'refit' : True,
    'level' : [50, 80, 95]
}
# may need to adjust terms
def cv_mstla(df: pd.DataFrame):
    sf = StatsForecast(
        models=[
            MSTL(season_length=[48, 336],
                 trend_forecaster=AutoARIMA(approximation=True, seasonal=False))
                 ],
        freq='30min', n_jobs=-1
    ) # feel free to remove season from AutoARIMA 
    cv_df = sf.cross_validation(
                df = df,
                n_windows = 101,
                **PARAMS)
    return cv_df

def cv_sarimax(df: pd.DataFrame):
    sf = StatsForecast(
        models=[AutoARIMA(season_length=48, d = 0, approximation=True)],
        freq='30min',
        n_jobs=-1
    )
    cv_df = sf.cross_validation(
                df = df,
                n_windows = 101,
                **PARAMS)
    return cv_df

def cv_tbats(df: pd.DataFrame):
    sf = StatsForecast(
        models=[AutoTBATS(season_length=[48, 336])],
        freq='30min',
        n_jobs=-1
    )
    cv_df = sf.cross_validation(
                df = df,
                h = 1,
                step_size = 53,
                n_windows = 101,
                input_size = WINDOW_SIZE,
                refit = True,
                level=[50, 80, 95])
    return cv_df

def cv_lightGBM(df: pd.DataFrame):
    df.columns = [re.sub(r'[\[\]\s\:"\',]+', '_', str(col)) for col in df.columns]
    mf = MLForecast(
        models=[lgb.LGBMRegressor(n_estimators=100, verbosity=-1)],
        freq='30min',
        lags = [1, 2, 48, 336]
    )
    cv_df = mf.cross_validation(
                df = df,
                n_windows = 101,
                static_features=[],
                prediction_intervals=PredictionIntervals(n_windows=5),
                **PARAMS)
    return cv_df

def cv_XGB(df: pd.DataFrame):
    df.columns = [re.sub(r'[\[\]\s\:"\',]+', '_', str(col)) for col in df.columns]
    mf = MLForecast(
        models=[XGBRegressor(n_estimators=100, learning_rate=0.1, enable_categorical=True,)],
        freq='30min',
        lags = [1, 2, 48, 336]
    )
    cv_df = mf.cross_validation(
                df = df,
                n_windows = 101,
                static_features=[],
                prediction_intervals=PredictionIntervals(n_windows=5),
                **PARAMS)
    return cv_df

def cv_CatBoost(df: pd.DataFrame):
    df.columns = [re.sub(r'[\[\]\s\:"\',]+', '_', str(col)) for col in df.columns]
    mf = MLForecast(
        models=[CatBoostRegressor(n_estimators=100, 
                                  learning_rate=0.1, 
                                  cat_features=CATEGORICALS, 
                                  verbose=0)],
        freq='30min',
        lags = [1, 2, 48, 336]
    )
    cv_df = mf.cross_validation(
                df = df,
                n_windows = 101,
                static_features=[],
                prediction_intervals=PredictionIntervals(n_windows=5),
                **PARAMS)
    return cv_df


def cv_NHITS(df: pd.DataFrame):
    df.columns = [re.sub(r'[\[\]\s\:"\',]+', '_', str(col)) for col in df.columns]
    futures = ['holiday', 'w_sin(1_48)',
       'w_cos(1_48)', 'w_sin(2_48)', 'w_cos(2_48)', 'w_sin(1_336)',
       'w_cos(1_336)', 'w_sin(2_336)', 'w_cos(2_336)']
    quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]

    models = [
        NHITS(
            h=1,
            input_size=336,
            loss=MQLoss(quantiles=quantiles),
            futr_exog_list=futures,
            max_steps=2000,
            early_stop_patience_steps=3,
            learning_rate=1e-4,
            val_check_steps=100,
            n_freq_downsample=[24, 12, 1],
            start_padding_enabled = True
        )
    ]
    nf = NeuralForecast(models=models, freq='30min')

    cv_df = nf.cross_validation(
        df=df,
        step_size=53,
        n_windows=101,
        val_size=336
    )    
    return cv_df

def cv_LSTM(df: pd.DataFrame):
    df.columns = [re.sub(r'[\[\]\s\:"\',]+', '_', str(col)) for col in df.columns]
    futures = ['holiday', 'w_sin(1_48)',
       'w_cos(1_48)', 'w_sin(2_48)', 'w_cos(2_48)', 'w_sin(1_336)',
       'w_cos(1_336)', 'w_sin(2_336)', 'w_cos(2_336)']
    quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    models = [
        LSTM(
            h=1,
            input_size=336,
            loss=MQLoss(quantiles=quantiles),
            futr_exog_list=futures,
            max_steps=2000,
            early_stop_patience_steps=2,
            learning_rate=1e-4,
            val_check_steps=500,
            accelerator='auto'
        )
    ]
    nf = NeuralForecast(models=models, freq='30min')

    cv_df = nf.cross_validation(
        df=df,
        step_size=53,
        n_windows=101,
        val_size=336
    )    
    return cv_df

def cv_TFT(df: pd.DataFrame):
    df.columns = [re.sub(r'[\[\]\s\:"\',]+', '_', str(col)) for col in df.columns]
    futures = ['holiday', 'w_sin(1_48)',
       'w_cos(1_48)', 'w_sin(2_48)', 'w_cos(2_48)', 'w_sin(1_336)',
       'w_cos(1_336)', 'w_sin(2_336)', 'w_cos(2_336)']
    quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    models = [
        TFT(
            h=1,
            input_size=336,
            loss=MQLoss(quantiles=quantiles),
            futr_exog_list=futures,
            max_steps=2000,
            early_stop_patience_steps=2,
            learning_rate=1e-4,
            val_check_steps=500,
            accelerator='auto',
            hidden_size=16,
            n_head=2,
            scaler_type='standard'
        )
    ]
    nf = NeuralForecast(models=models, freq='30min')

    cv_df = nf.cross_validation(
        df=df,
        step_size=53,
        n_windows=101,
        val_size=336,
        batch_size=16
    )    
    return cv_df

def run_mstla():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_mstla(dloader.LoadMSTLA())
    cv.to_csv(f'./baseline/cv_mstla.csv', index=False)
    print("MSTL ARIMA completed")

def run_sarimax():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_sarimax(dloader.LoadSARIMAX())
    cv.to_csv(f'./baseline/cv_sarimax.csv', index=False)
    print("SARIMAX completed")

def run_tbats():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_tbats(dloader.LoadTBATS())
    cv.to_csv(f'./baseline/cv_tbats.csv', index=False)
    print("TBATS completed")

def run_lightgbm():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_lightGBM(dloader.LoadLightGBM())
    cv.to_csv('./baseline/cv_lightgbm.csv', index=False)
    print("LightGBM completed")

def run_xgb():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_XGB(dloader.loadXGB())
    cv.to_csv('./baseline/cv_xgb.csv', index=False)
    print("XGBoost completed")

def run_catboost():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_CatBoost(dloader.loadCatBoost())
    cv.to_csv('./baseline/cv_catboost.csv', index=False)
    print("CatBoost completed")

def run_nhits():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_NHITS(dloader.LoadNHITS())
    cv.to_csv('./baseline/cv_nhits.csv', index=False)
    print("NHITS completed")

def run_LSTM():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_LSTM(dloader.LoadLSTM())
    cv.to_csv('./baseline/cv_lstm.csv', index=False)
    print("LSTM completed")

def run_TFT():
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_TFT(dloader.LoadTFT())
    cv.to_csv('./baseline/cv_tft.csv', index=False)
    print("TFT completed")

if __name__ == '__main__':
    run_sarimax()
    run_nhits()
    run_LSTM()
    run_TFT()

    tasks = [
        Process(target=run_mstla),
        Process(target=run_sarimax),
        Process(target=run_tbats),
        Process(target=run_lightgbm),
        Process(target=run_xgb),
        Process(target=run_catboost),
    ]
    
    for task in tasks:
        task.start()
    
    for task in tasks:
        task.join()
    
    print("All forecasts completed")
    
"""
# LightGBM
dloader = DataLoader()
lightgbmloader = dloader.LoadLightGBM()
cv = cv_lightGBM(*lightgbmloader)
cv.to_csv('cv_lightgbm.csv', index=False)

def forecast():
    # MSTL ARIMA
    dloader = DataLoader(pd.read_csv(TRAINING_SET), pd.read_csv(TESTING_SET))
    cv = cv_mstla(dloader.LoadMSTLA())
    cv.to_csv(f'./baseline/cv_mstla.csv', index=False)

    # SARIMAX 
    cv = cv_sarimax(dloader.LoadSARIMAX())
    cv.to_csv(f'./baseline/cv_sarimax.csv', index=False)

    # TBATS
    cv = cv_tbats(dloader.LoadTBATS()) 
    cv.to_csv(f'./baseline/cv_tbats.csv', index=False)


"""