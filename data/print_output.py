import os 
import time
import pandas as pd # pip install pandas
import numpy as np # pip install numpy
from sklearn.metrics import * # pip install scikit-learn

def logging_time(original_fn):
    """
    original_fn : 시간 측정할 함수
    decorator로 사용할 수 있음
    예시)
    @logging_time
    def model_run():
        ols = LinearRegression()
        ols.fit(x_train_ohe, y_train)
        y_pred = ols.predict(x_test_ohe)
        return y_pred
    출력 예시) Execution_Time[model_run]: 0.04 sec 
              array([]) # y_pred
    [출처] : https://kimjingo.tistory.com/85 
    """
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"Execution_Time[{original_fn.__name__}]: {end_time-start_time:.2f} sec")
        return result
    return wrapper_fn

def reg_eval(y_true, y_pred): # 회귀 성능 평가지표 출력
    """
    y_true (array-like) : 실제값 리스트
    y_pred (array-like) : 예측값 리스트
    """
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse:.2f}")
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.2f}")
    r2 = r2_score(y_true, y_pred)
    print(f"R^2: {r2:.2f}")

def class_eval(y_true, y_pred): # 분류 성능 평가지표 출력
    """
    y_true (array-like) : 실제값 리스트
    y_pred (array-like) : 예측값 리스트
    """
    precision = precision_score(y_true, y_pred)
    print(f"재현율: {precision:.2f}")
    recall = recall_score(y_true, y_pred)
    print(f"정밀도: {recall:.2f}")
    f1 = f1_score(y_true, y_pred)
    print(f"f1: {f1:.2f}")
    # cm = confusion_matrix(y_true, y_pred)
    # print("오차 행렬: ")
    # print(cm)