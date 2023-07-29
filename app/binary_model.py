import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import *
import joblib
import time

# 모델 로드
xgboost_model = joblib.load("models/binary_model.pkl")

# 입력된 csv file로 val set 생성 및 모델 평가
def evaluate_model(validation_file_path, target_class=1):
    
    # 밸리데이션 set을 csv 파일로 불러오기
    validation_data = pd.read_csv(validation_file_path)

    # Scaler 객체를 불러오기
    scaler = joblib.load('models/binary_scaler.pkl')

    # target_class 컬럼 분리
    validation_target = validation_data["target_class"].values
    validation_features = validation_data.drop(columns=["target_class"]).values

    # transform
    validation_features = scaler.transform(validation_features)
    #start time
    start_time = time.time()
    # validation_set으로 모델 평가
    predictions = xgboost_model.predict(validation_features)
    end_time = time.time()
    report = classification_report(validation_target, predictions, digits=3, output_dict=True)
    
    #소요 시간 계산
    Time_taken = end_time - start_time

    # 원하는 클래스(target_class)에 대한 결과 추출
    target_metrics = report[str(target_class)]

    # F1-score, 재현율(recall) 값 추출
    f1_score = target_metrics["f1-score"]
    precision = target_metrics["precision"]
    recall = target_metrics["recall"]

    return f1_score, precision, recall, Time_taken