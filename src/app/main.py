import uvicorn #pip install
import time #소요시간 출력
from fastapi import FastAPI, Request, Form, File, UploadFile #pip install
from fastapi.responses import HTMLResponse #pip install
from fastapi.templating import Jinja2Templates #pip install
from fastapi.staticfiles import StaticFiles #pip install
from pydantic import BaseModel 
import joblib
import xgboost as xgb
#--------------------------------------
import os
import regression_model 
import binary_model
import multi_model
#---------------------------------------
import numpy as np
from keras.optimizers import RMSprop
from sklearn.metrics import *
from tensorflow.keras.models import model_from_json
#---------------------------------------
#app 객체 선포
app = FastAPI()
# 이미지, 스타일시트 적용
app.mount("/static", StaticFiles(directory="static"), name="static")
# 템플릿 로드
templates = Jinja2Templates(directory="templates")
#time 반올림
def round_prediction_time(time_taken: float, decimal_places: int) -> float:
    return round(time_taken, decimal_places)
#------------------------------------------------------------------------------------------

#기본 페이지 라우트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#-----------------------------------------------------------------------------------------
#Abalone페이지 성능 검증 기능 구현
# Abalone 성능 페이지 라우트
@app.get("/abalone_p", response_class=HTMLResponse)
async def read_abalone(request: Request):
    return templates.TemplateResponse("abalone_perform_input.html", {"request": request})

# Abalone validation 페이지 라우트
@app.post("/abalone_perform_evaluate/")
async def evaluate_validation_set(request: Request, file: UploadFile = File(...)):
    # 업로드된 Csv file 임시 저장
    with open("validation_set.csv", "wb") as f:
        contents = await file.read()
        f.write(contents)

    # regression_model.py에서 evaluate_model 함수 호출
    mse, Time_taken = regression_model.evaluate_model("validation_set.csv")

    # # 임시 검증 세트 파일 삭제
    os.remove("validation_set.csv")

    # HTML 템플릿에 mse 값을 넘겨 웹페이지에 출력
    return templates.TemplateResponse("abalone_perform_evaluate.html", {"request": request, "mse": np.round(mse,2), "Time_taken": np.round(Time_taken,2)})

#----------------------------------------------------------------------------------------
#Abalone페이지 예측 기능 구현
# Abalone 페이지 라우트
@app.get("/abalone", response_class=HTMLResponse)
async def read_abalone(request: Request):
    return templates.TemplateResponse("abalone.html", {"request": request})

# 예측 결과를 담을 Pydantic 모델 정의
class AbalonePredictionResult(BaseModel):
    rings: float
    time_taken: float
    accuracy: float
    sex: str
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float

# 예측 수행 함수
def perform_abalone_prediction(sex: str, 
                               length: float, 
                               diameter: float, 
                               height: float,
                               whole_weight: float, 
                               shucked_weight: float, 
                               viscera_weight: float,
                               shell_weight: float) -> AbalonePredictionResult:
    
     # 모델 로드
    json_file = open('models/regression_m.json', 'rb')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/regression_ml.h5")
    loaded_model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.001))

    # 입력 데이터를 변환
    input_data = [[length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]]
    
    # 예측 시간 측정 시작
    start_time = time.time()

    # 모델을 사용하여 예측 수행
    rings = loaded_model.predict(input_data)[0]
    rounded_rings = np.round(rings).astype(int)

    # 예측 시간 측정 종료 및 계산
    end_time = time.time()
    time_taken = end_time - start_time
    rounded_time_taken = round_prediction_time(time_taken, 2)

    accuracy = 0.84

    return AbalonePredictionResult(
        rings=rounded_rings,
        time_taken=rounded_time_taken,
        accuracy=accuracy,
        sex=sex,
        length=length,
        diameter=diameter,
        height=height,
        whole_weight=whole_weight,
        shucked_weight=shucked_weight,
        viscera_weight=viscera_weight,
        shell_weight=shell_weight
    )

# Abalone 예측 페이지 라우트
@app.post("/predict_abalone", response_class=HTMLResponse)
async def predict_abalone(request: Request,
                          sex: str = Form(...),
                          length: float = Form(...),
                          diameter: float = Form(...),
                          height: float = Form(...),
                          whole_weight: float = Form(...),
                          shucked_weight: float = Form(...),
                          viscera_weight: float = Form(...),
                          shell_weight: float = Form(...)):
    prediction = perform_abalone_prediction(sex,
                                            length,
                                            diameter,
                                            height,
                                            whole_weight,
                                            shucked_weight,
                                            viscera_weight,
                                            shell_weight)
    
    return templates.TemplateResponse("abalone_predict.html", {"request": request, "prediction": prediction})


#------------------------------------------------------------------------------------------------
#Pulsar 페이지 성능 검증 기능 구현
# Pulsar 성능 페이지 라우트
@app.get("/pulsars_p", response_class=HTMLResponse)
async def read_abalone(request: Request):
    return templates.TemplateResponse("pulsars_perform_input.html", {"request": request})

# Pulsar validation 페이지 라우트
@app.post("/pulsars_perform_evaluate/")
async def evaluate_validation_set(request: Request, file: UploadFile = File(...)):
    # 업로드된 Csv file 임시 저장
    with open("validation_set.csv", "wb") as f:
        contents = await file.read()
        f.write(contents)

    # regression_model.py에서 evaluate_model 함수 호출
    f1_score, recall, Time_taken = binary_model.evaluate_model("validation_set.csv")

    # # 임시 검증 세트 파일 삭제
    os.remove("validation_set.csv")

    # HTML 템플릿에 mse와 mae 값을 넘겨 웹페이지에 출력
    return templates.TemplateResponse(
        "pulsars_perform_evaluate.html", {"request": request,
                                          "f1_score": np.round(f1_score,3),
                                          "recall":np.round(recall,3),
                                          "Time_taken":np.round(recall,2)})

#----------------------------------------------------------------------------------------
#Pulsars 기능 구현
# Pulsars 페이지 라우트
@app.get("/pulsars", response_class=HTMLResponse)
async def read_pulsars(request: Request):
    return templates.TemplateResponse("pulsars.html", {"request": request})

# 예측 결과를 담을 Pydantic 모델 정의
class PulsarPredictionResult(BaseModel):
    target: str
    time_taken: float
    f1_score: float
    mean_profile: float
    std_profile: float
    kurtosis_profile: float
    skewness_profile: float
    mean_dmsnr: float
    std_dmsnr: float
    kurtosis_dmsnr: float
    skewness_dmsnr: float

# 예측 수행 함수
def perform_pulsar_prediction(mean_profile: float,
                               std_profile: float,
                               kurtosis_profile: float,
                               skewness_profile: float,
                               mean_dmsnr: float,
                               std_dmsnr: float, 
                               kurtosis_dmsnr: float,
                               skewness_dmsnr: float) -> PulsarPredictionResult:
    
    # model 로드
    loaded_model = xgb.Booster()
    loaded_model.load_model('models/binary_model.xgb')

    # 입력 데이터를 DMatrix 형식으로 변환
    input_data = np.array([[mean_profile, std_profile, kurtosis_profile, skewness_profile, mean_dmsnr, std_dmsnr, kurtosis_dmsnr, skewness_dmsnr]])
    dinput = xgb.DMatrix(data=input_data)

    # 예측 시간 측정 시작
    start_time = time.time()

    # 모델을 사용하여 예측 수행
    predictions = loaded_model.predict(dinput)

    # 결과값에 따라 클래스를 할당
    target = "중성자별입니다" if predictions[0] >= 0.5 else "중성자별이 아닙니다"

    # 예측 시간 측정 종료 및 계산
    end_time = time.time()
    time_taken = end_time - start_time
    rounded_time_taken = round_prediction_time(time_taken, 2)

    f1_score = 0.876
    
    return PulsarPredictionResult(
        target=target, #중성자 별 여부
        time_taken=rounded_time_taken, #time_taken
        f1_score=f1_score,
        mean_profile=mean_profile,
        std_profile=std_profile,
        kurtosis_profile=kurtosis_profile,
        skewness_profile=skewness_profile,
        mean_dmsnr=mean_dmsnr,
        std_dmsnr=std_dmsnr,
        kurtosis_dmsnr=kurtosis_dmsnr,
        skewness_dmsnr=skewness_dmsnr
    )

# Pulsars 예측 페이지 라우트
@app.post("/predict_pulsars", response_class=HTMLResponse)
async def predict_pulsar(request: Request,
                          mean_profile: float = Form(...),
                          std_profile: float = Form(...),
                          kurtosis_profile: float = Form(...),
                          skewness_profile: float = Form(...),
                          mean_dmsnr: float = Form(...),
                          std_dmsnr: float = Form(...),
                          kurtosis_dmsnr: float = Form(...),
                          skewness_dmsnr: float = Form(...)):
    prediction = perform_pulsar_prediction(mean_profile,
                                            std_profile,
                                            kurtosis_profile,
                                            skewness_profile,
                                            mean_dmsnr,
                                            std_dmsnr,
                                            kurtosis_dmsnr,
                                            skewness_dmsnr)
    
    return templates.TemplateResponse("pulsars_predict.html", {"request": request, "prediction": prediction})


#-----------------------------------------------------------------------------
#steel 페이지 성능 검증 기능 구현
# steel 성능 페이지 라우트
@app.get("/steel_p", response_class=HTMLResponse)
async def read_abalone(request: Request):
    content = "steel 페이지입니다."
    return templates.TemplateResponse("steel_perform_input.html", {"request": request, "content": content})

# steel validation 페이지 라우트
@app.post("/steel_perform_evaluate/")
async def evaluate_validation_set(request: Request, file: UploadFile = File(...)):
    # 업로드된 Csv file 임시 저장
    with open("validation_set.csv", "wb") as f:
        contents = await file.read()
        f.write(contents)

    # regression_model.py에서 evaluate_model 함수 호출
    loss, accuracy, Time_taken = multi_model.evaluate_model("validation_set.csv")

    # # 임시 검증 세트 파일 삭제
    os.remove("validation_set.csv")

    # HTML 템플릿에 mse와 mae 값을 넘겨 웹페이지에 출력
    return templates.TemplateResponse("steel_perform_evaluate.html",
                                      {"request": request,
                                       "loss": np.round(loss,3),
                                       "accuracy": np.round(accuracy,3),
                                       "Time_taken":np.round(Time_taken,2)})

#----------------------------------------------------------------------------------------
#Steel 기능 구현
# Steel 페이지 라우트
@app.get("/steel", response_class=HTMLResponse)
async def read_steel(request: Request):
    content = "Steel 페이지입니다."
    return templates.TemplateResponse("steel.html", {"request": request, "content": content})


# 예측 결과를 담을 Pydantic 모델 정의
class SteelPredictionResult(BaseModel):
    target: str
    time_taken: float
    accuracy: float
    X_Minimum : float
    X_Maximum : float
    Y_Minimum : float
    Y_Maximum : float
    Pixels_Areas : float
    X_Perimeter : float
    Y_Perimeter : float
    Sum_of_Luminosity : float
    Minimum_of_Luminosity : float
    Maximum_of_Luminosity : float
    Length_of_Conveyer : float
    TypeOfSteel_A300 : float
    TypeOfSteel_A400 : float
    Steel_Plate_Thickness : float
    Edges_Index : float
    Empty_Index : float
    Square_Index : float
    Outside_X_Index : float
    Edges_X_Index : float
    Edges_Y_Index : float
    Outside_Global_Index : float
    LogOfAreas : float
    Log_X_Index : float
    Log_Y_Index : float
    Orientation_Index : float
    Luminosity_Index : float
    SigmoidOfAreas	 : float

# 예측 수행 함수
def perform_steel_prediction(
    X_Minimum : float,
    X_Maximum : float,
    Y_Minimum : float,
    Y_Maximum : float,
    Pixels_Areas : float,
    X_Perimeter : float,
    Y_Perimeter : float,
    Sum_of_Luminosity : float,
    Minimum_of_Luminosity : float,
    Maximum_of_Luminosity : float,
    Length_of_Conveyer : float,
    TypeOfSteel_A300 : float,
    TypeOfSteel_A400 : float,
    Steel_Plate_Thickness : float,
    Edges_Index : float,
    Empty_Index : float,
    Square_Index : float,
    Outside_X_Index : float,
    Edges_X_Index : float,
    Edges_Y_Index : float,
    Outside_Global_Index : float,
    LogOfAreas : float,
    Log_X_Index : float,
    Log_Y_Index : float,
    Orientation_Index : float,
    Luminosity_Index : float,
    SigmoidOfAreas	 : float) -> SteelPredictionResult:
    
    # 피클된 모델 로드
    json_file = open('models/multi_class_m.json', 'rb')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/multi_class_ml.h5")
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    # 입력 데이터를 변환
    input_data = [X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter, Sum_of_Luminosity,
                  Minimum_of_Luminosity, Maximum_of_Luminosity, Length_of_Conveyer,
                  Steel_Plate_Thickness, Edges_Index, Empty_Index, Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index,
                  Outside_Global_Index, LogOfAreas, Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index, SigmoidOfAreas]
    input_data = np.array(input_data)  # 입력 데이터를 numpy 배열로 변환
    input_data = np.expand_dims(input_data, axis=0)  # (1, n) 형태로 변환

    # 예측 시간 측정 시작
    start_time = time.time()

    # 모델을 사용하여 예측 수행
    target = loaded_model.predict(input_data)[0]

     # 결과 리스트에서 1이 존재하는 인덱스를 찾아 해당 인덱스에 해당하는 값을 출력
    target_labels = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    predicted_fault_index = np.where(target == 1)[0][0]
    predicted_fault = target_labels[predicted_fault_index]

    # 예측 시간 측정 종료 및 계산
    end_time = time.time()
    time_taken = end_time - start_time
    rounded_time_taken = round_prediction_time(time_taken, 2)

    accuracy = 0.697
    
    return SteelPredictionResult(
        target =predicted_fault,
        time_taken =rounded_time_taken,
        accuracy =accuracy, 
        X_Minimum =X_Minimum,
        X_Maximum =X_Maximum,
        Y_Maximum =Y_Maximum,
        Y_Minimum =Y_Minimum,
        Pixels_Areas =Pixels_Areas, 
        X_Perimeter =X_Perimeter, 
        Y_Perimeter =Y_Perimeter,
        Sum_of_Luminosity =Sum_of_Luminosity,
        Minimum_of_Luminosity =Minimum_of_Luminosity,
        Maximum_of_Luminosity =Maximum_of_Luminosity,
        Length_of_Conveyer =Length_of_Conveyer,
        TypeOfSteel_A300 =TypeOfSteel_A300,
        TypeOfSteel_A400 =TypeOfSteel_A400,
        Steel_Plate_Thickness =Steel_Plate_Thickness,
        Edges_Index =Edges_Index,
        Empty_Index =Empty_Index,
        Square_Index =Square_Index,
        Outside_X_Index =Outside_X_Index,
        Edges_X_Index =Edges_X_Index,
        Edges_Y_Index =Edges_Y_Index,
        Outside_Global_Index =Outside_Global_Index,
        LogOfAreas =LogOfAreas,
        Log_X_Index =Log_X_Index,
        Log_Y_Index =Log_Y_Index,
        Orientation_Index =Orientation_Index,
        Luminosity_Index =Luminosity_Index,
        SigmoidOfAreas =SigmoidOfAreas
        )

# Pulsars 예측 페이지 라우트
@app.post("/predict_steel", response_class=HTMLResponse)
async def predict_steel(request: Request,
                          X_Minimum : float= Form(...),
                          X_Maximum : float= Form(...),
                          Y_Minimum : float= Form(...),
                          Y_Maximum : float= Form(...),
                          Pixels_Areas : float= Form(...),
                          X_Perimeter : float= Form(...),
                          Y_Perimeter : float= Form(...),
                          Sum_of_Luminosity : float= Form(...),
                          Minimum_of_Luminosity : float= Form(...),
                          Maximum_of_Luminosity : float= Form(...),
                          Length_of_Conveyer : float= Form(...),
                          TypeOfSteel_A300 : float= Form(...),
                          TypeOfSteel_A400 : float= Form(...),
                          Steel_Plate_Thickness : float= Form(...),
                          Edges_Index : float= Form(...),
                          Empty_Index : float= Form(...),
                          Square_Index : float= Form(...),
                          Outside_X_Index : float= Form(...),
                          Edges_X_Index : float= Form(...),
                          Edges_Y_Index : float= Form(...),
                          Outside_Global_Index : float= Form(...),
                          LogOfAreas : float= Form(...),
                          Log_X_Index : float= Form(...),
                          Log_Y_Index : float= Form(...),
                          Orientation_Index : float= Form(...),
                          Luminosity_Index : float= Form(...),
                          SigmoidOfAreas : float= Form(...)):
    prediction = perform_steel_prediction(X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter, Sum_of_Luminosity,
                  Minimum_of_Luminosity, Maximum_of_Luminosity, Length_of_Conveyer, TypeOfSteel_A300, TypeOfSteel_A400,
                  Steel_Plate_Thickness, Edges_Index, Empty_Index, Square_Index, Outside_X_Index, Edges_X_Index, Edges_Y_Index,
                  Outside_Global_Index, LogOfAreas, Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index, SigmoidOfAreas)
    
    return templates.TemplateResponse("steel_predict.html", {"request": request, "prediction": prediction})
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)