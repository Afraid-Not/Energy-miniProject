# ========================
# 임포트 및 랜덤 시드 고정
# ========================
print(f"[11_00_2] 시작")
import pandas as pd
import numpy as np
import time
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

seed_file = "./Energy/11_submission/11_00_2.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 1 # seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# ========================
# 1. 데이터 로드
# ========================

data_path = './Energy/'
save_path = './Energy/11_submission/'
os.makedirs(save_path, exist_ok=True)

train = pd.read_csv(save_path + '2_SolarRadiation_train.csv', index_col=0)
test = pd.read_csv(save_path + '2_preprocessed_test.csv', index_col=0)
building = pd.read_csv(data_path + 'building_info.csv')

# print(train.columns)
# print(test.columns)
# ========================
# 1.2 feature 구분해놓기 
# ========================

# train['SolarRadiation(MJ/m2)'] 결측치 예측 모델용 피처
print(f"[2] 'SunshineHours(hr)', 'SolarRadiation(MJ/m2)' 예측")

# test['SunshineHours(hr)', 'SolarRadiation(MJ/m2)'] 예측 모델용 피처
feature_2nd_model = [
    'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'TotalFloorArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
    'PCSCapacity(kW)', 'SinHour', 'CosHour', 'DiscomfortIndex','FeelsLikeTemperature', 'PrecipitationChangeRate'
]
target_2nd_model = ['SunshineHours(hr)', 'SolarRadiation(MJ/m2)']

# ========================
# 2. test['SunshineHours(hr)', 'SolarRadiation(MJ/m2)'] 예측
# ========================

# 입력과 출력 분리
X_train_2 = train[feature_2nd_model]
y_train_2 = train[target_2nd_model]
X_test_2 = test[feature_2nd_model]

print("> test['SunshineHours(hr)'] 예측모델")
# 타겟 1: SunshineHours(hr)
X_tr1, X_val1, y_tr1, y_val1 = train_test_split(
    X_train_2, y_train_2['SunshineHours(hr)'], test_size=0.2, random_state=SEED
)

# 43308에서 early_stop
model1 = XGBRegressor(
    n_estimators=100000, learning_rate=0.03, max_depth=15,             
    num_leaves=300, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=1.0, random_state=SEED,
    n_jobs=-4, verbosity=0, early_stopping_rounds=100
)
model1.fit(
    X_tr1, y_tr1,
    eval_set=[(X_val1, y_val1)],
    # eval_metric='rmse'
)
val_model1 = model1.predict(X_val1)
rmse_model1 = np.sqrt(mean_squared_error(y_val1, val_model1))

pred_sun = model1.predict(X_test_2)

# 타겟 2: SolarRadiation(MJ/m2)
print("> test['SolarRadiation(MJ/m2)'] 예측모델")
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_2, y_train_2['SolarRadiation(MJ/m2)'], test_size=0.2, random_state=SEED
)

# 30301에서 early_stop
model2 = XGBRegressor(
    n_estimators=100000, learning_rate=0.03, max_depth=15,             
    num_leaves=300, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=1.0, random_state=SEED,
    n_jobs=-4, verbosity=0, early_stopping_rounds=100
)

model2.fit(
    X_tr2, y_tr2,
    eval_set=[(X_val2, y_val2)],
    # eval_metric='rmse',
)
val_model2 = model2.predict(X_val2)
rmse_model2 = np.sqrt(mean_squared_error(y_val2, val_model2))

pred_rad = model2.predict(X_test_2)

# 결과 저장
test['SunshineHours(hr)'] = np.round(pred_sun, 1)
test['SolarRadiation(MJ/m2)'] = np.round(pred_rad, 2)

filename = f'{SEED}_final_train.csv'
train.to_csv(save_path + filename)
test.to_csv(save_path + f'{SEED}_final_test.csv')

print(f"\n    'SunshineHours(hr)' RMSE : {rmse_model1}")
print(f"'SolarRadiation(MJ/m2)' RMSE : {rmse_model2}")

with open("./Energy/11_submission/11_00_2.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"train : {filename}\n")
    f.write(f" test : {SEED}_final_test.csv\n")
    f.write(f"일조 RMSE : {rmse_model1}\n")
    f.write(f"일사 RMSE : {rmse_model2}\n")
    f.write("="*40 + "\n")