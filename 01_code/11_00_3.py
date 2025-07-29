# ========================
# 임포트 및 랜덤 시드 고정
# ========================
print(f"[11_00_3] 시작")
import pandas as pd
import numpy as np
import time
import datetime
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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

seed_file = "./Energy/11_submission/11_00_3.json"

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

train_csv = pd.read_csv(save_path + '1_final_train.csv', index_col=0)
test_csv = pd.read_csv(save_path + '1_final_test.csv', index_col=0)
submit_csv = pd.read_csv(data_path + 'sample_submission.csv')

# print(train.columns)
# print(test.columns)
# Index(['BuildingID', 'DateTime', 'Temperature(C)', 'Precipitation(mm)',
#        'WindSpeed(m/s)', 'Humidity(%)', 'TotalFloorArea(m2)',
#        'CoolingArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
#        'PCSCapacity(kW)', 'Year', 'Month', 'Day', 'Hour', 'SinHour', 'CosHour',
#        'IsWeekend', 'IsHoliday', 'DiscomfortIndex', 'CoolingDegreeDay',
#        'FeelsLikeTemperature', 'IsWorkingHours', 'PrecipitationChangeRate',
#        'IsRainXWorkingHours', 'BuildingType_IDC', 'BuildingType_Etc',
#        'BuildingType_Public', 'BuildingType_DepartmentStore',
#        'BuildingType_Hospital', 'BuildingType_Commercial',
#        'BuildingType_Apartment', 'BuildingType_ResearchInstitute',
#        'BuildingType_School', 'BuildingType_Hotel', 'DayOfWeek_0',
#        'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4',
#        'DayOfWeek_5', 'DayOfWeek_6', 'SunshineHours(hr)',
#        'SolarRadiation(MJ/m2)'],
# exit()

# test['PowerConsumption(kWh)'] 최종 예측 모델용 피처
# feature_3rd_model = [
#     'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'TotalFloorArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
#     'PCSCapacity(kW)', 'SinHour', 'CosHour', 'DiscomfortIndex','FeelsLikeTemperature', 'PrecipitationChangeRate', 'SunshineHours(hr)', 'SolarRadiation(MJ/m2)'
# ]
feature_3rd_model = [
    'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'TotalFloorArea(m2)',
    'PrecipitationChangeRate', 'DayOfWeek_0',
    'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4',
    'DayOfWeek_5', 'DayOfWeek_6','CoolingArea(m2)', 'SolarPowerCapacity(kW)', 'SinHour', 
    'CosHour', 'Month', 'Day', 'DiscomfortIndex','FeelsLikeTemperature', 
    'SunshineHours(hr)', 'SolarRadiation(MJ/m2)'
]
target_3rd_model = ['PowerConsumption(kWh)']

# log_col = ['TotalFloorArea(m2)', 'CoolingArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)']
# train_csv[log_col] = np.log1p(train_csv[log_col])
# test_csv[log_col] = np.log1p(test_csv[log_col])


# ========================
# 3. test['PowerConsumption(kWh)'] 예측
# ========================

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

building_ids = train_csv['BuildingID'].unique()
submit_csv = pd.read_csv(data_path + 'sample_submission.csv')
submit_csv['answer'] = submit_csv['answer'].astype(float)

smape_dict = {}
final_preds = []

for b_id in tqdm(building_ids, desc="🏢 Building Loop"):

    # 건물별 데이터 분리
    train_b = train_csv[train_csv['BuildingID'] == b_id]
    test_b = test_csv[test_csv['BuildingID'] == b_id]
    submit_b = submit_csv[submit_csv['num_date_time'].str.startswith(f'{b_id}_')]

    X = train_b[feature_3rd_model]
    Y = train_b[target_3rd_model]
    # Y = np.log1p(train_b[target_3rd_model])
    test = test_b[feature_3rd_model]

    smape_list = []
    test_preds = []

    kfold = KFold(n_splits=5)
    kfold = TimeSeriesSplit(n_splits=5)
    for idx, (trn, val) in enumerate(kfold.split(X)):
        x_trn, x_val = X.iloc[trn], X.iloc[val]
        y_trn, y_val = Y.iloc[trn], Y.iloc[val]

        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=3.0,
            bagging_temperature=1.0,
            random_strength=1.0,
            od_type='Iter',
            od_wait=100,
            random_state=SEED,
            task_type="CPU",
            verbose=0
        )
        model.fit(x_trn, y_trn, eval_set=[(x_val, y_val)], use_best_model=True)
        val_pred = model.predict(x_val)
        smape_score = smape(y_val.to_numpy(), val_pred)
        smape_list.append(smape_score)
        print(f"[{b_id}] fold {idx} SMAPE: {smape_score:.4f}")

        preds = model.predict(test)
        test_preds.append(preds)

    # 결과 저장
    avg_smape = np.mean(smape_list)
    smape_dict[b_id] = avg_smape
    submit_csv.loc[submit_b.index, 'answer'] = np.mean(test_preds, axis=0)

    print(f"[{b_id}] SMAPE: {avg_smape:.4f}")

# 최종 결과 저장
mean_smape = np.mean(list(smape_dict.values()))
print(f"✅ 전체 평균 SMAPE: {mean_smape:.6f}")

today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{mean_smape:.6f}".replace('.', '_')
filename = f"({SEED})energy_({today})_({score_str}).csv"
submit_csv.to_csv(save_path + filename, index=False)

print(f"📁 저장 완료: {filename}")

# 로그 저장
with open(save_path + "11_00_3.txt", "a") as f:
    f.write(f"<SEED : {SEED}> Per-Building Run\n")
    f.write(f"{filename}\n")
    f.write(f"Mean SMAPE : {mean_smape:.6f}\n")
    f.write("="*40 + "\n")