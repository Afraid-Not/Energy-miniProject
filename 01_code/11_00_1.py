# ========================
# 임포트 및 랜덤 시드 고정
# ========================
print(f"[11_00_1] 시작")
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

seed_file = "./Energy/11_submission/11_00_1.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 2 # seed_state["seed"]
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
# 데이터 로드
# ========================

data_path = './Energy/'
save_path = './Energy/11_submission/'
os.makedirs(save_path, exist_ok=True)

train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)
building = pd.read_csv(data_path + 'building_info.csv')

# ========================
# 1. 전처리
# ========================

print(f"[1] 전처리")
for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building[col] = building[col].replace('-', 0).astype(float)

# 1. 태양광용량 ≥ 1000 → PCS = 90%, 10의 자리 반올림 (PCS가 0 또는 NaN인 경우만)
cond1 = (building['태양광용량(kW)'] >= 1000) & ((building['PCS용량(kW)'] == 0) | (building['PCS용량(kW)'].isna()))
building.loc[cond1, 'PCS용량(kW)'] = np.round(building.loc[cond1, '태양광용량(kW)'] * 0.9, -1)

# 2. 100 ≤ 태양광용량 < 1000 → PCS = 동일값, 10의 자리 반올림
cond2 = (building['태양광용량(kW)'] >= 100) & (building['태양광용량(kW)'] < 1000) & ((building['PCS용량(kW)'] == 0) | (building['PCS용량(kW)'].isna()))
building.loc[cond2, 'PCS용량(kW)'] = np.round(building.loc[cond2, '태양광용량(kW)'], -1)

# building.to_csv(save_path + 'new_bui.csv')

train = pd.merge(train, building, on='건물번호', how='left')
test = pd.merge(test, building, on='건물번호', how='left')


def feature_engineering_1(df):
    df['parsed_datetime'] = pd.to_datetime(df['일시'])
    df['연'] = df['parsed_datetime'].dt.year
    df['월'] = df['parsed_datetime'].dt.month
    df['일'] = df['parsed_datetime'].dt.day
    df['시간'] = df['parsed_datetime'].dt.hour
    df['요일'] = df['parsed_datetime'].dt.dayofweek
    df['sin_hour'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['시간'] / 24)
    df['주말 여부'] = ((df['요일'] == 5) | (df['요일'] == 6)).astype(int)
    df['공휴일'] = 0
    df.loc[(df['월'] == 6) & (df['일'] == 6), '공휴일'] = 1
    df.loc[(df['월'] == 8) & (df['일'] == 15), '공휴일'] = 1
    df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    df['냉방도일'] = df['기온(°C)'].apply(lambda x: max(0, x - 26))
    df['체감온도'] = 13.12 + 0.6215 * df['기온(°C)'] - 11.37 * (df['풍속(m/s)']**0.16) + 0.3965 * df['기온(°C)'] * (df['풍속(m/s)']**0.16)
    df.loc[df['풍속(m/s)'] == 0, '체감온도'] = df['기온(°C)']
    df['근무시간여부'] = ((df['시간'] >= 9) & (df['시간'] < 18)).astype(int)
    df['강수량변화율'] = df.groupby('건물번호')['강수량(mm)'].diff().fillna(0)
    df['is_rain_x_working_hours'] = df['강수량(mm)'].apply(lambda x: 1 if x > 0 else 0) * df['근무시간여부']
    df = pd.get_dummies(df, columns=['건물유형', '요일'])
    return df.drop(columns=['parsed_datetime'])

train = feature_engineering_1(train)
test = feature_engineering_1(test)

# print(train.columns)

column_name_mapping = {
    '건물번호': 'BuildingID', '일시': 'DateTime', '기온(°C)': 'Temperature(C)', '강수량(mm)': 'Precipitation(mm)', '풍속(m/s)': 'WindSpeed(m/s)',
    '습도(%)': 'Humidity(%)', '일조(hr)': 'SunshineHours(hr)', '일사(MJ/m2)': 'SolarRadiation(MJ/m2)', '전력소비량(kWh)': 'PowerConsumption(kWh)',
    '연면적(m2)': 'TotalFloorArea(m2)', '냉방면적(m2)': 'CoolingArea(m2)', '태양광용량(kW)': 'SolarPowerCapacity(kW)', 'ESS저장용량(kWh)': 'ESSCapacity(kWh)',
    'PCS용량(kW)': 'PCSCapacity(kW)', '연': 'Year', '월': 'Month', '일': 'Day', '시간': 'Hour', 'sin_hour': 'SinHour', 'cos_hour': 'CosHour', 
    '주말 여부': 'IsWeekend', '공휴일': 'IsHoliday', '불쾌지수': 'DiscomfortIndex', '냉방도일': 'CoolingDegreeDay', '체감온도': 'FeelsLikeTemperature',
    '근무시간여부': 'IsWorkingHours', '강수량변화율': 'PrecipitationChangeRate', 'is_rain_x_working_hours': 'IsRainXWorkingHours',
    '건물유형_IDC(전화국)': 'BuildingType_IDC', '건물유형_건물기타': 'BuildingType_Etc', '건물유형_공공': 'BuildingType_Public',
    '건물유형_백화점': 'BuildingType_DepartmentStore', '건물유형_병원': 'BuildingType_Hospital', '건물유형_상용': 'BuildingType_Commercial',
    '건물유형_아파트': 'BuildingType_Apartment', '건물유형_연구소': 'BuildingType_ResearchInstitute', '건물유형_학교': 'BuildingType_School',
    '건물유형_호텔': 'BuildingType_Hotel', '요일_0': 'DayOfWeek_0', '요일_1': 'DayOfWeek_1', '요일_2': 'DayOfWeek_2', '요일_3': 'DayOfWeek_3',
    '요일_4': 'DayOfWeek_4', '요일_5': 'DayOfWeek_5', '요일_6': 'DayOfWeek_6'
}

train.rename(columns=column_name_mapping, inplace=True)
test.rename(columns=column_name_mapping, inplace=True)

# print(train.columns)
# 'BuildingID', 'DateTime', 'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'SunshineHours(hr)',
# 'SolarRadiation(MJ/m2)', 'PowerConsumption(kWh)', 'TotalFloorArea(m2)', 'CoolingArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
# 'PCSCapacity(kW)', 'Year', 'Month', 'Day', 'Hour', 'SinHour', 'CosHour', 'IsWeekend', 'IsHoliday', 'DiscomfortIndex',
# 'FeelsLikeTemperature', 'IsWorkingHours', 'PrecipitationChangeRate', 'IsRainXWorkingHours', 'BuildingType_IDC', 'BuildingType_Etc',
# 'BuildingType_Public', 'BuildingType_DepartmentStore', 'BuildingType_Hospital', 'BuildingType_Commercial',
# 'BuildingType_Apartment', 'BuildingType_ResearchInstitute', 'BuildingType_School', 'BuildingType_Hotel', 'DayOfWeek_0',
# 'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6'

# ========================
# 1.2 feature 구분해놓기 
# ========================

# train['SolarRadiation(MJ/m2)'] 결측치 예측 모델용 피처

feature_1st_model = [
    'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'TotalFloorArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
    'PCSCapacity(kW)', 'SinHour', 'CosHour', 'DiscomfortIndex','FeelsLikeTemperature', 'PrecipitationChangeRate', 'SunshineHours(hr)', 'SolarRadiation(MJ/m2)'
]
target_1st_model = ['SolarRadiation(MJ/m2)']

# # test['SunshineHours(hr)', 'SolarRadiation(MJ/m2)'] 예측 모델용 피처
# feature_2nd_model = [
#     'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'TotalFloorArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
#     'PCSCapacity(kW)', 'SinHour', 'CosHour', 'DiscomfortIndex','FeelsLikeTemperature', 'PrecipitationChangeRate'
# ]
# target_2nd_model = ['SunshineHours(hr)', 'SolarRadiation(MJ/m2)']

# # test['PowerConsumption(kWh)'] 최종 예측 모델용 피처
# feature_3rd_model = [
#     'Temperature(C)', 'Precipitation(mm)', 'WindSpeed(m/s)', 'Humidity(%)', 'SunshineHours(hr)',
#     'SolarRadiation(MJ/m2)', 'TotalFloorArea(m2)', 'CoolingArea(m2)', 'SolarPowerCapacity(kW)', 'ESSCapacity(kWh)',
#     'PCSCapacity(kW)', 'Year', 'Month', 'Day', 'Hour', 'SinHour', 'CosHour', 'IsWeekend', 'IsHoliday', 'DiscomfortIndex',
#     'FeelsLikeTemperature', 'IsWorkingHours', 'PrecipitationChangeRate', 'IsRainXWorkingHours', 'BuildingType_IDC', 'BuildingType_Etc',
#     'BuildingType_Public', 'BuildingType_DepartmentStore', 'BuildingType_Hospital', 'BuildingType_Commercial',
#     'BuildingType_Apartment', 'BuildingType_ResearchInstitute', 'BuildingType_School', 'BuildingType_Hotel', 'DayOfWeek_0',
#     'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6'
# ]
# target_3rd_model = ['PowerConsumption(kWh)']

# ========================
# 2. train['SolarRadiation(MJ/m2)'] 결측치 예측
# ========================

abnormal_buildings = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

# 일사량이 낮인데도 0인 경우를 결측으로 간주
condition = (
    (train['BuildingID'].isin(abnormal_buildings)) &
    (train['Hour'].between(6, 20)) &
    (train['SolarRadiation(MJ/m2)'] == 0)
)

# 결측 마킹
train.loc[condition, 'SolarRadiation(MJ/m2)'] = np.nan

train_non_null = train[train['SolarRadiation(MJ/m2)'].notnull()]
train_null = train[train['SolarRadiation(MJ/m2)'].isnull()]

X_train = train_non_null[feature_1st_model].drop(columns=['SolarRadiation(MJ/m2)'])
y_train = train_non_null['SolarRadiation(MJ/m2)']
X_pred = train_null[feature_1st_model].drop(columns=['SolarRadiation(MJ/m2)'])

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED
)

# 모델 학습
model = LGBMRegressor(
    n_estimators=3000, learning_rate=0.03, max_depth=15,             
    num_leaves=300, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=1.0, reg_lambda=1.0, random_state=SEED,
    n_jobs=-4, verbosity=-1, early_stopping_rounds=100
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse')

# 예측 및 대입
train.loc[train['SolarRadiation(MJ/m2)'].isnull(), 'SolarRadiation(MJ/m2)'] = model.predict(X_pred)

import matplotlib.pyplot as plt

before_fix = condition.sum()
after_fix = train.loc[condition, 'SolarRadiation(MJ/m2)'].sum()

print(f"결측치로 처리된 개수: {before_fix}")
print(f"예측 후 일사량 총합: {after_fix:.2f} (0 이상이면 대체된 것)")

# # 예측 결과 분포 확인
# sns.histplot(train.loc[condition, 'SolarRadiation(MJ/m2)'], bins=30, kde=True)
# plt.title("Predicted Solar Radiation for Abnormal Buildings")
# plt.show()

# 예측
val_pred = model.predict(X_val)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"[Validation RMSE]: {rmse:.4f}")
filename = f'{SEED}_SolarRadiation_train.csv'
train.to_csv(save_path + filename)
test.to_csv(save_path + f'{SEED}_preprocessed_test.csv')
print("저장 완료")

with open("./Energy/11_submission/11_00_1.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"RMSE 점수 : {rmse}\n")
    f.write("="*40 + "\n")