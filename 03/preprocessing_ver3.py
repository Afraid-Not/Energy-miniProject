print("[PREPROCESSING]")

# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

data_path = './Energy/'
csv_path = './Energy/03/'
log_path = './Energy/03/log/'
trainer = './Energy/03/trainer/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

SEED = 1


def atmospheric_clarity_index(humidity, rainfall, dewpoint_diff):
    """일조량 예측에 특화된 대기 투명도 지수 (0~1, 높을수록 맑음)"""
    # 습도 효과
    humidity_clarity = max(0, (85 - humidity) / 85)
    
    # 강수 효과
    rain_clarity = max(0, np.exp(-rainfall * 0.5))
    
    # 이슬점 차이 효과
    dewpoint_clarity = min(1, dewpoint_diff / 5)
    
    return (humidity_clarity + rain_clarity + dewpoint_clarity) / 3

def calculate_dewpoint(temp, humidity):
    """이슬점 온도 계산 (Magnus 공식 사용) - 안전한 버전"""
    # 습도가 0이거나 음수인 경우 처리
    humidity = np.clip(humidity, 0.01, 100)  # 0.01% ~ 100% 범위로 제한
    
    a = 17.27
    b = 237.7
    
    # log 계산 시 0으로 나누기 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        dewpoint = (b * alpha) / (a - alpha)
    
    # 이상값 처리
    dewpoint = np.where(np.isfinite(dewpoint), dewpoint, temp - 10)  # 이상값은 기온-10도로 대체
    return dewpoint

def feature_engineering(df, is_train=True):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['date'] = pd.to_datetime(df['일시'])

    df['시각'] = df['date'].dt.hour                      # 시각(0~23)
    df['요일'] = df['date'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['월'] = df['date'].dt.month
    df['일'] = df['date'].dt.day
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['근무시간'] = df['시각'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)  # 주기적 패턴
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)
    df['SIN_일'] = np.sin(2 * np.pi * df['일'] / 31)  # 일의 주기적 패턴 (31일 기준)
    df['COS_일'] = np.cos(2 * np.pi * df['일'] / 31)
    df['SIN_월'] = np.sin(2 * np.pi * df['월'] / 12)  # 일의 주기적 패턴 (31일 기준)
    df['COS_월'] = np.cos(2 * np.pi * df['월'] / 12)
    df['SIN_요일'] = np.sin(2 * np.pi * df['요일'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_요일'] = np.cos(2 * np.pi * df['요일'] / 7)
    df['peak_time'] = df['시각'].apply(lambda x: 1 if 11 <= x <= 16 else 0) # Peak time (11 AM to 4 PM)

    # ======================
    # 기상/에너지 관련 파생 피처
    # ======================

    df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    df['태양광per냉방면적'] = df['태양광용량(kW)'] / (df['냉방면적(m2)'] + 1e-6)
    df['ESS설치여부'] = df['ESS저장용량(kWh)'].apply(lambda x: 1 if x > 0 else 0)
    df['PCS설치여부'] = df['PCS용량(kW)'].apply(lambda x: 1 if x > 0 else 0)
    df['설비밀도'] = (df['ESS저장용량(kWh)'] + df['PCS용량(kW)']) / (df['연면적(m2)'] + 1e-6)
    df['이슬점온도'] = calculate_dewpoint(df['기온(°C)'], df['습도(%)'])
    df['이슬점차이'] = df['기온(°C)'] - df['이슬점온도']  # 기온과 이슬점의 차이 (수증기 압력 관련)
    df['대기투명도'] = df.apply(lambda row: atmospheric_clarity_index(
        row['습도(%)'], row['강수량(mm)'], row['이슬점차이']), axis=1)

    ####### 강수 및 습도 관련 피쳐 #######
    df['rainy'] = (df['강수량(mm)'] > 0).astype(int)
    df['high_humidity'] = (df['습도(%)'] >= 85).astype(int)
    df['cloudy_or_rain'] = ((df['습도(%)'] >= 80) | (df['강수량(mm)'] > 0)).astype(int)
    df['humid_x_rain'] = df['습도(%)'] * df['강수량(mm)']
    df['discomfort_index'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    df['THI'] = 0.8 * df['기온(°C)'] + df['습도(%)'] * (df['기온(°C)'] - 14.4) / 100 + 46.4
    df['CDH'] = np.maximum(df['기온(°C)'] - 26, 0)
    
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    wind_speed = df['풍속(m/s)']
    df['perceived_temperature'] = temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00
    
    cloudy_based_on_humidity = np.clip((df['습도(%)'] - 30) / 7, 0, 10)
    df['cloudy_based_on_humidity'] = np.where(df['강수량(mm)'] > 0, 10, cloudy_based_on_humidity)

    ####### 태양 관련 피쳐 #######
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear
    
    df['SIN_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['COS_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # 원본 컬럼들 제거
    df = df.drop(['temp_date'], axis=1)
    
    latitude = 37.5665  # 서울 기준
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + df['day_of_year']) / 365))
    # 2) 일출/일몰 시각 계산
    lat_rad = np.deg2rad(latitude)
    dec_rad = np.deg2rad(declination)
    cos_ha = -np.tan(lat_rad) * np.tan(dec_rad)
    cos_ha = np.clip(cos_ha, -1, 1)  # 혹시 있을 오차 방지
    hour_angle = np.rad2deg(np.arccos(cos_ha))
    # 3) 일출/일몰 시간
    sunrise = 12 - hour_angle / 15
    sunset = 12 + hour_angle / 15
    df['sunrise_hour'] = sunrise
    df['sunset_hour'] = sunset
    df['daylight'] = ((df['시각'] >= df['sunrise_hour']) & (df['시각'] <= df['sunset_hour'])).astype(int)

    # 태양 고도각
    time_decimal = df['시각'] + 0.5  # 30분 기준 (정시 측정이라면 그냥 hour)
    solar_noon = (df['sunrise_hour'] + df['sunset_hour']) / 2
    hour_angle = 15 * (time_decimal - solar_noon)  # 시간각(°)
    # declination, latitude (radian)
    lat_rad = np.deg2rad(37.5665)
    dec_rad = np.deg2rad(declination)
    ha_rad = np.deg2rad(hour_angle)

    # 고도각
    df['solar_elevation'] = np.arcsin(
        np.sin(lat_rad) * np.sin(dec_rad) +
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    ) * 180 / np.pi
    df['solar_elevation'] = df['solar_elevation'].clip(lower=0) 
    
    G_sc = 0.0820  # 태양상수(MJ/m2/min)
    dr = 1 + 0.033 * np.cos(2 * np.pi * df['day_of_year'] / 365)  # 거리 계수
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec_rad))  # 일출/일몰 시각각(rad)
    I_0 = (24*60/np.pi) * G_sc * dr * (ws * np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.sin(ws))
    df['extraterrestrial_rad'] = I_0  # 단위: MJ/m2/day
    
    return df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

# 건물 정보 전처리
building_csv = pd.read_csv(data_path + 'building_info.csv')
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

# train & test 기본 전처리
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
train = feature_engineering(train, is_train=True)

test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')
test = feature_engineering(test, is_train=False)

# 지역 그룹 매핑
building_groups = [
    [28], [72], [19,58,75,91], [77], [24], [61,74,81], [32,42,65,79,99],
    [11,12,13,41,68,83,88], [20,26,44,45,70,100], [1,2,3,4,5,6,7,8,27,33,34,35,37,47,67,86,96],
    [71], [54,84], [17,18,29,30,31,40,43,48,49,51,52,53,60,63,64,76,78], [66], [85],
    [55,82], [15,16,39,59,73,92], [80,87], [89,90], [98], [50], [21,22,23],
    [46,93,94,95], [14,69], [57], [97], [36,38,56], [25,62], [9,10]
]

building_to_group = {}
for group_id, buildings in enumerate(building_groups):
    for building in buildings:
        building_to_group[building] = group_id

def add_region_group(df):
    df = df.copy()
    df['지역그룹ID'] = df['건물번호'].map(building_to_group)
    return df

train = add_region_group(train)
test = add_region_group(test)

# 일사량이 0인 건물들
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print("Train 컬럼:", len(train.columns))
print("Test 컬럼:", len(test.columns))
print("Train 고유 컬럼:", list(set(train.columns) - set(test.columns)))

def smape(actual, predicted):
    """SMAPE 계산"""
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))

# ===== 1단계: 일조 시간 XGBoost + LightGBM 앙상블 예측 =====
print("\n" + "="*60)
print("1단계: 일조 시간 XGBoost + LightGBM 앙상블 예측")
print("="*60)

sunshine_prediction_features = [
    # 기상 변수
    '기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)',
    # 구름/날씨 상태
    'cloudy_based_on_humidity', 'cloudy_or_rain', 'rainy', 
    'humid_x_rain', 'high_humidity', '대기투명도',
    # 태양 위치
    'sunrise_hour', 'sunset_hour', 'solar_elevation', 
    'extraterrestrial_rad', 'daylight',
    # 시간 정보
    'SIN_시', 'COS_시', 'SIN_일', 'COS_일', 'SIN_월', 'COS_월',
    'SIN_day_of_year', 'COS_day_of_year', 'day_of_year', '월', '시각',
    # 온도 관련
    '이슬점온도', '이슬점차이', 'perceived_temperature', 'THI', '불쾌지수',
    # 지역
    '건물번호', '지역그룹ID'
]

# 사용 가능한 피처만 선택
available_sunshine_features = [col for col in sunshine_prediction_features if col in train.columns]
print(f"일조 예측 사용 피처: {len(available_sunshine_features)}개")

# 데이터 준비
X = train[available_sunshine_features].fillna(0)
y = train['일조(hr)']
test_sunshine = test[available_sunshine_features].fillna(0)

# Train/Valid 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 (일조)
xgb_sunshine_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse',
    early_stopping_rounds=50,
)

# LightGBM 모델 (일조)
lgb_sunshine_model = LGBMRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("일조 XGBoost 모델 학습 중...")
xgb_sunshine_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print("일조 LightGBM 모델 학습 중...")
lgb_sunshine_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# 검증 예측
xgb_sunshine_val_pred = xgb_sunshine_model.predict(X_val)
lgb_sunshine_val_pred = lgb_sunshine_model.predict(X_val)

# 개별 모델 성능
xgb_sunshine_smape = smape(y_val, xgb_sunshine_val_pred)
lgb_sunshine_smape = smape(y_val, lgb_sunshine_val_pred)
xgb_sunshine_r2 = r2_score(y_val, xgb_sunshine_val_pred)
lgb_sunshine_r2 = r2_score(y_val, lgb_sunshine_val_pred)

print(f"\n=== 일조 개별 모델 성능 ===")
print(f"XGBoost - SMAPE: {xgb_sunshine_smape:.4f}, R2: {xgb_sunshine_r2:.4f}")
print(f"LightGBM - SMAPE: {lgb_sunshine_smape:.4f}, R2: {lgb_sunshine_r2:.4f}")

# 성능 기반 가중치 결정
if xgb_sunshine_smape < lgb_sunshine_smape:
    xgb_sunshine_weight, lgb_sunshine_weight = 0.7, 0.3
    better_sunshine_model = "XGBoost"
else:
    xgb_sunshine_weight, lgb_sunshine_weight = 0.3, 0.7
    better_sunshine_model = "LightGBM"

# 앙상블 검증 예측
sunshine_ensemble_val_pred = xgb_sunshine_weight * xgb_sunshine_val_pred + lgb_sunshine_weight * lgb_sunshine_val_pred
sunshine_ensemble_smape = smape(y_val, sunshine_ensemble_val_pred)
sunshine_ensemble_r2 = r2_score(y_val, sunshine_ensemble_val_pred)

print(f"\n=== 일조 앙상블 결과 ===")
print(f"가중치: XGBoost({xgb_sunshine_weight:.1f}) + LightGBM({lgb_sunshine_weight:.1f}) - {better_sunshine_model}이 더 우수")
print(f"앙상블 SMAPE: {sunshine_ensemble_smape:.4f}")
print(f"앙상블 R2: {sunshine_ensemble_r2:.4f}")

# 테스트 예측
xgb_sunshine_test_pred = xgb_sunshine_model.predict(test_sunshine)
lgb_sunshine_test_pred = lgb_sunshine_model.predict(test_sunshine)
sunshine_test_pred = xgb_sunshine_weight * xgb_sunshine_test_pred + lgb_sunshine_weight * lgb_sunshine_test_pred

# 일조 클리핑 (0~1)
sunshine_test_pred_clipped = np.clip(sunshine_test_pred, 0, 1)

print(f"\n=== 일조 예측값 범위 ===")
print(f"XGBoost: {xgb_sunshine_test_pred.min():.4f} ~ {xgb_sunshine_test_pred.max():.4f}")
print(f"LightGBM: {lgb_sunshine_test_pred.min():.4f} ~ {lgb_sunshine_test_pred.max():.4f}")
print(f"앙상블: {sunshine_test_pred.min():.4f} ~ {sunshine_test_pred.max():.4f}")
print(f"클리핑 후: {sunshine_test_pred_clipped.min():.4f} ~ {sunshine_test_pred_clipped.max():.4f}")

# test에 일조 추가
test = test.copy()
test['일조(hr)'] = sunshine_test_pred_clipped

# ===== 2단계: 일사량 XGBoost + LightGBM 앙상블 예측 =====
print("\n" + "="*60)
print("2단계: 일사량 XGBoost + LightGBM 앙상블 예측")
print("="*60)

solar_prediction_features = [
    # 기상 변수 (일조 포함)
    '기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)', '일조(hr)',
    # 구름/날씨 상태
    'cloudy_based_on_humidity', 'cloudy_or_rain', 'rainy',
    'humid_x_rain', 'high_humidity', '대기투명도',
    # 태양 위치
    'sunrise_hour', 'sunset_hour', 'solar_elevation', 
    'extraterrestrial_rad', 'daylight',
    # 시간 정보
    'SIN_시', 'COS_시', 'SIN_일', 'COS_일', 'SIN_월', 'COS_월',
    'SIN_day_of_year', 'COS_day_of_year', 'day_of_year', '월', '시각',
    # 온도 관련
    '이슬점온도', '이슬점차이', 'perceived_temperature', 'THI', '불쾌지수',
    # 지역
    '건물번호', '지역그룹ID'
]

# 사용 가능한 피처만 선택
available_solar_features = [col for col in solar_prediction_features if col in train.columns]
print(f"일사량 예측 사용 피처: {len(available_solar_features)}개")

# 데이터 준비
X_solar = train[available_solar_features].fillna(0)
y_solar = train['일사(MJ/m2)']
test_solar = test[available_solar_features].fillna(0)

# Train 일사량 최대값
solar_max = y_solar.max()
print(f"Train 일사량 최대값: {solar_max:.4f}")

# Train/Valid 분할
X_solar_train, X_solar_val, y_solar_train, y_solar_val = train_test_split(
    X_solar, y_solar, test_size=0.2, random_state=42
)

# XGBoost 모델 (일사량)
xgb_solar_model = XGBRegressor(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse',
    early_stopping_rounds=50,
)

# LightGBM 모델 (일사량)
lgb_solar_model = LGBMRegressor(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("일사량 XGBoost 모델 학습 중...")
xgb_solar_model.fit(
    X_solar_train, y_solar_train,
    eval_set=[(X_solar_train, y_solar_train), (X_solar_val, y_solar_val)],
    verbose=False
)

print("일사량 LightGBM 모델 학습 중...")
lgb_solar_model.fit(
    X_solar_train, y_solar_train,
    eval_set=[(X_solar_train, y_solar_train), (X_solar_val, y_solar_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# 검증 예측
xgb_solar_val_pred = xgb_solar_model.predict(X_solar_val)
lgb_solar_val_pred = lgb_solar_model.predict(X_solar_val)

# 제외할 건물번호들의 validation 인덱스 찾기
val_building_mask = ~X_solar_val['건물번호'].isin(zero_bnos)
print(f"\n일사량 SMAPE 계산에서 제외된 건물: {zero_bnos}")
print(f"검증 데이터 중 평가 대상: {val_building_mask.sum()}/{len(X_solar_val)} 개")

# 제외 건물을 뺀 SMAPE 계산
y_solar_val_filtered = y_solar_val[val_building_mask]
xgb_solar_val_pred_filtered = xgb_solar_val_pred[val_building_mask]
lgb_solar_val_pred_filtered = lgb_solar_val_pred[val_building_mask]

# 개별 모델 성능 (제외 건물 빼고 계산)
xgb_solar_smape = smape(y_solar_val_filtered, xgb_solar_val_pred_filtered)
lgb_solar_smape = smape(y_solar_val_filtered, lgb_solar_val_pred_filtered)
xgb_solar_r2 = r2_score(y_solar_val_filtered, xgb_solar_val_pred_filtered)
lgb_solar_r2 = r2_score(y_solar_val_filtered, lgb_solar_val_pred_filtered)

print(f"\n=== 일사량 개별 모델 성능 (제외 건물 빼고) ===")
print(f"XGBoost - SMAPE: {xgb_solar_smape:.4f}, R2: {xgb_solar_r2:.4f}")
print(f"LightGBM - SMAPE: {lgb_solar_smape:.4f}, R2: {lgb_solar_r2:.4f}")

# 성능 기반 가중치 결정
if xgb_solar_smape < lgb_solar_smape:
    xgb_solar_weight, lgb_solar_weight = 0.7, 0.3
    better_solar_model = "XGBoost"
else:
    xgb_solar_weight, lgb_solar_weight = 0.3, 0.7
    better_solar_model = "LightGBM"

# 앙상블 검증 예측 (제외 건물 빼고 계산)
solar_ensemble_val_pred_filtered = xgb_solar_weight * xgb_solar_val_pred_filtered + lgb_solar_weight * lgb_solar_val_pred_filtered
solar_ensemble_smape = smape(y_solar_val_filtered, solar_ensemble_val_pred_filtered)
solar_ensemble_r2 = r2_score(y_solar_val_filtered, solar_ensemble_val_pred_filtered)

print(f"\n=== 일사량 앙상블 결과 (제외 건물 빼고) ===")
print(f"가중치: XGBoost({xgb_solar_weight:.1f}) + LightGBM({lgb_solar_weight:.1f}) - {better_solar_model}이 더 우수")
print(f"앙상블 SMAPE: {solar_ensemble_smape:.4f}")
print(f"앙상블 R2: {solar_ensemble_r2:.4f}")

# 테스트 예측
xgb_solar_test_pred = xgb_solar_model.predict(test_solar)
lgb_solar_test_pred = lgb_solar_model.predict(test_solar)
solar_test_pred = xgb_solar_weight * xgb_solar_test_pred + lgb_solar_weight * lgb_solar_test_pred

# 일사량 클리핑 (0 ~ train_max)
solar_test_pred_clipped = np.clip(solar_test_pred, 0, solar_max)

print(f"\n=== 일사량 예측값 범위 ===")
print(f"XGBoost: {xgb_solar_test_pred.min():.4f} ~ {xgb_solar_test_pred.max():.4f}")
print(f"LightGBM: {lgb_solar_test_pred.min():.4f} ~ {lgb_solar_test_pred.max():.4f}")
print(f"앙상블: {solar_test_pred.min():.4f} ~ {solar_test_pred.max():.4f}")
print(f"클리핑 후: {solar_test_pred_clipped.min():.4f} ~ {solar_test_pred_clipped.max():.4f}")

# test에 일사량 추가
test['일사(MJ/m2)'] = solar_test_pred_clipped

# ===== 피처 중요도 출력 =====
print(f"\n=== 일조 XGBoost 피처 중요도 Top 10 ===")
sunshine_importance_df = pd.DataFrame({
    'feature': available_sunshine_features,
    'importance': xgb_sunshine_model.feature_importances_
}).sort_values('importance', ascending=False)
print(sunshine_importance_df.head(10).to_string(index=False))

print(f"\n=== 일사량 XGBoost 피처 중요도 Top 10 ===")
solar_importance_df = pd.DataFrame({
    'feature': available_solar_features,
    'importance': xgb_solar_model.feature_importances_
}).sort_values('importance', ascending=False)
print(solar_importance_df.head(10).to_string(index=False))

# ===== 최종 저장 =====
print(f"\n{'='*60}")
print("XGBoost + LightGBM 앙상블 일조/일사량 예측 완료!")
print(f"{'='*60}")

# 파일 저장
try:
    train.to_csv(trainer + 'ensemble_train.csv', index=False)
    test.to_csv(trainer + 'ensemble_test.csv', index=False)
    print("앙상블 결과 파일 저장 완료!")
except NameError:
    print("trainer 경로가 정의되지 않음. 수동으로 저장하세요:")
    print("train.to_csv('ensemble_train.csv', index=False)")
    print("test.to_csv('ensemble_test.csv', index=False)")

print(f"\n=== 최종 데이터 확인 ===")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Test에 추가된 컬럼: 일조(hr), 일사(MJ/m2)")

print(f"\n=== 최종 성능 요약 ===")
print(f"일조 앙상블 SMAPE: {sunshine_ensemble_smape:.4f}")
print(f"일사량 앙상블 SMAPE: {solar_ensemble_smape:.4f}")
print("XGBoost + LightGBM 앙상블로 안정적이고 정확한 예측 완료! 🎯")