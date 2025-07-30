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


building_csv = pd.read_csv(data_path + 'building_info.csv')
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

########################## train & test 기본 전처리 ############################
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
train = feature_engineering(train, is_train=True)

test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')
test = feature_engineering(test, is_train=False)

building_groups = [
    [28], [72], [19,58,75,91], [77], [24], [61,74,81], [32,42,65,79,99],
    [11,12,13,41,68,83,88], [20,26,44,45,70,100], [1,2,3,4,5,6,7,8,27,33,34,35,37,47,67,86,96],
    [71], [54,84], [17,18,29,30,31,40,43,48,49,51,52,53,60,63,64,76,78], [66], [85],
    [55,82], [15,16,39,59,73,92], [80,87], [89,90], [98], [50], [21,22,23],
    [46,93,94,95], [14,69], [57], [97], [36,38,56], [25,62], [9,10]
]

# 매핑 딕셔너리 생성
building_to_group = {}
for group_id, buildings in enumerate(building_groups):
    for building in buildings:
        building_to_group[building] = group_id

# 데이터프레임에 적용
def add_region_group(df):
    df['지역그룹ID'] = df['건물번호'].map(building_to_group)
    return df

# 사용법
train = add_region_group(train)
test = add_region_group(test)

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print(list(set(train.columns)))
print(list(set(test.columns)))
print(list(set(train.columns) - set(test.columns)))

sunshine_prediction_features = [
    # 핵심 기상 변수 (직접적 영향)
    '기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)',
    
    # 구름/날씨 상태 (매우 중요!)
    'cloudy_based_on_humidity', 'cloudy_or_rain', 'rainy', 
    'humid_x_rain', 'high_humidity', '대기투명도',
    
    # 태양 위치 정보 (물리적 제약)
    'sunrise_hour', 'sunset_hour', 'solar_elevation', 
    'extraterrestrial_rad', 'daylight',
    
    # 시간 정보 (계절성, 일주기)
    'SIN_시', 'COS_시', 'SIN_일', 'COS_일', 'SIN_월', 'COS_월',
    'SIN_day_of_year', 'COS_day_of_year', 'day_of_year', '월', '시각',
    
    # 온도 관련 파생변수
    '이슬점온도', '이슬점차이', 'perceived_temperature', 'THI', '불쾌지수',
    
    # 건물/지역 특성
    '건물번호', '지역그룹ID'
]

def smape(actual, predicted):
    """SMAPE 계산"""
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))

# 데이터 준비
X = train[sunshine_prediction_features].copy()
y = train['일조(hr)'].copy()
test_1 = test[sunshine_prediction_features].copy()

# 결측치 처리
X = X.fillna(0)
test_1 = test_1.fillna(0)

# Train/Validation 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# XGBoost 모델 생성
model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    eval_metric='rmse',
    early_stopping_rounds=50,
)

# 모델 학습
print("XGBoost 모델 학습 중...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# 예측
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
test_pred = model.predict(test_1)

# 평가
train_smape = smape(y_train, y_train_pred)
val_smape = smape(y_val, y_val_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\n=== 일조 시간 예측 결과 ===")
print(f"Train SMAPE: {train_smape:.4f}")
print(f"Valid SMAPE: {val_smape:.4f}")
print(f"Train R2: {train_r2:.4f}")
print(f"Valid R2: {val_r2:.4f}")

print(f"\n=== 예측값 범위 ===")
print(f"Train 예측: {y_train_pred.min():.4f} ~ {y_train_pred.max():.4f}")
print(f"Valid 예측: {y_val_pred.min():.4f} ~ {y_val_pred.max():.4f}")
print(f"Test 예측: {test_pred.min():.4f} ~ {test_pred.max():.4f}")

# 피처 중요도 출력 (상위 10개)
feature_importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n=== 피처 중요도 Top 10 ===")
print(importance_df.head(10).to_string(index=False))

# 일조 시간 범위 클리핑 (0~1)
test_pred_clipped = np.clip(test_pred, 0, 1)
test['일조(hr)'] = test_pred_clipped

print(f"\n=== 클리핑 후 Test 예측값 ===")
print(f"클리핑 전: {test_pred.min():.4f} ~ {test_pred.max():.4f}")
print(f"클리핑 후: {test_pred_clipped.min():.4f} ~ {test_pred_clipped.max():.4f}")
print("\n" + "="*60)
print("일사량 예측 시작")
print("="*60)

# 일사량 예측 피처 (일조 포함)
solar_prediction_features = [
    # 기상 변수
    '기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)', '일조(hr)',
    
    # 구름/날씨 상태  
    'cloudy_based_on_humidity', 'cloudy_or_rain', 'rainy',
    'humid_x_rain', 'high_humidity', '대기투명도',
    
    # 태양 위치 정보
    'sunrise_hour', 'sunset_hour', 'solar_elevation', 
    'extraterrestrial_rad', 'daylight',
    
    # 시간 정보
    'SIN_시', 'COS_시', 'SIN_일', 'COS_일', 'SIN_월', 'COS_월',
    'SIN_day_of_year', 'COS_day_of_year', 'day_of_year', '월', '시각',
    
    # 온도 관련
    '이슬점온도', '이슬점차이', 'perceived_temperature', 'THI', '불쾌지수',
    
    # 건물/지역
    '건물번호', '지역그룹ID'
]

# 사용 가능한 피처만 선택
available_solar_features = [col for col in solar_prediction_features if col in train.columns]
print(f"일사량 예측 사용 피처: {len(available_solar_features)}개")

# 일사량 데이터 준비
X_solar = train[available_solar_features].copy()
y_solar = train['일사(MJ/m2)'].copy()
test_solar = test[available_solar_features].copy()

# test에 예측된 일조 값 추가
if '일조(hr)' in available_solar_features:
    test_solar['일조(hr)'] = test_pred_clipped

# 결측치 처리
X_solar = X_solar.fillna(0)
test_solar = test_solar.fillna(0)

# Train에서 일사량 최대값 (클리핑용)
solar_max = y_solar.max()
print(f"Train 일사량 최대값: {solar_max:.4f}")

# Train/Valid 분할
X_solar_train, X_solar_val, y_solar_train, y_solar_val = train_test_split(
    X_solar, y_solar, test_size=0.2, random_state=42
)

# XGBoost 모델 (일사량용)
solar_model = XGBRegressor(
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

# 모델 학습
print("일사량 XGBoost 모델 학습 중...")
solar_model.fit(
    X_solar_train, y_solar_train,
    eval_set=[(X_solar_train, y_solar_train), (X_solar_val, y_solar_val)],
    verbose=False
)

# 예측
y_solar_train_pred = solar_model.predict(X_solar_train)
y_solar_val_pred = solar_model.predict(X_solar_val)
solar_test_pred = solar_model.predict(test_solar)

# 평가
solar_train_smape = smape(y_solar_train, y_solar_train_pred)
solar_val_smape = smape(y_solar_val, y_solar_val_pred)
solar_train_r2 = r2_score(y_solar_train, y_solar_train_pred)
solar_val_r2 = r2_score(y_solar_val, y_solar_val_pred)

print(f"\n=== 일사량 예측 결과 ===")
print(f"Train SMAPE: {solar_train_smape:.4f}")
print(f"Valid SMAPE: {solar_val_smape:.4f}")
print(f"Train R2: {solar_train_r2:.4f}")
print(f"Valid R2: {solar_val_r2:.4f}")

print(f"\n=== 일사량 예측값 범위 ===")
print(f"Train 예측: {y_solar_train_pred.min():.4f} ~ {y_solar_train_pred.max():.4f}")
print(f"Valid 예측: {y_solar_val_pred.min():.4f} ~ {y_solar_val_pred.max():.4f}")
print(f"Test 예측: {solar_test_pred.min():.4f} ~ {solar_test_pred.max():.4f}")

# 일사량 클리핑 (0 ~ train_max)
solar_test_pred_clipped = np.clip(solar_test_pred, 0, solar_max)

print(f"\n=== 클리핑 후 일사량 예측값 ===")
print(f"클리핑 전: {solar_test_pred.min():.4f} ~ {solar_test_pred.max():.4f}")
print(f"클리핑 후: {solar_test_pred_clipped.min():.4f} ~ {solar_test_pred_clipped.max():.4f}")

# 일사량 피처 중요도 (상위 10개)
solar_importance = solar_model.feature_importances_
solar_feature_names = X_solar.columns
solar_importance_df = pd.DataFrame({
    'feature': solar_feature_names,
    'importance': solar_importance
}).sort_values('importance', ascending=False)

print(f"\n=== 일사량 피처 중요도 Top 10 ===")
print(solar_importance_df.head(10).to_string(index=False))

# 최종 결과 저장 (선택사항)
test['일사(MJ/m2)'] = solar_test_pred_clipped

print(f"\n{'='*60}")
print("일조/일사량 예측 완료!")
print(f"{'='*60}")

train.to_csv(trainer + 'new_train.csv', index = False)
test.to_csv(trainer + 'new_test.csv', index = False)

