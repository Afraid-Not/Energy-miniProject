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

def smape(actual, predicted):
    """SMAPE 계산"""
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8))

# ===== 앙상블 피처 세트 정의 =====
# 모델 1: 물리적 기상 모델 (온도/습도/구름 중심)
weather_physics_features = [
    # 핵심 기상 변수
    '기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)',
    # 온도/습도 파생변수 (물리적 관계)
    '이슬점온도', '이슬점차이', 'perceived_temperature', 'THI', '불쾌지수',
    # 구름 상태 (습도 기반)
    'cloudy_based_on_humidity', 'high_humidity', 'humid_x_rain',
    # 태양 물리 정보
    'solar_elevation', 'extraterrestrial_rad', '대기투명도',
    'sunrise_hour', 'sunset_hour', 'daylight',
    # 기본 시간 (선형적)
    '시각', '월', 'day_of_year'
]

# 모델 2: 시공간 패턴 모델 (순환성과 지역성 중심)
temporal_spatial_features = [
    # 기본 기상 (다른 관점)
    '기온(°C)', '습도(%)', '강수량(mm)',
    # 순환 시간 피처 (주기성 강조)
    'SIN_시', 'COS_시', 'SIN_일', 'COS_일', 
    'SIN_월', 'COS_월', 'SIN_day_of_year', 'COS_day_of_year',
    # 날씨 패턴 (강수 중심)
    'cloudy_or_rain', 'rainy',
    # 지역/공간 정보
    '건물번호', '지역그룹ID',
    # 태양 시간 정보
    'sunrise_hour', 'sunset_hour', 'daylight',
    # 대기 상태
    '대기투명도'
]

def train_ensemble_prediction(train_data, test_data, target_col, model1_features, model2_features, model_name=""):
    """앙상블 예측 함수"""
    print(f"\n{'='*60}")
    print(f"{model_name} 앙상블 예측 시작")
    print(f"{'='*60}")
    
    # 사용 가능한 피처만 필터링
    available_model1_features = [f for f in model1_features if f in train_data.columns]
    available_model2_features = [f for f in model2_features if f in train_data.columns]
    
    print(f"모델1 (물리기상) 피처 수: {len(available_model1_features)}")
    print(f"모델2 (시공간패턴) 피처 수: {len(available_model2_features)}")
    print(f"공통 피처 수: {len(set(available_model1_features) & set(available_model2_features))}")
    
    # 데이터 준비
    X1 = train_data[available_model1_features].fillna(0)
    X2 = train_data[available_model2_features].fillna(0)
    y = train_data[target_col]
    
    test1 = test_data[available_model1_features].fillna(0)
    test2 = test_data[available_model2_features].fillna(0)
    
    # Train/Valid 분할
    X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y, test_size=0.2, random_state=42)
    X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y, test_size=0.2, random_state=42)
    
    # 모델 1: 물리적 기상 모델 (더 깊고 세밀)
    model1 = XGBRegressor(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=50,
        
    )
    
    # 모델 2: 시공간 패턴 모델 (더 일반화)
    model2 = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=123,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=50,
        
    )
    
    # 모델 학습
    print("모델1 (물리기상) 학습 중...")
    model1.fit(
        X1_train, y1_train,
        eval_set=[(X1_train, y1_train), (X1_val, y1_val)],
        verbose=False
    )
    
    print("모델2 (시공간패턴) 학습 중...")
    model2.fit(
        X2_train, y2_train,
        eval_set=[(X2_train, y2_train), (X2_val, y2_val)],
        verbose=False
    )
    
    # 검증 예측
    val_pred1 = model1.predict(X1_val)
    val_pred2 = model2.predict(X2_val)
    
    # 개별 모델 성능
    smape1 = smape(y1_val, val_pred1)
    smape2 = smape(y2_val, val_pred2)
    r2_1 = r2_score(y1_val, val_pred1)
    r2_2 = r2_score(y2_val, val_pred2)
    
    print(f"\n=== 개별 모델 검증 성능 ===")
    print(f"모델1 (물리기상) - SMAPE: {smape1:.4f}, R2: {r2_1:.4f}")
    print(f"모델2 (시공간패턴) - SMAPE: {smape2:.4f}, R2: {r2_2:.4f}")
    
    # 성능 기반 가중치 계산
    if smape1 < smape2:
        weight1, weight2 = 0.7, 0.3
        better_model = "모델1 (물리기상)"
    else:
        weight1, weight2 = 0.3, 0.7
        better_model = "모델2 (시공간패턴)"
    
    # 앙상블 검증 예측
    ensemble_val_pred = 0.5 * val_pred1 + 0.5 * val_pred2
    ensemble_smape = smape(y1_val, ensemble_val_pred)  # y1_val과 y2_val은 동일
    ensemble_r2 = r2_score(y1_val, ensemble_val_pred)
    
    print(f"\n=== 앙상블 결과 ===")
    print(f"가중치: 모델1({weight1:.1f}) + 모델2({weight2:.1f}) - {better_model}이 더 우수")
    print(f"앙상블 SMAPE: {ensemble_smape:.4f}")
    print(f"앙상블 R2: {ensemble_r2:.4f}")
    
    # 테스트 예측
    test_pred1 = model1.predict(test1)
    test_pred2 = model2.predict(test2)
    ensemble_test_pred = weight1 * test_pred1 + weight2 * test_pred2
    
    print(f"\n=== 테스트 예측값 범위 ===")
    print(f"모델1: {test_pred1.min():.4f} ~ {test_pred1.max():.4f}")
    print(f"모델2: {test_pred2.min():.4f} ~ {test_pred2.max():.4f}")
    print(f"앙상블: {ensemble_test_pred.min():.4f} ~ {ensemble_test_pred.max():.4f}")
    
    # 피처 중요도 출력
    print(f"\n=== 모델1 (물리기상) Top 5 피처 ===")
    imp1_df = pd.DataFrame({
        'feature': available_model1_features,
        'importance': model1.feature_importances_
    }).sort_values('importance', ascending=False)
    print(imp1_df.head(5).to_string(index=False))
    
    print(f"\n=== 모델2 (시공간패턴) Top 5 피처 ===")
    imp2_df = pd.DataFrame({
        'feature': available_model2_features,
        'importance': model2.feature_importances_
    }).sort_values('importance', ascending=False)
    print(imp2_df.head(5).to_string(index=False))
    
    return ensemble_test_pred, (model1, model2), (test_pred1, test_pred2)

# ===== 1단계: 일조 시간 앙상블 예측 =====
sunshine_ensemble_pred, sunshine_models, sunshine_individual = train_ensemble_prediction(
    train, test, '일조(hr)', weather_physics_features, temporal_spatial_features, "일조 시간"
)

# 일조 클리핑 (0~1)
sunshine_ensemble_pred_clipped = np.clip(sunshine_ensemble_pred, 0, 1)
print(f"\n=== 일조 클리핑 결과 ===")
print(f"클리핑 전: {sunshine_ensemble_pred.min():.4f} ~ {sunshine_ensemble_pred.max():.4f}")
print(f"클리핑 후: {sunshine_ensemble_pred_clipped.min():.4f} ~ {sunshine_ensemble_pred_clipped.max():.4f}")

# test에 일조 추가
test = test.copy()
test['일조(hr)'] = sunshine_ensemble_pred_clipped

# ===== 2단계: 일사량 앙상블 예측 =====
# 일사량 피처에 일조 추가
solar_weather_physics_features = weather_physics_features + ['일조(hr)']
solar_temporal_spatial_features = temporal_spatial_features + ['일조(hr)']

solar_ensemble_pred, solar_models, solar_individual = train_ensemble_prediction(
    train, test, '일사(MJ/m2)', solar_weather_physics_features, solar_temporal_spatial_features, "일사량"
)

# 일사량 클리핑 (0 ~ train_max)
solar_max = train['일사(MJ/m2)'].max()
solar_ensemble_pred_clipped = np.clip(solar_ensemble_pred, 0, solar_max)

print(f"\n=== 일사량 클리핑 결과 ===")
print(f"Train 일사량 최대값: {solar_max:.4f}")
print(f"클리핑 전: {solar_ensemble_pred.min():.4f} ~ {solar_ensemble_pred.max():.4f}")
print(f"클리핑 후: {solar_ensemble_pred_clipped.min():.4f} ~ {solar_ensemble_pred_clipped.max():.4f}")

# test에 일사량 추가
test['일사(MJ/m2)'] = solar_ensemble_pred_clipped

# ===== 최종 결과 =====
print(f"\n{'='*60}")
print("앙상블 일조/일사량 예측 완료!")
print(f"{'='*60}")

# 파일 저장
# try:
train.to_csv(trainer + 'new_train_ver1.csv', index=False)
test.to_csv(trainer + 'new_test_ver1.csv', index=False)
print("앙상블 결과 파일 저장 완료!")
# except NameError:
    # print("trainer 경로가 정의되지 않음. 수동으로 저장하세요:")
    # print("train.to_csv('new_test_ver1.csv', index=False)")
    # print("test.to_csv('ensemble_test.csv', index=False)")

print(f"\n=== 최종 데이터 확인 ===")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Test에 추가된 컬럼: 일조(hr), 일사(MJ/m2)")
print("\n앙상블 모델로 더 안정적이고 정확한 예측 완료! 🎯")