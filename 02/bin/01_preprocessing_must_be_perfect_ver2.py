print(f"[01_preprocessing] 시작")
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
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
csv_path = './Energy/02/'
log_path = './Energy/02/log/'
trainer = './Energy/02/trainer/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

seed_file = "./Energy/02/log/(preprocessing)SEED_COUNTS.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

# Optuna 최적화를 위한 기본 파라미터 (초기값)
def get_default_params():
    return {
        'xgb': {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1,
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': 0
        },
        'lgbm': {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': -1,
            'force_col_wise': True
        },
        'cat': {
            'loss_function': 'MAE',
            'iterations': 300,
            'learning_rate': 0.08,
            'depth': 5,
            'l2_leaf_reg': 2,
            'subsample': 0.9,
            'colsample_bylevel': 0.9,
            'random_strength': 0.5,
            'bagging_temperature': 0.5,
            'border_count': 32,
            'random_seed': SEED,
            'thread_count': -1,
            'verbose': False,
            'allow_writing_files': False
        }
    }

# Optuna 최적화 함수들
def create_xgb_objective(X_train, y_train, X_val, y_val, sample_weight=None):
    """XGBoost 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
            'gamma': trial.suggest_float('gamma', 0.01, 0.3),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, 
                  sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], 
                  verbose=False)
        
        pred = model.predict(X_val)
        rmse = smape(y_val, pred)
        return rmse
    
    return objective

def create_lgbm_objective(X_train, y_train, X_val, y_val, sample_weight=None):
    """LightGBM 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'num_leaves': trial.suggest_int('num_leaves', 15, 100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(stopping_rounds=30, verbose=False), log_evaluation(0)])
        
        pred = model.predict(X_val)
        rmse = smape(y_val, pred)
        return rmse
    
    return objective

def create_catboost_objective(X_train, y_train, X_val, y_val, sample_weight=None):
    """CatBoost 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'loss_function': 'MAE',
            'iterations': trial.suggest_int('iterations', 200, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 5.0),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),
            'border_count': trial.suggest_int('border_count', 16, 64),
            'random_seed': SEED,
            'thread_count': -1,
            'verbose': False,
            'allow_writing_files': False
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train,
                  sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)])
        
        pred = model.predict(X_val)
        rmse = smape(y_val, pred)
        return rmse
    
    return objective

def optimize_hyperparameters(X_train_sample, y_train_sample, X_val_sample, y_val_sample, 
                           model_type, sample_weight=None, n_trials=50):
    """
    하이퍼파라미터 최적화 실행
    Args:
        X_train_sample: 샘플링된 훈련 데이터
        y_train_sample: 샘플링된 훈련 타겟
        X_val_sample: 샘플링된 검증 데이터
        y_val_sample: 샘플링된 검증 타겟
        model_type: 'xgb', 'lgbm', 'catboost' 중 하나
        sample_weight: 가중치
        n_trials: 최적화 시행 횟수
    """
    print(f"    > {model_type.upper()} 하이퍼파라미터 최적화 시작 ({n_trials} trials)...")
    
    study = optuna.create_study(
        direction='minimize', 
        sampler=TPESampler(seed=SEED),
        study_name=f'{model_type}_optimization'
    )
    
    if model_type == 'xgb':
        objective = create_xgb_objective(X_train_sample, y_train_sample, X_val_sample, y_val_sample, sample_weight)
    elif model_type == 'lgbm':
        objective = create_lgbm_objective(X_train_sample, y_train_sample, X_val_sample, y_val_sample, sample_weight)
    elif model_type == 'catboost':
        objective = create_catboost_objective(X_train_sample, y_train_sample, X_val_sample, y_val_sample, sample_weight)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    study.optimize(objective, n_trials=n_trials)
    
    print(f"    > {model_type.upper()} 최적화 완료. Best RMSE: {study.best_value:.6f}")
    
    # 기본 파라미터와 병합 (키 매핑 처리)
    model_key_mapping = {'catboost': 'cat'}  # catboost -> cat 매핑
    param_key = model_key_mapping.get(model_type, model_type)
    
    default_params = get_default_params()[param_key]
    optimized_params = default_params.copy()
    optimized_params.update(study.best_params)
    
    return optimized_params, study.best_value

def sample_data_for_optimization(X_data, y_data, sample_ratio=0.3, min_samples=1000):
    """
    최적화를 위한 데이터 샘플링
    Args:
        X_data: 입력 데이터
        y_data: 타겟 데이터
        sample_ratio: 샘플링 비율 (기본 20%)
        min_samples: 최소 샘플 수
    """
    total_samples = len(X_data)
    sample_size = max(int(total_samples * sample_ratio), min_samples)
    sample_size = min(sample_size, total_samples)  # 전체 데이터 크기를 넘지 않도록
    
    # 무작위 샘플링
    sample_indices = np.random.choice(total_samples, size=sample_size, replace=False)
    
    X_sample = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
    y_sample = y_data.iloc[sample_indices] if hasattr(y_data, 'iloc') else y_data[sample_indices]
    
    print(f"    > 데이터 샘플링: {total_samples} -> {sample_size} ({sample_ratio*100:.1f}%)")
    
    return X_sample, y_sample, sample_indices

########################### 전처리 함수 정의 ############################

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

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

def calculate_apparent_temp(temp, humidity, wind_speed):
    """체감온도 계산"""
    return temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00

def estimate_visibility_advanced(temperature, humidity, rainfall, wind_speed):
    """다중 기상 요소를 고려한 가시거리 추정"""
    # 이슬점 계산
    dewpoint = temperature - ((100 - humidity) / 5)
    dewpoint_diff = abs(temperature - dewpoint)
    
    base_vis = 25.0
    
    # 이슬점 차이 효과 (작을수록 안개 가능성 높음)
    dewpoint_factor = min(1.0, dewpoint_diff / 3.0)
    
    # 습도 효과
    humidity_factor = max(0.2, (90 - humidity) / 70)
    
    # 강수 효과
    rain_factor = max(0.1, np.exp(-rainfall * 0.3))
    
    # 풍속 효과 (바람이 안개를 날려보냄)
    wind_factor = min(1.2, 0.8 + wind_speed * 0.1)
    
    visibility = base_vis * dewpoint_factor * humidity_factor * rain_factor * wind_factor
    return max(0.1, min(50.0, visibility))

def atmospheric_clarity_index(humidity, rainfall, dewpoint_diff):
    """일조량 예측에 특화된 대기 투명도 지수 (0~1, 높을수록 맑음)"""
    # 습도 효과
    humidity_clarity = max(0, (85 - humidity) / 85)
    
    # 강수 효과
    rain_clarity = max(0, np.exp(-rainfall * 0.5))
    
    # 이슬점 차이 효과
    dewpoint_clarity = min(1, dewpoint_diff / 5)
    
    return (humidity_clarity + rain_clarity + dewpoint_clarity) / 3

def simple_visibility_category(humidity, rainfall):
    """간단한 가시거리 카테고리 (0-4점)"""
    if rainfall > 5:  # 폭우
        return 0
    elif rainfall > 1:  # 비
        return 1
    elif humidity > 95:  # 안개 가능성
        return 1
    elif humidity > 85:  # 흐림
        return 2
    elif humidity > 70:  # 보통
        return 3
    else:  # 맑음
        return 4

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
    df['정오거리'] = (df['시각'] - 12).abs()  # 12시(정오) 기준 거리
    df['정오거리_INV'] = 1 / (df['정오거리'] + 1)
    df['peak_time'] = df['시각'].apply(lambda x: 1 if 11 <= x <= 16 else 0) # Peak time (11 AM to 4 PM)
    df['is_holiday'] = (((df['월'] == 6) & (df['일'] == 6)) | ((df['월'] == 8) & (df['일'] == 15))).astype(int)

    # ======================
    # 기상/에너지 관련 파생 피처
    # ======================
    df['이슬점온도'] = calculate_dewpoint(df['기온(°C)'], df['습도(%)'])
    df['이슬점차이'] = df['기온(°C)'] - df['이슬점온도']  # 기온과 이슬점의 차이 (수증기 압력 관련)

    # 이슬점 온도 계산
    df['체감온도'] = calculate_apparent_temp(df['기온(°C)'], df['습도(%)'], df['풍속(m/s)'])
    df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    df['태양광per냉방면적'] = df['태양광용량(kW)'] / (df['냉방면적(m2)'] + 1e-6)
    df['ESS설치여부'] = df['ESS저장용량(kWh)'].apply(lambda x: 1 if x > 0 else 0)
    df['PCS설치여부'] = df['PCS용량(kW)'].apply(lambda x: 1 if x > 0 else 0)
    df['설비밀도'] = (df['ESS저장용량(kWh)'] + df['PCS용량(kW)']) / (df['연면적(m2)'] + 1e-6)

    # 다중 기상 요소 고려한 고급 가시거리
    df['가시거리_고급'] = df.apply(lambda row: estimate_visibility_advanced(
        row['기온(°C)'], row['습도(%)'], row['강수량(mm)'], row['풍속(m/s)']), axis=1)
    # 대기 투명도 지수 (일조량 예측에 특화)
    df['대기투명도'] = df.apply(lambda row: atmospheric_clarity_index(
        row['습도(%)'], row['강수량(mm)'], row['이슬점차이']), axis=1)
    # 간단한 가시거리 카테고리
    df['가시거리_카테고리'] = df.apply(lambda row: simple_visibility_category(
        row['습도(%)'], row['강수량(mm)']), axis=1)
    visible_distance = pd.get_dummies(df['가시거리_카테고리'], prefix='가시거리')
    df = pd.concat([df, visible_distance], axis=1)
    
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
    df = df.drop(['temp_date', '가시거리_카테고리'], axis=1)
    
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
    #일출~일몰 대비 현재 시점의 상대적 위치
    df['solar_rel_pos'] = (df['시각'] + 0.5 - df['sunrise_hour']) / (df['sunset_hour'] - df['sunrise_hour'])
    df['solar_rel_pos'] = df['solar_rel_pos'].clip(0, 1)
    
    G_sc = 0.0820  # 태양상수(MJ/m2/min)
    dr = 1 + 0.033 * np.cos(2 * np.pi * df['day_of_year'] / 365)  # 거리 계수
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec_rad))  # 일출/일몰 시각각(rad)
    I_0 = (24*60/np.pi) * G_sc * dr * (ws * np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.sin(ws))
    df['extraterrestrial_rad'] = I_0  # 단위: MJ/m2/day
    
    return df

def add_lag_features(df, target_cols, lag_hours=[24, 48]):
    """전날(또는 며칠 전) 날씨 정보 추가"""
    df = df.copy()
    df = df.sort_values(['건물번호', '일시']).reset_index(drop=True)
    
    for col in target_cols:
        if col in df.columns:
            for lag in lag_hours:
                df[f'{col.split("(")[0]}_lag_{lag}h'] = df.groupby('건물번호')[col].shift(lag)
    
    return df

def add_lag_features_with_continuity(train_df, test_df, target_cols, lag_hours=[24, 48]):
    """
    train과 test의 연속성을 고려한 lag features 추가
    test 데이터의 lag는 train 마지막 데이터를 활용
    """
    print(f"    > 시계열 연속성을 고려한 lag 피처 생성 중...")
    
    # test 데이터 처리
    test_result = test_df.copy()
    test_result = test_result.sort_values(['건물번호', '일시']).reset_index(drop=True)
    
    # train 데이터도 정렬
    train_sorted = train_df.sort_values(['건물번호', '일시']).reset_index(drop=True)
    
    # 각 건물별로 train 마지막 데이터를 활용해서 test lag 생성
    for col in target_cols:
        if col in test_result.columns:
            for lag in lag_hours:
                lag_col_name = f'{col.split("(")[0]}_lag_{lag}h'
                
                # 먼저 test 내에서 lag 생성 (기본)
                test_result[lag_col_name] = test_result.groupby('건물번호')[col].shift(lag)
                
                # 각 건물별로 train 마지막 데이터로 초기값 채우기
                for building_num in test_result['건물번호'].unique():
                    # 해당 건물의 train 마지막 데이터
                    train_building = train_sorted[train_sorted['건물번호'] == building_num].copy()
                    if len(train_building) == 0:
                        continue
                        
                    # test에서 해당 건물 데이터
                    test_building_mask = test_result['건물번호'] == building_num
                    test_building_indices = test_result[test_building_mask].index.tolist()
                    
                    # lag 시간만큼 train 마지막 데이터로 채우기
                    if len(train_building) >= lag:
                        for i in range(min(lag, len(test_building_indices))):
                            test_idx = test_building_indices[i]
                            train_lag_idx = len(train_building) - lag + i
                            if train_lag_idx >= 0:
                                test_result.loc[test_idx, lag_col_name] = train_building.iloc[train_lag_idx][col]
    
    return test_result

def add_sunshine_rolling_features(df):
    """일조시간 rolling features 추가"""
    df = df.copy()
    
    # 일조시간 이전 3시간 rolling features
    if '일조(hr)' in df.columns:
        for lag in range(1, 4):
            df[f'일조_변화량_{lag}h'] = df.groupby('건물번호')['일조(hr)'].diff(periods=lag)
            df[f'일조_rolling_mean_{lag}h'] = df.groupby('건물번호')['일조(hr)'].rolling(window=lag, min_periods=1).mean().reset_index(0, drop=True)
            df[f'일조_rolling_sum_{lag}h'] = df.groupby('건물번호')['일조(hr)'].rolling(window=lag, min_periods=1).sum().reset_index(0, drop=True)
    
    df = df.fillna(0)
    return df

def add_insolation_rolling_features(df):
    """일사량 rolling features 추가 (저장 직전)"""
    df = df.copy()
    
    # 일사량 이전 3시간 rolling features
    if '일사(MJ/m2)' in df.columns:
        for lag in range(1, 4):
            df[f'일사_변화량_{lag}h'] = df.groupby('건물번호')['일사(MJ/m2)'].diff(periods=lag)
            df[f'일사_rolling_mean_{lag}h'] = df.groupby('건물번호')['일사(MJ/m2)'].rolling(window=lag, min_periods=1).mean().reset_index(0, drop=True)
            df[f'일사_rolling_sum_{lag}h'] = df.groupby('건물번호')['일사(MJ/m2)'].rolling(window=lag, min_periods=1).sum().reset_index(0, drop=True)
    
    df = df.fillna(0)
    return df

def add_weather_rolling_features(df):
    """기온, 풍속, 강수량, 습도 rolling features 추가"""
    df = df.copy()
    
    # 롤링 피처를 생성할 컬럼 목록
    weather_cols = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '이슬점온도', '체감온도']
    
    for col in weather_cols:
        if col in df.columns:
            for lag in range(1, 4):
                # 변화량: 이전 시점과의 차이
                df[f'{col.split("(")[0]}_변화량_{lag}h'] = df.groupby('건물번호')[col].diff(periods=lag)
                
                # 롤링 평균: 이전 n시간의 평균
                df[f'{col.split("(")[0]}_rolling_mean_{lag}h'] = df.groupby('건물번호')[col].rolling(window=lag, min_periods=1).mean().reset_index(0, drop=True)
                
                # 롤링 합계: 이전 n시간의 합계 (강수량에 특히 유용)
                df[f'{col.split("(")[0]}_rolling_sum_{lag}h'] = df.groupby('건물번호')[col].rolling(window=lag, min_periods=1).sum().reset_index(0, drop=True)
    
    df = df.fillna(0)
    return df

def add_weather_volatility_features(df):
    """날씨 급변 감지 피처 추가"""
    df = df.copy()
    
    # 기온/습도/풍속 급변 감지
    df['기온_급변'] = df.groupby('건물번호')['기온(°C)'].diff().abs().fillna(0)
    df['습도_급변'] = df.groupby('건물번호')['습도(%)'].diff().abs().fillna(0)
    df['풍속_급변'] = df.groupby('건물번호')['풍속(m/s)'].diff().abs().fillna(0)
    
    # 강수 시작/끝 감지
    prev_rain = df.groupby('건물번호')['강수량(mm)'].shift(1).fillna(0)
    df['강수시작'] = ((df['강수량(mm)'] > 0) & (prev_rain == 0)).astype(int)
    df['강수끝'] = ((df['강수량(mm)'] == 0) & (prev_rain > 0)).astype(int)
    
    # 강수 강도 분류 (NaN 처리 개선)
    df['강수강도'] = pd.cut(df['강수량(mm)'], 
                         bins=[-0.001, 0.1, 1, 5, 10, 100], 
                         labels=[0, 1, 2, 3, 4], 
                         include_lowest=True).fillna(0).astype(int)
    
    rainfall_intensity = pd.get_dummies(df['강수강도'], prefix='강수강도')
    df = pd.concat([df, rainfall_intensity], axis=1)
    df = df.drop(['강수강도'], axis=1)
    
    # 종합 날씨 불안정 지수
    df['날씨불안정지수'] = (df['기온_급변'] + df['습도_급변'] * 0.5 + df['풍속_급변'] * 2)
    
    # 시간대별 이상치 감지 (안전한 방식으로 수정)
    for col in ['기온(°C)', '습도(%)', '풍속(m/s)']:
        if col in df.columns:
            # 시간대별 통계 계산
            hourly_stats = df.groupby('시각')[col].agg(['mean', 'std']).reset_index()
            hourly_stats.columns = ['시각', f'{col}_hourly_mean', f'{col}_hourly_std']
            
            # 병합
            df = df.merge(hourly_stats, on='시각', how='left')
            
            # Z-score 계산 (0으로 나누기 방지)
            std_col = f'{col}_hourly_std'
            mean_col = f'{col}_hourly_mean'
            df[std_col] = df[std_col].fillna(1).replace(0, 1)  # 표준편차 0인 경우 1로 대체
            
            df[f'{col}_zscore'] = (df[col] - df[mean_col]) / df[std_col]
            df[f'{col}_이상치'] = (df[f'{col}_zscore'].abs() > 2).astype(int)
            
            # 임시 컬럼 제거
            df = df.drop([mean_col, std_col], axis=1)
    
    # 모든 NaN 값을 0으로 채우기
    df = df.fillna(0)
    return df

def identify_transition_periods(X_data):
    """새벽/저녁 전환 구간 식별"""
    if '시각' not in X_data.columns:
        return np.zeros(len(X_data))
    
    hour = X_data['시각'].values
    transition_mask = np.zeros(len(hour))
    
    transition_mask = np.where(
        ((hour >= 5) & (hour <= 6)) | ((hour >= 19) & (hour <= 20)), 3.0,  # 급격한 전환
                      np.where(
        ((hour >= 6) & (hour <= 8)) | ((hour >= 17) & (hour <= 19)), 2.0,  # 완만한 전환
            1.0  # 일반 구간
        )
    )
    
    return transition_mask

def add_transition_features(df):
    """전환 구간 특화 피처 추가"""
    df = df.copy()
    
    if '시각' not in df.columns:
        return df
    
    # 새벽/저녁 전환 구간 식별
    df['morning_transition'] = df['시각'].apply(lambda x: 1 if 5 <= x <= 8 else 0)
    df['evening_transition'] = df['시각'].apply(lambda x: 1 if 17 <= x <= 20 else 0)
    df['transition_intensity'] = df['시각'].apply(lambda x: 
        3.0 if (5 <= x <= 6) or (19 <= x <= 20) else  # 급격한 전환
        2.0 if (6 <= x <= 8) or (17 <= x <= 19) else  # 완만한 전환
        0.0  # 일반
    )
    
    # 일출/일몰 시간과의 거리 (계절별 변화 고려)
    month = df['월'] if '월' in df.columns else 6  # 기본값
    
    # 간단한 일출/일몰 시간 추정 (위도 37도 기준)
    if isinstance(month, pd.Series):
        sunrise_hour = 6.5 - 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
        sunset_hour = 18.5 + 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
    else:
        sunrise_hour = 6.5 - 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
        sunset_hour = 18.5 + 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
    
    df['sunrise_distance'] = np.abs(df['시각'] - sunrise_hour)
    df['sunset_distance'] = np.abs(df['시각'] - sunset_hour)
    df['min_transition_distance'] = np.minimum(df['sunrise_distance'], df['sunset_distance'])
    
    # 전환 구간에서의 기울기 변화 감지
    if '기온(°C)' in df.columns:
        df['temp_transition_signal'] = df['기온(°C)'] * df['transition_intensity']
    
    return df

def train_ensemble_models_enhanced(X_train, y_train, X_val, y_val, is_insolation=False, use_log_scale=False, optimize_params=False):
    """Peak + Transition 모델이 추가된 강화된 앙상블 모델 (Optuna 최적화 포함)"""
    
    # 전환 구간 피처 추가
    X_train_enhanced = add_transition_features(X_train)
    X_val_enhanced = add_transition_features(X_val)
    
    # 하락 구간 가중치 계산
    y_train_reset = y_train.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    
    if len(y_train_reset) > 1:
        decline_mask_train = y_train_reset.iloc[1:].values < y_train_reset.iloc[:-1].values
        decline_weights = np.ones(len(y_train_reset))
        decline_weights[1:] = np.where(decline_mask_train, 4.0, 1.0)
    else:
        decline_weights = np.ones(len(y_train_reset))
    
    # Peak 구간 가중치 계산
    if 'peak_time' in X_train_enhanced.columns:
        peak_mask_train = X_train_enhanced['peak_time'].values == 1
        peak_weights = np.where(peak_mask_train, 4.0, 1.0)
        print("    > Peak 시간대 가중치 적용")
    else:
        peak_weights = np.ones(len(y_train_reset))
        print("    > Peak_time 피처 없음, 일반 가중치 사용")
    
    # Transition 구간 가중치 계산 (새로 추가)
    transition_weights = identify_transition_periods(X_train_enhanced)
    print(f"    > Transition 구간 가중치 적용 (평균 가중치: {transition_weights.mean():.2f})")
    
    # log1p 스케일링 적용
    if use_log_scale:
        y_train_scaled = np.log1p(y_train_reset)
        y_val_scaled = np.log1p(y_val_reset)
        print("    > log1p 스케일링 적용")
    else:
        y_train_scaled = y_train_reset
        y_val_scaled = y_val_reset
    
    print("    > 6개 모델 앙상블 훈련... (하락 + Peak + Transition 가중치 적용)")
    
    # Optuna 하이퍼파라미터 최적화 수행
    optimized_params = {}
    if optimize_params:
        print("    > Optuna 하이퍼파라미터 최적화 시작...")
        
        # 20% 샘플링된 데이터로 최적화
        X_train_sample, y_train_sample, train_sample_indices = sample_data_for_optimization(X_train_enhanced, y_train_scaled)
        X_val_sample, y_val_sample, val_sample_indices = sample_data_for_optimization(X_val_enhanced, y_val_scaled)
        
        # 샘플링된 가중치
        decline_weights_sample = decline_weights[train_sample_indices]
        peak_weights_sample = peak_weights[train_sample_indices]
        transition_weights_sample = transition_weights[train_sample_indices]
        
        # XGBoost 최적화
        xgb_params_opt, xgb_score = optimize_hyperparameters(
            X_train_sample, y_train_sample, X_val_sample, y_val_sample,
            'xgb', sample_weight=decline_weights_sample, n_trials=30
        )
        optimized_params['xgb'] = xgb_params_opt
        
        # LightGBM 최적화
        lgbm_params_opt, lgbm_score = optimize_hyperparameters(
            X_train_sample, y_train_sample, X_val_sample, y_val_sample,
            'lgbm', sample_weight=decline_weights_sample, n_trials=30
        )
        optimized_params['lgbm'] = lgbm_params_opt
        
        # CatBoost 최적화
        cat_params_opt, cat_score = optimize_hyperparameters(
            X_train_sample, y_train_sample, X_val_sample, y_val_sample,
            'catboost', sample_weight=decline_weights_sample, n_trials=20
        )
        optimized_params['cat'] = cat_params_opt
        
        print(f"    > 최적화 완료: XGB({xgb_score:.6f}), LGBM({lgbm_score:.6f}), CAT({cat_score:.6f})")
        
    else:
        # 기본 파라미터 사용
        default_params = get_default_params()
        optimized_params['xgb'] = default_params['xgb'].copy()
        optimized_params['lgbm'] = default_params['lgbm'].copy()
        optimized_params['cat'] = default_params['cat'].copy()
    
    # 특화 모델용 파라미터 생성
    decline_params = optimized_params['xgb'].copy()
    decline_params.update({
        'max_depth': min(optimized_params['xgb'].get('max_depth', 6) + 2, 12),
        'learning_rate': optimized_params['xgb'].get('learning_rate', 0.05) * 0.6,
        'n_estimators': int(optimized_params['xgb'].get('n_estimators', 500) * 1.2),
        'gamma': optimized_params['xgb'].get('gamma', 0.1) * 0.5,
    })
    
    peak_params = optimized_params['xgb'].copy()
    peak_params.update({
        'max_depth': min(optimized_params['xgb'].get('max_depth', 6) + 1, 10),
        'learning_rate': optimized_params['xgb'].get('learning_rate', 0.05) * 0.8,
        'reg_alpha': optimized_params['xgb'].get('reg_alpha', 0.1) * 0.5,
    })
    
    transition_params = optimized_params['xgb'].copy()
    transition_params.update({
        'max_depth': min(optimized_params['xgb'].get('max_depth', 6) + 3, 15),
        'learning_rate': optimized_params['xgb'].get('learning_rate', 0.05) * 0.5,
        'n_estimators': int(optimized_params['xgb'].get('n_estimators', 500) * 1.5),
        'gamma': optimized_params['xgb'].get('gamma', 0.1) * 0.1,
        'min_child_weight': 1,
    })
    
    # 1. 기본 XGBoost
    xgb_model = XGBRegressor(**optimized_params['xgb'])
    xgb_model.fit(X_train_enhanced, y_train_scaled, 
                  sample_weight=decline_weights,
                  eval_set=[(X_val_enhanced, y_val_scaled)], 
                  verbose=False)
    
    # 2. 기본 LightGBM
    lgbm_model = LGBMRegressor(**optimized_params['lgbm'])
    lgbm_model.fit(X_train_enhanced, y_train_scaled,
                   sample_weight=decline_weights,
                   eval_set=[(X_val_enhanced, y_val_scaled)],
                   callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)])
    
    # 3. 하락 특화 모델
    decline_model = XGBRegressor(**decline_params)
    extreme_decline_weights = np.where(decline_weights > 1, 5.0, 1.0)
    decline_model.fit(X_train_enhanced, y_train_scaled,
                     sample_weight=extreme_decline_weights,
                     eval_set=[(X_val_enhanced, y_val_scaled)],
                     verbose=False)
    
    # 4. Peak 특화 모델
    peak_model = XGBRegressor(**peak_params)
    combined_peak_weights = decline_weights * peak_weights
    peak_model.fit(X_train_enhanced, y_train_scaled,
                  sample_weight=combined_peak_weights,
                  eval_set=[(X_val_enhanced, y_val_scaled)],
                  verbose=False)
    
    # 5. Transition 특화 모델 (새로 추가)
    transition_model = XGBRegressor(**transition_params)
    combined_transition_weights = decline_weights * transition_weights
    transition_model.fit(X_train_enhanced, y_train_scaled,
                        sample_weight=combined_transition_weights,
                        eval_set=[(X_val_enhanced, y_val_scaled)],
                        verbose=False)
    
    # 6. CatBoost 모델
    cat_model = CatBoostRegressor(**optimized_params['cat'])
    cat_model.fit(X_train_enhanced, y_train_scaled,
                  sample_weight=decline_weights,
                  eval_set=[(X_val_enhanced, y_val_scaled)])
    
    # 각 모델의 예측
    xgb_pred_val = xgb_model.predict(X_val_enhanced)
    lgbm_pred_val = lgbm_model.predict(X_val_enhanced)
    decline_pred_val = decline_model.predict(X_val_enhanced)
    peak_pred_val = peak_model.predict(X_val_enhanced)
    transition_pred_val = transition_model.predict(X_val_enhanced)
    cat_pred_val = cat_model.predict(X_val_enhanced)
    
    # 동적 가중치 계산
    val_decline_mask = y_val_reset.iloc[1:].values < y_val_reset.iloc[:-1].values
    val_decline_indicators = np.zeros(len(y_val_reset))
    val_decline_indicators[1:] = val_decline_mask.astype(float)
    
    # Peak 구간 감지
    if 'peak_time' in X_val_enhanced.columns:
        val_peak_indicators = X_val_enhanced['peak_time'].values
    else:
        val_peak_indicators = np.zeros(len(y_val_reset))
    
    # Transition 구간 감지
    val_transition_weights = identify_transition_periods(X_val_enhanced)
    val_transition_indicators = val_transition_weights > 1.5  # transition 가중치가 1.5 이상인 구간
    
    # 앙상블 예측 (6개 모델)
    ensemble_pred_val = np.zeros_like(xgb_pred_val)
    
    for i in range(len(ensemble_pred_val)):
        is_decline = val_decline_indicators[i] > 0
        is_peak = val_peak_indicators[i] > 0
        is_transition = val_transition_indicators[i]
        
        if is_transition and is_decline:  # 전환 + 하락
            # Transition과 Decline 모델에 최우선 가중치
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.08 + 
                                   lgbm_pred_val[i] * 0.12 + 
                                   decline_pred_val[i] * 0.35 + 
                                   peak_pred_val[i] * 0.1 + 
                                   transition_pred_val[i] * 0.25 + 
                                   cat_pred_val[i] * 0.1)
        elif is_transition:  # 전환 구간만
            # Transition 모델 최우선
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.1 + 
                                   lgbm_pred_val[i] * 0.15 + 
                                   decline_pred_val[i] * 0.15 + 
                                   peak_pred_val[i] * 0.15 + 
                                   transition_pred_val[i] * 0.35 + 
                                   cat_pred_val[i] * 0.1)
        elif is_decline and is_peak:  # 하락 + Peak
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.1 + 
                                   lgbm_pred_val[i] * 0.15 + 
                                   decline_pred_val[i] * 0.3 + 
                                   peak_pred_val[i] * 0.25 + 
                                   transition_pred_val[i] * 0.1 + 
                                   cat_pred_val[i] * 0.1)
        elif is_decline:  # 하락 구간만
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.15 + 
                                   lgbm_pred_val[i] * 0.2 + 
                                   decline_pred_val[i] * 0.35 + 
                                   peak_pred_val[i] * 0.1 + 
                                   transition_pred_val[i] * 0.1 + 
                                   cat_pred_val[i] * 0.1)
        elif is_peak:  # Peak 구간만
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.2 + 
                                   lgbm_pred_val[i] * 0.2 + 
                                   decline_pred_val[i] * 0.1 + 
                                   peak_pred_val[i] * 0.3 + 
                                   transition_pred_val[i] * 0.1 + 
                                   cat_pred_val[i] * 0.1)
        else:  # 일반 구간
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.2 + 
                                   lgbm_pred_val[i] * 0.2 + 
                                   decline_pred_val[i] * 0.15 + 
                                   peak_pred_val[i] * 0.15 + 
                                   transition_pred_val[i] * 0.15 + 
                                   cat_pred_val[i] * 0.15)
    
    # log 스케일 역변환 및 클리핑
    if use_log_scale:
        ensemble_pred_val = np.expm1(ensemble_pred_val)
        ensemble_pred_val = np.clip(ensemble_pred_val, -0.1 if not is_insolation else 0, None)
        if not is_insolation:
            ensemble_pred_val = np.maximum(ensemble_pred_val, 0)
    else:
        if is_insolation:
            ensemble_pred_val = np.clip(ensemble_pred_val, 0, None)
        else:
            # 일조시간: 전환 구간에서 더 관대한 처리
            ensemble_pred_val = np.clip(ensemble_pred_val, 0, 1.3)
            ensemble_pred_val = np.minimum(ensemble_pred_val, 1.0)
    
    final_mae = smape(y_val_reset, ensemble_pred_val)
    
    print(f"    > 최종 smape: {final_mae:.6f}")
    
    return {
        'models': (xgb_model, lgbm_model, decline_model, peak_model, transition_model, cat_model),
        'final_mae': final_mae,
        'use_log_scale': use_log_scale,
        'decline_indicators': val_decline_indicators,
        'peak_indicators': val_peak_indicators,
        'transition_indicators': val_transition_indicators,
        'optimized_params': optimized_params  # 최적화된 파라미터 저장
    }

def predict_with_ensemble_enhanced(models_dict, X_test, is_insolation=False):
    """Peak + Transition 모델이 포함된 6개 모델 앙상블 예측"""
    xgb_model, lgbm_model, decline_model, peak_model, transition_model, cat_model = models_dict['models']
    use_log_scale = models_dict.get('use_log_scale', False)
    
    # 전환 구간 피처 추가
    X_test_enhanced = add_transition_features(X_test)
    
    # 각 모델 예측
    xgb_pred = xgb_model.predict(X_test_enhanced)
    lgbm_pred = lgbm_model.predict(X_test_enhanced)
    decline_pred = decline_model.predict(X_test_enhanced)
    peak_pred = peak_model.predict(X_test_enhanced)
    transition_pred = transition_model.predict(X_test_enhanced)
    cat_pred = cat_model.predict(X_test_enhanced)
    
    # 구간 감지
    if 'peak_time' in X_test_enhanced.columns:
        peak_indicators = X_test_enhanced['peak_time'].values
    else:
        peak_indicators = np.zeros(len(xgb_pred))
    
    transition_weights = identify_transition_periods(X_test_enhanced)
    transition_indicators = transition_weights > 1.5
    
    # 동적 가중치 앙상블
    ensemble_pred = np.zeros_like(xgb_pred)
    
    for i in range(len(ensemble_pred)):
        is_peak = peak_indicators[i] > 0
        is_transition = transition_indicators[i]
        
        if is_transition:  # 전환 구간 (최우선)
            ensemble_pred[i] = (xgb_pred[i] * 0.1 + lgbm_pred[i] * 0.15 + 
                               decline_pred[i] * 0.15 + peak_pred[i] * 0.15 + 
                               transition_pred[i] * 0.35 + cat_pred[i] * 0.1)
        elif is_peak:  # Peak 구간
            ensemble_pred[i] = (xgb_pred[i] * 0.2 + lgbm_pred[i] * 0.2 + 
                               decline_pred[i] * 0.1 + peak_pred[i] * 0.3 + 
                               transition_pred[i] * 0.1 + cat_pred[i] * 0.1)
        else:  # 일반 구간
            ensemble_pred[i] = (xgb_pred[i] * 0.2 + lgbm_pred[i] * 0.2 + 
                               decline_pred[i] * 0.15 + peak_pred[i] * 0.15 + 
                               transition_pred[i] * 0.15 + cat_pred[i] * 0.15)
    
    # log 스케일 역변환 및 클리핑
    if use_log_scale:
        ensemble_pred = np.expm1(ensemble_pred)
        ensemble_pred = np.clip(ensemble_pred, -0.1 if not is_insolation else 0, None)
        if not is_insolation:
            ensemble_pred = np.maximum(ensemble_pred, 0)
    else:
        if is_insolation:
            ensemble_pred = np.clip(ensemble_pred, 0, None)
        else:
            ensemble_pred = np.clip(ensemble_pred, 0, 1.3)
            ensemble_pred = np.minimum(ensemble_pred, 1.0)
    
    return ensemble_pred

def add_sunshine_bin_onehot(df, col='일조(hr)', prefix='sun_bin'):
    """
    일조량을 구간화한 후, 구간별로 원-핫 인코딩된 피처를 추가합니다.
    
    Parameters:
    - df: 입력 DataFrame
    - col: 일조량 컬럼 이름 (기본: '일조(hr)')
    - prefix: 원핫 피처 prefix 이름 (기본: 'sun_bin')

    반환값:
    - df: 구간 힌트 피처들이 추가된 DataFrame
    """
    def sunshine_bin(x):
        if pd.isna(x): 
            return -1
        elif x == 0: 
            return 0
        elif x < 0.2: 
            return 1     # 매우 낮음
        elif x < 0.5: 
            return 2     # 약간 흐림
        elif x < 0.8: 
            return 3
        elif x < 1: 
            return 4     # 중간
        elif x == 1: 
            return 5
        else:
            return 5     # 1 초과하는 경우도 처리

    bin_col = f"{prefix}_cat"
    
    # 안전한 방식으로 구간화 적용
    df[bin_col] = df[col].apply(sunshine_bin)
    
    # NaN 처리 후 정수형 변환
    df[bin_col] = pd.to_numeric(df[bin_col], errors='coerce').fillna(-1).astype(int)

    # 원핫 인코딩
    onehot_df = pd.get_dummies(df[bin_col], prefix=prefix)
    df = pd.concat([df, onehot_df], axis=1)
    df.drop(columns=[bin_col], inplace=True)

    return df

def identify_feature_types(columns):
    """
    피처 타입을 식별하여 스케일링 여부 결정
    """
    # One-hot 인코딩된 피처들 (스케일링 X)
    onehot_patterns = ['요일_', 'sun_bin_', '_이상치', '_설치여부', '강수시작', '강수끝', '강수강도', 
                       '주말여부', '근무시간', 'peak_time', '건물번호_encoded']
    
    # 건물 유형 (스케일링 X)
    building_type_patterns = [col for col in columns if '건물유형_' in col]
    
    categorical_features = []
    continuous_features = []
    
    for col in columns:
        # One-hot 인코딩된 피처들
        if any(pattern in col for pattern in onehot_patterns):
            categorical_features.append(col)
        # 건물 유형
        elif any(pattern in col for pattern in building_type_patterns):
            categorical_features.append(col)
        # 연속형 피처들
        else:
            continuous_features.append(col)
    
    return categorical_features, continuous_features

def apply_feature_scaling(X_train, X_val, feature_cols, scaler_type='standard'):
    """
    연속형 피처에만 스케일링 적용
    """
    print(f"    > {scaler_type} 스케일링 적용 중...")
    
    # 피처 타입 식별
    categorical_features, continuous_features = identify_feature_types(feature_cols)
    
    print(f"    > 카테고리 피처 수: {len(categorical_features)}")
    print(f"    > 연속형 피처 수: {len(continuous_features)}")
    
    # 스케일러 선택
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    # 연속형 피처만 추출
    continuous_features_in_data = [col for col in continuous_features if col in X_train.columns]
    
    if len(continuous_features_in_data) > 0:
        # 연속형 피처에만 스케일링 적용
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        
        # 스케일러 학습 및 변환
        X_train_scaled[continuous_features_in_data] = scaler.fit_transform(X_train[continuous_features_in_data])
        X_val_scaled[continuous_features_in_data] = scaler.transform(X_val[continuous_features_in_data])
        
        print(f"    > {len(continuous_features_in_data)}개 연속형 피처에 스케일링 적용 완료")
        return X_train_scaled, X_val_scaled, scaler, continuous_features_in_data
    else:
        print(f"    > 스케일링할 연속형 피처가 없음")
        return X_train, X_val, None, []

def apply_test_scaling(X_test, scaler, continuous_features_in_data):
    """
    테스트 데이터에 동일한 스케일링 적용
    """
    if scaler is not None and len(continuous_features_in_data) > 0:
        X_test_scaled = X_test.copy()
        X_test_scaled[continuous_features_in_data] = scaler.transform(X_test[continuous_features_in_data])
        return X_test_scaled
    else:
        return X_test

def train_global_model_with_building_features(train_data, features, target_col, building_col='건물번호', 
                                            scaler_type='robust', optimize_params=False):
    """
    전체 데이터로 학습하되 건물별 특성을 반영하는 모델 훈련 (잔차학습 없음 + 스케일링 + Optuna 최적화)
    """
    print(f"전체 데이터로 {target_col} 모델 학습 시작... (건물번호 피처 포함, 잔차학습 없음, {scaler_type} 스케일링)")
    if optimize_params:
        print("    > Optuna 하이퍼파라미터 최적화 활성화")
    
    # 건물 번호를 라벨 인코딩으로 추가
    le = LabelEncoder()
    building_features = features + [building_col]
    X_full = train_data[building_features].copy()
    y_full = train_data[target_col]
    
    # 건물번호를 라벨 인코딩
    X_full[f'{building_col}_encoded'] = le.fit_transform(X_full[building_col])
    
    # 건물번호 원본은 제거하고 인코딩된 버전 사용
    feature_cols = [col for col in X_full.columns if col != building_col]
    X_full_final = X_full[feature_cols]
    
    # Train-Validation 분할
    X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
        X_full_final, y_full, test_size=0.2, random_state=SEED, shuffle=True
    )
    
    # 피처 스케일링 적용
    X_train_scaled, X_val_scaled, scaler, continuous_features_in_data = apply_feature_scaling(
        X_train_global, X_val_global, feature_cols, scaler_type
    )
    
    # 잔차학습 없는 앙상블 모델 학습 (Optuna 최적화 포함)
    global_models = train_ensemble_models_enhanced(
        X_train_scaled, y_train_global, X_val_scaled, y_val_global, 
        is_insolation=(target_col == '일사(MJ/m2)'), use_log_scale=True, optimize_params=optimize_params
    )
    
    return global_models, le, feature_cols, scaler, continuous_features_in_data

def predict_with_global_model(test_data, global_models, label_encoder, 
                            feature_cols, target_col, scaler, continuous_features_in_data, building_col='건물번호'):
    """
    전체 모델로 건물별 예측 수행 (잔차학습 없음 + 스케일링)
    """
    predictions = {}
    
    # 테스트 데이터에서 훈련에 사용된 건물들만 필터링
    train_buildings = set(label_encoder.classes_)
    test_buildings = set(test_data[building_col].unique())
    valid_buildings = train_buildings.intersection(test_buildings)
    
    # 테스트 데이터 준비
    test_encoded = test_data.copy()
    
    # 훈련에 없었던 건물은 평균값으로 대체
    unknown_buildings = test_buildings - train_buildings
    if unknown_buildings:
        # 가장 많은 건물의 인코딩값으로 대체 (보통 0)
        default_encoding = 0
        test_encoded.loc[test_encoded[building_col].isin(unknown_buildings), building_col] = label_encoder.classes_[default_encoding]
    
    # 라벨 인코딩 적용
    test_encoded[f'{building_col}_encoded'] = label_encoder.transform(test_encoded[building_col])
    X_test_final = test_encoded[feature_cols]
    
    # 테스트 데이터에 스케일링 적용
    X_test_scaled = apply_test_scaling(X_test_final, scaler, continuous_features_in_data)
    
    # 전체 모델로 예측 (잔차학습 없음)
    all_predictions = predict_with_ensemble_enhanced(
        global_models, X_test_scaled, 
        is_insolation=(target_col == '일사(MJ/m2)')
    )
    
    # 건물별로 예측 결과 저장
    for i, building_num in enumerate(test_data[building_col]):
        if building_num not in predictions:
            predictions[building_num] = []
        predictions[building_num].append(all_predictions[i])
    
    return predictions, all_predictions

def weather_consistency_correction_v2(predictions, weather_data):
    """개선된 날씨 논리 기반 일조량 보정 함수"""
    corrected = predictions.copy()
    
    # 기본 정보 추출
    humidity = weather_data['습도(%)']
    rainfall = weather_data['강수량(mm)']
    dewpoint_diff = weather_data.get('이슬점차이', humidity * 0)  # 없으면 0으로 처리
    
    # === 1. 강수 기반 보정 (가장 강력한 영향) ===
    # 강수량에 따른 점진적 감소
    rain_factor = np.where(rainfall > 0, 
                          np.clip(1 - np.exp(-rainfall * 2), 0.8, 1.0),  # 지수 감소
                          1.0)
    corrected *= (1 - rain_factor + 0.1)  # 최소 10%는 유지
    
    # === 2. 습도 기반 보정 (점진적 적용) ===
    # 습도 70% 이상부터 점진적 감소
    humidity_factor = np.where(humidity >= 70,
                              np.clip((humidity - 70) / 25, 0, 0.7),  # 최대 70% 감소
                              0)
    corrected *= (1 - humidity_factor)
    
    # === 3. 이슬점차이 기반 보정 (안개/구름 조건) ===
    # 이슬점차이 3도 미만일 때 점진적 감소
    dewpoint_factor = np.where(dewpoint_diff < 3,
                              np.clip((3 - dewpoint_diff) / 3 * 0.6, 0, 0.6),  # 최대 60% 감소
                              0)
    corrected *= (1 - dewpoint_factor)
    
    # === 4. 복합 조건 보정 ===
    # 극단적 조건: 강수 + 높은 습도 + 낮은 이슬점차이
    extreme_mask = (rainfall > 0.5) & (humidity >= 85) & (dewpoint_diff < 1)
    corrected[extreme_mask] = np.minimum(corrected[extreme_mask], 0.05)  # 최대 5%만 허용
    
    # 안개 조건: 높은 습도 + 낮은 이슬점차이 + 강수 없음
    fog_mask = (humidity >= 90) & (dewpoint_diff < 1) & (rainfall == 0)
    corrected[fog_mask] *= 0.2  # 안개로 인한 80% 감소
    
    # 소나기 조건: 낮은 습도 + 강수 (대류성 강수)
    shower_mask = (humidity < 65) & (rainfall > 0.1)
    corrected[shower_mask] *= 0.1  # 소나기 시 90% 감소
    
    # === 5. 최종 클리핑 및 최소값 보장 ===
    # 물리적으로 불가능한 값 제거
    corrected = np.clip(corrected, 0, 1)
    
    # 극단적 감소 방지 (최소 1% 보장)
    corrected = np.maximum(corrected, 0.01)
    
    return corrected

###########################################################################
# 메인 실행 코드
###########################################################################
print(f"[1] 전처리 시작 (시계열 연속성 고려 + Optuna 최적화)")

########################## building_info 전처리 ############################
building_csv = pd.read_csv(data_path + 'building_info.csv')
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

########################## train & test 기본 전처리 ############################
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')

train_all = feature_engineering(train, is_train=True)

test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

test_all = feature_engineering(test, is_train=False)

########################## 시계열 연속성 고려한 lag 피처 추가 ############################
print(f"[1.5] 시계열 연속성을 고려한 lag 피처 생성...")

# train에는 모든 weather 정보
weather_lag_cols_train = ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)', '일조(hr)', '일사(MJ/m2)']
train_all = add_lag_features(train_all, weather_lag_cols_train, lag_hours=[1, 2])

# test에는 기상 데이터만, train과의 연속성 고려
weather_lag_cols_test = ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)']
test_all = add_lag_features_with_continuity(train_all, test_all, weather_lag_cols_test, lag_hours=[1, 2])

print(f"    > Train-Test 연속성 lag 피처 생성 완료")

# 나머지 피처들 추가
train_all = add_weather_rolling_features(train_all)
train_all = add_sunshine_rolling_features(train_all)
train_all = add_sunshine_bin_onehot(train_all)
train_all = add_weather_volatility_features(train_all)

# train_all NaN → 각 컬럼의 중위값으로 채우기
train_all = train_all.fillna(train_all.median(numeric_only=True))
# test_all NaN → 각 컬럼의 중위값으로 채우기
test_all = test_all.fillna(test_all.median(numeric_only=True))

test_all = add_weather_rolling_features(test_all)
test_all = add_weather_volatility_features(test_all)

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print(f"[2] train zero_bno 일사 예측 시작 (시계열 연속성 + Optuna 최적화 앙상블 모델)")

# 예측에 사용할 피처 선택
insolation_features = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'peak_time', '이슬점온도', '이슬점차이', '체감온도', 
    'SIN_시', 'COS_시', '시각', '대기투명도',
    'cloudy_or_rain', 'humid_x_rain',  
    'cloudy_based_on_humidity', 'daylight',  
    'solar_elevation', 'solar_rel_pos', 'extraterrestrial_rad',
    '날씨불안정지수', 'discomfort_index', 'THI', 'CDH',
    '일조(hr)'
]
# '기온_급변', '습도_급변', '풍속_급변', 
insolation_features.extend([col for col in train_all.columns if 'sun_bin' in col])
insolation_features.extend([col for col in train_all.columns if '강수강도_' in col])
insolation_features.extend([col for col in train_all.columns if '가시거리_' in col])
insolation_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '이슬점' in col and ('변화량' in col or 'rolling' in col)])
insolation_features.extend([col for col in train_all.columns if '체감' in col and ('변화량' in col or 'rolling' in col)])
insolation_features.extend([col for col in train_all.columns if 'lag_' in col and ('기온' in col or '습도' in col or '풍속' in col or '강수량' in col)])
insolation_features = list(set(insolation_features))
# print(f"insolation_features : {insolation_features}")

# train zero_bno 일사량 예측 (daylight==1인 데이터만 사용)
train_insolation_data = train_all[~train_all['건물번호'].isin(zero_bnos) & (train_all['daylight'] == 1)].copy()

insolation_models, insolation_label_encoder, insolation_feature_cols, insolation_scaler, insolation_continuous_features = train_global_model_with_building_features(
    train_insolation_data, insolation_features, '일사(MJ/m2)', '건물번호', scaler_type='robust', optimize_params=True
)

print(f"전체 모델 학습 완료, 이제 건물별로 예측 진행...")

# 건물별 예측 수행 (daylight==1인 데이터만)
for bno in zero_bnos:
    building_data = train_all[(train_all['건물번호'] == bno) & (train_all['daylight'] == 1)].copy()
    
    if len(building_data) == 0:
        continue
    
    _, building_predictions = predict_with_global_model(
        building_data, insolation_models, insolation_label_encoder, 
        insolation_feature_cols, '일사(MJ/m2)', insolation_scaler, insolation_continuous_features, '건물번호'
    )
    
    # # 예측값에 날씨 기반 보정 적용
    # building_predictions_corrected = weather_consistency_correction_v2(
    #     building_predictions, 
    #     building_data[['습도(%)', '강수량(mm)', '이슬점차이']]  # 필요한 날씨 컬럼만 전달
    # )
    
    # 보정된 예측값을 원본 데이터에 반영 (daylight==1인 경우만)
    train_all.loc[building_data.index, '일사(MJ/m2)'] = building_predictions
    print(f"    > 건물 {bno}번: 예측 완료 ({building_predictions.min():.4f} ~ {building_predictions.max():.4f})")

# daylight를 1시간 지연시켜서 마스킹
daylight_lagged = train_all.groupby('건물번호')['daylight'].shift(1).fillna(0)
# zero_bnos에 해당하는 건물들에 대해 daylight 지연 패턴 적용
train_all.loc[(train_all['건물번호'].isin(zero_bnos)) & (daylight_lagged == 0), '일사(MJ/m2)'] = 0

train_all = add_insolation_rolling_features(train_all)

print(f"[3] train zero_bno 일사 예측 완료")

########################## test 일조시간 예측 ############################
print(f"[4] test 일조시간 예측 시작 (시계열 연속성 고려 + Optuna 최적화)")

sunshine_features = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'peak_time', '이슬점온도', '이슬점차이', 
    'SIN_시', 'COS_시', '시각', '대기투명도',
    'cloudy_or_rain', 'humid_x_rain', 'cloudy_based_on_humidity', 'daylight', 
    'solar_elevation', 'solar_rel_pos', 'extraterrestrial_rad',
    '날씨불안정지수',
]
# '기온_급변', '습도_급변', '풍속_급변',
sunshine_features.extend([col for col in train_all.columns if '강수강도_' in col])
sunshine_features.extend([col for col in train_all.columns if '가시거리_' in col])
sunshine_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '이슬점' in col and ('변화량' in col or 'rolling' in col)])
sunshine_features.extend([col for col in train_all.columns if 'lag_' in col and ('기온' in col or '습도' in col or '풍속' in col or '강수량' in col)])
sunshine_features = list(set(sunshine_features))
# print(f"sunshine_features : {sunshine_features}")

# daylight==1인 데이터만으로 일조시간 모델 훈련
train_sunshine_data = train_all[train_all['daylight'] == 1].copy()

sunshine_models, sunshine_label_encoder, sunshine_feature_cols, sunshine_scaler, sunshine_continuous_features = train_global_model_with_building_features(
    train_sunshine_data, sunshine_features, '일조(hr)', '건물번호', scaler_type='robust', optimize_params=True
)

# test에서 daylight==1인 데이터만 예측
test_daylight_data = test_all[test_all['daylight'] == 1].copy()
test_daylight_indices = test_daylight_data.index

if len(test_daylight_data) > 0:
    sunshine_predictions_dict, daylight_sunshine_predictions = predict_with_global_model(
        test_daylight_data, sunshine_models, sunshine_label_encoder, 
        sunshine_feature_cols, '일조(hr)', sunshine_scaler, sunshine_continuous_features, '건물번호'
    )
    
    # daylight==1인 데이터에만 예측값 할당
    test_all.loc[test_daylight_indices, '일조(hr)'] = daylight_sunshine_predictions

# daylight==0인 경우 일조시간을 0으로 설정
test_all.loc[test_all['daylight'] == 0, '일조(hr)'] = 0

test_all = add_sunshine_rolling_features(test_all)
test_all = add_sunshine_bin_onehot(test_all)
weather_lag_cols_sunshine = ['일조(hr)']
test_all = add_lag_features_with_continuity(train_all, test_all, weather_lag_cols_sunshine, lag_hours=[1, 2])
test_all = test_all.fillna(0)
print(f"[5] test 일조시간 예측 완료 - Validation smape: {sunshine_models['final_mae']:.6f}")

########################## test 일사량 예측 ############################
print(f"[6] test 일사량 예측 시작 (시계열 연속성 고려 + Optuna 최적화)")

insolation_features_test = sunshine_features + ['일조(hr)', '체감온도', 'discomfort_index', 'THI', 'CDH']
insolation_features_test.extend([col for col in test_all.columns if 'sun_bin' in col])
insolation_features_test.extend([col for col in test_all.columns if '체감' in col and ('변화량' in col or 'rolling' in col)])
insolation_features_test.extend([col for col in test_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features_test = list(set(insolation_features_test))
# print(f"insolation_features_test : {insolation_features_test}")

# daylight==1인 데이터만으로 일사량 모델 훈련
train_insolation_test_data = train_all[train_all['daylight'] == 1].copy()

insolation_test_models, insolation_test_label_encoder, insolation_test_feature_cols, insolation_test_scaler, insolation_test_continuous_features = train_global_model_with_building_features(
    train_insolation_test_data, insolation_features_test, '일사(MJ/m2)', '건물번호', scaler_type='robust', optimize_params=True
)

# test에서 daylight==1인 데이터만 예측
test_daylight_data_insolation = test_all[test_all['daylight'] == 1].copy()
test_daylight_indices_insolation = test_daylight_data_insolation.index

if len(test_daylight_data_insolation) > 0:
    insolation_predictions_dict, daylight_insolation_predictions = predict_with_global_model(
        test_daylight_data_insolation, insolation_test_models, insolation_test_label_encoder, 
        insolation_test_feature_cols, '일사(MJ/m2)', insolation_test_scaler, insolation_test_continuous_features, '건물번호'
    )
    
    # daylight==1인 데이터에만 예측값 할당
    test_all.loc[test_daylight_indices_insolation, '일사(MJ/m2)'] = daylight_insolation_predictions

# daylight 지연 패턴 적용
daylight_lagged = test_all.groupby('건물번호')['daylight'].shift(1).fillna(0)
test_all.loc[daylight_lagged == 0, '일사(MJ/m2)'] = 0

print(f"[7] test 일사량 예측 완료 - Validation smape: {insolation_test_models['final_mae']:.6f}")
test_all = add_insolation_rolling_features(test_all)

# 일조/일사량 예측 완료 후 해당 lag 피처들을 test 데이터에 추가
print(f"[7.5] test 데이터에 일조/일사량 lag 피처 추가...")
weather_lag_cols_insolation = ['일사(MJ/m2)']
test_all = add_lag_features_with_continuity(train_all, test_all, weather_lag_cols_insolation, lag_hours=[1, 2])
test_all = test_all.fillna(0)

# 결과 저장
train_all.to_csv(trainer + f'06_train_{SEED}_optuna.csv', index=False)
test_all.to_csv(trainer + f'06_test_{SEED}_optuna.csv', index=False)

print(train_all.shape)
print(test_all.shape)

print(f"[8] 저장 완료 (시계열 연속성 반영 + Optuna 최적화)")

# 결과 요약
print("\n" + "="*70)
print("최종 결과 요약 (시계열 연속성 + 전체 학습 + Optuna 최적화 앙상블 + daylight==1 조건)")
print("="*70)
print(f"Train 일사량 예측 smape: {insolation_models['final_mae']:.6f}")
print(f"Test 일조시간 예측 Validation smape: {sunshine_models['final_mae']:.6f}")
print(f"Test 일사량 예측 Validation smape: {insolation_test_models['final_mae']:.6f}")

# Optuna 최적화 결과 출력
if 'optimized_params' in insolation_models:
    print("\n[Optuna 최적화 결과]")
    for model_name, params in insolation_models['optimized_params'].items():
        print(f"{model_name.upper()} 최적 파라미터:")
        for key, value in params.items():
            if key not in ['random_state', 'n_jobs', 'verbosity', 'thread_count', 'verbose', 'allow_writing_files', 'force_col_wise']:
                print(f"  - {key}: {value}")

# 로그 파일에 결과 저장
with open(log_path + "(preprocessing)LOG.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"Train 일사량 예측 smape : {insolation_models['final_mae']:.6f}\n")
    f.write(f"Test 일조(hr)     smape : {sunshine_models['final_mae']:.6f}\n")
    f.write(f"Test 일사(MJ/m2)  smape : {insolation_test_models['final_mae']:.6f}\n")
    f.write("="*50 + "\n")
    
print(f"[preprocessing] 종료")