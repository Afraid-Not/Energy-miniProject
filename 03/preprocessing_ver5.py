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
import lightgbm as lgb

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
    
    
    # 습도 기반 구름량 (0-1, 기본)
    df['cloud_cover_humidity'] = np.clip((df['습도(%)'] - 30) / 60, 0, 1)
    
    # 이슬점 차이 기반 구름량 (더 정확)
    df['cloud_cover_dewpoint'] = np.exp(-df['이슬점차이'] / 4)
    df['cloud_cover_dewpoint'] = np.clip(df['cloud_cover_dewpoint'], 0, 1)
    
    # 강수 기반 구름량 
    df['cloud_cover_rain'] = np.where(df['강수량(mm)'] > 0, 1.0, 
                                      np.clip(df['강수량(mm)'] / 2, 0, 1))
    df['cloud_cover_combined'] = np.maximum.reduce([
        df['cloud_cover_humidity'],
        df['cloud_cover_dewpoint'], 
        df['cloud_cover_rain']
    ])
    df['cloud_cover_weighted'] = (
        0.4 * df['cloud_cover_humidity'] + 
        0.4 * df['cloud_cover_dewpoint'] + 
        0.2 * df['cloud_cover_rain']
    )

    df['low_cloud_prob'] = np.where(
        (df['습도(%)'] >= 85) & (df['이슬점차이'] <= 2), 1.0,
        np.clip((df['습도(%)'] - 70) / 15, 0, 1) * np.clip((3 - df['이슬점차이']) / 3, 0, 1)
    )
    df['mid_cloud_prob'] = np.where(
        (df['습도(%)'].between(60, 80)) & (df['이슬점차이'].between(2, 5)), 
        0.8, np.clip((75 - np.abs(df['습도(%)'] - 70)) / 75, 0, 1)
    )
    df['high_cloud_prob'] = np.where(
        (df['습도(%)'] < 60) & (df['이슬점차이'] > 5) & (df['대기투명도'] < 0.7),
        0.6, 0.0
    )
    df['cumulonimbus_prob'] = np.where(
        (df['강수량(mm)'] > 1) & (df['습도(%)'] >= 80) & (df['불쾌지수'] > 75),
        1.0, 0.0
    )

    # 구름량을 8단계로 세분화 (0=맑음, 8=완전흐림)
    df['cloud_okta'] = np.round(df['cloud_cover_combined'] * 8).astype(int)
    df['sky_condition'] = np.select([
        df['cloud_okta'] == 0,
        df['cloud_okta'].between(1, 2),
        df['cloud_okta'].between(3, 5),
        df['cloud_okta'].between(6, 7),
        df['cloud_okta'] == 8
    ], [
        0,  # 맑음 (Clear)
        1,  # 약간 흐림 (Few clouds)
        2,  # 구름 많음 (Scattered)
        3,  # 흐림 (Broken)
        4   # 완전 흐림 (Overcast)
    ], default=2)

    df['cloud_blocking_factor'] = df['cloud_cover_combined'] ** 0.7  # 지수 함수로 비선형 효과
    df['expected_sunshine_ratio'] = (1 - df['cloud_blocking_factor']) * df['daylight']
    df['solar_reduction_factor'] = np.select([
        df['low_cloud_prob'] > 0.7,   # 저층운이 많으면
        df['mid_cloud_prob'] > 0.7,   # 중층운이 많으면  
        df['high_cloud_prob'] > 0.5,  # 고층운이 많으면
        df['cumulonimbus_prob'] > 0.5 # 적란운이 있으면
    ], [
        0.8,  # 80% 감소
        0.6,  # 60% 감소
        0.3,  # 30% 감소
        0.9   # 90% 감소 (뇌우)
    ], default=df['cloud_cover_combined'] * 0.5)
    df['morning_fog_prob'] = np.where(
        (df['시각'].between(6, 9)) & (df['습도(%)'] >= 90) & (df['이슬점차이'] <= 1),
        1.0, 0.0
    )
    df['afternoon_cumulus_prob'] = np.where(
        (df['시각'].between(12, 16)) & (df['기온(°C)'] > 25) & (df['습도(%)'].between(60, 80)),
        0.7, 0.0
    )
    df['evening_stratus_prob'] = np.where(
        (df['시각'].between(17, 20)) & (df['습도(%)'] >= 75),
        0.6, 0.0
    )
    df['atmospheric_instability'] = (df['기온(°C)'] - 20) * (df['습도(%)'] / 100) * (1 - df['이슬점차이'] / 10)
    df['atmospheric_instability'] = np.clip(df['atmospheric_instability'], 0, 5)
    df['convective_cloud_prob'] = np.clip(df['atmospheric_instability'] / 3, 0, 1)
    df['wind_dispersion_factor'] = np.clip(1 - df['풍속(m/s)'] / 15, 0.5, 1.0)
    df['cloud_cover_wind_adjusted'] = df['cloud_cover_combined'] * df['wind_dispersion_factor']
    df['final_cloud_index'] = (
        0.3 * df['cloud_cover_humidity'] +
        0.3 * df['cloud_cover_dewpoint'] + 
        0.2 * df['cloud_cover_rain'] +
        0.1 * df['convective_cloud_prob'] +
        0.1 * df['low_cloud_prob']
    ) * df['wind_dispersion_factor']
    df['final_cloud_index'] = np.clip(df['final_cloud_index'], 0, 1)
    
    return df

def create_temporal_features_for_variables(train_df, test_df, variables, 
                                          time_col='일시', group_col='건물번호',
                                          lag_hours=[1, 2, 3], 
                                          rolling_windows=[3, 6, 12, 24],
                                          change_hours=[1, 3],
                                          std_windows=[3, 6],
                                          fill_method='median'):
    """
    시계열 연속성을 고려하여 여러 변수에 대해 lag/rolling 피처를 생성하는 함수
    
    Parameters:
    -----------
    train_df, test_df : DataFrame
        학습용, 테스트용 데이터프레임
    variables : list
        시계열 피처를 생성할 변수 리스트 (예: ['기온(°C)', '일조(hr)', '습도(%)'])
    time_col : str
        시간 컬럼명 (기본: '일시')
    group_col : str
        그룹 컬럼명 (기본: '건물번호')
    lag_hours : list
        lag 시간 리스트 (기본: [1, 2, 3])
    rolling_windows : list
        rolling window 크기 리스트 (기본: [3, 6, 12, 24])
    change_hours : list
        변화율 계산 시간 리스트 (기본: [1, 3])
    std_windows : list
        표준편차 window 크기 리스트 (기본: [3, 6])
    fill_method : str
        NaN 채우기 방법 ('median', 'mean', 'forward', 'backward')
        
    Returns:
    --------
    train_with_temporal, test_with_temporal : DataFrame
        시계열 피처가 추가된 학습용, 테스트용 데이터프레임
    created_features : list
        생성된 피처명 리스트
    """
    
    print(f"\n=== 시계열 피처 생성 시작 ===")
    print(f"대상 변수: {variables}")
    print(f"Lag: {lag_hours}시간, Rolling: {rolling_windows}시간")
    print(f"변화율: {change_hours}시간, Std: {std_windows}시간")
    
    # 원본 데이터 복사
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # 데이터 길이 저장
    train_length = len(train_copy)
    test_length = len(test_copy)
    print(f"Train: {train_length}개, Test: {test_length}개")
    
    # 구분용 컬럼 추가
    train_copy['data_type'] = 'train'
    test_copy['data_type'] = 'test'
    
    # 시간 기준 정렬
    train_sorted = train_copy.sort_values([group_col, time_col]).reset_index(drop=True)
    test_sorted = test_copy.sort_values([group_col, time_col]).reset_index(drop=True)
    
    # 데이터 결합
    combined_data = pd.concat([train_sorted, test_sorted], ignore_index=True)
    print(f"결합된 데이터: {len(combined_data)}개")
    
    def create_features_for_group(group_df):
        """그룹별 시계열 피처 생성"""
        group_df = group_df.copy()
        created_cols = []
        
        for var in variables:
            if var not in group_df.columns:
                print(f"Warning: {var} 컬럼이 존재하지 않습니다.")
                continue
                
            # 변수명 정리 (특수문자 제거 및 정규화)
            var_clean = var.replace('(°C)', '').replace('(%)', '').replace('(mm)', '').replace('(m/s)', '').replace('(hr)', '').replace('(MJ/m2)', '')
            var_clean = var_clean.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            
            # 한글 변수명 영어로 변환 (옵션)
            var_mapping = {
                '기온': 'temp',
                '습도': 'humidity', 
                '강수량': 'rainfall',
                '풍속': 'windspeed',
                '일조': 'sunshine',
                '일사': 'solar'
            }
            for kor, eng in var_mapping.items():
                if kor in var_clean:
                    var_clean = var_clean.replace(kor, eng)
            
            # 1. Lag 피처
            for lag in lag_hours:
                col_name = f'{var_clean}_lag{lag}h'
                group_df[col_name] = group_df[var].shift(lag)
                created_cols.append(col_name)
            
            # 2. Rolling 평균
            for window in rolling_windows:
                col_name = f'{var_clean}_roll{window}h'
                group_df[col_name] = group_df[var].rolling(window=window, min_periods=1).mean()
                created_cols.append(col_name)
            
            # 3. 변화율 피처
            for change_hour in change_hours:
                if f'{var_clean}_lag{change_hour}h' in group_df.columns:
                    col_name = f'{var_clean}_change{change_hour}h'
                    group_df[col_name] = group_df[var] - group_df[f'{var_clean}_lag{change_hour}h']
                    created_cols.append(col_name)
            
            # 4. Rolling 표준편차
            for std_window in std_windows:
                col_name = f'{var_clean}_roll{std_window}h_std'
                group_df[col_name] = group_df[var].rolling(window=std_window, min_periods=1).std()
                created_cols.append(col_name)
        
        return group_df, created_cols
    
    # 그룹별 피처 생성
    print("그룹별 시계열 피처 생성 중...")
    temporal_features_list = []
    all_created_features = []
    
    for group_id in tqdm(combined_data[group_col].unique()):
        group_data = combined_data[combined_data[group_col] == group_id].copy()
        group_data = group_data.sort_values(time_col).reset_index(drop=True)
        group_data_with_features, created_cols = create_features_for_group(group_data)
        temporal_features_list.append(group_data_with_features)
        
        if not all_created_features:  # 첫 번째 그룹에서만 컬럼명 저장
            all_created_features = created_cols
    
    # 결합
    combined_with_temporal = pd.concat(temporal_features_list, ignore_index=True)
    print(f"시계열 피처 추가 완료: {len(combined_with_temporal)}개")
    
    # NaN 처리
    print(f"\n=== NaN 값 처리 ({fill_method}) ===")
    for col in all_created_features:
        if col in combined_with_temporal.columns:
            nan_count = combined_with_temporal[col].isna().sum()
            if nan_count > 0:
                if fill_method == 'median':
                    fill_val = combined_with_temporal[col].median()
                    combined_with_temporal[col] = combined_with_temporal[col].fillna(fill_val)
                elif fill_method == 'mean':
                    fill_val = combined_with_temporal[col].mean()
                    combined_with_temporal[col] = combined_with_temporal[col].fillna(fill_val)
                elif fill_method == 'forward':
                    combined_with_temporal[col] = combined_with_temporal[col].fillna(method='ffill')
                    fill_val = "forward fill"
                elif fill_method == 'backward':
                    combined_with_temporal[col] = combined_with_temporal[col].fillna(method='bfill')
                    fill_val = "backward fill"
                
                if fill_method in ['median', 'mean']:
                    print(f"{col}: {nan_count}개 NaN → {fill_method}({fill_val:.4f})로 채움")
                else:
                    print(f"{col}: {nan_count}개 NaN → {fill_val}로 채움")
    
    # train/test 재분리
    print(f"\n=== train/test 재분리 ===")
    train_with_temporal = combined_with_temporal[combined_with_temporal['data_type'] == 'train'].copy()
    test_with_temporal = combined_with_temporal[combined_with_temporal['data_type'] == 'test'].copy()
    
    # data_type 컬럼 제거
    train_with_temporal = train_with_temporal.drop('data_type', axis=1)
    test_with_temporal = test_with_temporal.drop('data_type', axis=1)
    
    print(f"분리 후 Train: {len(train_with_temporal)}개")
    print(f"분리 후 Test: {len(test_with_temporal)}개")
    
    # 생성된 피처 정보 출력
    print(f"\n=== 생성된 시계열 피처 ({len(all_created_features)}개) ===")
    for i, col in enumerate(all_created_features):
        if i < 10:  # 처음 10개만 출력
            print(f"- {col}")
        elif i == 10:
            print(f"... 외 {len(all_created_features)-10}개")
            break
    
    return train_with_temporal, test_with_temporal, all_created_features

def train_xgb_lgb_ensemble(train_data, test_data, target_col, features, model_name=""):
    """XGBoost + LightGBM 앙상블 예측 함수 - 낮시간만 학습"""
    print(f"\n{'='*60}")
    print(f"{model_name} XGBoost + LightGBM 앙상블 예측 시작 (낮시간만 학습)")
    print(f"{'='*60}")
    
    # 사용 가능한 피처만 필터링
    available_features = [f for f in features if f in train_data.columns]
    print(f"사용 피처 수: {len(available_features)}")
    
    # 데이터 준비 (낮시간만)
    X = train_data[available_features].fillna(0)
    y = train_data[target_col]
    test_X = test_data[available_features].fillna(0)
    
    print(f"학습 데이터 크기: {len(X)}개 (낮시간만)")
    print(f"예측 데이터 크기: {len(test_X)}개 (낮시간만)")
    
    # Train/Valid 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost 모델
    xgb_model = XGBRegressor(
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
    
    # LightGBM 모델
    lgb_model = LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # 모델 학습
    print("XGBoost 모델 학습 중...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print("LightGBM 모델 학습 중...")
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # 검증 예측
    xgb_val_pred = xgb_model.predict(X_val)
    lgb_val_pred = lgb_model.predict(X_val)
    
    # 개별 모델 성능
    xgb_smape = smape(y_val, xgb_val_pred)
    lgb_smape = smape(y_val, lgb_val_pred)
    xgb_r2 = r2_score(y_val, xgb_val_pred)
    lgb_r2 = r2_score(y_val, lgb_val_pred)
    
    print(f"\n=== 개별 모델 검증 성능 (낮시간만) ===")
    print(f"XGBoost - SMAPE: {xgb_smape:.4f}, R2: {xgb_r2:.4f}")
    print(f"LightGBM - SMAPE: {lgb_smape:.4f}, R2: {lgb_r2:.4f}")
    
    # 성능 기반 가중치 계산
    if xgb_smape < lgb_smape:
        xgb_weight, lgb_weight = 0.7, 0.3
        better_model = "XGBoost"
    else:
        xgb_weight, lgb_weight = 0.3, 0.7
        better_model = "LightGBM"
    
    # 앙상블 검증 예측
    ensemble_val_pred = xgb_weight * xgb_val_pred + lgb_weight * lgb_val_pred
    ensemble_smape = smape(y_val, ensemble_val_pred)
    ensemble_r2 = r2_score(y_val, ensemble_val_pred)
    
    print(f"\n=== 앙상블 결과 (낮시간만) ===")
    print(f"가중치: XGBoost({xgb_weight:.1f}) + LightGBM({lgb_weight:.1f}) - {better_model}이 더 우수")
    print(f"앙상블 SMAPE: {ensemble_smape:.4f}")
    print(f"앙상블 R2: {ensemble_r2:.4f}")
    
    # 테스트 예측 (낮시간만)
    xgb_test_pred = xgb_model.predict(test_X)
    lgb_test_pred = lgb_model.predict(test_X)
    ensemble_test_pred = xgb_weight * xgb_test_pred + lgb_weight * lgb_test_pred
    
    print(f"\n=== 테스트 예측값 범위 (낮시간만) ===")
    print(f"XGBoost: {xgb_test_pred.min():.4f} ~ {xgb_test_pred.max():.4f}")
    print(f"LightGBM: {lgb_test_pred.min():.4f} ~ {lgb_test_pred.max():.4f}")
    print(f"앙상블: {ensemble_test_pred.min():.4f} ~ {ensemble_test_pred.max():.4f}")
    
    # 피처 중요도 출력 (XGBoost 기준)
    print(f"\n=== XGBoost 피처 중요도 Top 10 ===")
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10).to_string(index=False))
    
    return ensemble_test_pred, (xgb_model, lgb_model), test_data

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

# ===== 시계열 피처 생성 실행 =====
# 기온에 대해서만 시계열 피처 생성
train, test, temporal_features = create_temporal_features_for_variables(
    train_df=train,
    test_df=test, 
    variables=['기온(°C)', '습도(%)'],  # 기온만
    lag_hours=[1, 2, 3],
    rolling_windows=[1, 2, 3],
    change_hours=[1, 2, 3],
    std_windows=[3, 6],
    fill_method='median'
)

print("시계열 연속성을 고려한 lag/rolling 피처 생성 완료! 🕒📈")

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

# ===== 낮시간 데이터 필터링 =====
print(f"\n=== 낮시간(daylight=1) 필터링 ===")
print(f"전체 Train 데이터: {len(train)}개")
print(f"전체 Test 데이터: {len(test)}개")

# 낮시간만 필터링
train_daylight = train[train['daylight'] == 1].copy()
test_daylight = test[test['daylight'] == 1].copy()

print(f"낮시간 Train 데이터: {len(train_daylight)}개 ({len(train_daylight)/len(train)*100:.1f}%)")
print(f"낮시간 Test 데이터: {len(test_daylight)}개 ({len(test_daylight)/len(test)*100:.1f}%)")

# ===== 공통 피처 세트 정의 =====
# 일조 시간 예측용 피처 리스트
sunshine_prediction_features = [
    # 핵심 기상 변수
    '기온(°C)', '습도(%)', 
    
    # 이슬점 관련 (구름 예측 핵심)
    '이슬점온도', '이슬점차이', 
    
    # 대기 상태
    '대기투명도', 'perceived_temperature',
    
    # 강수/습도 관련
    'cloudy_or_rain', 'humid_x_rain',
    
    # 태양 위치 정보
    'solar_elevation', 'extraterrestrial_rad', 
    'daylight',
    
    # 시간 정보 (순환)
    'SIN_시', 'COS_시', 
    'SIN_day_of_year', 'COS_day_of_year',
    
    # 구름량 관련 (핵심!)
    'cloud_cover_dewpoint',        # 가장 정확한 구름량
    'cloud_cover_combined',        # 종합 구름량
    'expected_sunshine_ratio',     # 예상 일조율
    'cloud_blocking_factor',       # 구름 차단 효과
    'low_cloud_prob',             # 저층운 (일조에 큰 영향)
    'wind_dispersion_factor',     # 바람 분산 효과
    'final_cloud_index',          # 최종 구름 지수
    
    # 시간대별 구름 패턴
    'morning_fog_prob',           # 아침 안개
    'afternoon_cumulus_prob',     # 오후 적운
    
    # 지역 정보
    '건물번호', '지역그룹ID'
]

# 일사량 예측용 피처 리스트 (일조 포함)
solar_prediction_features = [
    # 기상 변수 (일조 추가!)
    '기온(°C)', '습도(%)', '일조(hr)',
    
    # 이슬점 관련
    '이슬점온도', '이슬점차이',
    
    # 대기 상태
    '대기투명도', 'perceived_temperature', 
    
    # 강수/습도 관련
    'cloudy_or_rain', 'humid_x_rain',
    
    # 태양 위치 정보 (일사량에 직접 영향)
    'solar_elevation', 'extraterrestrial_rad',
    'daylight',
    
    # 시간 정보
    'SIN_시', 'COS_시',
    'SIN_day_of_year', 'COS_day_of_year',
    
    'final_cloud_index',
    
    # 대기 안정도 (일사 산란 영향)
    'atmospheric_instability',
    'convective_cloud_prob',
    
    # 지역 정보
    '건물번호', '지역그룹ID'
]

patterns = ['temp_roll', 'temp_lag', 'temp_change', 'humidity_roll', 'humidity_lag', 'humidity_change']
solar_patterns = ['temp_roll', 'temp_lag', 'temp_change', 
                  'humidity_roll', 'humidity_lag', 'humidity_change',
                  'sunshine_roll', 'sunshine_lag', 'sunshine_change']

temp_cols = [col for col in train.columns if any(pattern in col for pattern in patterns)]
solar_temp_cols = [col for col in train.columns if any(pattern in col for pattern in patterns)]
sunshine_prediction_features.extend(temp_cols)
solar_prediction_features.extend(solar_temp_cols)










# ===== 1단계: 일조 시간 XGBoost + LightGBM 앙상블 예측 (낮시간만) =====
sunshine_ensemble_pred, sunshine_models, test_daylight_with_pred = train_xgb_lgb_ensemble(
    train_daylight, test_daylight, '일조(hr)', sunshine_prediction_features, "일조 시간"
)

sunshine_ensemble_pred_clipped = np.clip(sunshine_ensemble_pred, 0, 1)
print(f"\n=== 일조 클리핑 결과 ===")
print(f"클리핑 전: {sunshine_ensemble_pred.min():.4f} ~ {sunshine_ensemble_pred.max():.4f}")
print(f"클리핑 후: {sunshine_ensemble_pred_clipped.min():.4f} ~ {sunshine_ensemble_pred_clipped.max():.4f}")

# test 전체에 일조 추가 (낮시간은 예측값, 밤시간은 0)
test_copy = test.copy()
test_copy['일조(hr)'] = 0  # 모든 시간을 0으로 초기화
test_copy.loc[test_copy['daylight'] == 1, '일조(hr)'] = sunshine_ensemble_pred_clipped  # 낮시간만 예측값 입력

print(f"\n=== 전체 Test 데이터 일조 범위 ===")
print(f"밤시간 일조: {test_copy[test_copy['daylight'] == 0]['일조(hr)'].unique()}")
print(f"낮시간 일조: {test_copy[test_copy['daylight'] == 1]['일조(hr)'].min():.4f} ~ {test_copy[test_copy['daylight'] == 1]['일조(hr)'].max():.4f}")

# ===== 일조 rolling 피처 생성 =====
print(f"\n{'='*60}")
print("일조 예측 완료 후 일조 rolling 피처 생성 시작!")
print(f"{'='*60}")

# train에도 일조가 있으므로 train과 test 모두에 일조 rolling 피처 생성
train_with_sunshine_features, test_with_sunshine_features, sunshine_temporal_features = create_temporal_features_for_variables(
    train_df=train,  # 원본 train (일조 포함)
    test_df=test_copy,  # 일조 예측이 추가된 test
    variables=['일조(hr)'],  # 일조만
    lag_hours=[1, 2, 3],
    rolling_windows=[1, 2, 3, 6],
    change_hours=[1, 2],
    std_windows=[3, 6],
    fill_method='median'
)

print("일조 rolling 피처 생성 완료! 이제 일사량 예측에 활용 가능 🌞📈")

# ===== 일사량 예측용 피처 리스트 업데이트 (일조 rolling 피처 추가) =====
enhanced_solar_prediction_features = solar_prediction_features + sunshine_temporal_features

print(f"\n=== 일사량 예측 피처 업데이트 ===")
print(f"기존 일사량 피처: {len(solar_prediction_features)}개")
print(f"일조 rolling 피처: {len(sunshine_temporal_features)}개")
print(f"총 일사량 예측 피처: {len(enhanced_solar_prediction_features)}개")

# 추가된 일조 rolling 피처 확인
print(f"\n=== 추가된 일조 rolling 피처 ===")
for feat in sunshine_temporal_features[:10]:  # 처음 10개만 출력
    print(f"- {feat}")
if len(sunshine_temporal_features) > 10:
    print(f"... 외 {len(sunshine_temporal_features)-10}개")

# ===== 2단계: 일사량 XGBoost + LightGBM 앙상블 예측 (낮시간만, 일조 rolling 피처 포함) =====
# train 낮시간 데이터 준비 (일조 rolling 피처 포함)
train_daylight_with_sunshine = train_with_sunshine_features[train_with_sunshine_features['daylight'] == 1].copy()

# test 낮시간 데이터 준비 (예측된 일조 + rolling 피처 포함)
test_daylight_with_sunshine = test_with_sunshine_features[test_with_sunshine_features['daylight'] == 1].copy()

solar_ensemble_pred, solar_models, _ = train_xgb_lgb_ensemble(
    train_daylight_with_sunshine, test_daylight_with_sunshine, '일사(MJ/m2)', 
    enhanced_solar_prediction_features, "일사량 (일조 rolling 포함)"
)

# 일사량 클리핑 (0 ~ train_max)
solar_max = train['일사(MJ/m2)'].max()
solar_ensemble_pred_clipped = np.clip(solar_ensemble_pred, 0, solar_max)

print(f"\n=== 일사량 클리핑 결과 ===")
print(f"Train 일사량 최대값: {solar_max:.4f}")
print(f"클리핑 전: {solar_ensemble_pred.min():.4f} ~ {solar_ensemble_pred.max():.4f}")
print(f"클리핑 후: {solar_ensemble_pred_clipped.min():.4f} ~ {solar_ensemble_pred_clipped.max():.4f}")

# test 전체에 일사량 추가 (낮시간은 예측값, 밤시간은 0)
test_with_sunshine_features['일사(MJ/m2)'] = 0  # 모든 시간을 0으로 초기화
test_with_sunshine_features.loc[test_with_sunshine_features['daylight'] == 1, '일사(MJ/m2)'] = solar_ensemble_pred_clipped  # 낮시간만 예측값 입력

print(f"\n=== 전체 Test 데이터 일사량 범위 ===")
print(f"밤시간 일사량: {test_with_sunshine_features[test_with_sunshine_features['daylight'] == 0]['일사(MJ/m2)'].unique()}")
print(f"낮시간 일사량: {test_with_sunshine_features[test_with_sunshine_features['daylight'] == 1]['일사(MJ/m2)'].min():.4f} ~ {test_with_sunshine_features[test_with_sunshine_features['daylight'] == 1]['일사(MJ/m2)'].max():.4f}")

# ===== 일사량 rolling 피처 생성 =====
print(f"\n{'='*60}")
print("일사량 예측 완료 후 일사량 rolling 피처 생성 시작!")
print(f"{'='*60}")

# 최종적으로 train과 test 모두에 일사량 rolling 피처까지 생성
final_train, final_test, solar_temporal_features = create_temporal_features_for_variables(
    train_df=train_with_sunshine_features,  # 일조 rolling 피처가 포함된 train
    test_df=test_with_sunshine_features,   # 일조 + 일사량 예측이 추가된 test
    variables=['일사(MJ/m2)'],  # 일사량만
    lag_hours=[1, 2, 3],
    rolling_windows=[1, 2, 3, 6],
    change_hours=[1, 2],
    std_windows=[3, 6],
    fill_method='median'
)

print("일사량 rolling 피처 생성 완료! 📈⚡")

# ===== 최종 결과 =====
print(f"\n{'='*60}")
print("낮시간 필터링 XGBoost + LightGBM 앙상블 일조/일사량 예측 완료!")
print("일조 및 일사량 rolling 피처까지 모두 생성 완료!")
print(f"{'='*60}")

# 파일 저장
final_train.to_csv(trainer + 'new_train_ver5.csv', index=False)
final_test.to_csv(trainer + 'new_test_ver5.csv', index=False)
print("낮시간 필터링 XGBoost + LightGBM 앙상블 결과 파일 저장 완료!")

print(f"\n=== 최종 데이터 확인 ===")
print(f"Train shape: {final_train.shape}")
print(f"Test shape: {final_test.shape}")
print(f"Test에 추가된 컬럼: 일조(hr), 일사(MJ/m2) + rolling 피처들")

print(f"\n=== 생성된 피처 요약 ===")
print(f"✅ 일조 rolling 피처: {len(sunshine_temporal_features)}개")
print(f"✅ 일사량 rolling 피처: {len(solar_temporal_features)}개")
print(f"✅ 총 추가 시계열 피처: {len(sunshine_temporal_features) + len(solar_temporal_features)}개")

print(f"\n=== 낮시간 학습 + Rolling 피처 효과 ===")
print("✅ 밤시간 노이즈 제거 - 일조/일사량 0인 데이터 학습에서 제외")
print("✅ 낮시간 패턴 집중 학습 - 태양광 관련 피처들의 효과 극대화")
print("✅ 물리적 일관성 - 밤시간은 자동으로 0, 낮시간만 예측값 적용")
print("✅ XGBoost + LightGBM 앙상블 - 두 알고리즘의 장점 결합")
print("✅ 일조 rolling 피처 - 과거 일조 패턴으로 일사량 예측 정확도 향상")
print("✅ 일사량 rolling 피처 - 시계열 연속성 확보로 최종 모델링 성능 향상")

print(f"\n=== 학습 데이터 통계 ===")
print(f"전체 Train: {len(train)}개 → 낮시간 Train: {len(train_daylight)}개 (압축률: {len(train_daylight)/len(train)*100:.1f}%)")
print(f"전체 Test: {len(test)}개 → 낮시간 Test: {len(test_daylight)}개 (압축률: {len(test_daylight)/len(test)*100:.1f}%)")
print("낮시간만 학습하여 더 정확하고 효율적인 XGBoost + LightGBM 앙상블 예측 완료! 🌞🎯⚡")