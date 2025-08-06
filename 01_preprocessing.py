print(f"[preprocessing] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import math
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

warnings.filterwarnings('ignore')
pd.set_option('max_colwidth', None)

data_path = './Energy/'
log_path = './Energy/02/log/'
trainer = './Energy/02/new_csv/'
save_path = './Energy/02/submission/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

# 현재 seed 값 사용
SEED = 1 

random.seed(SEED)
np.random.seed(SEED)

print(f"[1] 전처리")

train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
""" 
print(train.columns)
['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
'일사(MJ/m2)', '전력소비량(kWh)']

print(test_csv.columns)
['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
 """

###################### 피쳐 엔지니어링 #####################
###################### 피쳐 엔지니어링 #####################
###################### 피쳐 엔지니어링 #####################
###################### 피쳐 엔지니어링 #####################


def create_day_pattern_mapping(train_df):
    """
    train 데이터에서 건물별 요일별 dayoff/hot_day 패턴을 학습하여 매핑 테이블 생성
    """
    # dayofweek 컬럼이 없으면 생성
    if 'dayofweek' not in train_df.columns:
        train_df['date'] = pd.to_datetime(train_df['일시'])
        train_df['dayofweek'] = train_df['date'].dt.dayofweek
    
    # 요일별 전력소비량 평균 계산 (건물번호별)
    building_dayofweek_avg = train_df.groupby(['건물번호', 'dayofweek'])['전력소비량(kWh)'].mean().reset_index()
    building_dayofweek_avg.columns = ['건물번호', 'dayofweek', 'avg_power_by_dayofweek']
    
    # 건물별 전체 평균 계산
    building_total_avg = train_df.groupby('건물번호')['전력소비량(kWh)'].mean().reset_index()
    building_total_avg.columns = ['건물번호', 'total_avg_power']
    
    # 두 평균값을 merge
    day_pattern_mapping = building_dayofweek_avg.merge(building_total_avg, on='건물번호')
    
    # 요일별 평균이 전체 평균보다 10% 이상 낮은 경우를 찾기 (dayoff)
    day_pattern_mapping['dayoff'] = (day_pattern_mapping['avg_power_by_dayofweek'] < 
                                    day_pattern_mapping['total_avg_power'] * 0.9).astype(int)
    
    # 요일별 평균이 전체 평균보다 10% 이상 높은 경우를 찾기 (hot_day)
    day_pattern_mapping['hot_day'] = (day_pattern_mapping['avg_power_by_dayofweek'] > 
                                     day_pattern_mapping['total_avg_power'] * 1.1).astype(int)
    
    return day_pattern_mapping[['건물번호', 'dayofweek', 'dayoff', 'hot_day']]

def apply_day_pattern_features(df, day_pattern_mapping):
    """
    미리 생성된 day_pattern_mapping을 사용해서 dayoff/hot_day 피쳐를 추가
    """
    # dayofweek 컬럼이 없으면 생성
    if 'dayofweek' not in df.columns:
        df['date'] = pd.to_datetime(df['일시'])
        df['dayofweek'] = df['date'].dt.dayofweek
    
    df_with_features = df.merge(
        day_pattern_mapping, 
        on=['건물번호', 'dayofweek'], 
        how='left'
    )
    
    # 매핑되지 않은 경우 0으로 채우기
    df_with_features['dayoff'] = df_with_features['dayoff'].fillna(0).astype(int)
    df_with_features['hot_day'] = df_with_features['hot_day'].fillna(0).astype(int)
    
    return df_with_features

def add_dayoff_hotday_feature(df):
    """
    건물번호별로 요일별 전력소비량 평균을 계산하고, 
    평균보다 10% 이상 낮은 요일을 'dayoff'로 표시
    평균보다 10% 이상 높은 요일을 'hot_day'로 표시
    (train 데이터에만 사용)
    """
    # 요일별 전력소비량 평균 계산 (건물번호별)
    building_dayofweek_avg = df.groupby(['건물번호', 'dayofweek'])['전력소비량(kWh)'].mean().reset_index()
    building_dayofweek_avg.columns = ['건물번호', 'dayofweek', 'avg_power_by_dayofweek']
    
    # 건물별 전체 평균 계산
    building_total_avg = df.groupby('건물번호')['전력소비량(kWh)'].mean().reset_index()
    building_total_avg.columns = ['건물번호', 'total_avg_power']
    
    # 두 평균값을 merge
    day_pattern_info = building_dayofweek_avg.merge(building_total_avg, on='건물번호')
    
    # 요일별 평균이 전체 평균보다 10% 이상 낮은 경우를 찾기 (dayoff)
    day_pattern_info['is_dayoff'] = (day_pattern_info['avg_power_by_dayofweek'] < 
                                    day_pattern_info['total_avg_power'] * 0.9).astype(int)
    
    # 요일별 평균이 전체 평균보다 10% 이상 높은 경우를 찾기 (hot_day)
    day_pattern_info['is_hot_day'] = (day_pattern_info['avg_power_by_dayofweek'] > 
                                     day_pattern_info['total_avg_power'] * 1.1).astype(int)
    
    # 원본 데이터에 merge하여 컬럼들 추가
    df_with_features = df.merge(
        day_pattern_info[['건물번호', 'dayofweek', 'is_dayoff', 'is_hot_day']], 
        on=['건물번호', 'dayofweek'], 
        how='left'
    )
    
    # dayoff와 hot_day 컬럼 추가
    df_with_features['dayoff'] = df_with_features['is_dayoff'].fillna(0).astype(int)
    df_with_features['hot_day'] = df_with_features['is_hot_day'].fillna(0).astype(int)
    df_with_features.drop(['is_dayoff', 'is_hot_day'], axis=1, inplace=True)
    
    return df_with_features


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

# 기존 engineering_features 함수에 dayoff 기능을 통합한 버전
def engineering_features_with_dayoff(df, is_train=True):
    ####### 시간 관련 피쳐 #######
    df['date'] = pd.to_datetime(df['일시'])
    df['hour'] = df['date'].dt.hour                      # 시각(0~23)
    df['dayofweek'] = df['date'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # 공휴일 체크 함수
    def is_holiday(row):
        month = row['month']
        day = row['day']
        # 현충일(6월 6일)과 광복절(8월 15일) 체크
        if (month == 6 and day == 6) or (month == 8 and day == 15):
            return True
        return False
    
    # 공휴일 여부 컬럼 추가
    df['is_holiday'] = df.apply(is_holiday, axis=1).astype(int)
    
    # 주말 또는 공휴일을 weekend로 처리
    df['is_weekend'] = ((df['dayofweek'] >= 5) | (df['is_holiday'] == 1)).astype(int)
    
    df['is_working'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  # 주기적 패턴
    df['COS_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['SIN_day'] = np.sin(2 * np.pi * df['day'] / 31)  # 일의 주기적 패턴 (31일 기준)
    df['COS_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['SIN_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['peak_time'] = df['hour'].apply(lambda x: 1 if 11 <= x <= 16 else 0) # Peak time (11 AM to 4 PM)
    
    # dayoff와 hot_day 피쳐 추가 (train 데이터에만 적용)
    if is_train and '전력소비량(kWh)' in df.columns:
        df = add_dayoff_hotday_feature(df)
    
    ####### 강수 및 습도 관련 피쳐 #######
    df['rainy'] = (df['강수량(mm)'] > 0).astype(int)
    df['high_humidity'] = (df['습도(%)'] >= 85).astype(int)
    df['cloudy_or_rain'] = ((df['습도(%)'] >= 80) | (df['강수량(mm)'] > 0)).astype(int)
    df['humid_x_rain'] = df['습도(%)'] * df['강수량(mm)']
    df['discomfort_index'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    # 쾌적지수(THI)
    df['THI'] = 0.8 * df['기온(°C)'] + df['습도(%)'] * (df['기온(°C)'] - 14.4) / 100 + 46.4
    df['CDH'] = np.maximum(df['기온(°C)'] - 26, 0)
    # 풍속 감안 체감온도(WCT)
    df['WCT'] = 13.12 + 0.6215*df['기온(°C)'] - 11.37*(df['풍속(m/s)']**0.16) + 0.3965*df['기온(°C)']*(df['풍속(m/s)']**0.16)
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    wind_speed = df['풍속(m/s)']
    df['perceived_temperature'] = temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00
    a = 17.27
    b = 237.7
    T = df['기온(°C)']
    RH = df['습도(%)']
    gamma = (a * T) / (b + T) + np.log(RH / 100)
    df['dew_point'] = (b * gamma) / (a - gamma)
    
    cloudy_based_on_humidity = np.clip((df['습도(%)'] - 30) / 7, 0, 10)
    df['cloudy_based_on_humidity'] = np.where(df['강수량(mm)'] > 0, 10, cloudy_based_on_humidity)

    ####### 태양 관련 피쳐 #######
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear

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
    df['daylight'] = ((df['hour'] >= df['sunrise_hour']) & (df['hour'] <= df['sunset_hour'])).astype(int)

    # 태양 고도각
    time_decimal = df['hour'] + 0.5  # 30분 기준 (정시 측정이라면 그냥 hour)
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
    df['solar_rel_pos'] = (df['hour'] + 0.5 - df['sunrise_hour']) / (df['sunset_hour'] - df['sunrise_hour'])
    df['solar_rel_pos'] = df['solar_rel_pos'].clip(0, 1)
    
    G_sc = 0.0820  # 태양상수(MJ/m2/min)
    dr = 1 + 0.033 * np.cos(2 * np.pi * df['day_of_year'] / 365)  # 거리 계수
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec_rad))  # 일출/일몰 시각각(rad)
    I_0 = (24*60/np.pi) * G_sc * dr * (ws * np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.sin(ws))
    df['extraterrestrial_rad'] = I_0  # 단위: MJ/m2/day
    
    df['dew_point_temperature'] = calculate_dewpoint(df['기온(°C)'], df['습도(%)'])

    return df

def create_weather_features(train_df, test_df):
    """
    train과 test 데이터에 대해 lag, rolling, 급변 피쳐를 생성
    """
    # 기상 피쳐들
    weather_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    
    # 건물별로 데이터를 합치고 피쳐 생성
    all_buildings = sorted(train_df['건물번호'].unique())
    
    processed_train_list = []
    processed_test_list = []
    
    for building_num in tqdm(all_buildings, desc="Processing buildings"):
        # 해당 건물의 train, test 데이터 추출
        building_train = train_df[train_df['건물번호'] == building_num].copy().sort_values('date')
        building_test = test_df[test_df['건물번호'] == building_num].copy().sort_values('date')
        
        # train과 test를 시간순으로 합치기
        building_combined = pd.concat([building_train, building_test], ignore_index=True).sort_values('date')
        
        # 2. 24시간 lag 피쳐 생성
        for col in weather_cols:
            building_combined[f'{col}_lag24'] = building_combined[col].shift(24)
        
        # 3. Rolling 피쳐 생성 (1, 2, 3, 6시간)
        rolling_windows = [1, 2, 3, 6]
        
        for col in weather_cols:
            for window in rolling_windows:
                # Rolling mean
                building_combined[f'{col}_rolling_mean_{window}h'] = building_combined[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                building_combined[f'{col}_rolling_std_{window}h'] = building_combined[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
                
                # Rolling max
                building_combined[f'{col}_rolling_max_{window}h'] = building_combined[col].rolling(
                    window=window, min_periods=1
                ).max()
                
                # Rolling min
                building_combined[f'{col}_rolling_min_{window}h'] = building_combined[col].rolling(
                    window=window, min_periods=1
                ).min()
        
        # 4. 급변 피쳐 생성
        building_combined = add_weather_volatility_features_single_building(building_combined)
        
        # 5. NaN 값 처리 (train 맨앞 24시간의 lag 피쳐들)
        for col in weather_cols:
            lag_col = f'{col}_lag24'
            if building_combined[lag_col].isna().any():
                # 해당 피쳐의 평균값으로 채우기
                mean_val = building_combined[col].mean()
                building_combined[lag_col] = building_combined[lag_col].fillna(mean_val)
        
        # 6. train과 test로 다시 분리
        train_len = len(building_train)
        
        processed_building_train = building_combined.iloc[:train_len].copy()
        processed_building_test = building_combined.iloc[train_len:].copy()
        
        processed_train_list.append(processed_building_train)
        processed_test_list.append(processed_building_test)
    
    # 모든 건물의 데이터를 다시 합치기
    final_train = pd.concat(processed_train_list, ignore_index=True)
    final_test = pd.concat(processed_test_list, ignore_index=True)
    

    
    return final_train, final_test


def add_weather_volatility_features_single_building(df):
    """단일 건물에 대한 날씨 급변 감지 피처 추가"""
    df = df.copy()
    
    # 기본 급변 피쳐들
    weather_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    
    for col in weather_cols:
        # rapid_change 컬럼 생성 (절댓값 차이)
        df[f'rapid_change_{col}'] = df[col].diff().abs().fillna(0)
    
    # 기존 급변 감지 피쳐들도 추가
    df['temp_rapid_change'] = df['기온(°C)'].diff().abs().fillna(0)
    df['humid_rapid_change'] = df['습도(%)'].diff().abs().fillna(0)
    df['wind_rapid_change'] = df['풍속(m/s)'].diff().abs().fillna(0)
    
    # 강수 시작/끝 감지
    prev_rain = df['강수량(mm)'].shift(1).fillna(0)
    df['rain_start'] = ((df['강수량(mm)'] > 0) & (prev_rain == 0)).astype(int)
    df['rain_end'] = ((df['강수량(mm)'] == 0) & (prev_rain > 0)).astype(int)
    
    # 강수 강도 분류
    df['rainfall_intensity'] = pd.cut(df['강수량(mm)'], 
                         bins=[-0.001, 0.1, 1, 5, 10, 100], 
                         labels=[0, 1, 2, 3, 4], 
                         include_lowest=True).fillna(0).astype(int)
    
    # 종합 날씨 불안정 지수
    df['weather_II'] = (df['temp_rapid_change'] + df['humid_rapid_change'] * 0.5 + df['wind_rapid_change'] * 2)
    
    # 시간대별 이상치 감지
    for col in ['기온(°C)', '습도(%)', '풍속(m/s)']:
        if col in df.columns and 'hour' in df.columns:
            # 시간대별 통계 계산
            hourly_stats = df.groupby('hour')[col].agg(['mean', 'std']).reset_index()
            hourly_stats.columns = ['hour', f'{col}_hourly_mean', f'{col}_hourly_std']
            
            # 병합
            df = df.merge(hourly_stats, on='hour', how='left')
            
            # Z-score 계산 (0으로 나누기 방지)
            std_col = f'{col}_hourly_std'
            mean_col = f'{col}_hourly_mean'
            df[std_col] = df[std_col].fillna(1).replace(0, 1)  # 표준편차 0인 경우 1로 대체
            
            df[f'{col}_zscore'] = (df[col] - df[mean_col]) / df[std_col]
            df[f'{col}_outliers'] = (df[f'{col}_zscore'].abs() > 2).astype(int)
            
            # 임시 컬럼 제거
            df = df.drop([mean_col, std_col], axis=1)
    
    # 모든 NaN 값을 0으로 채우기
    df = df.fillna(0)
    return df

# 1. train 데이터 전처리 (dayoff/hot_day 피쳐 생성됨)
train_processed = engineering_features_with_dayoff(train_csv.copy(), is_train=True)

# 2. train 데이터에서 day_pattern_mapping 생성
day_pattern_mapping = create_day_pattern_mapping(train_processed)

# 3. test 데이터 전처리 및 day_pattern 적용
test_processed = engineering_features_with_dayoff(test_csv.copy(), is_train=False)
test_processed = apply_day_pattern_features(test_processed, day_pattern_mapping)

# 피쳐 엔지니어링 실행
train_with_features, test_with_features = create_weather_features(train_processed, test_processed)
test_with_features = test_with_features.drop(['일조(hr)','일사(MJ/m2)', '전력소비량(kWh)'], axis=1)

# 결과 확인

# 최종 데이터 저장
print(f"Feature engineering completed!")
print(f"Train shape: {train_with_features.shape}")
print(f"Test shape: {test_with_features.shape}")

# CSV 저장
train_with_features.to_csv(f'{trainer}train_with_weather_features.csv')
test_with_features.to_csv(f'{trainer}test_with_weather_features.csv')

print(train_with_features[['건물번호','일시', 'date', 'hour']].head())
print(train_with_features[['건물번호','일시', 'date', 'hour']].tail())
print(test_with_features[['건물번호','일시', 'date', 'hour']].head())
print(test_with_features[['건물번호','일시', 'date', 'hour']].tail())
print(f"파일 저장 완료!")

print(f"- {trainer}train_with_weather_features.csv")
print(f"- {trainer}test_with_weather_features.csv")

train = train_with_features.copy()
test = test_with_features.copy()

pd.set_option('max_colwidth', None)
print(train.columns)
print(test.columns)

