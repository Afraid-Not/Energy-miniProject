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
log_path = './Energy/100/log/'
trainer = './Energy/100/new_csv/'
save_path = './Energy/100/submission/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

seed_file = "./Energy/100/log/(SEED_COUNT)preprocessing.json"

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

print(f"[1] 전처리")

train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
""" 
print(train.columns)
['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
'일사(MJ/m2)', '전력소비량(kWh)']
 """

######################기본 파생#####################
######################기본 파생#####################
######################기본 파생#####################

def engineering_features(df) :
    ####### 시간 관련 피쳐 #######
    df['date'] = pd.to_datetime(df['일시'])
    df['hour'] = df['date'].dt.hour                      # 시각(0~23)
    df['dayofweek'] = df['date'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['is_working'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  # 주기적 패턴
    df['COS_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['SIN_day'] = np.sin(2 * np.pi * df['day'] / 31)  # 일의 주기적 패턴 (31일 기준)
    df['COS_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['SIN_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['peak_time'] = df['hour'].apply(lambda x: 1 if 11 <= x <= 16 else 0) # Peak time (11 AM to 4 PM)
    
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
    
    return df

train = engineering_features(train_csv)
test = engineering_features(test_csv)

print(f"기본 파생 이후 train.shape : {train.shape}")
print(f"기본 파생 이후  test.shape : {test.shape}")

######################변화량 파생#####################
######################변화량 파생#####################
######################변화량 파생#####################

def create_change_features(train, test):
    # 변화량을 계산할 컬럼들
    change_columns = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)']
    # 변화량 시간 간격 (시간 단위)
    time_intervals = [1, 3, 5, 12, 24]
    
    # 건물번호별로 처리
    buildings = train['건물번호'].unique()
    
    train_with_changes = []
    test_with_changes = []
    
    for building_num in tqdm(buildings, desc="Processing buildings"):
        # 해당 건물의 train과 test 데이터 추출
        building_train = train[train['건물번호'] == building_num].copy()
        building_test = test[test['건물번호'] == building_num].copy()
        
        # train과 test를 시간순으로 연결 (train 뒤에 test)
        combined_df = pd.concat([building_train, building_test], ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # 각 컬럼별, 각 시간 간격별로 변화량 계산
        for col in change_columns:
            # 해당 컬럼의 평균값 계산 (빈 값 채우기용)
            col_mean = combined_df[col].mean()
            
            for interval in time_intervals:
                change_col_name = f'{col}_change_{interval}h'
                
                # 변화량 계산: 현재값 - interval시간 전 값
                combined_df[change_col_name] = combined_df[col] - combined_df[col].shift(interval)
        
        # train과 test 부분으로 다시 분리
        train_len = len(building_train)
        building_train_updated = combined_df.iloc[:train_len].copy()
        building_test_updated = combined_df.iloc[train_len:].copy()
        
        train_with_changes.append(building_train_updated)
        test_with_changes.append(building_test_updated)
    
    # 모든 건물의 데이터를 다시 합치기
    train_final = pd.concat(train_with_changes, ignore_index=True)
    test_final = pd.concat(test_with_changes, ignore_index=True)
    
    # 전체 데이터에서 변화량 피쳐들의 NaN 값을 해당 피쳐의 평균값으로 채우기
    change_features = [col for col in train_final.columns if '_change_' in col]
    
    # train과 test를 합쳐서 전체 변화량 통계 계산
    combined_all = pd.concat([train_final, test_final], ignore_index=True)
    
    for change_col in change_features:
        # 전체 데이터의 변화량 평균 계산
        change_mean = combined_all[change_col].mean()
        
        # train과 test 모두에서 NaN 값을 평균값으로 채우기
        train_final[change_col] = train_final[change_col].fillna(change_mean)
        test_final[change_col] = test_final[change_col].fillna(change_mean)
    
    # 원래 순서로 정렬
    train_final = train_final.sort_values(['건물번호', 'date']).reset_index(drop=True)
    test_final = test_final.sort_values(['건물번호', 'date']).reset_index(drop=True)
    
    return train_final, test_final

train_final, test_final = create_change_features(train, test)

print(f"기본 파생 이후 train.shape : {train_final.shape}")
print(f"결측치 확인 : {train_final.isna().sum()}")
print(f"기본 파생 이후  test.shape : {test_final.shape}")
print(f"결측치 확인 : {test_final.isna().sum()}")

test_final = test_final.drop(['일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'], axis=1)

train_final.to_csv(trainer + '00_train.csv', index=False)
test_final.to_csv(trainer + '00_test.csv', index=False)

print(f"Train : {trainer}00_train.csv")
print(f" Test : {trainer}00_test.csv")
print(f"저장 완료")