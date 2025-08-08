print(f"[VER 8] ACTIVATE")

# !! SEED RECOMMENDATION : 50, 71

#region IMPORT

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
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

py_path = './Energy/03/'
log_path = './Energy/03/log/'
os.makedirs(log_path, exist_ok=True)
version = 'ver8'
seed_file = f"./Energy/03/log/({version})SEED_COUNTS.json"

# SEED 관리
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 50 # seed_state["seed"]

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

print(f"[Current Run SEED]: {SEED}\n")

# 전역변수 관리
N_SPLIT= 3
N_TRIAL= 30
SAMPLE_SIZE = 0.3
print("[variables]")
print(f"N_SPLIT {N_SPLIT} | N_TRIAL {N_TRIAL} | SAMPLE_SIZE {SAMPLE_SIZE}\n")

#endregion IMPORT

print(f"[STEP 1] PREPROCESSING")

#region PREPROCESSING

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def load_data(train_path, test_path, building_path, sub_path):
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    submission = pd.read_csv(sub_path)
    building_info = pd.read_csv(building_path)

    mapping = {
        '건물번호':'building_num',
        '기온(°C)':'temperature',
        '습도(%)':'humidity',
        '풍속(m/s)':'windspeed',
        '강수량(mm)':'precipitation',
        '일조(hr)':'sunshine',
        '일사(MJ/m2)':'solar',
        '전력소비량(kWh)':'power_consumption',
    }

    mapping_building = {
        '건물번호':'building_num',
        '건물유형': 'building_type',
        '연면적(m2)': 'all_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'pvc',
        'ESS저장용량(kWh)': 'ess',
        'PCS용량(kW)': 'pcs',
    }
    
    def apply_mapping(df):
        return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    train = apply_mapping(train)
    test = apply_mapping(test)
    
    building_info = building_info.rename(columns=mapping_building)
    
    building_col = ['pvc', 'ess', 'pcs']
    for i in building_col:
        building_info[i] = building_info[i].replace('-', np.nan).astype(float)
    building_info = building_info.fillna(0)
    
    train = pd.merge(train, building_info, on='building_num', how='left')
    test = pd.merge(test, building_info, on='building_num', how='left')

    return train, test, submission

def feature_engineering(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['date'] = pd.to_datetime(df['일시'])

    df['hour'] = df['date'].dt.hour                      # 시각(0~23)
    df['dow'] = df['date'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['SIN_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  # 주기적 패턴
    df['COS_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['SIN_day'] = np.sin(2 * np.pi * df['day'] / 31)  # 일의 주기적 패턴 (31일 기준)
    df['COS_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['SIN_month'] = np.sin(2 * np.pi * df['month'] / 12)  # 일의 주기적 패턴 (31일 기준)
    df['COS_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['SIN_dow'] = np.sin(2 * np.pi * df['dow'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_dow'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear
    
    df['SIN_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['COS_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
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
    df['daylight'] = ((df['hour'] >= df['sunrise_hour']) & (df['hour'] <= df['sunset_hour'])).astype(int)
    
    time_decimal = df['hour'] + 0.5  # 30분 기준 (정시 측정이라면 그냥 hour)
    solar_noon = (df['sunrise_hour'] + df['sunset_hour']) / 2
    hour_angle = 15 * (time_decimal - solar_noon)  # 시간각(°)
    # declination, latitude (radian)
    lat_rad = np.deg2rad(37.5665)
    dec_rad = np.deg2rad(declination)
    ha_rad = np.deg2rad(hour_angle)
    df['solar_elevation'] = np.arcsin(
    np.sin(lat_rad) * np.sin(dec_rad) +
    np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    ) * 180 / np.pi
    
    temp = df['temperature']
    humidity = df['humidity']
    wind_speed = df['windspeed']
    df['PT'] = temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00
    df['CDH'] = np.maximum(df['temperature'] - 26, 0)
    df['DI'] = 0.81 * df['temperature'] + 0.01 * df['humidity'] * (0.99 * df['temperature'] - 14.3) + 46.3
    df['PVC_per_CA'] = df['pvc'] / (df['cooling_area'] + 1e-6)
    df['ESS_installation'] = df['ess'].apply(lambda x: 1 if x > 0 else 0)
    df['PCS_installation'] = df['pcs'].apply(lambda x: 1 if x > 0 else 0)
    df['Facility_Density'] = (df['ess'] + df['pcs']) / (df['all_area'] + 1e-6)
    
    yr = df['date'].dt.year.astype(str)
    season_start = pd.to_datetime(yr + "-06-01")
    season_end   = pd.to_datetime(yr + "-09-01")
    in_season = (df['date'] >= season_start) & (df['date'] < season_end)
    period_sec = (season_end - season_start).dt.total_seconds()
    pos = (df['date'] - season_start).dt.total_seconds() / period_sec  # 0~1
    phi = 2 * np.pi * pos
    df['SIN_summer'] = np.where(in_season, np.sin(phi), np.nan)
    df['COS_summer'] = np.where(in_season, np.cos(phi), np.nan)

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

    def add_region_group(x):
        x = x.copy()
        x['groupID'] = x['building_num'].map(building_to_group)
        return x

    df = add_region_group(df)
    
    return df

def add_data(
    df,
    cols=('temperature', 'precipitation', 'windspeed', 'humidity'), 
    copies=2, 
    jitter=0.10, 
    seed=SEED, 
    round_decimals=1,
    sort_by=('building_num', 'date'), 
    cast_back=False
):
    
    rng = np.random.default_rng(seed)
    present = [c for c in cols if c in df.columns]
    frames = [df]

    for _ in range(copies):
        new = df.copy()
        # 각 컬럼별로 행 단위 스케일링
        for c in present:
            scale = rng.uniform(1 - jitter, 1 + jitter, len(df))
            vals = (df[c].to_numpy() * scale)
            vals = np.round(vals, round_decimals)
            if cast_back and np.issubdtype(df[c].dtype, np.integer):
                vals = vals.astype(df[c].dtype)
            new[c] = vals
        frames.append(new)

    out = pd.concat(frames, ignore_index=True)
    if sort_by is not None:
        out = out.sort_values(list(sort_by)).reset_index(drop=True)
        
    return out

def peak_holidays(df, is_train=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['일시'])
    df['dow']  = df['date'].dt.weekday  # 0=월 ... 6=일

    # 기본값
    df['holidays'] = 0
    df['peak'] = 0

    # --- 건물별 '정기 휴무 요일' 지정 ---
    df.loc[(df['building_num']==2) & (df['dow']==5), 'holidays'] = 1       # 토
    df.loc[(df['building_num']==3) & (df['dow'].isin([5,6])), 'holidays'] = 1  # 토/일
    df.loc[(df['building_num']==4) & (df['dow']==0), 'holidays'] = 1       # 월
    df.loc[(df['building_num']==5) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==6) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==7) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['building_num']==8) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==10) & (df['dow']==0), 'holidays'] = 1
    df.loc[(df['building_num']==12) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==13) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==14) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==15) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==16) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==17) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==18) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['building_num']==20) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==21) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==22) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==23) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==24) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==37) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==38) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==39) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==42) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==43) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==45) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==46) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==47) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==48) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==49) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==50) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==51) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==52) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==53) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==55) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==56) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==60) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==61) & (df['dow'].isin([1,2])), 'holidays'] = 1
    df.loc[(df['building_num']==62) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==64) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==66) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==67) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==68) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==69) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==72) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==75) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==80) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==81) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==83) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==86) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==87) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==90) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==94) & (df['dow'].isin([5,6])), 'holidays'] = 1


    df.loc[(df['building_num']==2) & (df['dow']==3), 'peak'] = 1           # 목
    df.loc[(df['building_num']==3) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==4) & (df['dow']==4), 'peak'] = 1           # 금
    df.loc[(df['building_num']==6) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==7) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==8) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==9) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==10) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==11) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==12) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==13) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==14) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==15) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==16) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==17) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==18) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==19) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==20) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==21) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==22) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==23) & (df['dow']==1), 'peak'] = 1
    df.loc[(df['building_num']==24) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==25) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==26) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==27) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==28) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==29) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==30) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==31) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==32) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==33) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==34) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==35) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==36) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==37) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==38) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==39) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==40) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==41) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==42) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==43) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==44) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==45) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==46) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==47) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==48) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==49) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==50) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==51) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==52) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==53) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==54) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==55) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==56) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==57) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==58) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==59) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==60) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==61) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==62) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==63) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==64) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==65) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==66) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==67) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==68) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==69) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==70) & (df['dow']==1), 'peak'] = 1
    df.loc[(df['building_num']==71) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==72) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==73) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==74) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==75) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==76) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==77) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==78) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==79) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==80) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==81) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==82) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['building_num']==83) & (df['dow']==1), 'peak'] = 1
    df.loc[(df['building_num']==84) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==85) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==86) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==87) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['building_num']==88) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==89) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==90) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==91) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==92) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==93) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==94) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['building_num']==95) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==96) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['building_num']==97) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['building_num']==98) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==99) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['building_num']==100) & (df['dow']==6), 'peak'] = 1


    if is_train :
        nat = df['date'].dt.strftime('%m-%d').isin(['06-06','08-15'])
        df.loc[nat, 'holidays'] = 1

        df.loc[(df['building_num']==19) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 1
        df.loc[(df['building_num']==27) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['building_num']==29) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-24','07-10','07-28','08-10'])), 'holidays'] = 1
        df.loc[(df['building_num']==32) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-24','07-08','07-22','08-12'])), 'holidays'] = 1
        df.loc[(df['building_num']==38) & (df['date'].dt.strftime('%m-%d').isin(['06-07'])), 'holidays'] = 1
        df.loc[(df['building_num']==40) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['building_num']==45) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 1
        df.loc[(df['building_num']==54) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 1
        df.loc[(df['building_num']==56) & (df['date'].dt.strftime('%m-%d').isin(['06-07','08-16'])), 'holidays'] = 1
        df.loc[(df['building_num']==59) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['building_num']==63) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['building_num']==74) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01'])), 'holidays'] = 1
        df.loc[(df['building_num']==79) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 1
        df.loc[(df['building_num']==94) & (df['date'].dt.strftime('%m-%d').isin(['06-07','08-16'])), 'holidays'] = 1
        df.loc[(df['building_num']==95) & (df['date'].dt.strftime('%m-%d').isin(['07-08','08-05'])), 'holidays'] = 1
    
    else :
        df.loc[(df['building_num']==27) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['building_num']==29) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 1
        df.loc[(df['building_num']==32) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 1
        df.loc[(df['building_num']==40) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['building_num']==59) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['building_num']==63) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['building_num']==74) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 1        
    
    return df

intervals = {
    6:  [(2024081500, 2024081900)],
    7:  [(2024070710, 2024070811), (2024071214, 2024080603)],
    8:  [(2024072109, 2024072111)],
    12: [(2024072108, 2024072111)],
    17: [(2024062515, 2024062609)],
    19: [(2024073114, 2024073115)],
    25: [(2024070411, 2024070415)],
    26: [(2024061713, 2024061812)],
    29: [(2024061522, 2024061523), (2024062700, 2024062701)],
    36: [(2024060100, 2024060923)],
    40: [(2024071400, 2024071401)],
    41: [(2024062201, 2024062204), (2024071714, 2024071715)],
    43: [(2024061017, 2024061018), (2024081216, 2024081217)],
    44: [(2024063000, 2024063002), (2024063000, 2024063002)],
    52: [(2024081000, 2024081002)],
    53: [(2024061417, 2024061707), (2024081816, 2024081907)],
    57: [(2024060100, 2024060721)],
    67: [(2024061017, 2024061018), (2024072600, 2024072723), (2024081216, 2024081217)],
    68: [(2024062823, 2024062901)],
    70: [(2024060409, 2024060508)],
    72: [(2024061100, 2024061102), (2024072110, 2024072111)],
    76: [(2024062012, 2024062016)],
    79: [(2024081903, 2024081905)],
    80: [(2024070609, 2024070615), (2024070811, 2024070813), (2024072009, 2024072013)],
    88: [(2024082306, 2024082308)],
    89: [(2024071208, 2024071209)],
    92: [(2024071718, 2024071721)],
    94: [(2024072620, 2024080507)],
    95: [(2024070800, 2024070821), (2024080510, 2024080604)],
    99: [(2024071005, 2024071007)],
}

singles = {
    5:  [2024080312],
    20: [2024060110],
    30: [2024071320, 2024072500],
    73: [2024070822],
    76: [2024060313],
    77: [2024080617],
    78: [2024071714],
    81: [2024071714],
    90: [2024060518],
    97: [2024071714],
    98: [2024061315],
}

def build_stats_features(
    train, test,
    target_col='power_consumption',
    building_col='building_num',
    hour_col='hour', dow_col='dow', holiday_col='holidays',
    ddof=1,
    apply_dow_ratio=True,          # True면 요일 가중치 곱해서 집계
    mode='byb'                     # 'all'이면 ratio에 -0.005 보정
):
    """
    - 공통 피처 생성: dow/hour/holidays 없으면 date(또는 '일시')로부터 생성
    - 집계: (building,hour,dow) mean/std, (building,hour,holiday) mean/std,
            (building,hour) mean/std, (building) mean/std
    - 결측 백필: dow_hour -> holiday -> hour -> building -> global
    - target 값 자체는 수정하지 않음(집계용 내부 복사본에만 요일가중 적용)
    """
    tr = train.copy()
    te = test.copy()

    # ---- 글로벌 백업 값
    global_mean = tr[target_col].mean()
    global_std  = tr[target_col].std(ddof=ddof)

    # ---- 집계 입력용 트레인 복사본(여기에만 옵션 적용)
    tr_feat = tr[[building_col, hour_col, dow_col, holiday_col, target_col]].copy()

    # 2) 요일 가중치 적용(집계 전용)
    pc = tr_feat[target_col].to_numpy(dtype=float)
    if apply_dow_ratio:
        ratio = np.array([0.985, 0.98, 0.98, 0.995, 0.995, 0.99, 0.99], dtype=float)
        if mode == 'all':
            ratio = ratio - 0.005

        idx = tr_feat[dow_col].to_numpy()
        idx = idx.astype(int)

        # dow가 0~6 또는 1~7 둘 다 허용
        if np.all(np.isin(idx, np.arange(1, 8))):
            idx = idx - 1
        elif not np.all(np.isin(idx, np.arange(0, 7))):
            raise ValueError(f"{dow_col}는 0~6 또는 1~7 이어야 합니다. unique={np.unique(idx)}")

        pc = pc * ratio[idx]

    # ---- groupby 집계 (mean / std)
    tmp = tr_feat[[building_col, hour_col, dow_col, holiday_col]].copy()
    tmp['__pc'] = pc

    # by building, hour, dow  (★ std 추가)
    g1 = (tmp.groupby([building_col, hour_col, dow_col], as_index=False)
            .agg(dow_hour_mean=('__pc', 'mean'),
                 dow_hour_std =('__pc', lambda x: x.std(ddof=ddof))))

    # by building, hour, holiday
    g2 = (tmp.groupby([building_col, hour_col, holiday_col], as_index=False)
            .agg(holiday_mean=('__pc', 'mean'),
                 holiday_std =('__pc', lambda x: x.std(ddof=ddof))))

    # by building, hour
    g3 = (tmp.groupby([building_col, hour_col], as_index=False)
            .agg(hour_mean=('__pc', 'mean'),
                 hour_std =('__pc', lambda x: x.std(ddof=ddof))))

    # by building
    gb = (tmp.groupby([building_col], as_index=False)
            .agg(building_mean=('__pc', 'mean'),
                 building_std =('__pc', lambda x: x.std(ddof=ddof))))

    # ---- 머지 도우미
    def _attach(df):
        out = df.merge(g1, on=[building_col, hour_col, dow_col], how='left')
        out = out.merge(g2, on=[building_col, hour_col, holiday_col], how='left')
        out = out.merge(g3, on=[building_col, hour_col], how='left')
        out = out.merge(gb, on=[building_col], how='left')
        return out

    tr = _attach(tr)
    te = _attach(te)

    # ---- 결측 백필 체인
    for df in (tr, te):
        df['dow_hour_mean'] = (
            df['dow_hour_mean']
              .fillna(df['holiday_mean'])
              .fillna(df['hour_mean'])
              .fillna(df['building_mean'])
              .fillna(global_mean)
        )
        df['dow_hour_std'] = (
            df['dow_hour_std']
              .fillna(df['holiday_std'])
              .fillna(df['hour_std'])
              .fillna(df['building_std'])
              .fillna(global_std)
        )
        df['holiday_mean'] = df['holiday_mean'].fillna(df['hour_mean']).fillna(df['building_mean']).fillna(global_mean)
        df['holiday_std']  = df['holiday_std'].fillna(df['hour_std']).fillna(df['building_std']).fillna(global_std)
        df['hour_mean']    = df['hour_mean'].fillna(df['building_mean']).fillna(global_mean)
        df['hour_std']     = df['hour_std'].fillna(df['building_std']).fillna(global_std)

    return tr, te

def shift_power_from_6h_and_trim(
    df: pd.DataFrame,
    bno: int = 87,
    start_str: str = "20240629 01",
    hours: int = 6,
    id_col: str = "building_num",
    time_col: str = "date",
    time_fallback_col: str = "일시",
    target_col: str = 'power_consumption'
) -> pd.DataFrame:
    """
    [건물번호=bno]의 start_str 시각부터 끝까지 target_col 값을 6시간 뒤로 이동.
    이후 (1) 시작부 첫 6시간(01~06시) 행 삭제, (2) 끝부분 소스가 없는 마지막 6개 행 삭제.

    다른 컬럼은 수정하지 않으며, 행 삭제는 해당 구간에 한정됨.
    """

    out = df.copy()

    # 0) datetime 보장
    if time_col not in out.columns:
        if time_fallback_col not in out.columns:
            raise KeyError(f"'{time_col}'도 '{time_fallback_col}'도 없습니다.")
        # 포맷 우선 시도(YYYYMMDD HH), 실패 시 infer
        try:
            out[time_col] = pd.to_datetime(out[time_fallback_col].astype(str),
                                           format="%Y%m%d %H", errors="raise")
        except Exception:
            out[time_col] = pd.to_datetime(out[time_fallback_col], errors="coerce")
        if out[time_col].isna().any():
            raise ValueError(f"'{time_fallback_col}' 파싱 실패가 있습니다.")

    start_dt = pd.to_datetime(start_str, format="%Y%m%d %H")

    # 1) 건물 87, 시간 정렬
    m_bno = (out[id_col] == bno)
    idx_bno_sorted = out[m_bno].sort_values(time_col).index

    # 2) 시작 시점 이후 인덱스
    idx_after = out.loc[idx_bno_sorted][out.loc[idx_bno_sorted, time_col] >= start_dt].index
    n_after = len(idx_after)
    if n_after == 0:
        # 옮길 대상 없음
        return out

    # 3) 전력소비량을 +6h로 이동 (행은 그대로 두고 값만 옮김)
    #    -> after 구간 내에서 위치 기반 이동: i(소스) -> i+hours(목적지)
    if n_after > hours:
        src_idx  = idx_after[:-hours]
        dest_idx = idx_after[hours:]
        # 값 복사: 목적지에 소스 값을 덮어씀
        out.loc[dest_idx, target_col] = out.loc[src_idx, target_col].to_numpy()

    # 4) 행 삭제 대상 구성
    drop_idx = set()

    # (a) 시작부 6개(01~06시) 행 삭제 — 존재하는 만큼만
    drop_idx.update(idx_after[:min(hours, n_after)])

    # (b) 끝부분 6개 행 삭제 — 존재하는 만큼만
    drop_idx.update(idx_after[max(0, n_after - hours):])

    # 5) 실제 삭제
    if drop_idx:
        out = out.drop(index=list(drop_idx)).reset_index(drop=True)

    return out

def drop_outlier_times(df, intervals=None, singles=None, inclusive='both'):

    df = df.copy()

    s = df['일시'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    if s.str.len().eq(10).all():   # YYYYMMDDHH
        df['dt'] = pd.to_datetime(s, format='%Y%m%d%H', errors='coerce')
    else:
        df['dt'] = pd.to_datetime(df['일시'], errors='coerce', infer_datetime_format=True)

    mask = pd.Series(False, index=df.index)

    # 2) 구간 제거
    if intervals:
        for bno, spans in intervals.items():
            for start, end in spans:
                sdt = pd.to_datetime(str(start), format='%Y%m%d%H')
                edt = pd.to_datetime(str(end),   format='%Y%m%d%H')
                mask |= (df['building_num'] == bno) & df['dt'].between(sdt, edt, inclusive=inclusive)

    # 3) 단일 시각 제거
    if singles:
        for bno, tlist in singles.items():
            for t in tlist:
                tdt = pd.to_datetime(str(t), format='%Y%m%d%H')
                mask |= (df['building_num'] == bno) & (df['dt'] == tdt)

    removed = df.loc[mask].sort_values(['building_num','dt'])
    kept    = df.loc[~mask].reset_index(drop=True)
    removed = removed.drop(['dt'], axis=1)
    kept = kept.drop(['dt'], axis=1)
    
    return kept, removed

train_path = './Energy/train.csv'
test_path = './Energy/test.csv'
building_path = './Energy/building_info.csv'
sub_path = './Energy/sample_submission.csv'
train, test, submission = load_data(train_path, test_path, building_path, sub_path)

print(f"    Before Feature Engineering - train : {train.shape} | test : {test.shape}")  # (204000, 15) (16800, 12)
train = feature_engineering(train)
test = feature_engineering(test)
train = peak_holidays(train)
test = peak_holidays(test, is_train=False)
train, test = build_stats_features(train, test, mode='all', apply_dow_ratio=True)

train = shift_power_from_6h_and_trim(train)
train_clean, train_removed = drop_outlier_times(train, intervals, singles, inclusive='both')

train_clean = add_data(train_clean)

test = test.assign(
    sunshine=test.get('sunshine', 0.0),
    solar=test.get('solar', 0.0),
    power_consumption=test.get('power_consumption', 0.0),
)
print(f"    After Feature Engineering  - train : {train_clean.shape} | test : {test.shape}\n")  # (612000, 55) (16800, 55)

#region
# print(train.columns)
# Index(['building_num', '일시', 'temperature', 'precipitation', 'windspeed',
#        'humidity', 'sunshine', 'solar', 'power_consumption', 'building_type',
#        'all_area', 'cooling_area', 'pvc', 'ess', 'pcs', 'date', 'hour', 'dow',
#        'month', 'day', 'SIN_hour', 'COS_hour', 'SIN_day', 'COS_day',
#        'SIN_month', 'COS_month', 'SIN_dow', 'COS_dow', 'day_of_year',
#        'SIN_day_of_year', 'COS_day_of_year', 'sunrise_hour', 'sunset_hour',
#        'daylight', 'PT', 'CDH', 'DI', 'PVC_per_CA', 'ESS_installation',
#        'PCS_installation', 'Facility_Density', 'SIN_summer', 'COS_summer',
#        'groupID', 'holidays', 'peak', 'dow_hour_mean', 'dow_hour_std',
#        'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
#        'building_std','solar_elevation'],
#       dtype='object')
# print(test.columns)
# Index(['building_num', '일시', 'temperature', 'precipitation', 'windspeed',
#        'humidity', 'building_type', 'all_area', 'cooling_area', 'pvc', 'ess',
#        'pcs', 'date', 'hour', 'dow', 'month', 'day', 'SIN_hour', 'COS_hour',
#        'SIN_day', 'COS_day', 'SIN_month', 'COS_month', 'SIN_dow', 'COS_dow',
#        'day_of_year', 'SIN_day_of_year', 'COS_day_of_year', 'sunrise_hour',
#        'sunset_hour', 'daylight', 'PT', 'CDH', 'DI', 'PVC_per_CA',
#        'ESS_installation', 'PCS_installation', 'Facility_Density',
#        'SIN_summer', 'COS_summer', 'groupID', 'holidays', 'peak',
#        'dow_hour_mean', 'dow_hour_std', 'holiday_mean', 'holiday_std',
#        'hour_mean', 'hour_std', 'building_mean', 'building_std', 'solar_elevation'],
#       dtype='object')
# exit()
#endregion

#endregion PREPROCESSING

print(f"[STEP 2] INTERPOLATION")

#region INTERPOLATION

def impute_solar_for_zero_bnos_cv_optuna(
    train: pd.DataFrame,
    zero_bnos,
    bno_col: str = "building_num",
    target_col: str = "solar",
    daylight_col: str = "daylight",
    drop_cols=None,
    seed: int = SEED,
    n_splits: int = 5,
    n_trials: int = 30,        # 요구사항: 30회
    sample_frac: float = 0.3,  # 요구사항: 30% 샘플링
):
    """
    [LGBM-only] train의 zero_bnos 건물에서 'solar'가 0으로 잘못 기록된 구간을 모델로 보간.
    반환: (train_filled, oof_smape_best)

    - Optuna: 학습 데이터 30% 샘플로 30회 튜닝(KFold 5-fold OOF SMAPE 최소화)
              (LightGBM 하이퍼파라미터만 최적화)
    - 최종: best params로 전체 데이터에서 5-fold OOF 재계산하여 점수 반환
    - 예측 반영: zero_bnos & daylight==1 위치에 LGBM 예측치 대입, zero_bnos & daylight==0 은 0
    """

    # ---- helper: optuna 진행바 (best만 표시)
    def _optimize_with_pbar(objective, n_trials, desc, seed):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        pbar = tqdm(total=n_trials, desc=desc, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {postfix}", leave=False)
        def _cb(study, trial):
            if study.best_value is not None:
                pbar.set_postfix_str(f"best={study.best_value:.4f}")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        pbar.write(f"    [{desc}] SMAPE: {study.best_value:.4f}")
        pbar.close()
        return study

    drop_cols = set(drop_cols or [])

    # 학습 데이터: zero_bnos 제외 + 주간만 사용
    mask_train = (~train[bno_col].isin(zero_bnos)) & (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # 피처 선택: 숫자형만, target/drop 제외
    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col})
        and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not feature_cols:
        raise ValueError("    학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    # ---- Optuna 샘플링(30%)
    df_sample = df_train.sample(frac=sample_frac, random_state=seed) if 0 < sample_frac < 1 else df_train
    Xs = df_sample[feature_cols]
    ys = df_sample[target_col].astype(float)

    # ===== Optuna objective: 5-fold LGBM OOF SMAPE =====
    def objective(trial):
        lgb_params = dict(
            n_estimators=trial.suggest_int("lgb_n_estimators", 400, 3000),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
            max_depth=trial.suggest_categorical("lgb_max_depth", [-1, 6, 8, 10]),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 160),
            subsample=trial.suggest_float("lgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 0.0, 5.0),
            reg_alpha=trial.suggest_float("lgb_reg_alpha", 0.0, 2.0),
            max_bin=trial.suggest_int("lgb_max_bin", 63, 255),
            random_state=seed, n_jobs=-1, verbosity=-1,
            objective="l1",
        )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_pred = np.zeros(len(df_sample), dtype=float)

        for tr_idx, val_idx in kf.split(Xs):
            X_tr, X_val = Xs.iloc[tr_idx], Xs.iloc[val_idx]
            y_tr, y_val = ys.iloc[tr_idx], ys.iloc[val_idx]

            med = X_tr.median()
            X_tr_f = X_tr.fillna(med)
            X_val_f = X_val.fillna(med)

            lgb = LGBMRegressor(**lgb_params)
            lgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                eval_metric="l1",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )

            pred = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
            oof_pred[val_idx] = np.clip(pred, 0, None)

        return smape(ys.to_numpy(), oof_pred)

    # ---- 최적화
    study = _optimize_with_pbar(objective, n_trials=n_trials, desc="optuna(solar-impute LGBM)", seed=seed)
    best_params = study.best_params

    # ===== best params로 FULL 데이터 5-fold OOF 재계산 =====
    lgb_best = {
        "n_estimators": best_params["lgb_n_estimators"],
        "learning_rate": best_params["lgb_learning_rate"],
        "num_leaves": best_params["lgb_num_leaves"],
        "max_depth": best_params["lgb_max_depth"],
        "min_child_samples": best_params["lgb_min_child_samples"],
        "subsample": best_params["lgb_subsample"],
        "colsample_bytree": best_params["lgb_colsample_bytree"],
        "reg_lambda": best_params["lgb_reg_lambda"],
        "reg_alpha": best_params.get("lgb_reg_alpha", 0.0),
        "max_bin": best_params.get("lgb_max_bin", 255),
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
        "objective": "l1",
    }

    X = df_train[feature_cols]
    y_full = df_train[target_col].astype(float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(df_train), dtype=float)
    lgb_best_iters = []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        lgb = LGBMRegressor(**lgb_best)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        pv = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        oof_pred[val_idx] = np.clip(pv, 0, None)

        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_smape_best = float(smape(y_full.to_numpy(), oof_pred))

    # ===== 전체 데이터로 최종 재학습 후 zero_bnos 주간 예측 적용 =====
    med_full = X.median()
    X_full = X.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_best["n_estimators"])

    lgb_final = LGBMRegressor(**{**lgb_best, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_full)

    # 예측 반영: zero_bnos & 주간만 예측, 야간은 0
    train_filled = train.copy()
    m_day = (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 1)
    if m_day.any():
        feats = train_filled.loc[m_day, feature_cols].fillna(med_full)
        pred = lgb_final.predict(feats, num_iteration=getattr(lgb_final, "best_iteration_", None))
        train_filled.loc[m_day, target_col] = np.clip(pred, 0, None)

    train_filled.loc[(train_filled[bno_col].isin(zero_bnos)) &
                     (train_filled[daylight_col] == 0), target_col] = 0.0

    print(f"    [impute:LGBM] OOF SMAPE: {oof_smape_best:.6f} | best_n_estimators≈{lgb_final_n}")

    return train_filled, oof_smape_best

def train_predict_test_target_cv_optuna(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,                 # 'sunshine' 또는 'solar'
    exclude_bnos=None,               # 예: zero_bnos (train 학습에서 제외)
    bno_col: str = "building_num",
    daylight_col: str = "daylight",
    drop_cols=None,
    seed: int = 42,
    n_splits: int = 5,
    n_trials: int = 30,              # 요구: 30회
    sample_frac: float = 0.3,        # 요구: 30% 샘플링
    restrict_daylight: bool = True   # True면 주간만 학습/예측, 야간 0
):
    """
    [LGBM-only] sunshine/solar 모두 LightGBM 단일 모델로 CV 튜닝 및 예측
    반환: (test_pred_series, oof_smape)
    """

    # ---- Optuna 로그 off + 진행바에 best만 표시
    def _optimize_with_pbar(objective, n_trials, desc, seed):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        pbar = tqdm(total=n_trials, desc=desc, dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {postfix}", leave=False)
        def _cb(study, trial):
            if study.best_value is not None:
                pbar.set_postfix_str(f"best={study.best_value:.4f}")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        pbar.write(f"    [{desc}] SMAPE: {study.best_value:.4f}")
        pbar.close()
        return study

    drop_cols = set(drop_cols or [])
    exclude_bnos = set(exclude_bnos or [])

    # ---- 학습 데이터 마스크
    mask_train = (~train[bno_col].isin(exclude_bnos))
    if restrict_daylight:
        mask_train &= (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # ---- 피처: 숫자형 & 드롭/타깃 제외 (train/test 공통 교집합 권장)
    common_cols = (set(df_train.columns) & set(test.columns)) - drop_cols - {target_col}
    feature_cols = [c for c in sorted(common_cols) if pd.api.types.is_numeric_dtype(df_train[c])]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    # ---- Optuna 샘플링
    df_sample = df_train.sample(frac=sample_frac, random_state=seed) if 0 < sample_frac < 1 else df_train
    Xs = df_sample[feature_cols]
    ys = df_sample[target_col].astype(float)

    # ===== Optuna objective: LGBM 5-fold OOF SMAPE =====
    def objective(trial):
        lgb_params = dict(
            n_estimators=trial.suggest_int("lgb_n_estimators", 400, 3000),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
            max_depth=trial.suggest_categorical("lgb_max_depth", [-1, 6, 8, 10]),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 160),
            subsample=trial.suggest_float("lgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 0.0, 5.0),
            reg_alpha=trial.suggest_float("lgb_reg_alpha", 0.0, 2.0),
            max_bin=trial.suggest_int("lgb_max_bin", 63, 255),
            random_state=seed, n_jobs=-1, verbosity=-1,
            objective="l1",
        )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_pred = np.zeros(len(df_sample), dtype=float)

        for tr_idx, val_idx in kf.split(Xs):
            X_tr, X_val = Xs.iloc[tr_idx], Xs.iloc[val_idx]
            y_tr, y_val = ys.iloc[tr_idx], ys.iloc[val_idx]

            med = X_tr.median()
            X_tr_f = X_tr.fillna(med)
            X_val_f = X_val.fillna(med)

            lgb = LGBMRegressor(**lgb_params)
            lgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                eval_metric="l1",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )
            pred = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
            oof_pred[val_idx] = np.clip(pred, 0, None)

        return smape(ys.to_numpy(), oof_pred)

    # ---- 최적화
    study = _optimize_with_pbar(objective, n_trials=n_trials, desc=f"optuna(LGBM {target_col})", seed=seed)
    best_params = study.best_params

    # ===== best params로 FULL 데이터 5-fold OOF & 재학습 =====
    lgb_best = {
        "n_estimators": best_params["lgb_n_estimators"],
        "learning_rate": best_params["lgb_learning_rate"],
        "num_leaves": best_params["lgb_num_leaves"],
        "max_depth": best_params["lgb_max_depth"],
        "min_child_samples": best_params["lgb_min_child_samples"],
        "subsample": best_params["lgb_subsample"],
        "colsample_bytree": best_params["lgb_colsample_bytree"],
        "reg_lambda": best_params["lgb_reg_lambda"],
        "reg_alpha": best_params.get("lgb_reg_alpha", 0.0),
        "max_bin": best_params.get("lgb_max_bin", 255),
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
        "objective": "l1",
    }

    X = df_train[feature_cols]
    y_full = df_train[target_col].astype(float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(df_train), dtype=float)
    lgb_best_iters = []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        lgb = LGBMRegressor(**lgb_best)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )
        pv = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        oof_pred[val_idx] = np.clip(pv, 0, None)

        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_smape = float(smape(y_full.to_numpy(), oof_pred))

    # ===== FULL 재학습 후 test 예측 =====
    med_full = X.median()
    X_full = X.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_best["n_estimators"])
    lgb_final = LGBMRegressor(**{**lgb_best, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_full)

    # test 예측
    test_pred = pd.Series(index=test.index, dtype=float)
    if restrict_daylight:
        m_day = (test[daylight_col] == 1)
        feats = test.loc[m_day, feature_cols].fillna(med_full)
        if len(feats) > 0:
            p = lgb_final.predict(feats, num_iteration=getattr(lgb_final, "best_iteration_", None))
            test_pred.loc[m_day] = np.clip(p, 0, 1) if target_col == 'sunshine' else np.clip(p, 0, None)
        test_pred.loc[~m_day] = 0.0
    else:
        feats = test[feature_cols].fillna(med_full)
        p = lgb_final.predict(feats, num_iteration=getattr(lgb_final, "best_iteration_", None))
        test_pred[:] = np.clip(p, 0, 1) if target_col == 'sunshine' else np.clip(p, 0, None)

    print(f"[{target_col}] LGBM-only OOF SMAPE: {oof_smape:.6f} | best_n_estimators≈{lgb_final_n}")
    return test_pred, oof_smape

drop_cols = [
    "building_num", "building_type", "all_area", "cooling_area",
    "pvc", "ess", "pcs",
    "PVC_per_CA", "ESS_installation", "PCS_installation",
    "Facility_Density", "groupID",
    
    "일시", "date",
    
    "dow_hour_mean", "dow_hour_std", "holiday_mean", "holiday_std",
    "hour_mean", "hour_std", "building_mean", "building_std", "peak",
    
    'power_consumption', 'solar'
]

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print("    [Train-Solar]")

train_filled, smape_score = impute_solar_for_zero_bnos_cv_optuna(
    train=train_clean,
    zero_bnos=zero_bnos,
    drop_cols=drop_cols,
    seed=SEED,            # 있으면
    n_splits=N_SPLIT,
    n_trials=N_TRIAL,
    sample_frac=SAMPLE_SIZE,
)

print("    > SMAPE :", smape_score)

# 공통 드롭(타깃은 함수 내부에서 자동 제외)
drop_cols_for_sunshine = [
    "building_num", "building_type", "all_area", "cooling_area",
    "pvc", "ess", "pcs",
    "PVC_per_CA", "ESS_installation", "PCS_installation",
    "Facility_Density", "groupID",
    
    "일시", "date",
    
    "dow_hour_mean", "dow_hour_std", "holiday_mean", "holiday_std",
    "hour_mean", "hour_std", "building_mean", "building_std", "peak",
    
    "PT", "CDH", "DI",
    
    'power_consumption','solar','sunshine'
]

drop_cols_for_solar = [
    "building_num", "building_type", "all_area", "cooling_area",
    "pvc", "ess", "pcs",
    "PVC_per_CA", "ESS_installation", "PCS_installation",
    "Facility_Density", "groupID",
    
    "일시", "date",
    
    "dow_hour_mean", "dow_hour_std", "holiday_mean", "holiday_std",
    "hour_mean", "hour_std", "building_mean", "building_std", "peak",
    
    'power_consumption','solar'
]

print("    [Test-Sunshine]")

test['sunshine'], oof_sunshine = train_predict_test_target_cv_optuna(
    train=train_filled, 
    test=test, 
    target_col='sunshine',
    exclude_bnos=None,    # ← 또는 아예 인자 삭제
    drop_cols=drop_cols_for_sunshine,
    seed=SEED, 
    n_splits=N_SPLIT, 
    n_trials=N_TRIAL, 
    sample_frac=SAMPLE_SIZE,
    restrict_daylight=True,
)

print("    > SMAPE :", oof_sunshine)

print("    [Test-Solar]")

test['solar'], oof_solar = train_predict_test_target_cv_optuna(
    train=train_filled, 
    test=test, 
    target_col='solar',
    exclude_bnos=None,
    drop_cols=drop_cols_for_solar,
    seed=SEED,
    n_splits=N_SPLIT, 
    n_trials=N_TRIAL, 
    sample_frac=SAMPLE_SIZE,
    restrict_daylight=True,
)

print("    > SMAPE :", oof_solar)

def _stats(x, ddof=1):
    x = np.asarray(x, dtype=float)
    return dict(
        mean=np.nanmean(x),
        var=np.nanvar(x, ddof=ddof),
        std=np.nanstd(x, ddof=ddof),
    )

print("\nPreprocessing Session Completed\n")
print("[Description]")

print("[Train - Zero_bnos]")
for i in zero_bnos:
    s = train_filled.loc[train_filled['building_num'] == i, 'solar']
    if len(s) == 0:
        print(f"Building Number: {i}  (no rows)")
        continue
    print(f"Building Number: {i}")
    print(f"Min ~ Max : {s.min():.6f} ~ {s.max():.6f}")
print(f"> SMAPE : {smape_score:.6f}")

# ---- Sunshine 분포 비교 (train vs test 예측)
print("\n[Test - Sunshine]")
tr_sun = train_filled['sunshine']
te_sun = test['sunshine']  # ← 너가 예측을 여기에 넣었다고 가정
st_tr  = _stats(tr_sun, ddof=1)
st_te  = _stats(te_sun, ddof=1)
print("Stats (train sunshine vs test sunshine)")
print(f"- train : mean={st_tr['mean']:.6f}, var={st_tr['var']:.6f}, std={st_tr['std']:.6f}")
print(f"- test  : mean={st_te['mean']:.6f}, var={st_te['var']:.6f}, std={st_te['std']:.6f}")
print(f"> SMAPE : {oof_sunshine:.6f}")

# ---- Solar 분포 비교 (train vs test 예측)
print("\n[Test - Solar]")
tr_sol = train_filled['solar']
te_sol = test['solar']     # ← 너가 예측을 여기에 넣었다고 가정
st_tr  = _stats(tr_sol, ddof=1)
st_te  = _stats(te_sol, ddof=1)
print("Stats (train solar vs test solar)")
print(f"- train : mean={st_tr['mean']:.6f}, var={st_tr['var']:.6f}, std={st_tr['std']:.6f}")
print(f"- test  : mean={st_te['mean']:.6f}, var={st_te['var']:.6f}, std={st_te['std']:.6f}")
print(f"> SMAPE : {oof_solar:.6f}\n")

def make_building_features(
    df: pd.DataFrame,
    date_col: str = "date",
    bno_col: str = "building_num",
    sunshine_col: str = "sunshine",   # 0~1 또는 시간값(시간 합계는 0~24)
    solar_col: str = "solar",         # MJ/m²
    pvc_col: str = "pvc",             # kW
    ess_col: str = "ess",             # kWh
    pcs_col: str = "pcs",             # kW
    pr: float = 0.80,                 # Performance Ratio
) -> pd.DataFrame:
    """
    일별 sunshine/solar 합계와 PV 발전량(추정)을 만들고, 시간별로도 분배한 피처를 추가.
    - pv_day_kwh_est  = pvc(kW) * (sum(solar)/3.6 kWh/m²) * PR
    - pv_hour_kwh_est = pv_day_kwh_est * (solar / day_sum_solar)   (day_sum_solar=0이면 0)
    """
    out = df.copy()

    # 0) 날짜 정규화: 일 단위 키 컬럼 생성 (문자열 'date_col'이 반드시 컬럼명)
    if date_col not in out.columns:
        raise KeyError(f"'{date_col}' 컬럼이 없습니다.")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    day_col = "__day"  # 내부 키 컬럼명
    out[day_col] = out[date_col].dt.normalize()  # 모두 00:00:00로 정규화

    # 1) 일별 합계(건물,일자)
    grp_cols = [bno_col, day_col]
    if bno_col not in out.columns:
        raise KeyError(f"'{bno_col}' 컬럼이 없습니다.")
    if sunshine_col not in out.columns:
        out[sunshine_col] = 0.0
    if solar_col not in out.columns:
        out[solar_col] = 0.0

    day_agg = (
        out.groupby(grp_cols, dropna=False)
           .agg(
               sunshine_day_hours=(sunshine_col, lambda s: s.sum(min_count=1)),
               solar_day_mj_m2   =(solar_col,    lambda s: s.sum(min_count=1)),
           )
           .reset_index()
    )
    day_agg["solar_day_kwh_m2"] = day_agg["solar_day_mj_m2"] / 3.6

    # 2) 기존 동명/임시 key_* 컬럼 제거 후 병합
    out = out.drop(columns=[c for c in ["sunshine_day_hours","solar_day_mj_m2","solar_day_kwh_m2"]
                            if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in out.columns if c.startswith("key_")], errors="ignore")

    # 3) 안전 병합 (왼쪽 m:1)
    #   on 에는 반드시 "컬럼명" 리스트를 전달
    out = out.merge(day_agg, on=grp_cols, how="left", validate="m:1")

    # 4) 클리핑
    out["sunshine_day_hours"] = out["sunshine_day_hours"].clip(lower=0, upper=24)

    # 5) 일별 PV 발전량(kWh) 추정
    out["pv_day_kwh_est"] = (
        out[pvc_col].astype(float)
        * out["solar_day_kwh_m2"].astype(float)
        * float(pr)
        if pvc_col in out.columns else 0.0
    )
    out["pv_day_kwh_est"] = pd.Series(out["pv_day_kwh_est"], index=out.index).fillna(0).clip(lower=0)

    # 6) 시간별 배분: 동일 키(grp_cols)로 day_sum_solar 계산
    day_sum_solar = out.groupby(grp_cols, dropna=False)[solar_col].transform("sum")
    num = out[solar_col].to_numpy(dtype=float)
    den = day_sum_solar.to_numpy(dtype=float)
    w = np.divide(num, den, out=np.zeros(len(out), dtype=float), where=den > 0)
    out["pv_hour_kwh_est"] = out["pv_day_kwh_est"].to_numpy(dtype=float) * w
    out["pv_hour_kwh_est"] = out["pv_hour_kwh_est"].fillna(0).clip(lower=0)

    # 7) 파생 지표
    out["pvc_per_day"]  = (out[pvc_col].astype(float) * out["sunshine_day_hours"].astype(float)
                           if pvc_col in out.columns else 0.0)
    out["solar_by_pvc"] = (out[solar_col].astype(float) * out[pvc_col].astype(float)
                           if pvc_col in out.columns else 0.0)

    # 8) ESS 관련 (선택)
    if pcs_col in out.columns:
        out["pv_to_ess_kwh_cap"] = np.minimum(
            out["pv_hour_kwh_est"].to_numpy(dtype=float),
            out[pcs_col].astype(float).clip(lower=0).to_numpy(dtype=float),
        )
    if ess_col in out.columns and pvc_col in out.columns:
        out["ess_hours_at_pvc"] = np.divide(
            out[ess_col].astype(float).to_numpy(dtype=float),
            np.maximum(out[pvc_col].astype(float).to_numpy(dtype=float), 1e-6),
        )

    return out

train_filled = make_building_features(train_filled)
test = make_building_features(test)

#region(데이터 정보2)
# print(train_clean.columns)
# Index(['building_num', '일시', 'temperature', 'precipitation', 'windspeed',
#        'humidity', 'sunshine', 'solar', 'power_consumption', 'building_type',
#        'all_area', 'cooling_area', 'pvc', 'ess', 'pcs', 'date', 'hour', 'dow',
#        'month', 'day', 'SIN_hour', 'COS_hour', 'SIN_day', 'COS_day',
#        'SIN_month', 'COS_month', 'SIN_dow', 'COS_dow', 'day_of_year',
#        'SIN_day_of_year', 'COS_day_of_year', 'sunrise_hour', 'sunset_hour',
#        'daylight', 'solar_elevation', 'PT', 'CDH', 'DI', 'PVC_per_CA',
#        'ESS_installation', 'PCS_installation', 'Facility_Density',
#        'SIN_summer', 'COS_summer', 'groupID', 'holidays', 'peak',
#        'dow_hour_mean', 'dow_hour_std', 'holiday_mean', 'holiday_std',
#        'hour_mean', 'hour_std', 'building_mean', 'building_std',
#        'sunshine_day_hours', 'solar_day_mj_m2', 'solar_day_kwh_m2',
#        'pv_day_kwh_est', 'pv_hour_kwh_est', 'pvc_per_day', 'solar_by_pvc',
#        'pv_to_ess_kwh_cap', 'ess_hours_at_pvc'],
#       dtype='object')
# print(test.columns)
# Index(['building_num', '일시', 'temperature', 'precipitation', 'windspeed',
#        'humidity', 'building_type', 'all_area', 'cooling_area', 'pvc', 'ess',
#        'pcs', 'date', 'hour', 'dow', 'month', 'day', 'SIN_hour', 'COS_hour',
#        'SIN_day', 'COS_day', 'SIN_month', 'COS_month', 'SIN_dow', 'COS_dow',
#        'day_of_year', 'SIN_day_of_year', 'COS_day_of_year', 'sunrise_hour',
#        'sunset_hour', 'daylight', 'solar_elevation', 'PT', 'CDH', 'DI',
#        'PVC_per_CA', 'ESS_installation', 'PCS_installation',
#        'Facility_Density', 'SIN_summer', 'COS_summer', 'groupID', 'holidays',
#        'peak', 'dow_hour_mean', 'dow_hour_std', 'holiday_mean', 'holiday_std',
#        'hour_mean', 'hour_std', 'building_mean', 'building_std', 'sunshine',
#        'solar', 'power_consumption', 'sunshine_day_hours', 'solar_day_mj_m2',
#        'solar_day_kwh_m2', 'pv_day_kwh_est', 'pv_hour_kwh_est', 'pvc_per_day',
#        'solar_by_pvc', 'pv_to_ess_kwh_cap', 'ess_hours_at_pvc'],
#       dtype='object')
#endregion(데이터 정보2)

#endregion INTERPOLATION

print(f"[STEP 3] MODEL")

#region MODEL

print(f"    Train for MODEL | {train_filled.shape}")
print(f"     Test for MODEL | {test.shape}\n")

# -------------------- helpers --------------------

def _optimize_with_pbar(objective, n_trials, desc, seed):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    pbar = tqdm(total=n_trials, desc=desc, dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {postfix}", leave=False)
    def _cb(study, trial):
        if study.best_value is not None:
            pbar.set_postfix_str(f"best={study.best_value:.4f}")
        pbar.update(1)
    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    pbar.write(f"    [{desc}] SMAPE: {study.best_value:.4f}")
    pbar.close()
    return study

def _valid_features(train_df, test_df, cols):
    cols = [c for c in cols if (c in train_df.columns) and (c in test_df.columns)]
    # 숫자형만 사용
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
    return cols

def auto_select_log_cols(df, cols):
    """양수이며 스케일 큰/왜도 큰 피처에 log1p 적용 권장"""
    log_cols = []
    for c in cols:
        x = pd.to_numeric(df[c], errors='coerce')
        if x.isna().all(): 
            continue
        x = x[x.notna()]
        if x.min() < 0:  # 음수 있으면 스킵
            continue
        q5, q95 = np.percentile(x, [5, 95])
        rng_ok = (q95 / (q5 + 1e-6)) > 20 or x.max() > 1_000
        if rng_ok:
            log_cols.append(c)
    # 확실히 큰 스케일 후보는 강제 포함(있을 때만)
    forced = ['all_area','cooling_area','pvc','ess','pcs',
              'pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
              'solar_by_pvc','Facility_Density']
    for c in forced:
        if c in cols and c not in log_cols:
            log_cols.append(c)
    return log_cols

def _make_X(df, features, log_cols):
    X = df[features].copy()
    for c in log_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c].astype(float), 0, None))
    return X

def _train_one_model_cv_optuna(
    train_df, target_col, feature_cols, seed=42,
    n_splits=5, n_trials=30, sample_frac=0.3,
):
    # --- 유효 피처/로그 후보 선정 ---
    feature_cols = _valid_features(train_df, train_df, feature_cols)
    if not feature_cols:
        raise ValueError("유효한 feature가 없습니다.")

    log_cols = auto_select_log_cols(train_df, feature_cols)
    # pandas Series 로 유지(아래 .loc/.iloc 사용)
    y_log = pd.Series(
        np.log1p(np.clip(train_df[target_col].astype(float), 0, None)),
        index=train_df.index
    )
    X_all = _make_X(train_df, feature_cols, log_cols)

    # --- 튜닝용 샘플 ---
    if 0 < sample_frac < 1:
        sample_idx = train_df.sample(frac=sample_frac, random_state=seed).index
    else:
        sample_idx = train_df.index
    Xs, ys = X_all.loc[sample_idx], y_log.loc[sample_idx]

    # ---------- (1) XGB 파라미터 튜닝 ----------
    def objective_model(trial):
        xgb_params = dict(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("xgb_eta", 0.03, 0.2, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 7),
            subsample=trial.suggest_float("xgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 2.0, 10.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.0, 3.0),
            random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mae",
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_pred = np.zeros(len(Xs), dtype=float)

        for tr_idx, val_idx in kf.split(Xs):
            X_tr, X_val = Xs.iloc[tr_idx], Xs.iloc[val_idx]
            y_tr, y_val = ys.iloc[tr_idx], ys.iloc[val_idx]

            med = X_tr.median()
            X_tr_f, X_val_f = X_tr.fillna(med), X_val.fillna(med)

            xgb = XGBRegressor(**xgb_params, early_stopping_rounds=50,)
            xgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                verbose=False
            )
            # 예측은 로그→원복 후 SMAPE
            oof_pred[val_idx] = np.clip(np.expm1(xgb.predict(X_val_f)), 0, None)

        return smape(np.expm1(ys.to_numpy()), oof_pred)

    study_params = _optimize_with_pbar(objective_model, n_trials=n_trials, desc="optuna(model)", seed=seed)
    bp = study_params.best_params

    xgb_best = {
        "n_estimators": bp["xgb_n_estimators"],
        "learning_rate": bp["xgb_eta"],
        "max_depth": bp["xgb_max_depth"],
        "subsample": bp["xgb_subsample"],
        "colsample_bytree": bp["xgb_colsample_bytree"],
        "min_child_weight": bp["xgb_min_child_weight"],
        "reg_lambda": bp["xgb_reg_lambda"],
        "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae",
    }

    # ---------- (2) Full 데이터 OOF ----------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(X_all), dtype=float)
    xgb_best_iters = []

    for tr_idx, val_idx in kf.split(X_all):
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]

        med = X_tr.median()
        X_tr_f, X_val_f = X_tr.fillna(med), X_val.fillna(med)

        xgb = XGBRegressor(**xgb_best, early_stopping_rounds=50,)
        xgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            verbose=False
        )
        oof_pred[val_idx] = np.clip(np.expm1(xgb.predict(X_val_f)), 0, None)
        xgb_best_iters.append(getattr(xgb, "best_iteration", None))

    oof_smape = float(smape(np.expm1(y_log.to_numpy()), oof_pred))

    # ---------- (3) 최종 재학습 ----------
    med_full = X_all.median()
    X_full = X_all.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]), xgb_best["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_best, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y_log, verbose=False)

    # 반환 시그니처 유지(호출부 안전): lgb_final=None, weight=(1.0, 0.0)
    return (
        xgb_final, None, med_full, feature_cols, log_cols,
        1.0, 0.0,
        oof_smape,
        pd.Series(oof_pred, index=train_df.index)
    )

def _predict_with_models(df, models_pack, clip0=True):
    xgb_final, _lgb_final, med, feats, log_cols, _w_xgb, _w_lgb = models_pack
    X = _make_X(df, feats, log_cols).fillna(med)
    pred = np.expm1(xgb_final.predict(X))
    if clip0:
        pred = np.clip(pred, 0, None)
    return pred

# ------------------------------ feature lists ------------------------------

pvc_features = [
    'building_type','groupID','holidays','peak',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'solar','solar_day_kwh_m2',
    'all_area','cooling_area',
    'ess','pcs',
    'pvc','pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
    'solar_by_pvc','pv_to_ess_kwh_cap','ess_hours_at_pvc',
    'PVC_per_CA','Facility_Density',
]

no_pvc_features = [
    'building_type','groupID','holidays','peak',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'solar','solar_day_kwh_m2',
    'all_area','cooling_area',
]
 
# -------------------- 1) 세그먼트 모델 (PVC/ESS/No-PVC) --------------------

def model_segmented(train_filled, test, target='power_consumption', seed=SEED):
    print("  [model_segmented] activate")

    # PVC = pvc>0, No-PVC = pvc<=0
    pvc_pos_train = (train_filled.get('pvc', 0).fillna(0) > 0)
    seg_pvc_tr  = pvc_pos_train
    seg_none_tr = ~pvc_pos_train

    pvc_pos_test = (test.get('pvc', 0).fillna(0) > 0)
    seg_pvc_te   = pvc_pos_test
    seg_none_te  = ~pvc_pos_test

    oof_series_full = pd.Series(np.nan, index=train_filled.index)
    preds = pd.Series(0.0, index=test.index)
    oofs  = {}

    def _unpack_train_result(res):
        # 9개 리턴 (신버전)
        if len(res) == 9:
            xgb, _lgb, med, feats, logs, _wx, _wl, oof_smape, oof_series = res
        # 7개 리턴 (구버전)
        elif len(res) == 7:
            xgb, _lgb, med, feats, logs, oof_smape, oof_series = res
        else:
            raise ValueError(f"_train_one_model_cv_optuna returned {len(res)} values; expected 7 or 9.")
        # XGB ONLY 강제
        return (xgb, None, med, feats, logs, 1.0, 0.0, float(oof_smape), oof_series)

    # === PVC ===
    tr_pvc = train_filled.loc[seg_pvc_tr]
    if len(tr_pvc) > 50 and seg_pvc_te.any():
        print(f"    [PVC Building]")
        res = _train_one_model_cv_optuna(
            tr_pvc, target, _valid_features(train_filled, test, pvc_features),
            seed=seed, n_splits=N_SPLIT, n_trials=N_TRIAL, sample_frac=SAMPLE_SIZE
        )
        xgb, lgb, med, feats, logs, w_xgb, w_lgb, oof, oof_series = _unpack_train_result(res)
        preds.loc[seg_pvc_te] = _predict_with_models(
            test.loc[seg_pvc_te], (xgb, lgb, med, feats, logs, w_xgb, w_lgb)
        )
        oofs['PVC'] = oof
        oof_series_full.loc[tr_pvc.index] = oof_series
        print(f"    > SMAPE : {oof:.6f}\n")
    else:
        print(f"    [PVC] skip (train_n={len(tr_pvc)}, test_n={seg_pvc_te.sum()})")

    # === No-PVC ===
    tr_none = train_filled.loc[seg_none_tr]
    if len(tr_none) > 50 and seg_none_te.any():
        print(f"    [No-PVC Building]")
        res = _train_one_model_cv_optuna(
            tr_none, target, _valid_features(train_filled, test, no_pvc_features),
            seed=seed, n_splits=N_SPLIT, n_trials=N_TRIAL, sample_frac=SAMPLE_SIZE
        )
        xgb, lgb, med, feats, logs, w_xgb, w_lgb, oof, oof_series = _unpack_train_result(res)
        preds.loc[seg_none_te] = _predict_with_models(
            test.loc[seg_none_te], (xgb, lgb, med, feats, logs, w_xgb, w_lgb)
        )
        oofs['NoPVC'] = oof
        oof_series_full.loc[tr_none.index] = oof_series
        print(f"    > SMAPE : {oof:.6f}")
    else:
        print(f"    [No-PVC] skip (train_n={len(tr_none)}, test_n={seg_none_te.sum()})")

    return preds.values, oofs, oof_series_full

# -------------------- 2) 유형별 모델 --------------------

def model_by_type(train_filled, test, target='power_consumption', type_col='building_type', seed=SEED):
    print("  [model_by_type] activate")
    pred = pd.Series(0.0, index=test.index)
    type_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_filled.index)
    for t in sorted(test[type_col].dropna().unique()):
        print(f"    [{t}]")
        tr_t = train_filled[train_filled[type_col] == t]
        te_t = test[test[type_col] == t]
        if len(tr_t) < 50:
            continue
        feats = _valid_features(train_filled, test, pvc_features)  # 포괄적 피처
        xgb,lgb,med,fs,logs,w_x,w_l,oof,oof_series = _train_one_model_cv_optuna(
            tr_t, target, feats,
            seed=seed, 
            n_splits=N_SPLIT, 
            n_trials=N_TRIAL, 
            sample_frac=SAMPLE_SIZE
        )
        # XGB ONLY 강제
        lgb, w_x, w_l = None, 1.0, 0.0
        pred.loc[te_t.index] = _predict_with_models(te_t, (xgb,lgb,med,fs,logs,w_x,w_l))
        type_scores[str(t)] = oof
        oof_series_full.loc[tr_t.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return pred.values, type_scores, oof_series_full

# -------------------- 3) 전체 모델 --------------------

def model_global(train_filled, test, target='power_consumption', seed=SEED):
    print("  [model_global] activate")
    
    feats = _valid_features(train_filled, test, pvc_features)
    xgb,lgb,med,fs,logs,w_x,w_l,oof,oof_series = _train_one_model_cv_optuna(
        train_filled, target, feats,
        seed=seed, 
        n_splits=N_SPLIT, 
        n_trials=N_TRIAL,
        sample_frac=SAMPLE_SIZE
    )
    # XGB ONLY 강제
    lgb, w_x, w_l = None, 1.0, 0.0
    pred = _predict_with_models(test, (xgb,lgb,med,fs,logs,w_x,w_l))
    print(f"    > SMAPE : {oof}")
    return pred, oof, oof_series

# -------------------- 4) 건물번호별 모델 (구현만, 사용 X) --------------------

def model_by_building_num(train_filled, test, target='power_consumption', bno_col='building_num', seed=SEED):
    print("  [model_by_building_num] activate")
    
    pred = pd.Series(0.0, index=test.index)
    b_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_filled.index)
    feats = _valid_features(train_filled, test, pvc_features)
    for b in sorted(test[bno_col].dropna().unique()):
        print(f"    [Building Num] {b}")
        
        tr_b = train_filled[train_filled[bno_col] == b]
        te_b = test[test[bno_col] == b]
        if len(tr_b) < 30:
            continue
        xgb,lgb,med,fs,logs,w_x,w_l,oof,oof_series = _train_one_model_cv_optuna(
            tr_b, target, feats,
            seed=seed, 
            n_splits=N_SPLIT,
            n_trials=N_TRIAL, 
            sample_frac=SAMPLE_SIZE
        )
        # XGB ONLY 강제
        lgb, w_x, w_l = None, 1.0, 0.0
        pred.loc[te_b.index] = _predict_with_models(te_b, (xgb,lgb,med,fs,logs,w_x,w_l))
        oof_series_full.loc[tr_b.index] = oof_series
        b_scores[str(b)] = oof
        print(f"    > SMAPE : {oof}")
        
    return pred.values, b_scores, oof_series_full

# ==================== RUN: 4모델 예측 후 앙상블 ====================
target = 'power_consumption'

print("[M1] Segmented (PVC/No-PVC)")
pred1, oof1, oof1_s = model_segmented(train_filled, test, target=target, seed=SEED)
avg1 = sum(oof1.values()) / len(oof1)   
print(f"[Segmented Model] mean SMAPE : {avg1}\n")

print("[M2] By Type")
pred2, oof2, oof2_s = model_by_type(train_filled, test, target=target, type_col='building_type', seed=SEED)
avg2 = sum(oof2.values()) / len(oof2)   
print(f"[By Type Model] mean SMAPE : {avg2}\n")

print("[M3] Global")
pred3, oof3, oof3_s = model_global(train_filled, test, target=target, seed=SEED)
print(f"[Global Model] mean SMAPE : {oof3}\n")

print("[M4] By BuildingNum")
pred4, oof4, oof4_s = model_by_building_num(train_filled, test, target=target, bno_col='building_num', seed=SEED)
avg4 = sum(oof4.values()) / len(oof4)   
print(f"[By BuildingNum Model] mean SMAPE : {avg4}\n")

# endregion MODEL

print(f"[STEP 4] ENSEMBLE")

#region ENSEMBLE

def _fit_ensemble_weights_smape_optuna(
    oof_df, y,
    base_weights=None,
    n_trials=60,
    seed=73,
    l2=1e-3,
    row_frac=1.0,
    dtype=np.float32,
    silence_optuna=True,
    show_progress=True,
    print_final=True,
    fix_last_weight=0.25,   # M>=4이면 마지막 가중치 고정
    min_head_weight=0.15,   # ★ w1,w2,w3의 최소 가중치
):
    """
    return: (w_best, best_smape)
    - M>=4: w_{M-1}=fix_last_weight로 고정, w1..w3는 각각 >= min_head_weight.
            남은 합(remain_sum - 3*min_head_weight)은 Optuna가 분배(정규화 기반).
    - M<4: 기존과 동일(전 가중치 자유 최적화).
    """

    # --- 유효 행 & 데이터 준비
    mask_any = oof_df.notna().any(axis=1).to_numpy()
    P_raw = oof_df.loc[mask_any].to_numpy(dtype)
    yv    = np.asarray(y.loc[mask_any], dtype=dtype)
    V     = (~np.isnan(P_raw)).astype(dtype)
    P     = np.nan_to_num(P_raw, nan=0.0)
    N, M  = P.shape

    # 행 서브샘플
    rng = np.random.default_rng(seed)
    if 0 < row_frac < 1.0:
        idx = rng.choice(N, size=int(max(1000, N*row_frac)), replace=False)
        P, V, yv = P[idx], V[idx], yv[idx]
        N = len(idx)

    eps = dtype(1e-12)

    use_fixed_last = (fix_last_weight is not None) and (M >= 4)
    fixed_last = dtype(fix_last_weight) if use_fixed_last else None
    remain_sum = dtype(1.0) - (fixed_last if use_fixed_last else dtype(0.0))
    H = (M - 1) if use_fixed_last else M  # 최적화할 가중치 개수(헤드 길이)

    # --- w1..w3 최소 가중치 벡터(m) 구성 (M>=4일 때만 적용)
    m = np.zeros(H, dtype=dtype)
    if use_fixed_last:
        K = min(3, H)  # 보통 3
        m[:K] = dtype(min_head_weight)
        sum_min = m.sum()
        # 불가능한 제약 방지: 남은 합이 음수면 최소값을 스케일 다운
        if sum_min > remain_sum:
            scale = float(remain_sum / (sum_min + eps))
            m *= dtype(scale)
        sum_min = m.sum()
        R = remain_sum - sum_min  # 자유롭게 분배할 잔여 합
    else:
        R = dtype(1.0)  # 전체를 자유 분배

    def _smape_vec(true, pred):
        return dtype(200.0) * np.nanmean(
            np.abs(pred - true) / (np.abs(pred) + np.abs(true) + eps)
        ).astype(dtype)

    def _build_w_from_raw(w_raw_head):
        # w_raw_head∈R^H → 정규화해 R를 분배, 최소치 m을 보장
        s = w_raw_head.sum()
        u = (np.ones(H, dtype=dtype) / dtype(H)) if s <= 0 else (w_raw_head / s)
        w_head = m + R * u  # 각 요소 ≥ m_i, 합 = sum(m) + R = remain_sum (또는 1.0)
        if use_fixed_last:
            w = np.empty(M, dtype=dtype)
            w[:H] = w_head
            w[M-1] = fixed_last
        else:
            w = w_head
        return w

    def objective(trial):
        w_raw = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(H)], dtype=dtype)
        w = _build_w_from_raw(w_raw)

        denom = V @ w
        num   = P @ w
        pred  = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)

        sm = _smape_vec(yv, pred)
        if l2 and l2 > 0:
            if use_fixed_last:
                # 제약을 반영한 '균등' 기준: m + R/H (헤드), 마지막은 fixed_last
                eq_head = m + (R / dtype(H))
                eq_vec = np.empty(M, dtype=dtype)
                eq_vec[:H] = eq_head
                eq_vec[M-1] = fixed_last
                sm = sm + dtype(l2) * np.sum((w - eq_vec) ** 2, dtype=dtype)
            else:
                eq = dtype(1.0 / M)
                sm = sm + dtype(l2) * np.sum((w - eq) ** 2, dtype=dtype)
        return float(sm)

    sampler = TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # 초기 후보 enqueue: equal / base
    # equal → 헤드 균등 분배
    study.enqueue_trial({f"w{i}": 1.0 for i in range(H)})
    if base_weights is not None:
        bw = np.asarray(base_weights, dtype=float)[:M]
        bw = np.clip(bw, 0, None)
        if bw.sum() > 0:
            if use_fixed_last:
                # 마지막 고정 + 헤드 최소보장 하에 재정규화
                head = bw[:H]
                s = head.sum()
                u = (np.ones(H) / H) if s <= 0 else (head / s)
                # 최소치 m을 보장하고 남은 합 R을 u로 분배
                head_adj = m.astype(float) + float(R) * u
                bw[:H] = head_adj
                bw[M-1] = float(fixed_last)
            else:
                bw = bw / bw.sum()
            study.enqueue_trial({f"w{i}": float(bw[i]) for i in range(H)})

    # Optuna 로그 & 진행바
    prev_level = None
    if silence_optuna:
        prev_level = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    pbar = tqdm(total=n_trials, leave=False, desc="Weight Search", disable=not show_progress)
    best_seen = np.inf
    def _callback(study_obj, trial):
        nonlocal best_seen
        if study_obj.best_value < best_seen:
            best_seen = study_obj.best_value
        if not pbar.disable:
            pbar.set_postfix(best=f"{best_seen:.4f}")
            pbar.update(1)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[_callback], gc_after_trial=True)
    finally:
        pbar.close()
        if silence_optuna and prev_level is not None:
            optuna.logging.set_verbosity(prev_level)

    # best w 재계산(제약 재적용)
    w_best_raw_head = np.array([study.best_trial.params.get(f"w{i}", 1.0) for i in range(H)], dtype=dtype)
    w_best = _build_w_from_raw(w_best_raw_head)

    # 최종 점수
    denom = V @ w_best
    num   = P @ w_best
    pred  = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)
    best_val = float(200.0 * np.nanmean(np.abs(pred - yv) / (np.abs(pred) + np.abs(yv) + 1e-12)))

    if print_final:
        print(f"    [ens_weight/optuna] best SMAPE: {best_val:.4f}  weights={np.round(w_best.astype(float),4)}")

    return w_best.astype(float), best_val

def _combine_test_preds_with_weights(pred_list, w_best):
    """
    pred_list: [pred1, pred2, pred3, (pred4)] 각 길이 T의 1D array-like, NaN 허용
    w_best: 길이 M의 가중치(합=1)
    반환: pred_final (길이 T)
    """
    import numpy as np
    P = np.vstack([np.asarray(p, dtype=float) for p in pred_list]).T  # [T, M]
    W = np.broadcast_to(np.asarray(w_best, dtype=float), P.shape)     # [T, M]
    # 행별로 예측이 NaN인 모델은 가중치 0
    valid = ~np.isnan(P)
    W = np.where(valid, W, 0.0)
    wsum = W.sum(axis=1, keepdims=True)
    # 가중치 합 0(그 행에선 전부 NaN)인 경우는 0으로 남김
    W_norm = np.divide(W, wsum, out=np.zeros_like(W), where=wsum > 0)
    P_filled = np.nan_to_num(P, nan=0.0)
    return (P_filled * W_norm).sum(axis=1)

def _combine_oof_with_weights(oof_df, w_best):
    """OOF DataFrame을 행별 결측 재정규화로 결합 → 예측 벡터 반환(denom<=0인 행은 NaN)."""
    P_raw = oof_df.to_numpy(float)                 # (N,M)
    V     = (~np.isnan(P_raw)).astype(float)       # (N,M)
    P     = np.nan_to_num(P_raw, nan=0.0)
    W     = np.broadcast_to(np.asarray(w_best, float), P.shape)
    W     = np.where(V > 0, W, 0.0)
    wsum  = W.sum(axis=1, keepdims=True)
    Wn    = np.divide(W, wsum, out=np.zeros_like(W), where=wsum > 0)
    return (np.nan_to_num(P, nan=0.0) * Wn).sum(axis=1)   # (N,)

def _safe_scope():
    """globals + locals 병합 스코프."""
    scope = {}
    scope.update(globals())
    try:
        scope.update(locals())
    except Exception:
        pass
    return scope

# 초기 가중(베이스라인)
W1, W2, W3, W4 = 0.25, 0.25, 0.25, 0.25

# ===== 최종 앙상블 OOF SMAPE (Optuna) =====
scope = _safe_scope()
oof_data = {}
for key, var in (('m1', 'oof1_s'), ('m2', 'oof2_s'), ('m3', 'oof3_s'), ('m4', 'oof4_s')):
    if var in scope:
        oof_data[key] = scope[var]

if not oof_data:
    raise RuntimeError("OOF 시리즈가 없습니다. (oof1_s/oof2_s/oof3_s/oof4_s 중 최소 1개 필요)")

oof_df = pd.DataFrame(oof_data)
cols = list(oof_df.columns)

# 베이스라인 가중치 준비(길이에 맞춰 자르기)
base = [W1, W2, W3, W4][:len(cols)]

# 1) Optuna로 최적 가중치
w_best, best_oof = _fit_ensemble_weights_smape_optuna(
    oof_df=oof_df,
    y=train_filled['power_consumption'],
    base_weights=base,
    n_trials=150,         # 필요에 따라 40~200 조절
    seed=SEED,
    l2=1e-3,
    row_frac=0.5,        # 속도/메모리 더 줄이려면 0.5 같은 값
    dtype=np.float32,
    fix_last_weight=0.25
)

# 2) 최종 OOF 점수(재검) — 행별 재정규화 동일
mask_any = oof_df.notna().any(axis=1).to_numpy()
oof_vec  = _combine_oof_with_weights(oof_df, w_best)      # (N,)
valid    = ~np.isnan(oof_vec)
y_eval   = train_filled.loc[mask_any, 'power_consumption'].to_numpy()[valid]
p_eval   = oof_vec[valid]
final_oof_smape = smape(y_eval, p_eval)
print(f"    [OOF] Final Ensemble SMAPE (best): {final_oof_smape:.4f}")

def _to_scalar_or_keep(x):
    try:
        return float(x)
    except Exception:
        return x

wmap = {c: _to_scalar_or_keep(w) for c, w in zip(cols, w_best)}

def fmt(c):
    v = wmap.get(c, "-")
    # 숫자면 소수 4자리
    if isinstance(v, (int, float, np.floating)):
        return f"{float(v):.4f}"
    # 리스트/배열이면 간단히 요약
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=float).ravel()
        head = ", ".join(f"{x:.4f}" for x in arr[:3])
        tail = "" if len(arr) <= 3 else ", ..."
        return f"[{head}{tail}] (len={len(arr)})"
    # dict면 key 몇 개만 보여줌
    if isinstance(v, dict):
        items = list(v.items())
        head = ", ".join(f"{k}:{float(val):.4f}" for k, val in items[:3] if isinstance(val, (int,float,np.floating)))
        more = "" if len(items) <= 3 else ", ..."
        return "{" + head + more + f"}} (|keys|={len(items)})"
    # 그 외 타입은 문자열로
    return str(v)

print(f"    > Ensemble Weight (M1): {fmt('m1')}")
print(f"    > Ensemble Weight (M2): {fmt('m2')}")
print(f"    > Ensemble Weight (M3): {fmt('m3')}")
print(f"    > Ensemble Weight (M4): {fmt('m4')}\n")

# 3) 테스트 결합
pred_list = []
for v in ('pred1', 'pred2', 'pred3', 'pred4'):
    if v in scope:
        pred_list.append(np.asarray(scope[v], float))
if not pred_list:
    raise RuntimeError("테스트 예측(pred1~pred4)이 없습니다.")

pred_final = _combine_test_preds_with_weights(pred_list, w_best)

#endregion ENSEMBLE

print(f"[STEP 5] SAVE")

#region SAVE

#region Total Description

print(f"\nModel Session Completed")
print(f"")
print(f"<Total Description>")
print(f"[Current Run SEED]: {SEED}")
print(f"")
print(f"[STEP1] PREPROCESSING")
print(f"    Before Feature Engineering ")
print(f"    - train : {train.shape} | test : {test.shape}")
print(f"    After  Feature Engineering ")
print(f"    - train : {train.shape} | test : {test.shape}")
print(f"")
print(f"[STEP2] INTERPOLATION")
print(f"  [Train - Solar]")
print(f"    > SMAPE {smape_score:.6f}")
print(f"  [Test  - Sunshine]")
print(f"    > SMAPE {oof_sunshine:.6f}")
print(f"  [Test  - Solar]")
print(f"    > SMAPE {oof_solar:.6f}")
print(f"")

# === 세그먼트 출력: PVC / No-PVC 로 통일 ===
print(f"[STEP3] MODEL")
print(f"  [M1] Segmented (PVC / No-PVC) - Weight {fmt('m1')}")
if isinstance(oof1, dict):
    if 'PVC'   in oof1 and oof1['PVC']   is not None:   print(f"    [PVC Building]   SMAPE {oof1['PVC']:.6f}")
    if 'NoPVC' in oof1 and oof1['NoPVC'] is not None:   print(f"    [No-PV Building] SMAPE {oof1['NoPVC']:.6f}")
else:
    print("    [Segmented scores] N/A")
print(f"  [Segmented] TOTAL SMAPE {avg1:.6f}")
print(f"")

print(f"  [M2] By Type - Weight {fmt('m2')}")
if isinstance(oof2, dict) and len(oof2) > 0:
    for t, sc in sorted(oof2.items(), key=lambda x: str(x[0])):
        print(f"    [{t}] SMAPE {sc:.6f}" if sc is not None else f"    [{t}] SMAPE N/A")
else:
    print("  [Type scores] N/A")
print(f"  [Type] TOTAL SMAPE {avg2:.6f}")
print(f"")

print(f"  [M3] Global - Weight {fmt('m3')}")
print(f"  [GLOBAL] TOTAL SMAPE {oof3:.6f}\n")

# M4가 있을 때만
if 'oof4' in scope or 'oof4' in locals():
    print(f"  [M4] By BuildingNum - Weight {fmt('m4')}")
    print(f"  [BuildingNum] TOTAL SMAPE {avg4:.6f}")

print(f"")
print(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}")
print(f"  > Ensemble Weight (M1): {fmt('m1')}")
print(f"  > Ensemble Weight (M2): {fmt('m2')}")
print(f"  > Ensemble Weight (M3): {fmt('m3')}")
print(f"  > Ensemble Weight (M4): {fmt('m4')}")

#endregion

save_path = f'./Energy/03/{SEED}_submission_{version}/'
os.makedirs(save_path , exist_ok=True )

submission['answer'] = pred_final

score_str = ("nan" if np.isnan(final_oof_smape) else f"{final_oof_smape:.4f}").replace('.', '_')
filename = f"03_SEED_{SEED}_SMAPE_{score_str}_{version}.csv"
submission.to_csv(save_path + filename, index=False)
print(f"FILENAME : {filename}")

#region LOG 저장부

with open(save_path + f"(LOG)model_{version}.txt", "a") as f:
    f.write(f"<SEED :{SEED}> ({version})\n")
    f.write(f"{filename}\n")
    f.write(f"[Train-Solar] SMAPE {smape_score:.6f}\n")
    f.write(f"[Test-Sunshine] SMAPE {oof_sunshine:.6f}\n")
    f.write(f"[Test-Solar] SMAPE {oof_solar:.6f}\n")
    f.write(f"[Segmented] SMAPE {avg1:.6f}\n")
    f.write(f"[Type] SMAPE {avg2:.6f}\n")
    f.write(f"[Global] SMAPE {oof3:.6f}\n")
    f.write(f"[By BuildingNum] SMAPE {avg4:.6f}\n")
    f.write(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}\n")
    f.write("="*40 + "\n")
        
with open(log_path + f"(LOG)model_{version}.txt", "a") as f:
    f.write(f"<SEED :{SEED}> ({version})\n")
    f.write(f"{filename}\n")
    f.write(f"[Train-Solar] SMAPE {smape_score:.6f}\n")
    f.write(f"[Test-Sunshine] SMAPE {oof_sunshine:.6f}\n")
    f.write(f"[Test-Solar] SMAPE {oof_solar:.6f}\n")
    f.write(f"[Segmented] SMAPE {avg1:.6f}\n")
    f.write(f"[Type] SMAPE {avg2:.6f}\n")
    f.write(f"[Global] SMAPE {oof3:.6f}\n")
    f.write(f"[By BuildingNum] SMAPE {avg4:.6f}\n")
    f.write(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}\n")
    f.write("="*40 + "\n")

#endregion(LOG 저장부)

#endregion SAVE

print(f"[VER 8] COMPLETED")