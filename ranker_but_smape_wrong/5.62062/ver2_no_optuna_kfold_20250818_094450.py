print(f"[PROTOTYPE_ver2] ACTIVATE")
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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

log_path = './Energy/03/log/'
os.makedirs(log_path, exist_ok=True)

seed_file = "./Energy/03/log/(prototype_ver2)SEED_COUNTS.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED =50# seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)
import datetime
save_path = f'./Energy/03/{SEED}_submission_ver2/'
os.makedirs(save_path , exist_ok=True )

import shutil
from pathlib import Path
def backup_self(dest_dir: str | Path = None, add_timestamp: bool = True) -> Path:
    src = Path(__file__).resolve()
    # 목적지 폴더: 환경변수 SELF_BACKUP_DIR > 인자 > ./_backup
    dest_root = Path(
        os.getenv("SELF_BACKUP_DIR") or dest_dir or (src.parent / "_backup")
    ).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    name = src.name
    if add_timestamp:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{src.stem}_{ts}{src.suffix}"

    dst = dest_root / name
    shutil.copy2(src, dst)   # 메타데이터 보존
    return dst

# 실행 즉시 백업
if __name__ == "__main__":
    saved = backup_self(dest_dir=save_path)  # 예: ./_backup/스크립트명_YYYYMMDD_HHMMSS.py
    print(f"[self-backup] saved -> {saved}\n")


random.seed(SEED)
np.random.seed(SEED)

py_path = './Energy/03/'

#######################################################################
#######################################################################
#######################################################################
#######################################################################

print("[STEP 1] PREPROCESSING")

#region함수정의

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
    mode='byb'                     # 'all'이면 ratio에 -0.005 보정(기존 fast_v2 호환)
):
    """
    - 공통 피처 생성: dow/hour 없으면 date로부터 생성
    - 집계: (building,hour,dow) mean/std, (building,hour,holiday) mean/std, (building,hour) mean/std,
            (building) mean/std
    - 결측 백필: dow_hour -> holiday -> hour -> building -> global
    - 주의: target 값 자체는 수정하지 않음(집계용 내부 복사본에만 요일가중/피크휴일 반영)
    """
    tr = train.copy()
    te = test.copy()

    # ---- 글로벌 백업 값
    global_mean = tr[target_col].mean()
    global_std  = tr[target_col].std(ddof=ddof)

    # ---- 집계 입력용 트레인 복사본(여기에만 옵션 적용)
    tr_feat = tr.copy()
    
    # 2) 요일 가중치 적용(집계 전용)
    pc = tr_feat[target_col].to_numpy(dtype=float)
    if apply_dow_ratio:
        ratio = np.array([0.985, 0.98, 0.98, 0.995, 0.995, 0.99, 0.99], dtype=float)
        if mode == 'all':
            ratio = ratio - 0.005
        idx = tr_feat[dow_col].to_numpy(dtype=int)
        pc = pc * ratio[idx]

    # ---- groupby 집계 (mean / std)
    tmp = tr_feat[[building_col, hour_col, dow_col, holiday_col]].copy()
    tmp['__pc'] = pc

    g1 = (tmp.groupby([building_col, hour_col, dow_col], as_index=False)['__pc']
             .agg(dow_hour_mean='mean', dow_hour_std=lambda x: x.std(ddof=ddof)))
    g2 = (tmp.groupby([building_col, hour_col, holiday_col], as_index=False)['__pc']
             .agg(holiday_mean='mean', holiday_std=lambda x: x.std(ddof=ddof)))
    g3 = (tmp.groupby([building_col, hour_col], as_index=False)['__pc']
             .agg(hour_mean='mean', hour_std=lambda x: x.std(ddof=ddof)))
    gb = (tmp.groupby([building_col], as_index=False)['__pc']
             .agg(building_mean='mean', building_std=lambda x: x.std(ddof=ddof)))

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

#endregion

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
train = add_data(train)
train, test = build_stats_features(train, test, mode='byb')
test = test.assign(
    sunshine=test.get('sunshine', 0.0),
    solar=test.get('solar', 0.0),
    power_consumption=test.get('power_consumption', 0.0),
)
print(f"    After Feature Engineering  - train : {train.shape} | test : {test.shape}")  # (612000, 55) (16800, 55)

#region(데이터 정보)
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

print("[STEP 2] INTERPOLATION")

def impute_solar_for_zero_bnos_cv_optuna(   # ← 함수명 그대로 유지
    train: pd.DataFrame,
    zero_bnos,
    bno_col: str = "building_num",
    target_col: str = "solar",
    daylight_col: str = "daylight",
    drop_cols=None,
    seed: int = 73,
    n_splits: int = 5,
    n_trials: int = 20,        # ← 더 이상 사용하지 않지만 시그니처 호환을 위해 남김
    sample_frac: float = 0.3,  # ← 더 이상 사용하지 않음
    w_xgb: float = 0.5,
    w_lgb: float = 0.5,
    time_col: str | None = None,   # 선택: 시간 정렬 열
    xgb_params: dict | None = None,
    lgb_params: dict | None = None,
):
    """
    train의 zero_bnos 건물에서 주간(daylight==1) 구간의 잘못(0) 기록된 'solar'를 모델로 보간.
    - Optuna 제거, 5-fold TimeSeriesSplit OOF로 평가/학습
    - 전처리: fold마다 결측을 train-fold 중앙값으로 대치
    - 최종: fold별 best_iteration 평균으로 n_estimators 조정 후 전체 재학습
    - 적용: zero_bnos & daylight==1 위치에 예측치 대입, 야간은 0
    반환: (train_filled, oof_smape)
    """
    # ---- SMAPE
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    drop_cols = set(drop_cols or [])

    # 학습 데이터: zero_bnos 제외 + 주간
    mask_train = (~train[bno_col].isin(zero_bnos)) & (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # 시간 정렬(가능하면 권장)
    if time_col is not None and time_col in df_train.columns:
        df_train = df_train.sort_values(time_col).reset_index(drop=True)

    # 피처 선택: 숫자형만, target/drop 제외
    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col}) and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    X = df_train[feature_cols]
    y = df_train[target_col].astype(float)

    # 기본 파라미터(보수적, 규제·서브샘플 포함) — 필요시 인자로 override
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=4.0,
            reg_lambda=1.0, random_state=seed, n_jobs=-1,
            tree_method="hist", eval_metric="mae"
        )
    if lgb_params is None:
        lgb_params = dict(
            n_estimators=1500, learning_rate=0.05, num_leaves=63,
            max_depth=-1, min_child_samples=60, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, max_bin=255,
            random_state=seed, n_jobs=-1, verbosity=-1
        )

    # ===== TimeSeriesSplit OOF =====
    # tscv = TimeSeriesSplit(n_splits=n_splits)
    tscv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(df_train), dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # fold-wise 결측 대치(누수 방지)
        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        # XGB
        xgb = XGBRegressor(**xgb_params, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        # LGBM
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        pred = w_xgb * px + w_lgb * pl
        oof_pred[val_idx] = np.clip(pred, 0, None)

        # best iters 기록
        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_smape = float(smape(y.to_numpy(), oof_pred))

    # ===== 전체 데이터 재학습 (best_iteration 평균 반영) =====
    med_full = X.median()
    X_full = X.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_params["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_params["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_params, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_params, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y)

    # ===== zero_bnos 주간만 예측 적용, 야간은 0 =====
    train_filled = train.copy()
    m_day = (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 1)
    if m_day.any():
        feats = train_filled.loc[m_day, feature_cols].fillna(med_full)
        px = xgb_final.predict(feats)
        pl = lgb_final.predict(feats, num_iteration=getattr(lgb_final, "best_iteration_", None))
        pred = w_xgb * px + w_lgb * pl
        train_filled.loc[m_day, target_col] = np.clip(pred, 0, None)

    train_filled.loc[
        (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 0),
        target_col
    ] = 0.0

    return train_filled, oof_smape

def train_predict_test_target_cv_optuna(  # ← 이름 유지
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,                 # 'sunshine' 또는 'solar'
    exclude_bnos=None,               # 예: zero_bnos (train 학습에서 제외)
    bno_col: str = "building_num",
    daylight_col: str = "daylight",
    drop_cols=None,
    seed: int = 42,
    n_splits: int = 5,
    n_trials: int = 20,              # (미사용) 시그니처 호환용
    sample_frac: float = 0.3,        # (미사용)
    w_xgb: float = 0.5,
    w_lgb: float = 0.5,
    restrict_daylight: bool = True,  # True면 주간만 학습/예측, 야간 0
    time_col: str | None = None,     # 시간 정렬 열(권장)
    xgb_params: dict | None = None,  # 하이퍼파라미터 override
    lgb_params: dict | None = None,
):
    """
    test의 target_col(sunshine/solar)을 예측한다.
    - Optuna 제거, TimeSeriesSplit(n_splits) OOF로 평가
    - fold별 결측은 train-fold 중앙값으로 채움(누수 방지)
    - fold별 best_iteration 평균으로 최종 n_estimators 조정 후 전체 재학습
    - restrict_daylight=True이면 주간만 예측, 야간은 0
    반환: (test_pred_series, oof_smape)
    """
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                             (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    drop_cols = set(drop_cols or [])
    exclude_bnos = set(exclude_bnos or [])

    # ---- 학습 데이터 마스크
    mask_train = (~train[bno_col].isin(exclude_bnos))
    if restrict_daylight:
        mask_train &= (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # 시간 정렬(있으면 권장)
    if time_col is not None and time_col in df_train.columns:
        df_train = df_train.sort_values(time_col).reset_index(drop=True)

    # ---- 피처 선택: 숫자형 & 드롭/타깃 제외 (+ test 공통 컬럼만 유지)
    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col})
        and pd.api.types.is_numeric_dtype(df_train[c])
        and c in test.columns
    ]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols/test 컬럼 일치 여부를 확인하세요.")

    X = df_train[feature_cols]
    y = df_train[target_col].astype(float)

    # ---- 기본 파라미터(보수적, 규제/샘플링 포함) — 필요시 인자로 override
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=4.0,
            reg_lambda=1.0, random_state=seed, n_jobs=-1,
            tree_method="hist", eval_metric="mae"
        )
    if lgb_params is None:
        lgb_params = dict(
            n_estimators=1500, learning_rate=0.05, num_leaves=63,
            max_depth=-1, min_child_samples=60, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, max_bin=255,
            random_state=seed, n_jobs=-1, verbosity=-1
        )

    # ===== TimeSeriesSplit OOF =====
    # tscv = TimeSeriesSplit(n_splits=n_splits)
    tscv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(df_train), dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # fold-wise 결측 대치
        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        # XGB
        xgb = XGBRegressor(**xgb_params, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        # LGBM
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        pred = w_xgb * px + w_lgb * pl
        # 타깃 타입에 맞춰 클리핑
        if target_col == 'sunshine':
            pred = np.clip(pred, 0, 1)
        else:
            pred = np.clip(pred, 0, None)

        oof_pred[val_idx] = pred

        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_smape = float(smape(y.to_numpy(), oof_pred))

    # ===== 전체 데이터 재학습 (best_iteration 평균 반영) =====
    med_full = X.median()
    X_full = X.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_params["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_params["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_params, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_params, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y)

    # ===== test 예측 =====
    test_pred = pd.Series(index=test.index, dtype=float)

    if restrict_daylight:
        m_day = (test[daylight_col] == 1)
        feats_day = test.loc[m_day, feature_cols].fillna(med_full)
        if len(feats_day) > 0:
            px = xgb_final.predict(feats_day)
            pl = lgb_final.predict(feats_day, num_iteration=getattr(lgb_final, "best_iteration_", None))
            pred = w_xgb * px + w_lgb * pl
            if target_col == 'sunshine':
                pred = np.clip(pred, 0, 1)
            else:
                pred = np.clip(pred, 0, None)
            test_pred.loc[m_day] = pred
        test_pred.loc[~m_day] = 0.0
    else:
        feats = test[feature_cols].fillna(med_full)
        px = xgb_final.predict(feats)
        pl = lgb_final.predict(feats, num_iteration=getattr(lgb_final, "best_iteration_", None))
        pred = w_xgb * px + w_lgb * pl
        if target_col == 'sunshine':
            pred = np.clip(pred, 0, 1)
        else:
            pred = np.clip(pred, 0, None)
        test_pred[:] = pred

    return test_pred, oof_smape

drop_cols = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'peak', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'power_consumption'
]

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

train_filled, smape_score = impute_solar_for_zero_bnos_cv_optuna(
    train=train,
    zero_bnos=zero_bnos,
    drop_cols=drop_cols,
    seed=SEED,            # 있으면
    n_splits=3,
    n_trials=1,
    sample_frac=0.01
)

print("    > SMAPE :", smape_score)

# 공통 드롭(타깃은 함수 내부에서 자동 제외)
drop_cols_for_sunshine = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'peak', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'solar','power_consumption'
]

drop_cols_for_solar = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'peak', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'power_consumption'
]

test['sunshine'], oof_sunshine = train_predict_test_target_cv_optuna(
    train=train_filled, test=test, target_col='sunshine',
    exclude_bnos=None,    # ← 또는 아예 인자 삭제
    drop_cols=drop_cols_for_sunshine,
    seed=SEED, n_splits=3, n_trials=1, sample_frac=0.01,
    restrict_daylight=True
)

print("    > SMAPE :", oof_sunshine)

test['solar'], oof_solar = train_predict_test_target_cv_optuna(
    train=train_filled, test=test, target_col='solar',
    exclude_bnos=None,
    drop_cols=drop_cols_for_solar,
    seed=SEED, n_splits=3, n_trials=1, sample_frac=0.01,
    restrict_daylight=True
)

print("    > SMAPE :", oof_solar)

def _stats(x, ddof=1):
    x = np.asarray(x, dtype=float)
    return dict(
        mean=np.nanmean(x),
        var=np.nanvar(x, ddof=ddof),
        std=np.nanstd(x, ddof=ddof),
    )

print("\nPreprocessing Session Completed")
print("Description")

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

train_filled = shift_power_from_6h_and_trim(train_filled)
train_clean, train_removed = drop_outlier_times(train, intervals, singles, inclusive='both')

def make_building_features(
    df: pd.DataFrame,
    date_col: str = "date",
    bno_col: str = "building_num",
    sunshine_col: str = "sunshine",   # 0~1 (또는 시간단위 hour값)로 가정
    solar_col: str = "solar",         # MJ/m²
    pvc_col: str = "pvc",             # kW (설비용량)
    ess_col: str = "ess",             # kWh (저장용량)
    pcs_col: str = "pcs",             # kW  (충·방전 파워 한도)
    pr: float = 0.80,                 # Performance Ratio (효율·손실 포함)
) -> pd.DataFrame:
    """
    일별 sunshine/solar 합계와 PV 발전량(추정)을 만들고, 시간별로도 배분한 피처를 추가.
    - pv_day_kwh_est = pvc(kW) * (sum(solar)/3.6 kWh/m²) * PR
    - pv_hour_kwh_est = pv_day_kwh_est * (solar / day_sum_solar)  (일중 분배)
    """
    out = df.copy()

    # 0) 날짜 정규화 (일 단위 키)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    day_key = out[date_col].dt.floor("D")

    # 1) 일별 합계(건물별-날짜별)
    # sunshine은 보통 hr 합계(0~24), solar는 일사량(MJ/m²) 합계 → kWh/m² 변환
    grp_keys = [bno_col, day_key]
    day_sunshine = out.groupby(grp_keys, dropna=False)[sunshine_col].sum(min_count=1).rename("sunshine_day_hours")
    day_solar_mj = out.groupby(grp_keys, dropna=False)[solar_col].sum(min_count=1).rename("solar_day_mj_m2")
    day_solar_kwh = (day_solar_mj / 3.6).rename("solar_day_kwh_m2")

    # 2) 일별 피처 머지
    out = out.join(day_sunshine, on=grp_keys)
    out = out.join(day_solar_mj, on=grp_keys)
    out = out.join(day_solar_kwh, on=grp_keys)

    # 품질보정: sunshine_day_hours는 0~24로 클립 (이상치 보호)
    out["sunshine_day_hours"] = out["sunshine_day_hours"].clip(lower=0, upper=24)

    # 3) 일별 PV 발전량(kWh) 추정: pvc(kW) * (일일 일사량 kWh/m²) * PR
    #  - 면적/효율은 PR에 흡수되어 있음(현실에서 PR~0.75~0.85)
    out["pv_day_kwh_est"] = out[pvc_col].astype(float) * out["solar_day_kwh_m2"].astype(float) * float(pr)
    out["pv_day_kwh_est"] = out["pv_day_kwh_est"].fillna(0).clip(lower=0)

    # 4) 시간별 배분: solar 비중으로 일중 분배
    #    시간이별 비중 w = solar / sum_day(solar), pv_hour = pv_day * w
    #    day_sum_solar가 0이면 전체 0
    day_sum_solar = out.groupby(grp_keys, dropna=False)[solar_col].transform("sum")
    w = np.divide(out[solar_col].to_numpy(float), day_sum_solar.to_numpy(float),
                  out=np.zeros(len(out), dtype=float), where=day_sum_solar.to_numpy(float) > 0)
    out["pv_hour_kwh_est"] = out["pv_day_kwh_est"].to_numpy(float) * w
    out["pv_hour_kwh_est"] = out["pv_hour_kwh_est"].fillna(0).clip(lower=0)

    # 5) 네가 시작하던 파생들(안전하게 완성)
    #    - pvc_per_day: 설비용량(kW)*그날 sunshine 합계(시간) → kWh에 비례하는 간단 지표
    out["pvc_per_day"]   = out[pvc_col].astype(float) * out["sunshine_day_hours"].astype(float)
    #    - solar_by_pvc: 순간 일사량 x 설비용량 → 시점별 생성 잠재력 proxy
    out["solar_by_pvc"]  = out[solar_col].astype(float) * out[pvc_col].astype(float)

    # 6) ESS 관련 간단 파생(선택): 시간별 충전 가능량(PCS로 제한, 1h 가정)
    if pcs_col in out.columns:
        out["pv_to_ess_kwh_cap"] = np.minimum(out["pv_hour_kwh_est"], out[pcs_col].astype(float).clip(lower=0))
    if ess_col in out.columns:
        # 설비용량만큼 몇 시간 방전 가능한지(rough proxy)
        out["ess_hours_at_pvc"] = np.divide(out[ess_col].astype(float), np.maximum(out[pvc_col].astype(float), 1e-6))

    return out

train_clean = make_building_features(train_clean)
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

#######################################################################
#######################################################################
#######################################################################
#######################################################################

print("[STEP 3] MODEL")
print(f"    Train for MODEL | {train_clean.shape}")
print(f"    Test for MODEL  | {test.shape}\n")

# -------------------- helpers --------------------

# --- 그대로 사용 ---
def _valid_features(train_df, test_df, cols):
    cols = [c for c in cols if (c in train_df.columns) and (c in test_df.columns)]
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
        if x.min() < 0:
            continue
        q5, q95 = np.percentile(x, [5, 95])
        rng_ok = (q95 / (q5 + 1e-6)) > 20 or x.max() > 1_000
        if rng_ok:
            log_cols.append(c)
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

# --- Optuna 제거 + TimeSeriesSplit 적용 ---
def _train_one_model_cv_optuna(   # ← 이름 유지(호환)
    train_df, target_col, feature_cols, seed=42,
    n_splits=5, n_trials=30, sample_frac=0.3,   # ← n_trials/sample_frac 미사용(호환용)
    w_xgb=0.5, w_lgb=0.5,
):
    """
    타깃은 log1p로 학습하고 예측은 expm1로 복원.
    - Optuna 완전 제거.
    - 교차검증은 TimeSeriesSplit(n_splits).
    - 각 fold에서 결측은 train-fold 중앙값으로 대치(누수 방지).
    - fold별 best_iteration 평균으로 최종 n_estimators 조정 후 전체 재학습.
    - OOF는 원 스케일로 반환.
    ※ 데이터는 '시간순으로 정렬되어 있다'고 가정(TimeSeriesSplit은 행 순서 기준).
    """
    # 피처 정제
    feature_cols = _valid_features(train_df, train_df, feature_cols)
    if not feature_cols:
        raise ValueError("유효한 feature가 없습니다.")

    # 자동 log1p 대상 선정
    log_cols = auto_select_log_cols(train_df, feature_cols)

    # 타깃 log1p
    y_log = np.log1p(np.clip(train_df[target_col].astype(float), 0, None))
    X_all = _make_X(train_df, feature_cols, log_cols)

    # 보수적인 기본 하이퍼(규제/서브샘플 포함)
    xgb_params = dict(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=4.0,
        reg_lambda=1.0, random_state=seed, n_jobs=-1,
        tree_method="hist", eval_metric="mae"
    )
    lgb_params = dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, min_child_samples=60, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, max_bin=255,
        random_state=seed, n_jobs=-1, verbosity=-1
    )

    # ===== TimeSeriesSplit OOF =====
    # tscv = TimeSeriesSplit(n_splits=n_splits)
    tscv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X_all), dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tscv.split(X_all):
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        xgb = XGBRegressor(**xgb_params, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        pred = w_xgb * np.expm1(px) + w_lgb * np.expm1(pl)
        oof_pred[val_idx] = np.clip(pred, 0, None)

        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    oof_smape = float(smape(np.expm1(y_log.to_numpy()), oof_pred))

    # ===== 전체 재학습(best_iteration 평균 반영) =====
    med_full = X_all.median()
    X_full = X_all.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_params["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_params["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_params, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y_log, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_params, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_log)

    # OOF를 원 스케일로 시리즈 반환(인덱스 일치)
    oof_series = pd.Series(oof_pred, index=X_all.index, name=f"OOF_{target_col}")

    return (
        xgb_final, lgb_final, med_full, feature_cols, log_cols,
        oof_smape, oof_series
    )

# --- 그대로 사용(필요시 가중치 수정 가능) ---
def _predict_with_models(df, models_pack, clip0=True):
    xgb_final, lgb_final, med, feats, log_cols = models_pack
    X = _make_X(df, feats, log_cols).fillna(med)
    px = xgb_final.predict(X)
    pl = lgb_final.predict(X)
    pred = 0.5 * np.expm1(px) + 0.5 * np.expm1(pl)
    if clip0:
        pred = np.clip(pred, 0, None)
    return pred

# -------------------- feature lists (from your code) --------------------
pvc_features = [
    'building_type','groupID','holidays','peak',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'solar','solar_day_kwh_m2','sunshine_day_hours',
    'all_area','cooling_area',
    'pvc','pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
    'solar_by_pvc','PVC_per_CA','Facility_Density',
]

ess_features = [
    'building_type','groupID','holidays','peak',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'solar','solar_day_kwh_m2','sunshine_day_hours',
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
    'solar','solar_day_kwh_m2','sunshine_day_hours',
    'all_area','cooling_area',
]

# -------------------- 1) 세그먼트 모델 (PVC/ESS/No-PVC) --------------------
def model_segmented(train_clean, test, target='power_consumption', seed=SEED):
    print("  [model_segmented] activate")
    # 세그먼트 정의(상호 배타)
    pvc_pos_train = (train_clean.get('pvc', 0).fillna(0) > 0)
    pcs_pos_train = (train_clean.get('pcs', 0).fillna(0) > 0)
    seg_ess_tr  = pvc_pos_train & pcs_pos_train
    seg_pvc_tr  = pvc_pos_train & (~pcs_pos_train)
    seg_none_tr = ~pvc_pos_train

    pvc_pos_test = (test.get('pvc', 0).fillna(0) > 0)
    pcs_pos_test = (test.get('pcs', 0).fillna(0) > 0)
    seg_ess_te  = pvc_pos_test & pcs_pos_test
    seg_pvc_te  = pvc_pos_test & (~pcs_pos_test)
    seg_none_te = ~pvc_pos_test

    oof_series_full = pd.Series(np.nan, index=train_clean.index)
    preds = pd.Series(0.0, index=test.index)
    oofs  = {}

    # ESS
    tr_ess = train_clean.loc[seg_ess_tr]
    if len(tr_ess) > 50 and seg_ess_te.any():
        print(f"    [PCS Building]")
        xgb,lgb,med,feats,logs,oof, oof_series  = _train_one_model_cv_optuna(
            tr_ess, target, _valid_features(train_clean,test,ess_features),
            seed=seed, n_splits=3, n_trials=1, sample_frac=0.05
        )
        preds.loc[seg_ess_te] = _predict_with_models(test.loc[seg_ess_te], (xgb,lgb,med,feats,logs))
        oofs['ESS'] = oof
        oof_series_full.loc[tr_ess.index] = oof_series
        print(f"    > SMAPE : {oof}")

    # PVC only
    tr_pvc = train_clean.loc[seg_pvc_tr]
    if len(tr_pvc) > 50 and seg_pvc_te.any():
        print(f"    [PV Building]")
        xgb,lgb,med,feats,logs,oof, oof_series  = _train_one_model_cv_optuna(
            tr_pvc, target, _valid_features(train_clean,test,pvc_features),
            seed=seed, n_splits=3, n_trials=1, sample_frac=0.05
        )
        preds.loc[seg_pvc_te] = _predict_with_models(test.loc[seg_pvc_te], (xgb,lgb,med,feats,logs))
        oofs['PVC'] = oof
        oof_series_full.loc[tr_ess.index] = oof_series
        print(f"    > SMAPE : {oof}")

    # No-PVC
    tr_none = train_clean.loc[seg_none_tr]
    if len(tr_none) > 50 and seg_none_te.any():
        print(f"    [No-PV Building]")
        xgb,lgb,med,feats,logs,oof, oof_series  = _train_one_model_cv_optuna(
            tr_none, target, _valid_features(train_clean,test,no_pvc_features),
            seed=seed, n_splits=3, n_trials=1, sample_frac=0.05
        )
        preds.loc[seg_none_te] = _predict_with_models(test.loc[seg_none_te], (xgb,lgb,med,feats,logs))
        oofs['NoPVC'] = oof
        oof_series_full.loc[tr_ess.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return preds.values, oofs, oof_series_full 

# -------------------- 2) 유형별 모델 --------------------
def model_by_type(train_clean, test, target='power_consumption', type_col='building_type', seed=SEED):
    print("  [model_by_type] activate")
    pred = pd.Series(0.0, index=test.index)
    type_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_clean.index)
    for t in sorted(test[type_col].dropna().unique()):
        print(f"    [Building Type] {t}")
        tr_t = train_clean[train_clean[type_col] == t]
        te_t = test[test[type_col] == t]
        if len(tr_t) < 50:
            continue
        feats = _valid_features(train_clean, test, pvc_features)  # 포괄적 피처
        xgb,lgb,med,fs,logs,oof, oof_series = _train_one_model_cv_optuna(tr_t, target, feats,
                                                             seed=seed, n_splits=3, n_trials=1, sample_frac=0.05)
        pred.loc[te_t.index] = _predict_with_models(te_t, (xgb,lgb,med,fs,logs))
        type_scores[str(t)] = oof
        oof_series_full.loc[tr_t.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return pred.values, type_scores, oof_series_full

# -------------------- 3) 전체 모델 --------------------
def model_global(train_clean, test, target='power_consumption', seed=SEED):
    print("  [model_global] activate")
    
    feats = _valid_features(train_clean, test, pvc_features)
    xgb,lgb,med,fs,logs,oof, oof_series = _train_one_model_cv_optuna(train_clean, target, feats,
                                                         seed=seed, n_splits=3, n_trials=1, sample_frac=0.05)
    pred = _predict_with_models(test, (xgb,lgb,med,fs,logs))
    print(f"    > SMAPE : {oof}")
    return pred, oof, oof_series

# -------------------- 4) 건물번호별 모델 (구현만, 사용 X) --------------------
def model_by_building_num(train_clean, test, target='power_consumption', bno_col='building_num', seed=42):
    print("  [model_by_building_num] activate")
    
    pred = pd.Series(0.0, index=test.index)
    b_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_clean.index)
    feats = _valid_features(train_clean, test, pvc_features)
    for b in sorted(test[bno_col].dropna().unique()):
        print(f"    [Building Type] {b}")
        
        tr_b = train_clean[train_clean[bno_col] == b]
        te_b = test[test[bno_col] == b]
        if len(tr_b) < 30:
            continue
        xgb,lgb,med,fs,logs,oof, oof_series = _train_one_model_cv_optuna(tr_b, target, feats,
                                                             seed=seed, n_splits=3, n_trials=1, sample_frac=0.05)
        pred.loc[te_b.index] = _predict_with_models(te_b, (xgb,lgb,med,fs,logs))
        oof_series_full.loc[tr_b.index] = oof_series
        b_scores[str(b)] = oof
        print(f"    > SMAPE : {oof}")
        
    return pred.values, b_scores, oof_series

# ==================== RUN: 4모델 예측 후 앙상블 ====================
target = 'power_consumption'

print("[M1] Segmented (ESS/PVC/NoPVC)")
pred1, oof1, oof1_s = model_segmented(train_clean, test, target=target, seed=SEED)
avg1 = sum(oof1.values()) / len(oof1)   
print(f"[Segmented Model] SMAPE : {avg1}\n")

print("[M2] By Type")
pred2, oof2, oof2_s = model_by_type(train_clean, test, target=target, type_col='building_type', seed=SEED)
avg2 = sum(oof2.values()) / len(oof2)   
print(f"[By Type Model] SMAPE : {avg2}\n")

print("[M3] Global")
pred3, oof3, oof3_s = model_global(train_clean, test, target=target, seed=SEED)
print(f"[Global Model] SMAPE : {oof3}\n")

print("[M4] By BuildingNum")
pred4, oof4, oof4_s = model_by_building_num(train_clean, test, target=target, bno_col='building_num', seed=SEED)
avg4 = sum(oof4.values()) / len(oof4)   
print(f"[By BuildingNum Model] SMAPE : {avg4}\n")

# 앙상블 가중 (원하면 조정)
W1, W2, W3, W4 = 0.25, 0.25, 0.25, 0.25
pred_final = W1*pred1 + W2*pred2 + W3*pred3 + W4*pred4

# ===== 최종 앙상블 OOF SMAPE =====
oof_df = pd.DataFrame({
    'm1': oof1_s,   # 세그먼트 OOF (행별로 값/NaN)
    'm2': oof2_s,   # 유형별 OOF
    'm3': oof3_s,   # 글로벌 OOF
    'm4': oof4_s,   # 번호별 OOF
})
w = np.array([W1, W2, W3, W4], dtype=float)
W = pd.DataFrame(np.broadcast_to(w, oof_df.shape), index=oof_df.index, columns=oof_df.columns)

# 각 행에서 예측이 NaN인 모델 가중치는 0으로
W = W.where(oof_df.notna(), 0.0)
wsum = W.sum(axis=1)
# 0인 행(모두 NaN) 제외
mask = (wsum > 0) & oof_df.notna().any(axis=1)

# 행별 가중 재정규화 후 앙상블 OOF
W_norm = W.div(wsum, axis=0).where(mask, 0.0)
oof_ens = (oof_df.fillna(0.0) * W_norm).sum(axis=1)

final_oof_smape = smape(train_clean.loc[mask, 'power_consumption'].to_numpy(),
                        oof_ens.loc[mask].to_numpy())
print(f"[OOF] Final Ensemble SMAPE: {final_oof_smape}\n")



submission['answer'] = pred_final

today = datetime.datetime.now().strftime('%Y%m%d')
score_str = ("nan" if np.isnan(final_oof_smape) else f"{final_oof_smape:.4f}").replace('.', '_')
filename = f"(proto){SEED}_03_{today}_SMAPE_{score_str}.csv"
submission.to_csv(save_path + filename, index=False)

with open(save_path + "(LOG)model_ver2.txt", "a") as f:
    f.write(f"<SEED :{SEED}> ver2\n")
    f.write(f"{filename}\n")
    f.write(f"[Segmented] SMAPE {avg1:.6f}\n")
    f.write(f"[Type] SMAPE {avg2:.6f}\n")
    f.write(f"[Global] SMAPE {oof3:.6f}\n")
    f.write(f"[By BuildingNum] SMAPE {avg4:.6f}\n")
    f.write(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}\n")
    f.write("="*40 + "\n")
        
with open(log_path + "(LOG)model_ver2.txt", "a") as f:
    f.write(f"<SEED :{SEED}> ver2\n")
    f.write(f"{filename}\n")
    f.write(f"[Segmented] SMAPE {avg1:.6f}\n")
    f.write(f"[Type] SMAPE {avg2:.6f}\n")
    f.write(f"[Global] SMAPE {oof3:.6f}\n")
    f.write(f"[By BuildingNum] SMAPE {avg4:.6f}\n")
    f.write(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}\n")
    f.write("="*40 + "\n")

print(f"[PROTOTYPE_ver2] COMPLETED")
