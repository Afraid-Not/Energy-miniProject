version = "LETS_GO"

print(f"[{version}] ACTIVATE\n")

#region IMPORT
# ========== Standard Library ==========
import json
import os
import random
import shutil
import warnings
from pathlib import Path
import hashlib

# ========== Third-Party ==========
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress
from tqdm import tqdm

# Scikit-learn
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

# Gradient Boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


log_path = './Energy/03/log/'
os.makedirs(log_path, exist_ok=True)

seed_file = f"./Energy/03/log/({version})SEED_COUNTS.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 74}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

#endregion IMPORT

SEED = 775 #seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

#region BASIC OPTIONS
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)
import datetime
save_path = f'./Energy/03/{SEED}_submission_{version}/'
os.makedirs(save_path , exist_ok=True )

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

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

py_path = './Energy/03/'

#endregion BASIC OPTIONS

N_SPLITS = 5
N_TRIALS = 25
N_SPLITS_MODEL = 5
N_TRIALS_MODEL = 25
SAMPLE_SIZE = 0.5
SAMPLE_SIZE_MODEL = 0.5
SAMPLE_SIZE_BNUM = 0.7
AUGMENTATION = 2

print(f"[Preprocessing Variable]")
print(f"N_SPLITS {N_SPLITS} | N_TRIALS {N_TRIALS}")
print(f"SAMPLE_SIZE {SAMPLE_SIZE} | AUGMENTATION {60/AUGMENTATION} times")
print(f"[Model Variable]")
print(f"N_SPLITS_MODEL {N_SPLITS_MODEL} | N_TRIALS_MODEL {N_TRIALS_MODEL}")
print(f"SAMPLE_SIZE_MODEL {SAMPLE_SIZE_MODEL} | SAMPLE_SIZE_BNUM {SAMPLE_SIZE_BNUM}\n")

print("[STEP 1] PREPROCESSING")

#region STEP 1

#region FUNCTIONS
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

def date_features(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['date'] = pd.to_datetime(df['일시'])

    df['minute'] = df['date'].dt.minute
    df['hour'] = df['date'].dt.hour                      # 시각(0~23)
    df['dow'] = df['date'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    df = df.sort_values(['building_num', 'date']).reset_index(drop=True)

    # 0을 NaN으로 처리 (보간 대상)
    df[['windspeed','humidity']] = df[['windspeed','humidity']].replace(0, np.nan)

    # 함수 정의: 앞뒤 평균, 없으면 ffill
    def fill_mean_ffill(s: pd.Series):
        # interpolate(method='linear')로 앞뒤 평균 대체
        s = s.interpolate(method='linear', limit_direction='both')
        # 그래도 NaN 남은 건 ffill → bfill까지 해주면 완벽
        s = s.ffill().bfill()
        return s

    # 건물번호별 적용
    df[['windspeed','humidity']] = (
        df.groupby('building_num')[['windspeed','humidity']]
        .transform(fill_mean_ffill)
    )

    
    return df

def feature_engineering(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    _total_min = ((pd.to_numeric(df['hour'], errors='coerce') * 60) +
                pd.to_numeric(df['minute'], errors='coerce')) % 1440

    _theta = 2 * np.pi * (_total_min / 1440.0)

    df['SIN_Time'] = np.sin(_theta).astype('float32')
    df['COS_Time'] = np.cos(_theta).astype('float32')
    df['SIN_minute'] = np.sin(2 * np.pi * df['minute'] / 60)  # 주기적 패턴
    df['COS_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['SIN_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  # 주기적 패턴
    df['COS_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    days_in_month = df['month'].map({6: 30, 7: 31, 8: 31}).astype('int16')

    # day가 1부터 시작한다고 가정. (0부터 시작하고 싶으면 df['day']-1 대신 df['day'] 사용)
    theta = 2 * np.pi * (df['day'] - 1) / days_in_month

    df['SIN_day'] = np.sin(theta).astype('float32')
    df['COS_day'] = np.cos(theta).astype('float32')
        
    df['SIN_month'] = np.sin(2 * np.pi * df['month'] / 12)  # 일의 주기적 패턴 (31일 기준)
    df['COS_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['SIN_dow'] = np.sin(2 * np.pi * df['dow'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_dow'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear
    
    df['SIN_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['COS_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    df = df.drop(['temp_date'], axis=1)
    
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

def upsample_20min_linear(
    df: pd.DataFrame,
    time_col: str,
    group_cols=None,
    numeric_cols=None,
    copy_non_numeric_from='prev',
    enforce_hour_gap=False,
    minutes=15
) -> pd.DataFrame:
    df = df.copy()
    group_cols = group_cols or []

    # 시간 파싱
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        try:
            df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
        except Exception:
            df[time_col] = pd.to_datetime(df[time_col])

    # 수치 컬럼 자동 선택
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.difference(group_cols).tolist()

    non_num_cols = [c for c in df.columns if c not in numeric_cols + [time_col]]

    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).reset_index(drop=True)
        rows = []
        n = len(g)
        for i in range(n - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]

            # 원본 행
            base = cur.copy()
            base['is_interpolated'] = 0
            rows.append(base)

            delta = nxt[time_col] - cur[time_col]
            if enforce_hour_gap and delta != pd.Timedelta(hours=1):
                continue

            step = pd.Timedelta(minutes=minutes)
            m = int(delta / step)  # 몇 개의 20분 스텝이 들어가는가
            # 내부 지점만 추가 (다음 시각과 중복 방지)
            for j in range(1, m):
                t = cur[time_col] + j * step
                frac = (t - cur[time_col]) / delta  # 시간 비율과 일치

                new = cur.copy()
                new[time_col] = t
                for col in numeric_cols:
                    new[col] = cur[col] + float(frac) * (nxt[col] - cur[col])
                new['is_interpolated'] = 1

                if copy_non_numeric_from == 'next':
                    for col in non_num_cols:
                        new[col] = nxt[col]
                elif copy_non_numeric_from is None:
                    for col in non_num_cols:
                        new[col] = pd.NA
                # 'prev'면 cur 값 유지

                rows.append(new)

        # 마지막 원본 행
        last = g.iloc[-1].copy()
        last['is_interpolated'] = 0
        rows.append(last)
        return pd.DataFrame(rows)

    if group_cols:
        out = (df.groupby(group_cols, group_keys=False)
                 .apply(_one_group)
                 .sort_values(group_cols + [time_col])
                 .reset_index(drop=True))
    else:
        out = _one_group(df).sort_values(time_col).reset_index(drop=True)

    out['minute'] = out[time_col].dt.minute.astype('int16')
    return out

def peak_holidays(df, is_train=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['일시'])
    df['dow']  = df['date'].dt.weekday  # 0=월 ... 6=일

    # 기본값
    df['holidays'] = 0
    # df['peak'] = 0

    # --- 건물별 '정기 휴무 요일' 지정 ---
    df.loc[(df['building_num']==2) & (df['dow']==5), 'holidays'] = 1       # 토
    df.loc[(df['building_num']==3) & (df['dow'].isin([5,6])), 'holidays'] = 1  # 토/일
    # df.loc[(df['building_num']==4) & (df['dow']==0), 'holidays'] = 1       # 월
    df.loc[(df['building_num']==5) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==6) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==7) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['building_num']==8) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # df.loc[(df['building_num']==10) & (df['dow']==0), 'holidays'] = 1
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
    # 33?
    df.loc[(df['building_num']==37) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==38) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==39) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==42) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==43) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==44) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # df.loc[(df['building_num']==45) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==46) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==47) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==48) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==49) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # df.loc[(df['building_num']==50) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==51) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==52) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==53) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==55) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # df.loc[(df['building_num']==56) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==60) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==61) & (df['dow'].isin([1,2,3,4,5])), 'holidays'] = 1    #확인
    df.loc[(df['building_num']==62) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # df.loc[(df['building_num']==64) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==66) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==67) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==68) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==69) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==72) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==75) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==80) & (df['dow'].isin([5,6])), 'holidays'] = 1
    # df.loc[(df['building_num']==81) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==82) & (df['dow'].isin([0])), 'holidays'] = 1
    df.loc[(df['building_num']==83) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==86) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==87) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==90) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['building_num']==94) & (df['dow'].isin([5,6])), 'holidays'] = 1

    if is_train :
        nat = df['date'].dt.strftime('%m-%d').isin(['06-06','08-15'])
        df.loc[nat, 'holidays'] = 1

        # 33은 이상하니까
        df.loc[(df['building_num']==33) & (df['date'].dt.strftime('%m-%d').isin(['06-08','07-05','07-06','07-07','08-23'])), 'holidays'] = 2
        
        df.loc[(df['building_num']==19) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 2
        df.loc[(df['building_num']==27) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==29) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-23','07-10','07-28','08-10'])), 'holidays'] = 2
        df.loc[(df['building_num']==32) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-24','07-08','07-22','08-12'])), 'holidays'] = 2
        # df.loc[(df['building_num']==38) & (df['date'].dt.strftime('%m-%d').isin(['06-07'])), 'holidays'] = 1
        df.loc[(df['building_num']==40) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==45) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 2
        df.loc[(df['building_num']==54) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 2
        # df.loc[(df['building_num']==56) & (df['date'].dt.strftime('%m-%d').isin(['06-07','08-16'])), 'holidays'] = 1
        df.loc[(df['building_num']==59) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==63) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 2
        df.loc[(df['building_num']==74) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01'])), 'holidays'] = 2
        df.loc[(df['building_num']==79) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 2
        # df.loc[(df['building_num']==94) & (df['date'].dt.strftime('%m-%d').isin(['06-07','08-16'])), 'holidays'] = 2
        df.loc[(df['building_num']==95) & (df['date'].dt.strftime('%m-%d').isin(['07-08','08-05'])), 'holidays'] = 2
    
    else :
        df.loc[(df['building_num']==27) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==29) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==32) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 2
        df.loc[(df['building_num']==40) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==59) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==63) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 2
        df.loc[(df['building_num']==74) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 2        
    
    return df

intervals = {
    5: [(2024080407, 2024080408)], #
    6: [(2024081500, 2024081900)],
    7: [(2024070710, 2024070811), (2024071214, 2024080603)],
    8: [(2024072108, 2024072111)], #
    9: [(2024061210, 2024061211)], #
    12: [(2024072109, 2024072111), (2024082408, 2024082410)], #
    17: [(2024062515, 2024062609)],
    19: [(2024073113, 2024073116)], #
    20: [(2024060110, 2024060111)], #
    25: [(2024070412, 2024070414)], #
    26: [(2024061714, 2024061811)], #
    28: [(2024071714, 2024071715), (2024060906, 2024060920)], #
    29: [(2024061522, 2024061523), (2024062700, 2024062701)],
    36: [(2024060100, 2024060923)],
    38: [(2024071714, 2024071715)], #
    40: [(2024071400, 2024071401)],
    41: [(2024062201, 2024062204), (2024071714, 2024071715)],
    43: [(2024061017, 2024061018), (2024081216, 2024081217)],
    44: [(2024060612, 2024060613), (2024063000, 2024063002)], #
    52: [(2024081000, 2024081002)],
    53: [(2024061417, 2024061707), (2024081816, 2024081907)],
    57: [(2024060100, 2024060721)],
    60: [(2024071714, 2024071715)], #
    62: [(2024071714, 2024071715)], #
    65: [(2024060100, 2024060823)], #
    67: [(2024061017, 2024061018), (2024072600, 2024072800), (2024080115, 2024080116), (2024081216, 2024081217)], #
    68: [(2024062823, 2024062901)],
    69: [(2024071714, 2024071715)], #
    70: [(2024060409, 2024060509)], #
    72: [(2024061100, 2024061102), (2024072110, 2024072111)],
    76: [(2024062013, 2024062016)], #
    78: [(2024071712, 2024071713)], #
    79: [(2024081903, 2024081905)],
    80: [(2024070609, 2024070615), (2024070811, 2024070820), (2024072009, 2024072013)], #
    88: [(2024082306, 2024082308)],
    89: [(2024071208, 2024071210)], #
    90: [(2024060517, 2024060518)], #
    92: [(2024071714, 2024071723)], #
    94: [(2024072620, 2024080507)],
    95: [(2024080510, 2024080511)], #
    97: [(2024071713, 2024071715)], #
    98: [(2024061314, 2024061315)], #
    99: [(2024071005, 2024071007)],
}

singles = {
    3: [2024071714], #
    12: [2024071714], #
    18: [2024061117, 2024071714, 2024080815], #
    30: [2024071320, 2024072500],
    31: [2024071714], #
    42: [2024071714], #
    46: [2024071714], #
    47: [2024071714], #
    50: [2024070514, 2024080815], #
    51: [2024072917], #
    55: [2024071714], #
    57: [2024062104], #
    73: [2024070822],
    76: [2024060313, 2024082221], #
    77: [2024080617],
    78: [2024071714],
    81: [2024062714, 2024071714], #
    82: [2024071714], #
    83: [2024071714], #
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

    g1 = (tmp.groupby([building_col, hour_col, dow_col], as_index=False)
            .agg(dow_hour_mean=('__pc', 'mean'),
                dow_hour_std =('__pc', lambda x: x.std(ddof=ddof))))

    g2 = (tmp.groupby([building_col, hour_col, holiday_col], as_index=False)
            .agg(holiday_mean=('__pc', 'mean'),
                holiday_std =('__pc', lambda x: x.std(ddof=ddof))))

    g3 = (tmp.groupby([building_col, hour_col], as_index=False)
            .agg(hour_mean=('__pc', 'mean'),
                hour_std =('__pc', lambda x: x.std(ddof=ddof))))

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

def attach_solar_geometry(df, *, 
                          lat=37.5665,       # 위도(서울)
                          lon=126.9780,      # 경도(서울 시청 기준)
                          tz_hours=9.0):     # KST=UTC+9
    df = df.copy()

    # --- 1) day_of_year 필요
    if 'day_of_year' not in df.columns:
        # '일시'가 YYYYMMDDHH.. 형태라면:
        temp_date = pd.to_datetime(df['일시'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df['day_of_year'] = temp_date.dt.dayofyear
    n = df['day_of_year'].to_numpy(dtype=float)

    # --- 2) 태양 적위 δ
    delta_deg = 23.44 * np.sin(np.deg2rad(360.0 * (284.0 + n) / 365.0))
    delta = np.deg2rad(delta_deg)

    # --- 3) 시간 보정: EoT + 경도 차이
    # B는 라디안; EoT(분) ≈ 9.87 sin(2B) - 7.53 cos B - 1.5 sin B
    B = 2.0 * np.pi * (n - 81.0) / 364.0
    eot_min = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)  # 분
    LSTM = 15.0 * tz_hours  # 표준자오선(한국=135°E)
    tc_min = 4.0 * (lon - LSTM) + eot_min   # 분, Local Solar Time = clock + tc_min/60
    tc_h = tc_min / 60.0                    # 시간

    # --- 4) 일출/일몰 (먼저 '태양시' 기준 계산)
    phi = np.deg2rad(lat)
    cosH0 = -np.tan(phi) * np.tan(delta)
    H0 = np.arccos(np.clip(cosH0, -1.0, 1.0))     # 라디안
    H0h = (12.0 / np.pi) * H0                     # 시간
    sunrise_solar = 12.0 - H0h
    sunset_solar  = 12.0 + H0h

    # --- 5) 표준시(KST)로 변환: clock = solar - tc_h
    sunrise = sunrise_solar - tc_h
    sunset  = sunset_solar  - tc_h

    df['sunrise_hour'] = sunrise
    df['sunset_hour']  = sunset

    # --- 6) daylight 구간 정렬: sunshine가 (t-1, t] 누적이므로 겹치면 1
    h = df['hour'].to_numpy(dtype=float)
    df['daylight_prev'] = ((h > sunrise) & ((h - 1.0) < sunset)).astype('int8')

    # (참고) 순간판정/다음시간 누적이 필요하면 같이 보관
    df['daylight_instant'] = ((h >= sunrise) & (h < sunset)).astype('int8')
    df['daylight_next']    = (((h + 1.0) > sunrise) & (h < sunset)).astype('int8')

    # --- 7) 태양 고도: 구간 중앙 t-0.5h, '태양시' = clock + tc_h
    t_mid_clock = h - 0.5
    lst_mid = t_mid_clock + tc_h          # Local Solar Time
    H_deg = 15.0 * (lst_mid - 12.0)
    H = np.deg2rad(H_deg)
    elev = np.arcsin(np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(H))
    df['solar_elevation'] = np.rad2deg(elev)

    return df

#endregion

train_path = './Energy/train.csv'
test_path = './Energy/test.csv'
building_path = './Energy/building_info.csv'
sub_path = './Energy/sample_submission.csv'
train, test, submission = load_data(train_path, test_path, building_path, sub_path)

print(f"    Before Feature Engineering - train : {train.shape} | test : {test.shape}")  # (204000, 15) (16800, 12)
train = date_features(train)
test = date_features(test)

numeric_cols= [
    'temperature','humidity', 'windspeed',
]

train = upsample_20min_linear(train, time_col="date", group_cols=["building_num"], numeric_cols=numeric_cols, minutes=AUGMENTATION)

train = feature_engineering(train)
test = feature_engineering(test)

train = peak_holidays(train)
test = peak_holidays(test, is_train=False)

KOREA_MAINLAND_LON = 127.7

train = attach_solar_geometry(train, lat=37.5665, lon=KOREA_MAINLAND_LON, tz_hours=9.0)
test  = attach_solar_geometry(test,  lat=37.5665, lon=KOREA_MAINLAND_LON, tz_hours=9.0)

train, test = build_stats_features(train, test, mode='byb')

test = test.assign(
    sunshine=test.get('sunshine', 0.0),
    solar=test.get('solar', 0.0),
    power_consumption=test.get('power_consumption', 0.0),
    is_interpolated=test.get('is_interpolated', 0.0),
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

#endregion STEP 1

print("[STEP 2] INTERPOLATION")

#region STEP 2

def _to_py(o):
    """numpy/scalar를 JSON 직렬화 가능한 파이썬 기본형으로 변환"""
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if hasattr(o, "item"):
        try: return o.item()
        except Exception: pass
    if isinstance(o, (np.floating, np.integer)):
        return o.tolist()
    return o

def _feat_sig(feature_cols):
    s = ",".join(map(str, sorted(feature_cols)))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def _cache_paths(cache_dir, prefix, target_col, feature_cols, cache_key=None):
    key = cache_key or f"{prefix}-{target_col}-feat{_feat_sig(feature_cols)}"
    d = os.path.join(cache_dir, "optuna")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{key}.json")
    return p, key

def _save_best_params(path, key, best_params, best_value, extra=None):
    payload = {
        "key": key,
        "best_value": _to_py(best_value),
        "best_params": _to_py(best_params),
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra: payload["extra"] = _to_py(extra)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_best_params(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params"), data.get("best_value", None)

def impute_solar_for_zero_bnos_cv_optuna(
    train: pd.DataFrame,
    zero_bnos,
    bno_col: str = "building_num",
    target_col: str = "solar",
    daylight_col: str = "daylight_prev",
    drop_cols=None,
    seed: int = SEED,
    n_splits: int = 5,
    n_trials: int = 20,        # 요구사항: 30회 (외부에서 30으로 넣어 호출)
    sample_frac: float = 0.3,  # 요구사항: 30% 샘플링
    w_xgb: float = 0.5,
    w_lgb: float = 0.5,
    # ▼ 추가 옵션
    reuse_optuna: bool = True,
    cache_dir: str = "./Energy/03",
    cache_key: str | None = None,
):
    """
    train의 zero_bnos 건물의 'solar' 잘못(0) 기록 구간을 모델로 보간.
    반환: (train_filled, oof_smape_best)
    - Optuna: 학습 데이터의 30% 샘플로 n_trials회 튜닝(TimeSeriesSplit OOF SMAPE 최소화)
    - 캐시 사용: reuse_optuna=True면 저장된 best_params를 재사용(없으면 학습 후 저장)
    - 중요: TSS에서 검증 예측을 '원본 행 순서'로 되돌려 SMAPE 순서 불일치 방지
    """
    def _smape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                             (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    METRIC_EXCLUDE_BNOS = {9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98}

    # ---- 시간 컬럼 자동 탐지 + 정렬 도우미
    def _detect_time_col(df: pd.DataFrame):
        for c in ["date", "일시", "datetime", "timestamp", "ts", "time"]:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if s.notna().any():
                    return c, s
        return None, None

    def _time_order_index(df: pd.DataFrame) -> pd.Index:
        c, s = _detect_time_col(df)
        if c is None:
            # 시간 컬럼이 없으면 원래 인덱스 순서 유지
            return df.index
        # 안정 정렬(mergesort)로 시간 오름차순
        return s.sort_values(kind="mergesort").index

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

    # 학습 데이터: zero_bnos 제외 + 주간만
    mask_train = (~train[bno_col].isin(zero_bnos)) & (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # 피처 선택
    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col}) and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    # ---- Optuna 샘플 생성
    df_sample = df_train.sample(frac=sample_frac, random_state=seed) if 0 < sample_frac < 1 else df_train
    # 시간 순서 정렬(순서 중요!)
    ord_idx = _time_order_index(df_sample)
    Xs_ord = df_sample.loc[ord_idx, feature_cols]
    ys_ord = df_sample.loc[ord_idx, target_col].astype(float)

    # ===== Optuna objective: TSS OOF SMAPE =====
    def objective(trial):
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
        lgb_params = dict(
            n_estimators=trial.suggest_int("lgb_n_estimators", 400, 2000),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.03, 0.2, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
            max_depth=trial.suggest_categorical("lgb_max_depth", [-1, 6, 8, 10]),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 120),
            subsample=trial.suggest_float("lgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 0.0, 3.0),
            max_bin=trial.suggest_int("lgb_max_bin", 63, 255),
            random_state=seed, n_jobs=-1, verbosity=-1,
        )

        tss = TimeSeriesSplit(n_splits=n_splits)
        N = len(Xs_ord)
        oof_pred_ord = np.zeros(N, dtype=float)

        for tr_idx, val_idx in tss.split(np.arange(N)):
            X_tr, X_val = Xs_ord.iloc[tr_idx], Xs_ord.iloc[val_idx]
            y_tr, y_val = ys_ord.iloc[tr_idx], ys_ord.iloc[val_idx]

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
            oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        # 원래 순서로 재배치
        oof_series = pd.Series(oof_pred_ord, index=df_sample.loc[ord_idx].index)
        oof_pred = oof_series.reindex(df_sample.index).to_numpy()

        mask_metric = ~df_sample[bno_col].isin(METRIC_EXCLUDE_BNOS)
        if mask_metric.sum() == 0:
            return _smape(df_sample[target_col].to_numpy(), oof_pred)
        return _smape(df_sample.loc[mask_metric, target_col].to_numpy(),
                      oof_series.reindex(df_sample.index)[mask_metric].to_numpy())

    # ====== 캐시 확인/로드 또는 최적화 수행 ======
    cache_path, cache_tag = _cache_paths(cache_dir, prefix="impute-solar", target_col=target_col,
                                         feature_cols=feature_cols, cache_key=cache_key)
    best_params = None
    if reuse_optuna and os.path.exists(cache_path):
        try:
            best_params, _best_value = _load_best_params(cache_path)
            if not all(k in best_params for k in ["xgb_n_estimators","xgb_eta","lgb_n_estimators","lgb_learning_rate"]):
                best_params = None
        except Exception:
            best_params = None

    if best_params is None:
        study = _optimize_with_pbar(objective, n_trials=n_trials, desc="optuna(solar-impute)", seed=seed)
        best_params = study.best_params
        _save_best_params(cache_path, cache_tag, best_params, study.best_value,
                          extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed})

    # ===== best params로 FULL 데이터 TSS OOF 재계산 =====
    # 시간 정렬
    ord_idx_full = _time_order_index(df_train)
    X_ord = df_train.loc[ord_idx_full, feature_cols]
    y_ord = df_train.loc[ord_idx_full, target_col].astype(float)

    xgb_best = {
        "n_estimators": int(best_params["xgb_n_estimators"]),
        "learning_rate": float(best_params["xgb_eta"]),
        "max_depth": int(best_params["xgb_max_depth"]),
        "subsample": float(best_params["xgb_subsample"]),
        "colsample_bytree": float(best_params["xgb_colsample_bytree"]),
        "min_child_weight": float(best_params["xgb_min_child_weight"]),
        "reg_lambda": float(best_params["xgb_reg_lambda"]),
        "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae",
    }
    lgb_best = {
        "n_estimators": int(best_params["lgb_n_estimators"]),
        "learning_rate": float(best_params["lgb_learning_rate"]),
        "num_leaves": int(best_params["lgb_num_leaves"]),
        "max_depth": int(best_params["lgb_max_depth"]),
        "min_child_samples": int(best_params["lgb_min_child_samples"]),
        "subsample": float(best_params["lgb_subsample"]),
        "colsample_bytree": float(best_params["lgb_colsample_bytree"]),
        "reg_lambda": float(best_params["lgb_reg_lambda"]),
        "max_bin": int(best_params.get("lgb_max_bin", 255)),
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
    }

    tss = TimeSeriesSplit(n_splits=n_splits)
    Nfull = len(X_ord)
    oof_pred_ord = np.zeros(Nfull, dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tss.split(np.arange(Nfull)):
        X_tr, X_val = X_ord.iloc[tr_idx], X_ord.iloc[val_idx]
        y_tr, y_val = y_ord.iloc[tr_idx], y_ord.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        xgb = XGBRegressor(**xgb_best, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        lgb = LGBMRegressor(**lgb_best)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    # 원래 순서로 재배치 후 SMAPE
    oof_series = pd.Series(oof_pred_ord, index=df_train.loc[ord_idx_full].index)
    oof_pred_full = oof_series.reindex(df_train.index).to_numpy()

    mask_metric_full = ~df_train[bno_col].isin(METRIC_EXCLUDE_BNOS).to_numpy()
    if mask_metric_full.sum() == 0:
        oof_smape_best = float(_smape(df_train[target_col].to_numpy(), oof_pred_full))
    else:
        oof_smape_best = float(_smape(df_train.loc[mask_metric_full, target_col].to_numpy(),
                                      oof_series.reindex(df_train.index)[mask_metric_full].to_numpy()))

    # ===== 최종 재학습(전구간) =====
    med_full = X_ord.median()
    X_full = X_ord.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_best["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_best["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_best, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y_ord, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_best, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_ord)

    train_filled = train.copy()
    m_day = (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 1)
    if m_day.any():
        # 최종 학습에서 사용한 중앙값으로 결측 대체
        feats = train_filled.loc[m_day, feature_cols].fillna(med_full)
        px = xgb_final.predict(feats)
        pl = lgb_final.predict(feats)
        pred = w_xgb * px + w_lgb * pl
        train_filled.loc[m_day, target_col] = np.clip(pred, 0, None)

    train_filled.loc[
        (train_filled[bno_col].isin(zero_bnos)) & (train_filled[daylight_col] == 0),
        target_col
    ] = 0.0

    return train_filled, oof_smape_best

def train_predict_test_target_cv_optuna(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,                 # 'sunshine' 또는 'solar'
    exclude_bnos=None,
    bno_col: str = "building_num",
    daylight_col: str = "daylight_prev",
    drop_cols=None,
    seed: int = 42,
    n_splits: int = 5,
    n_trials: int = 20,              # 외부에서 30으로 호출
    sample_frac: float = 0.3,
    w_xgb: float = 0.5,
    w_lgb: float = 0.5,
    restrict_daylight: bool = True,
    # ▼ 추가
    reuse_optuna: bool = True,
    cache_dir: str = "./Energy/03",
    cache_key: str | None = None,
):
    def smape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                             (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    # ---- 시간 컬럼 자동 탐지 + 정렬 도우미
    def _detect_time_col(df: pd.DataFrame):
        for c in ["date", "일시", "datetime", "timestamp", "ts", "time"]:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                if s.notna().any():
                    return c, s
        return None, None

    def _time_order_index(df: pd.DataFrame) -> pd.Index:
        c, s = _detect_time_col(df)
        if c is None:
            return df.index
        return s.sort_values(kind="mergesort").index

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

    # 학습 데이터
    mask_train = (~train[bno_col].isin(exclude_bnos))
    if restrict_daylight:
        mask_train &= (train[daylight_col] == 1)
    df_train = train.loc[mask_train].copy()

    # 피처
    feature_cols = [
        c for c in df_train.columns
        if c not in (drop_cols | {target_col}) and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not feature_cols:
        raise ValueError("학습에 사용할 숫자형 feature가 없습니다. drop_cols를 확인하세요.")

    # 샘플 + 시간정렬
    df_sample = df_train.sample(frac=sample_frac, random_state=seed) if 0 < sample_frac < 1 else df_train
    ord_idx = _time_order_index(df_sample)
    Xs_ord = df_sample.loc[ord_idx, feature_cols]
    ys_ord = df_sample.loc[ord_idx, target_col].astype(float)

    # objective (TSS로 OOF)
    def objective(trial):
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
        lgb_params = dict(
            n_estimators=trial.suggest_int("lgb_n_estimators", 400, 2000),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.03, 0.2, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
            max_depth=trial.suggest_categorical("lgb_max_depth", [-1, 6, 8, 10]),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 120),
            subsample=trial.suggest_float("lgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 0.0, 3.0),
            max_bin=trial.suggest_int("lgb_max_bin", 63, 255),
            random_state=seed, n_jobs=-1, verbosity=-1,
        )

        tss = TimeSeriesSplit(n_splits=n_splits)
        N = len(Xs_ord)
        oof_pred_ord = np.zeros(N, dtype=float)

        for tr_idx, val_idx in tss.split(np.arange(N)):
            X_tr, X_val = Xs_ord.iloc[tr_idx], Xs_ord.iloc[val_idx]
            y_tr, y_val = ys_ord.iloc[tr_idx], ys_ord.iloc[val_idx]

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
            oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        # 원래 순서로 되돌려 SMAPE
        oof_series = pd.Series(oof_pred_ord, index=df_sample.loc[ord_idx].index)
        oof_pred = oof_series.reindex(df_sample.index).to_numpy()
        return smape(df_sample[target_col].to_numpy(), oof_pred)

    # ====== 캐시 확인/로드 또는 최적화 수행 ======
    cache_path, cache_tag = _cache_paths(cache_dir, prefix=f"predict-{target_col}",
                                         target_col=target_col, feature_cols=feature_cols,
                                         cache_key=cache_key)
    best_params = None
    if reuse_optuna and os.path.exists(cache_path):
        try:
            best_params, _best_value = _load_best_params(cache_path)
            if not all(k in best_params for k in ["xgb_n_estimators","xgb_eta","lgb_n_estimators","lgb_learning_rate"]):
                best_params = None
        except Exception:
            best_params = None

    if best_params is None:
        study = _optimize_with_pbar(objective, n_trials=n_trials, desc=f"optuna({target_col})", seed=seed)
        best_params = study.best_params
        _save_best_params(cache_path, cache_tag, best_params, study.best_value,
                          extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed})

    # ===== FULL 데이터로 TSS OOF & 최종 모델 =====
    # 시간 정렬
    ord_idx_full = _time_order_index(df_train)
    X_ord = df_train.loc[ord_idx_full, feature_cols]
    y_ord = df_train.loc[ord_idx_full, target_col].astype(float)

    xgb_best = {
        "n_estimators": int(best_params["xgb_n_estimators"]),
        "learning_rate": float(best_params["xgb_eta"]),
        "max_depth": int(best_params["xgb_max_depth"]),
        "subsample": float(best_params["xgb_subsample"]),
        "colsample_bytree": float(best_params["xgb_colsample_bytree"]),
        "min_child_weight": float(best_params["xgb_min_child_weight"]),
        "reg_lambda": float(best_params["xgb_reg_lambda"]),
        "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae",
    }
    lgb_best = {
        "n_estimators": int(best_params["lgb_n_estimators"]),
        "learning_rate": float(best_params["lgb_learning_rate"]),
        "num_leaves": int(best_params["lgb_num_leaves"]),
        "max_depth": int(best_params["lgb_max_depth"]),
        "min_child_samples": int(best_params["lgb_min_child_samples"]),
        "subsample": float(best_params["lgb_subsample"]),
        "colsample_bytree": float(best_params["lgb_colsample_bytree"]),
        "reg_lambda": float(best_params["lgb_reg_lambda"]),
        "max_bin": int(best_params.get("lgb_max_bin", 255)),
        "random_state": seed, "n_jobs": -1, "verbosity": -1,
    }

    tss = TimeSeriesSplit(n_splits=n_splits)
    Nfull = len(X_ord)
    oof_pred_ord = np.zeros(Nfull, dtype=float)
    xgb_best_iters, lgb_best_iters = [], []

    for tr_idx, val_idx in tss.split(np.arange(Nfull)):
        X_tr, X_val = X_ord.iloc[tr_idx], X_ord.iloc[val_idx]
        y_tr, y_val = y_ord.iloc[tr_idx], y_ord.iloc[val_idx]

        med = X_tr.median()
        X_tr_f = X_tr.fillna(med)
        X_val_f = X_val.fillna(med)

        xgb = XGBRegressor(**xgb_best, early_stopping_rounds=50)
        xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

        lgb = LGBMRegressor(**lgb_best)
        lgb.fit(
            X_tr_f, y_tr,
            eval_set=[(X_val_f, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )

        px = xgb.predict(X_val_f)
        pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
        oof_pred_ord[val_idx] = np.clip(w_xgb * px + w_lgb * pl, 0, None)

        xgb_best_iters.append(getattr(xgb, "best_iteration", None))
        lgb_best_iters.append(getattr(lgb, "best_iteration_", None))

    # 원래 순서로 되돌려 OOF SMAPE 계산
    oof_series = pd.Series(oof_pred_ord, index=df_train.loc[ord_idx_full].index)
    oof_smape = float(smape(df_train[target_col].to_numpy(),
                            oof_series.reindex(df_train.index).to_numpy()))

    # 최종 재학습(전구간)
    med_full = X_ord.median()
    X_full = X_ord.fillna(med_full)

    def _safe_iter(avg, default):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default
        try:
            return int(max(100, round(avg)))
        except Exception:
            return default

    xgb_final_n = _safe_iter(np.nanmean([i for i in xgb_best_iters if i is not None]),
                             xgb_best["n_estimators"])
    lgb_final_n = _safe_iter(np.nanmean([i for i in lgb_best_iters if i is not None]),
                             lgb_best["n_estimators"])

    xgb_final = XGBRegressor(**{**xgb_best, "n_estimators": xgb_final_n})
    xgb_final.fit(X_full, y_ord, verbose=False)

    lgb_final = LGBMRegressor(**{**lgb_best, "n_estimators": lgb_final_n})
    lgb_final.fit(X_full, y_ord)

    # 예측
    test_pred = pd.Series(index=test.index, dtype=float)
    if restrict_daylight:
        m_day = (test[daylight_col] == 1)
        feats = test.loc[m_day, feature_cols].fillna(med_full)
        if len(feats) > 0:
            px = xgb_final.predict(feats)
            pl = lgb_final.predict(feats)
            pred = w_xgb * px + w_lgb * pl
            if target_col == 'sunshine':
                test_pred.loc[m_day] = np.clip(pred, 0, 1)
            else:
                test_pred.loc[m_day] = np.clip(pred, 0, None)
        test_pred.loc[~m_day] = 0.0
    else:
        feats = test[feature_cols].fillna(med_full)
        px = xgb_final.predict(feats)
        pl = lgb_final.predict(feats)
        pred = w_xgb * px + w_lgb * pl
        if target_col == 'sunshine':
            test_pred[:] = np.clip(pred, 0, 1)
        else:
            test_pred[:] = np.clip(pred, 0, None)

    return test_pred, oof_smape

drop_cols = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'power_consumption', 'building_num'
]

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

train_filled, smape_score = impute_solar_for_zero_bnos_cv_optuna(
    train=train,
    zero_bnos=zero_bnos,
    drop_cols=drop_cols,
    seed=50,
    n_splits=N_SPLITS,
    n_trials=N_TRIALS,
    sample_frac=SAMPLE_SIZE,
    reuse_optuna=True,
    cache_key = "train-solar-impute"
)

print("    > SMAPE :", smape_score)

# 공통 드롭(타깃은 함수 내부에서 자동 제외)
drop_cols_for_sunshine = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'solar','power_consumption', 'building_num'
]

drop_cols_for_solar = [
    '일시','building_type','groupID','all_area','cooling_area','pvc','ess','pcs',
    'date','PT','CDH','DI','PVC_per_CA','ESS_installation','PCS_installation',
    'Facility_Density','sunrise_hour','sunset_hour',
    'holidays', 'dow_hour_mean', 'dow_hour_std',
    'holiday_mean', 'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
    'building_std', 'power_consumption','building_num'
]

test['sunshine'], oof_sunshine = train_predict_test_target_cv_optuna(
    train=train_filled, test=test, target_col='sunshine',
    exclude_bnos=None,
    drop_cols=drop_cols_for_sunshine,
    seed=SEED, 
    n_splits=N_SPLITS, 
    n_trials=N_TRIALS, 
    sample_frac=SAMPLE_SIZE,
    restrict_daylight=True,
    reuse_optuna=True,
    cache_key = "test-sunshine-predict",
)

print("    > SMAPE :", oof_sunshine)

test['solar'], oof_solar = train_predict_test_target_cv_optuna(
    train=train_filled, test=test, target_col='solar',
    exclude_bnos=None,
    drop_cols=drop_cols_for_solar,
    seed=701, 
    n_splits=N_SPLITS, 
    n_trials=N_TRIALS, 
    sample_frac=SAMPLE_SIZE,
    restrict_daylight=True,
    reuse_optuna=True,
    cache_key = "test-solar-predict",
)

print("    > SMAPE :", oof_solar)

#endregion STEP 2

print("\nPreprocessing Session Completed")
print("Description")

#region DESCRIPTION/FEATURES
def _stats(x, ddof=1):
    x = np.asarray(x, dtype=float)
    return dict(
        mean=np.nanmean(x),
        var=np.nanvar(x, ddof=ddof),
        std=np.nanstd(x, ddof=ddof),
    )

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
te_sun = test['sunshine']
st_tr  = _stats(tr_sun, ddof=1)
st_te  = _stats(te_sun, ddof=1)
print("Stats (train sunshine vs test sunshine)")
print(f"- train : mean={st_tr['mean']:.6f}, var={st_tr['var']:.6f}, std={st_tr['std']:.6f}")
print(f"- test  : mean={st_te['mean']:.6f}, var={st_te['var']:.6f}, std={st_te['std']:.6f}")
print(f"> SMAPE : {oof_sunshine:.6f}")

# ---- Solar 분포 비교 (train vs test 예측)
print("\n[Test - Solar]")
tr_sol = train_filled['solar']
te_sol = test['solar']
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
train_clean, train_removed = drop_outlier_times(train_filled, intervals, singles, inclusive='both')

 
def make_building_features(
    df: pd.DataFrame,
    date_col: str = "date",
    bno_col: str = "building_num",
    sunshine_col: str = "sunshine",
    solar_col: str = "solar",
    pvc_col: str = "pvc",
    ess_col: str = "ess",
    pcs_col: str = "pcs",
    pr: float = 0.80,
) -> pd.DataFrame:
    out = df.copy()

    # 0) 날짜 정규화 + 일 단위 키 컬럼화
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["_day"] = out[date_col].dt.floor("D")
    grp_keys = [bno_col, "_day"]

    # 1) 일별 합계(건물별-날짜별) DataFrame으로 한 번에 만들기
    day_agg = (
        out.groupby(grp_keys, dropna=False)
           .agg(
               sunshine_day_hours=(sunshine_col, "sum"),
               solar_day_mj_m2=(solar_col, "sum"),
           )
           .reset_index()
    )
    day_agg["solar_day_kwh_m2"] = day_agg["solar_day_mj_m2"] / 3.6

    # 2) 일별 피처 머지 (join 대신 merge, on=컬럼명)
    out = out.merge(day_agg, on=grp_keys, how="left")

    # 품질보정
    out["sunshine_day_hours"] = out["sunshine_day_hours"].clip(lower=0, upper=24)

    # 3) 일별 PV 발전량(kWh) 추정
    out["pv_day_kwh_est"] = (
        out[pvc_col].astype(float)
        * out["solar_day_kwh_m2"].astype(float)
        * float(pr)
    ).fillna(0).clip(lower=0)

    # 4) 시간별 배분: solar 비중으로 분배 (day_sum_solar=0이면 0)
    day_sum_solar = (
        out.groupby(grp_keys, dropna=False)[solar_col]
           .transform("sum")
    )
    num = out[solar_col].to_numpy(dtype=float)
    den = day_sum_solar.to_numpy(dtype=float)
    w = np.divide(num, den, out=np.zeros(len(out), dtype=float), where=den > 0)
    out["pv_hour_kwh_est"] = (out["pv_day_kwh_est"].to_numpy(dtype=float) * w)
    out["pv_hour_kwh_est"] = out["pv_hour_kwh_est"].fillna(0).clip(lower=0)

    # 5) 파생
    out["pvc_per_day"]  = out[pvc_col].astype(float) * out["sunshine_day_hours"].astype(float)
    out["solar_by_pvc"] = out[solar_col].astype(float) * out[pvc_col].astype(float)

    # 6) ESS/PCS 파생
    if pcs_col in out.columns:
        out["pv_to_ess_kwh_cap"] = np.minimum(
            out["pv_hour_kwh_est"], out[pcs_col].astype(float).clip(lower=0)
        )
    if ess_col in out.columns:
        out["ess_hours_at_pvc"] = np.divide(
            out[ess_col].astype(float),
            np.maximum(out[pvc_col].astype(float), 1e-6)
        )

    # 임시 키 제거(원하면 유지)
    out = out.drop(columns=["_day"])
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

#endregion DESCRIPTION/FEATURES

print("[STEP 3] MODEL")
print(f"    Train for MODEL | {train_clean.shape}")
print(f"    Test for MODEL  | {test.shape}\n")

#region STEP 3

#region MODEL FUNCTIONS
# -------------------- helpers --------------------
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

def _split_best_params(best_params: dict):
    if not isinstance(best_params, dict):
        return {}, {}
    xgb = {k[4:]: v for k, v in best_params.items() if k.startswith("xgb_")}
    lgb = {k[4:]: v for k, v in best_params.items() if k.startswith("lgb_")}
    # 키 정규화
    if "eta" in xgb: xgb["learning_rate"] = xgb.pop("eta")
    if "lr"  in xgb: xgb["learning_rate"] = xgb.pop("lr")
    if "lr"  in lgb: lgb["learning_rate"] = lgb.pop("lr")
    return xgb, lgb

def _train_one_model_cv_optuna(   # ← 이름 유지(호환)
    train_df, target_col, feature_cols, seed=42,
    n_splits=5, n_trials=30, sample_frac=0.3,
    w_xgb=0.5, w_lgb=0.5,
    # ▼ 캐시/재사용 옵션 추가
    reuse_optuna: bool = True,
    cache_dir: str = "./Energy/03",
    cache_key: str | None = None,
    cache_prefix: str = "tss",       # 파일명 prefix
):
    """
    TSS 교차검증 + Optuna로 XGB/LGB 하이퍼파라미터 튜닝 (SMAPE 최소화) + 결과 캐시.
    - log1p 학습 / expm1 복원
    - best_params, best_value(SMAPE), xgb_final_n/lgb_final_n 저장/재사용
    """

    # ---------- 준비 ----------
    feat_ok = [c for c in feature_cols
               if (c in train_df.columns) and pd.api.types.is_numeric_dtype(train_df[c])]
    if not feat_ok:
        raise ValueError("유효한 feature가 없습니다.")
    log_cols = auto_select_log_cols(train_df, feat_ok)

    y_all_raw = (
        pd.to_numeric(train_df[target_col], errors='coerce')
        .fillna(0).astype(float)
        .clip(lower=0)
    )
    X_all = _make_X(train_df, feat_ok, log_cols)

    # ---------- CV 1회 실행 ----------
    def _run_cv_once(xgb_params, lgb_params, sample_frac_for_metric=1.0, collect_best_iters=False):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_xgb = np.full(len(X_all), np.nan, dtype=float)
        oof_lgb = np.full(len(X_all), np.nan, dtype=float)
        was_val = np.zeros(len(X_all), dtype=bool)
        iters_x, iters_l = [], []

        for tr_idx, val_idx in tscv.split(X_all):
            X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
            med = X_tr.median()
            X_tr_f = X_tr.fillna(med)
            X_val_f = X_val.fillna(med)

            y_tr = np.log1p(y_all_raw.iloc[tr_idx].to_numpy())
            y_val = np.log1p(y_all_raw.iloc[val_idx].to_numpy())

            xgb = XGBRegressor(
                **xgb_params,
                random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="mae",
                early_stopping_rounds=50
            )
            xgb.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

            lgb = LGBMRegressor(
                **lgb_params,
                random_state=seed, n_jobs=-1
            )
            lgb.fit(
                X_tr_f, y_tr,
                eval_set=[(X_val_f, y_val)],
                eval_metric="l1",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )

            px = xgb.predict(X_val_f)
            pl = lgb.predict(X_val_f, num_iteration=getattr(lgb, "best_iteration_", None))
            oof_xgb[val_idx] = np.expm1(px).clip(min=0)
            oof_lgb[val_idx] = np.expm1(pl).clip(min=0)
            was_val[val_idx] = True

            if collect_best_iters:
                iters_x.append(getattr(xgb, "best_iteration", None))
                iters_l.append(getattr(lgb, "best_iteration_", None))

        mask = was_val
        pred_mix = w_xgb * oof_xgb[mask] + w_lgb * oof_lgb[mask]

        idx = np.where(mask)[0]
        if sample_frac_for_metric < 1.0 and len(idx) > 200:
            k = max(200, int(len(idx) * sample_frac_for_metric))
            rng = np.random.default_rng(seed)
            sub = np.sort(rng.choice(idx, size=k, replace=False))
            sm = float(smape(y_all_raw.to_numpy()[sub],
                             (w_xgb * oof_xgb + w_lgb * oof_lgb)[sub]))
        else:
            sm = float(smape(y_all_raw.to_numpy()[mask], pred_mix))

        return sm, (w_xgb * oof_xgb + w_lgb * oof_lgb), was_val, iters_x, iters_l

    # ---------- Optuna + 캐시 ----------
    # 캐시 경로 생성: prefix/target/featSig + (사용자가 준 cache_key가 있으면 그것을 우선 사용)
    cache_path, cache_tag = _cache_paths(
        cache_dir=cache_dir,
        prefix=cache_prefix,
        target_col=target_col,
        feature_cols=feat_ok,
        cache_key=cache_key
    )

    best_params_combined = None
    best_value = None
    xgb_final_n_cached = None
    lgb_final_n_cached = None

    # 캐시 로드
    if reuse_optuna and os.path.exists(cache_path):
        try:
            best_params_combined, best_value = _load_best_params(cache_path)
            # extra의 final_n도 있으면 회수
            with open(cache_path, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            _extra = _payload.get("extra", {})
            xgb_final_n_cached = _extra.get("xgb_final_n", None)
            lgb_final_n_cached = _extra.get("lgb_final_n", None)
        except Exception:
            best_params_combined = None

    # 없으면 튜닝
    if best_params_combined is None:
        from optuna.samplers import TPESampler
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        pbar = tqdm(total=n_trials, desc="[optuna(hparams)]", ncols=80,
                    bar_format="{l_bar}{bar}| {postfix}", leave=False)

        def objective(trial):
            xgb_params = dict(
                lr=trial.suggest_float("xgb_lr", 0.02, 0.2, log=True),
                max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
                min_child_weight=trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
                subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
                n_estimators=trial.suggest_int("xgb_n_estimators", 800, 4000),
            )
            lgb_params = dict(
                lr=trial.suggest_float("lgb_lr", 0.02, 0.2, log=True),
                num_leaves=trial.suggest_int("lgb_num_leaves", 31, 255),
                min_child_samples=trial.suggest_int("lgb_min_child_samples", 20, 120),
                subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("lgb_reg_alpha", 0.0, 5.0),
                max_bin=trial.suggest_int("lgb_max_bin", 127, 511),
                n_estimators=trial.suggest_int("lgb_n_estimators", 800, 4000),
                verbosity=-1,
            )
            # 이름 접두부를 붙여서 하나의 dict로 기록되게 함
            combined = {f"xgb_{k}": v for k, v in xgb_params.items()}
            combined.update({f"lgb_{k}": v for k, v in lgb_params.items()})

            sm, *_ = _run_cv_once(
                # 내부 실행을 위해서는 원래 키로 복원
                xgb_params={**xgb_params, "learning_rate": xgb_params.pop("lr")},
                lgb_params={**lgb_params, "learning_rate": lgb_params.pop("lr")},
                sample_frac_for_metric=max(0.1, min(1.0, float(sample_frac))),
                collect_best_iters=False
            )
            # 진행바 best만
            try:
                pbar.set_postfix_str(f"best_smape: {min(study.best_value, sm):.6f}")
            except Exception:
                pbar.set_postfix_str("best_smape: n/a")
            # trial user_attrs에 기록(선택)
            trial.set_user_attr("combined_params", combined)
            return sm

        def _cb(study, trial):
            pbar.update(1)

        try:
            study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
        finally:
            try:
                pbar.set_postfix_str(f"best_smape: {study.best_value:.6f}")
            except Exception:
                pass
            pbar.close()

        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise RuntimeError("Optuna: 완료된 trial이 없습니다.")

        # study.best_params는 접두부 없는 키. 우리는 접두부 달린 dict를 저장
        # 가장 좋은 trial에서 user_attrs로 꺼낸다(없으면 접두부 달아 변환)
        best_trial = study.best_trial
        best_params_combined = best_trial.user_attrs.get("combined_params", None)
        if best_params_combined is None:
            # fallback: study.best_params를 접두부로 감싸 저장
            bp = study.best_params
            xgb_bp = {f"xgb_{k}": v for k, v in bp.items() if k.startswith("xgb_") or k in {
                "xgb_lr","xgb_max_depth","xgb_min_child_weight","xgb_subsample","xgb_colsample_bytree",
                "xgb_reg_lambda","xgb_reg_alpha","xgb_n_estimators"
            }}
            lgb_bp = {f"lgb_{k}": v for k, v in bp.items() if k.startswith("lgb_") or k in {
                "lgb_lr","lgb_num_leaves","lgb_min_child_samples","lgb_subsample","lgb_colsample_bytree",
                "lgb_reg_lambda","lgb_reg_alpha","lgb_max_bin","lgb_n_estimators"
            }}
            best_params_combined = {**xgb_bp, **lgb_bp}

        best_value = float(study.best_value)
        _save_best_params(
            cache_path, cache_tag, best_params_combined, best_value,
            extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed}
        )

    # ---------- 베스트 파라미터로 OOF 재계산 ----------
    best_xgb_raw, best_lgb_raw = _split_best_params(best_params_combined)
    # 실행용 정규화
    best_xgb = {
        **best_xgb_raw,
        "learning_rate": best_xgb_raw.get("learning_rate", best_xgb_raw.get("lr", 0.1)),
    }
    best_lgb = {
        **best_lgb_raw,
        "learning_rate": best_lgb_raw.get("learning_rate", best_lgb_raw.get("lr", 0.1)),
        "verbosity": -1,
    }

    oof_smape, oof_pred, was_val, itx, itl = _run_cv_once(
        best_xgb, best_lgb, sample_frac_for_metric=1.0, collect_best_iters=True
    )
    oof_series = pd.Series(oof_pred, index=X_all.index, name=f"OOF_{target_col}")

    # ---------- 최종 재학습 ----------
    med_full = X_all.median()
    X_full = X_all.fillna(med_full)
    y_full_log = np.log1p(y_all_raw.to_numpy())

    def _safe_iter(avg, default_min=100):
        if avg is None or (isinstance(avg, float) and np.isnan(avg)):
            return default_min
        try:
            return int(max(default_min, round(avg)))
        except Exception:
            return default_min

    # 캐시에 final_n 있으면 재사용, 없으면 fold 평균
    xgb_final_n = _safe_iter(np.nanmean([i for i in itx if i is not None]))
    lgb_final_n = _safe_iter(np.nanmean([i for i in itl if i is not None]))
    if isinstance(xgb_final_n_cached, (int, float)) and xgb_final_n_cached > 0:
        xgb_final_n = int(xgb_final_n_cached)
    if isinstance(lgb_final_n_cached, (int, float)) and lgb_final_n_cached > 0:
        lgb_final_n = int(lgb_final_n_cached)

    xgb_final = XGBRegressor(
        **{**best_xgb, "n_estimators": int(xgb_final_n),
           "random_state": seed, "n_jobs": -1, "tree_method": "hist", "eval_metric": "mae"}
    )
    xgb_final.fit(X_full, y_full_log, verbose=False)

    lgb_final = LGBMRegressor(
        **{**best_lgb, "n_estimators": int(lgb_final_n),
           "random_state": seed, "n_jobs": -1}
    )
    lgb_final.fit(X_full, y_full_log)

    # 최종 n 저장(덮어쓰기)
    try:
        _save_best_params(
            cache_path, cache_tag, best_params_combined, best_value,
            extra={"n_splits": n_splits, "sample_frac": sample_frac, "seed": seed,
                   "xgb_final_n": int(xgb_final_n), "lgb_final_n": int(lgb_final_n)}
        )
    except Exception:
        pass

    return (
        xgb_final, lgb_final, med_full, feat_ok, log_cols,
        float(oof_smape), oof_series
    )

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
    'building_type','groupID','holidays', 'is_imputed',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year','minute',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'SIN_Time', 'COS_Time', 'SIN_minute', 'COS_minute',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    # 'solar','solar_day_kwh_m2','sunshine_day_hours',
    'all_area','cooling_area',
    # 'ess','pcs', 'pvc',
    'pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
    'solar_by_pvc',
    'PVC_per_CA','Facility_Density',
]

ess_features = [
    'building_type','groupID','holidays', 'is_imputed',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year','minute',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'SIN_Time', 'COS_Time', 'SIN_minute', 'COS_minute',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    # 'solar','solar_day_kwh_m2','sunshine_day_hours',
    'all_area','cooling_area',
    # 'ess','pcs', 'pvc',
    'pv_day_kwh_est','pv_hour_kwh_est','pvc_per_day',
    'solar_by_pvc','pv_to_ess_kwh_cap','ess_hours_at_pvc',
    'PVC_per_CA','Facility_Density',
]

no_pvc_features = [
    'building_type','groupID','holidays','is_imputed',
    'temperature','precipitation','windspeed','humidity',
    'hour','dow','day','day_of_year','minute',
    'SIN_hour','COS_hour','SIN_day','COS_day','SIN_month','COS_month',
    'SIN_dow','COS_dow','SIN_day_of_year','COS_day_of_year','SIN_summer','COS_summer',
    'SIN_Time', 'COS_Time', 'SIN_minute', 'COS_minute',
    'solar_elevation','PT','CDH','DI',
    'dow_hour_mean','dow_hour_std','holiday_mean','holiday_std',
    'hour_mean','hour_std','building_mean','building_std',
    'solar',
    'all_area','cooling_area',
    # 'solar_day_kwh_m2','sunshine_day_hours',
]

# -------------------- 1) 세그먼트 모델 (PVC/ESS/No-PVC) --------------------
def model_segmented(train_clean, test, target='power_consumption', seed=SEED, n_splits=5, n_trials=20, sample_size=0.3):
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
    
    pvc_for_seg = pvc_features.copy()
    pvc_for_seg.append('building_num')
    
    ess_for_seg = ess_features.copy()
    ess_for_seg.append('building_num')
    
    no_pvc_for_seg = no_pvc_features.copy()
    no_pvc_for_seg.append('building_num')

    # ESS
    tr_ess = train_clean.loc[seg_ess_tr]
    if len(tr_ess) > 50 and seg_ess_te.any():
        print(f"    [PCS Building]")
        xgb,lgb,med,feats,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_ess, target, _valid_features(train_clean, test, ess_for_seg),
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key="seg-ESS", cache_prefix="seg"
        )
        preds.loc[seg_ess_te] = _predict_with_models(test.loc[seg_ess_te], (xgb,lgb,med,feats,logs))
        oofs['ESS'] = oof
        oof_series_full.loc[tr_ess.index] = oof_series
        print(f"    > SMAPE : {oof}")

    # PVC only
    tr_pvc = train_clean.loc[seg_pvc_tr]
    if len(tr_pvc) > 50 and seg_pvc_te.any():
        print(f"    [PV Building]")
        xgb,lgb,med,feats,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_pvc, target, _valid_features(train_clean, test, pvc_for_seg),
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key="seg-PVC", cache_prefix="seg"
        )
        preds.loc[seg_pvc_te] = _predict_with_models(test.loc[seg_pvc_te], (xgb,lgb,med,feats,logs))
        oofs['PVC'] = oof
        oof_series_full.loc[tr_pvc.index] = oof_series 
        print(f"    > SMAPE : {oof}")

    # No-PVC
    tr_none = train_clean.loc[seg_none_tr]
    if len(tr_none) > 50 and seg_none_te.any():
        print(f"    [No-PV Building]")
        xgb,lgb,med,feats,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_none, target, _valid_features(train_clean, test, no_pvc_for_seg),
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key="seg-NoPVC", cache_prefix="seg"
        )
        preds.loc[seg_none_te] = _predict_with_models(test.loc[seg_none_te], (xgb,lgb,med,feats,logs))
        oofs['NoPVC'] = oof
        oof_series_full.loc[tr_none.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return preds.values, oofs, oof_series_full 

def model_by_type(train_clean, test, target='power_consumption', type_col='building_type', seed=SEED, n_splits=5, n_trials=20, sample_size=0.3):
    print("  [model_by_type] activate")
    pred = pd.Series(0.0, index=test.index)
    ess_for_type = ess_features.copy()
    ess_for_type.append('building_num')
    type_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_clean.index)
    for t in sorted(test[type_col].dropna().unique()):
        print(f"    [Building Type] {t}")
        tr_t = train_clean[train_clean[type_col] == t]
        te_t = test[test[type_col] == t]
        if len(tr_t) < 50:
            continue
        feats = _valid_features(train_clean, test, ess_for_type)  # 포괄적 피처
        xgb,lgb,med,fs,logs,oof,oof_series = _train_one_model_cv_optuna(
            tr_t, target, feats,
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key=f"type-{t}", cache_prefix="type"
        )
        pred.loc[te_t.index] = _predict_with_models(te_t, (xgb,lgb,med,fs,logs))
        type_scores[str(t)] = oof
        oof_series_full.loc[tr_t.index] = oof_series
        print(f"    > SMAPE : {oof}")

    return pred.values, type_scores, oof_series_full

def model_by_building_num(train_clean, test, target='power_consumption', bno_col='building_num', pvc_col='pvc', pcs_col='pcs', seed=SEED, n_splits=5, n_trials=20, sample_size=0.7):
    print("  [model_by_building_num] activate (per-building, ESS/PVC/NoPVC aware)")

    pred = pd.Series(0.0, index=test.index, dtype=float)
    b_scores = {}
    oof_series_full = pd.Series(np.nan, index=train_clean.index, dtype=float)

    # 세그먼트별 피처셋 준비 (ess_features 미정의 시 pvc_features로 폴백)
    try:
        _ess_features = ess_features
    except NameError:
        _ess_features = None  # 없으면 아래에서 pvc_features로 대체

    # 테스트에 등장하는 건물만 대상으로 학습/예측
    for b in sorted(test[bno_col].dropna().unique()):
        tr_b = train_clean[train_clean[bno_col] == b]
        te_b = test[test[bno_col] == b]
        if len(tr_b) < 30:
            print(f"    [Building {b}] skip (train rows < 30)")
            continue

        # --- 건물 단위 ESS/PVC/NoPVC 판정 ---
        has_pvc_train = (pvc_col in tr_b.columns) and (tr_b[pvc_col].fillna(0) > 0).any()
        has_pvc_test  = (pvc_col in te_b.columns) and (te_b[pvc_col].fillna(0) > 0).any()
        has_pvc = bool(has_pvc_train or has_pvc_test)

        has_pcs_train = (pcs_col in tr_b.columns) and (tr_b[pcs_col].fillna(0) > 0).any()
        has_pcs_test  = (pcs_col in te_b.columns) and (te_b[pcs_col].fillna(0) > 0).any()
        has_pcs = bool(has_pcs_train or has_pcs_test)

        if has_pvc and has_pcs:
            seg = 'ESS'
            feat_list = _ess_features if _ess_features is not None else pvc_features
        elif has_pvc:
            seg = 'PVC'
            feat_list = pvc_features
        else:
            seg = 'NoPVC'
            feat_list = no_pvc_features

        # --- 피처 유효성 필터링 ---
        feats = _valid_features(train_clean, test, feat_list)
        print(f"    [Building {b}] segment={seg} | features={len(feats)}")

        if len(feats) == 0:
            print(f"      > skip: no valid features for building {b}")
            continue

        # --- 학습(TSS 기반) & 예측
        xgb, lgb, med, fs, logs, oof, oof_series = _train_one_model_cv_optuna(
            tr_b, target, feats,
            seed=seed, n_splits=n_splits, n_trials=n_trials, sample_frac=sample_size,
            reuse_optuna=True, cache_dir="./Energy/03", cache_key=f"bno-{b}-{seg}", cache_prefix="bno"
        )
        pred.loc[te_b.index] = _predict_with_models(te_b, (xgb, lgb, med, fs, logs))
        oof_series_full.loc[tr_b.index] = oof_series
        b_scores[str(b)] = oof
        print(f"      > SMAPE : {oof:.6f}")

    return pred.values, b_scores, oof_series_full

#endregion MODEL FUNCTIONS

# ==================== RUN: 3모델 예측 후 앙상블 ====================
target = 'power_consumption'

print("[M1] Segmented (ESS/PVC/NoPVC)")
pred1, oof1, oof1_s = model_segmented(
    train_clean, test, target=target, seed=SEED,
    n_splits=N_SPLITS_MODEL, n_trials=N_TRIALS_MODEL, sample_size=SAMPLE_SIZE_MODEL
)
avg1 = sum(oof1.values()) / len(oof1)
print(f"[Segmented Model] SMAPE : {avg1}\n")

print("[M2] By Type")
pred2, oof2, oof2_s = model_by_type(
    train_clean, test, target=target, type_col='building_type', seed=SEED,
    n_splits=N_SPLITS_MODEL, n_trials=N_TRIALS_MODEL, sample_size=SAMPLE_SIZE_MODEL
)
avg2 = sum(oof2.values()) / len(oof2)
print(f"[By Type Model] SMAPE : {avg2}\n")

print("[M4] By BuildingNum (ESS/PVC/NoPVC AWARE)")
pred4, oof4, oof4_s = model_by_building_num(
    train_clean, test, target=target, bno_col='building_num', seed=SEED,
    n_splits=N_SPLITS_MODEL, n_trials=N_TRIALS_MODEL, sample_size=SAMPLE_SIZE_BNUM
)
avg4 = sum(oof4.values()) / len(oof4)
print(f"[By BuildingNum Model] SMAPE : {avg4}\n")

#endregion STEP 3

print("[STEP 4] SAVE")

#region STEP 4
# ===== 앙상블 가중 (M1/M2/M4만 사용) =====
W1, W2, W4 = 1/3, 1/3, 1/3
pred_final = W1*pred1 + W2*pred2 + W4*pred4

# ===== 최종 앙상블 OOF SMAPE =====
oof_df = pd.DataFrame({
    'm1': oof1_s,   # 세그먼트 OOF
    'm2': oof2_s,   # 유형별 OOF
    'm4': oof4_s,   # 번호별 OOF
})

w = np.array([W1, W2, W4], dtype=float)
W = pd.DataFrame(np.broadcast_to(w, oof_df.shape), index=oof_df.index, columns=oof_df.columns)

# NaN 예측엔 가중치 0
W = W.where(oof_df.notna(), 0.0)
wsum = W.sum(axis=1)
mask = (wsum > 0) & oof_df.notna().any(axis=1)

# 행별 가중 재정규화 후 앙상블 OOF
W_norm = W.div(wsum, axis=0).where(mask, 0.0)
oof_ens = (oof_df.fillna(0.0) * W_norm).sum(axis=1)

final_oof_smape = smape(
    train_clean.loc[mask, 'power_consumption'].to_numpy(),
    oof_ens.loc[mask].to_numpy()
)
print(f"[OOF] Final Ensemble SMAPE: {final_oof_smape}\n")

submission['answer'] = pred_final

today = datetime.datetime.now().strftime('%Y%m%d')
score_str = ("nan" if np.isnan(final_oof_smape) else f"{final_oof_smape:.4f}").replace('.', '_')
filename = f"{SEED}_SMAPE_{score_str}_{version}.csv"
submission.to_csv(save_path + filename, index=False)

with open(save_path + f"(LOG)model_{version}.txt", "a") as f:
    f.write(f"{version}\n")
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"[Preprocessing Variable]\n")
    f.write(f"N_SPLITS {N_SPLITS} | N_TRIALS {N_TRIALS}\n")
    f.write(f"SAMPLE_SIZE {SAMPLE_SIZE} | AUGMENTATION {60/AUGMENTATION} times\n")
    f.write(f"[Model Variable]\n")
    f.write(f"N_SPLITS_MODEL {N_SPLITS_MODEL} | N_TRIALS_MODEL {N_TRIALS_MODEL}\n")
    f.write(f"SAMPLE_SIZE_MODEL {SAMPLE_SIZE_MODEL} | SAMPLE_SIZE_BNUM {SAMPLE_SIZE_BNUM}\n")
    f.write(f"[Segmented] SMAPE {avg1:.6f}\n")
    f.write(f"[Type] SMAPE {avg2:.6f}\n")
    f.write(f"[By BuildingNum] SMAPE {avg4:.6f}\n")
    f.write(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}\n")
    f.write("="*40 + "\n")
        
with open(log_path + f"(LOG)model_{version}.txt", "a") as f:
    f.write(f"{version}\n")
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"[Preprocessing Variable]\n")
    f.write(f"N_SPLITS {N_SPLITS} | N_TRIALS {N_TRIALS}\n")
    f.write(f"SAMPLE_SIZE {SAMPLE_SIZE} | AUGMENTATION {60/AUGMENTATION} times\n")
    f.write(f"[Model Variable]\n")
    f.write(f"N_SPLITS_MODEL {N_SPLITS_MODEL} | N_TRIALS_MODEL {N_TRIALS_MODEL}\n")
    f.write(f"SAMPLE_SIZE_MODEL {SAMPLE_SIZE_MODEL} | SAMPLE_SIZE_BNUM {SAMPLE_SIZE_BNUM}\n")
    f.write(f"[Segmented] SMAPE {avg1:.6f}\n")
    f.write(f"[Type] SMAPE {avg2:.6f}\n")
    f.write(f"[By BuildingNum] SMAPE {avg4:.6f}\n")
    f.write(f"[Final Ensemble] SMAPE {final_oof_smape:.6f}\n")
    f.write("="*40 + "\n")
    
#endregion STEP 4

print(f"[{version}] COMPLETED")
