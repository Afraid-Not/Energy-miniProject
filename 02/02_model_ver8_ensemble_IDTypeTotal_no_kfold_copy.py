print(f"[preprocessing] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import warnings
import datetime
import math
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


data_path = './Energy/'
csv_path = './Energy/02/'
log_path = './Energy/02/log/'
trainer = './Energy/02/trainer/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

seed_file = "./Energy/02/log/(model_ver8)SEED_COUNTS.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 1 #seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)

building_csv = pd.read_csv(data_path + 'building_info.csv')
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

########################## train & test 기본 전처리 ############################
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')

test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

#region(모델용 전처리 부분)

def peak_holidays(df, is_train=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['일시'])
    df['dow']  = df['date'].dt.weekday  # 0=월 ... 6=일

    # 기본값
    df['holidays'] = 0
    df['peak'] = 0

    # --- 건물별 '정기 휴무 요일' 지정 ---
    df.loc[(df['건물번호']==2) & (df['dow']==5), 'holidays'] = 1       # 토
    df.loc[(df['건물번호']==3) & (df['dow'].isin([5,6])), 'holidays'] = 1  # 토/일
    df.loc[(df['건물번호']==4) & (df['dow']==0), 'holidays'] = 1       # 월
    df.loc[(df['건물번호']==5) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==6) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==7) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['건물번호']==8) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==10) & (df['dow']==0), 'holidays'] = 1
    df.loc[(df['건물번호']==12) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==13) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==14) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==15) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==16) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==17) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==18) & (df['dow']==6), 'holidays'] = 1
    df.loc[(df['건물번호']==20) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==21) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==22) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==23) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==24) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==37) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==38) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==39) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==42) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==43) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==45) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==46) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==47) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==48) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==49) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==50) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==51) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==52) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==53) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==55) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==56) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==60) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==61) & (df['dow'].isin([1,2])), 'holidays'] = 1
    df.loc[(df['건물번호']==62) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==64) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==66) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==67) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==68) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==69) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==72) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==75) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==80) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==81) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==83) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==86) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==87) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==90) & (df['dow'].isin([5,6])), 'holidays'] = 1
    df.loc[(df['건물번호']==94) & (df['dow'].isin([5,6])), 'holidays'] = 1


    df.loc[(df['건물번호']==2) & (df['dow']==3), 'peak'] = 1           # 목
    df.loc[(df['건물번호']==3) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==4) & (df['dow']==4), 'peak'] = 1           # 금
    df.loc[(df['건물번호']==6) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==7) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==8) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==9) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==10) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==11) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==12) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==13) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==14) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==15) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==16) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==17) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==18) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==19) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==20) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==21) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==22) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==23) & (df['dow']==1), 'peak'] = 1
    df.loc[(df['건물번호']==24) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==25) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==26) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==27) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==28) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==29) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==30) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==31) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==32) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==33) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==34) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==35) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==36) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==37) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==38) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==39) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==40) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==41) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==42) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==43) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==44) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==45) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==46) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==47) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==48) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==49) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==50) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==51) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==52) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==53) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==54) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==55) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==56) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==57) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==58) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==59) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==60) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==61) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==62) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==63) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==64) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==65) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==66) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==67) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==68) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==69) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==70) & (df['dow']==1), 'peak'] = 1
    df.loc[(df['건물번호']==71) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==72) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==73) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==74) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==75) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==76) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==77) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==78) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==79) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==80) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==81) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==82) & (df['dow']==5), 'peak'] = 1
    df.loc[(df['건물번호']==83) & (df['dow']==1), 'peak'] = 1
    df.loc[(df['건물번호']==84) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==85) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==86) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==87) & (df['dow']==0), 'peak'] = 1
    df.loc[(df['건물번호']==88) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==89) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==90) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==91) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==92) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==93) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==94) & (df['dow']==3), 'peak'] = 1
    df.loc[(df['건물번호']==95) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==96) & (df['dow']==6), 'peak'] = 1
    df.loc[(df['건물번호']==97) & (df['dow']==2), 'peak'] = 1
    df.loc[(df['건물번호']==98) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==99) & (df['dow']==4), 'peak'] = 1
    df.loc[(df['건물번호']==100) & (df['dow']==6), 'peak'] = 1


    if is_train :
        nat = df['date'].dt.strftime('%m-%d').isin(['06-06','08-15'])
        df.loc[nat, 'holidays'] = 1

        df.loc[(df['건물번호']==19) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 1
        df.loc[(df['건물번호']==27) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['건물번호']==29) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-24','07-10','07-28','08-10'])), 'holidays'] = 1
        df.loc[(df['건물번호']==32) & (df['date'].dt.strftime('%m-%d').isin(['06-10','06-24','07-08','07-22','08-12'])), 'holidays'] = 1
        df.loc[(df['건물번호']==38) & (df['date'].dt.strftime('%m-%d').isin(['06-07'])), 'holidays'] = 1
        df.loc[(df['건물번호']==40) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['건물번호']==45) & (df['date'].dt.strftime('%m-%d').isin(['06-10','07-08','08-19'])), 'holidays'] = 1
        df.loc[(df['건물번호']==54) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 1
        df.loc[(df['건물번호']==56) & (df['date'].dt.strftime('%m-%d').isin(['06-07','08-16'])), 'holidays'] = 1
        df.loc[(df['건물번호']==59) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['건물번호']==63) & (df['date'].dt.strftime('%m-%d').isin(['06-09','06-23','07-14','07-28','08-11'])), 'holidays'] = 1
        df.loc[(df['건물번호']==74) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01'])), 'holidays'] = 1
        df.loc[(df['건물번호']==79) & (df['date'].dt.strftime('%m-%d').isin(['06-17','07-01','08-19'])), 'holidays'] = 1
        df.loc[(df['건물번호']==94) & (df['date'].dt.strftime('%m-%d').isin(['06-07','08-16'])), 'holidays'] = 1
        df.loc[(df['건물번호']==95) & (df['date'].dt.strftime('%m-%d').isin(['07-08','08-05'])), 'holidays'] = 1
    
    else :
        df.loc[(df['건물번호']==27) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['건물번호']==29) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 1
        df.loc[(df['건물번호']==32) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 1
        df.loc[(df['건물번호']==40) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['건물번호']==59) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['건물번호']==63) & (df['date'].dt.strftime('%m-%d').isin(['08-25'])), 'holidays'] = 1
        df.loc[(df['건물번호']==74) & (df['date'].dt.strftime('%m-%d').isin(['08-26'])), 'holidays'] = 1        
    
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

def drop_outlier_times(df, intervals=None, singles=None, inclusive='both'):
    """
    intervals: {건물번호: [(start,end), ...]}   # 구간 제거
    singles  : {건물번호: [ts, ts, ...]}       # 단일 시각 제거
    시간 포맷은 YYYYMMDDHH (예: 2024070411)
    """
    df = df.copy()

    # 1) 일시 -> datetime
    # 숫자/문자 섞여도 안전하게 변환
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
                mask |= (df['건물번호'] == bno) & df['dt'].between(sdt, edt, inclusive=inclusive)

    # 3) 단일 시각 제거
    if singles:
        for bno, tlist in singles.items():
            for t in tlist:
                tdt = pd.to_datetime(str(t), format='%Y%m%d%H')
                mask |= (df['건물번호'] == bno) & (df['dt'] == tdt)

    removed = df.loc[mask].sort_values(['건물번호','dt'])
    kept    = df.loc[~mask].reset_index(drop=True)
    removed = removed.drop(['dt'], axis=1)
    kept = kept.drop(['dt'], axis=1)
    return kept, removed

def SMAPE(true, pred):
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) * 200

def shift_power_from_6h_and_trim(
    df: pd.DataFrame,
    bno: int = 87,
    start_str: str = "20240629 01",
    hours: int = 6,
    id_col: str = "건물번호",
    time_col: str = "date",
    time_fallback_col: str = "일시",
    target_col: str = "전력소비량(kWh)"
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

def _apply_peak_holidays(df, *, is_train=True):
    tmp = df.copy()

    # 호환 컬럼 확보
    if '건물번호' not in tmp.columns and 'building' in tmp.columns:
        tmp['건물번호'] = tmp['building']
    if '일시' not in tmp.columns:
        for c in ('date_time', 'date'):
            if c in tmp.columns:
                tmp['일시'] = tmp[c]; break
        else:
            raise KeyError("`일시`/`date_time`/`date` 중 하나가 필요합니다.")

    ph = peak_holidays(tmp, is_train=is_train)  # 사용자 제공 함수

    out = df.copy()
    out['holiday'] = ph['holidays'].astype('int8').to_numpy()
    out['peak']    = ph['peak'].astype('int8').to_numpy()

    # dow/hour/date 보정
    dt = pd.to_datetime(ph['date'], errors='coerce')
    if 'dow'  not in out.columns:  out['dow']  = dt.dt.weekday
    if 'hour' not in out.columns:  out['hour'] = dt.dt.hour
    if 'date' not in out.columns:  out['date'] = dt.dt.date

    if 'building' not in out.columns and '건물번호' in df.columns:
        out['building'] = df['건물번호'].to_numpy()

    return out

def mean_std_fast_v2(train, test, mode, use_peak_holidays=True):
    """peak_holidays 기반 휴일/피크 적용 + 원래 집계 로직 (강제 휴일 룰 제거)"""
    train = train.copy()
    test  = test.copy()

    # 기본 컬럼 보정
    def _base_fix(df):
        df = df.copy()
        if 'building' not in df.columns and '건물번호' in df.columns:
            df['building'] = df['건물번호']
        # 날짜 → dow/hour/date
        if 'dow' not in df.columns or 'hour' not in df.columns or 'date' not in df.columns:
            dtcol = None
            for c in ('date_time', '일시', 'date'):
                if c in df.columns:
                    dtcol = c; break
            if dtcol is None:
                raise KeyError("`dow/hour/date` 생성을 위해 `date_time`/`일시`/`date` 중 하나가 필요합니다.")
            dt = pd.to_datetime(df[dtcol], errors='coerce')
            if 'dow'  not in df.columns:  df['dow']  = dt.dt.weekday
            if 'hour' not in df.columns:  df['hour'] = dt.dt.hour
            if 'date' not in df.columns:  df['date'] = dt.dt.date
        if 'holiday' not in df.columns:
            df['holiday'] = 0
        return df

    train = _base_fix(train)
    test  = _base_fix(test)

    # peak_holidays 적용 (집계 전에)
    if use_peak_holidays:
        train = _apply_peak_holidays(train, is_train=True)
        test  = _apply_peak_holidays(test,  is_train=False)

    # 요일 가중 보정
    ratio = np.array([0.985, 0.98, 0.98, 0.995, 0.995, 0.99, 0.99], dtype=float)
    if mode == 'all':
        ratio = ratio - 0.005
    if mode in ('byb', 'all'):
        idx = train['dow'].to_numpy(dtype=int)
        train['전력소비량(kWh)'] = train['전력소비량(kWh)'].to_numpy() * ratio[idx]

    # 집계 (train 기준)
    g1 = (train.groupby(['building','hour','dow'], as_index=False)['전력소비량(kWh)']
                .mean().rename(columns={'전력소비량(kWh)':'dow_hour_mean'}))
    g2 = (train.groupby(['building','hour','holiday'], as_index=False)['전력소비량(kWh)']
                .agg(holiday_mean='mean', holiday_std=lambda x: x.std(ddof=0)))
    g3 = (train.groupby(['building','hour'], as_index=False)['전력소비량(kWh)']
                .agg(hour_mean='mean', hour_std=lambda x: x.std(ddof=0)))

    def _attach(df):
        df = df.merge(g1, on=['building','hour','dow'], how='left')
        df = df.merge(g2, on=['building','hour','holiday'], how='left')
        df = df.merge(g3, on=['building','hour'], how='left')
        return df

    train = _attach(train)
    test  = _attach(test)

    # 타입 정리
    for df in (train, test):
        df['holiday'] = df['holiday'].astype('int8')
        if 'peak' in df.columns:
            df['peak'] = df['peak'].astype('int8')

    return train, test

def calculate_apparent_temp(temp, humidity, wind_speed):
    """체감온도 계산"""
    return temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00

def add_data(
    df,
    cols=('기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)'),
    copies=2,
    jitter=0.10,          # ±10%
    seed=SEED,
    round_decimals=1,
    sort_by=('건물번호', 'date'),
    cast_back=False       # True면 원래 dtype으로 복원 시도
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

def date_feat(df) :
    df['date'] = pd.to_datetime(df['일시'])
    df['dow'] = df['date'].dt.dayofweek 
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['dow'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['is_workhour'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  # 주기적 패턴
    df['COS_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['SIN_day'] = np.sin(2 * np.pi * df['day'] / 31)  # 일의 주기적 패턴 (31일 기준)
    df['COS_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['SIN_month'] = np.sin(2 * np.pi * df['month'] / 12)  # 일의 주기적 패턴 (31일 기준)
    df['COS_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['SIN_dow'] = np.sin(2 * np.pi * df['dow'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_dow'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['정오거리'] = (df['hour'] - 12).abs()  # 12시(정오) 기준 거리
    df['정오거리_INV'] = 1 / (df['hour'] + 1)
    
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear
    
    df['SIN_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['COS_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # yr = df['date'].dt.year.astype(str)
    # season_start = pd.to_datetime(yr + "-06-01")
    # season_end   = pd.to_datetime(yr + "-09-01")
    # in_season = (df['date'] >= season_start) & (df['date'] < season_end)

    # delta_days = (df['date'] - season_start).dt.days
    # week_idx = np.floor_divide(delta_days, 7) + 1                # 1,2,3,...
    # weeks_total = np.ceil((season_end - season_start).dt.days / 7).astype(int)  # 보통 14

    # df['week_idx'] = np.where(in_season, week_idx, np.nan)       # 시즌 밖 NaN (원하면 .fillna(0))

    # # 주차를 0~1 정규화 후 sin/cos
    # pos_week = (week_idx - 1) / weeks_total
    # phi_week = 2 * np.pi * pos_week
    # df['SIN_week_idx'] = np.where(in_season, np.sin(phi_week), np.nan)
    # df['COS_week_idx'] = np.where(in_season, np.cos(phi_week), np.nan)

    # # --- 주(월~일) 내부 위치 sin/cos (dow+hour 기반) ---
    # pos_inweek = (df['dow'] * 24 + df['hour']) / (7 * 24)        # 0~1
    # phi_inweek = 2 * np.pi * pos_inweek
    # df['SIN_inweek'] = np.where(in_season, np.sin(phi_inweek), np.nan)
    # df['COS_inweek'] = np.where(in_season, np.cos(phi_inweek), np.nan)

    # # (옵션) 정수 주차가 필요하면:
    # df['week_idx'] = df['week_idx'].astype('Int64')
    
    # --- 6/1 ~ 9/1을 한 주기로 본 sin/cos (시즌 밖은 NaN) ---
    yr = df['date'].dt.year.astype(str)
    season_start = pd.to_datetime(yr + "-06-01")
    season_end   = pd.to_datetime(yr + "-09-01")
    in_season = (df['date'] >= season_start) & (df['date'] < season_end)
    period_sec = (season_end - season_start).dt.total_seconds()
    pos = (df['date'] - season_start).dt.total_seconds() / period_sec  # 0~1
    phi = 2 * np.pi * pos
    df['SIN_summer'] = np.where(in_season, np.sin(phi), np.nan)
    df['COS_summer'] = np.where(in_season, np.cos(phi), np.nan)
    
    df['체감온도'] = calculate_apparent_temp(df['기온(°C)'], df['습도(%)'], df['풍속(m/s)'])
    df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    df['태양광per냉방면적'] = df['태양광용량(kW)'] / (df['냉방면적(m2)'] + 1e-6)
    df['ESS설치여부'] = df['ESS저장용량(kWh)'].apply(lambda x: 1 if x > 0 else 0)
    df['PCS설치여부'] = df['PCS용량(kW)'].apply(lambda x: 1 if x > 0 else 0)
    df['설비밀도'] = (df['ESS저장용량(kWh)'] + df['PCS용량(kW)']) / (df['연면적(m2)'] + 1e-6)

    # 원본 컬럼들 제거
    df = df.drop(['temp_date',], axis=1)
    return df

#endregion

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

train = date_feat(train)
test = date_feat(test)

train_path = './Energy/02/trainer/06_train_135_ver3.csv'
test_path = './Energy/02/trainer/06_test_135_ver3.csv'
train_interpolated = pd.read_csv(train_path)
test_interpolated = pd.read_csv(test_path)

train['일사(MJ/m2)'] = train_interpolated['일사(MJ/m2)'].copy()
test['일조(hr)'] = test_interpolated['일조(hr)'].copy()
test['일사(MJ/m2)'] = test_interpolated['일사(MJ/m2)'].copy()

train = shift_power_from_6h_and_trim(train)

train, test = mean_std_fast_v2(train, test, mode='byb', use_peak_holidays=True)

train_clean, train_removed = drop_outlier_times(train, intervals, singles, inclusive='both')
addSeed = SEED #random.randint(1,1000)
train_clean = add_data(train_clean, seed=addSeed)

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

#region(모델부분 Hybrid+Global: (Type=LGBM) + (BNO=XGB+LGBM) + (GlobalFull=LGBM))
# === Hybrid+Global (no KFold): Type=LGBM (holdout), BNO=XGB+LGB (holdout), GlobalFull=LGBM; final 3-way ensemble ===
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# External vars assumed: SEED, data_path, train_path, test_path, addSeed
TARGET   = '전력소비량(kWh)'
TYPE_COL = '건물유형'
BNO_COL  = '건물번호'

# --- Weights ---
W_XGB, W_LGB = 0.5, 0.5                      # per-building-number inner ensemble
W_FINAL_TYPE, W_FINAL_BUILD, W_FINAL_GLOBAL = 0.35, 0.45, 0.20  # final 3-way ensemble

TIME_AWARE = False  # True면 'date' 기준으로 앞 90% train, 뒤 10% val

def SMAPE(true, pred):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 200

# 1) Features
common = set(train_clean.columns) & set(test.columns)
drop_cols = {TARGET, '일시', 'date', '일조(hr)', 'month', TYPE_COL, BNO_COL}
print(drop_cols)
feature_cols = [c for c in sorted(common - drop_cols) if pd.api.types.is_numeric_dtype(train_clean[c])]
print(f"[Info] #features: {len(feature_cols)}")

# 2) Buffers
test_pred_type   = np.zeros(len(test), dtype=float)   # Type LGB (holdout)
test_pred_build  = np.zeros(len(test), dtype=float)   # BNO XGB+LGB (holdout)
test_pred_global = np.zeros(len(test), dtype=float)   # Global-Full LGB

metrics_type, metrics_build = [], []
type_group_means, build_group_means = [], []

# —— Predict helpers
def _predict_lgb(model, X):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        return model.predict(X, num_iteration=best_iter), best_iter
    else:
        return model.predict(X), model.n_estimators

def _predict_xgb(model, X):
    best_iter = getattr(model, "best_iteration", None)
    used_trees = (best_iter + 1) if best_iter is not None else model.n_estimators
    if best_iter is not None:
        return model.predict(X, iteration_range=(0, used_trees)), used_trees, best_iter
    else:
        return model.predict(X), used_trees, best_iter

# 3) Global backup (90/10 split) models
def fit_global_lgb():
    df = train_clean.sort_values('date') if 'date' in train_clean.columns else train_clean.copy()
    X = df[feature_cols]; y = np.log1p(df[TARGET])

    split = int(len(df) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)

    lgb = LGBMRegressor(
        n_estimators=4000, learning_rate=0.03,
        num_leaves=256, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )

    val_log, _ = _predict_lgb(lgb, X_val)
    smape = SMAPE(np.expm1(y_val), np.clip(np.expm1(val_log), 0, None))
    print(f"[Global LGB (split)] val SMAPE={smape:.3f}  best_it={getattr(lgb,'best_iteration_',None)}")
    return lgb, med, smape

def fit_global_xgb():
    df = train_clean.sort_values('date') if 'date' in train_clean.columns else train_clean.copy()
    X = df[feature_cols]; y = np.log1p(df[TARGET])

    split = int(len(df) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)

    xgb = XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        eval_metric="mae", early_stopping_rounds=100,
        tree_method="hist", n_jobs=-1, reg_lambda=1.0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_log, _, best_it = _predict_xgb(xgb, X_val)
    smape = SMAPE(np.expm1(y_val), np.clip(np.expm1(val_log), 0, None))
    print(f"[Global XGB (split)] val SMAPE={smape:.3f}  best_it={best_it}")
    return xgb, med, smape

global_lgb_split, global_med_split, global_val_smape_split = fit_global_lgb()
global_xgb_split, _                 , global_val_smape_xgb   = fit_global_xgb()  # med 동일 구조이므로 global_med_split 사용

# ===== 4-A) Type-level (LGBM only, single holdout) =====
for t in test[TYPE_COL].unique():
    tr_type = train_clean[train_clean[TYPE_COL] == t]
    te_type = test[test[TYPE_COL] == t]

    if len(tr_type) < max(50, 60):
        X_te = te_type[feature_cols].fillna(global_med_split)
        preds_log, _ = _predict_lgb(global_lgb_split, X_te)
        preds = np.clip(np.expm1(preds_log), 0, None)
        test_pred_type[te_type.index] = preds
        metrics_type.append({"group": str(t), "split": "global", "rows": len(tr_type), "val_rows": 0, "val_smape": np.nan, "used_global": 1})
        print(f"[Type {t}] rows={len(tr_type):4d} -> use GLOBAL LGBM")
        continue

    if 'date' in tr_type.columns and TIME_AWARE:
        tr_type = tr_type.sort_values('date')
        split = int(len(tr_type) * 0.9)
        g_tr, g_val = tr_type.iloc[:split], tr_type.iloc[split:]
    else:
        g_tr, g_val = train_test_split(tr_type, test_size=0.1, random_state=SEED, shuffle=True)

    X_tr, y_tr = g_tr[feature_cols], np.log1p(g_tr[TARGET])
    X_val, y_val = g_val[feature_cols], np.log1p(g_val[TARGET])

    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)

    lgb = LGBMRegressor(
        n_estimators=4000, learning_rate=0.03,
        num_leaves=256, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )

    val_log, _ = _predict_lgb(lgb, X_val)
    smape = SMAPE(np.expm1(y_val), np.clip(np.expm1(val_log), 0, None))
    type_group_means.append(smape)
    print(f"[Type {t}] holdout val SMAPE(LGB)={smape:.3f}  best_it={getattr(lgb,'best_iteration_',None)}")

    metrics_type.append({
        "group": str(t), "split": "holdout", "rows": len(tr_type),
        "val_rows": len(X_val), "val_smape": smape, "used_global": 0
    })

    X_te = te_type[feature_cols].fillna(med)
    preds_log, _ = _predict_lgb(lgb, X_te)
    preds = np.clip(np.expm1(preds_log), 0, None)
    test_pred_type[te_type.index] = preds

# ===== 4-B) Building-number level (XGB+LGB, single holdout) =====
for bno in test[BNO_COL].unique():
    tr_b = train_clean[train_clean[BNO_COL] == bno]
    te_b = test[test[BNO_COL] == bno]

    if len(tr_b) < max(50, 60):
        X_te = te_b[feature_cols].fillna(global_med_split)
        px_log, _, _ = _predict_xgb(global_xgb_split, X_te)
        pl_log, _    = _predict_lgb(global_lgb_split, X_te)
        preds = W_XGB * np.clip(np.expm1(px_log), 0, None) + W_LGB * np.clip(np.expm1(pl_log), 0, None)
        test_pred_build[te_b.index] = preds
        metrics_build.append({"group": str(bno), "split": "global", "rows": len(tr_b), "val_rows": 0, "val_smape": np.nan, "used_global": 1})
        print(f"[BNO {bno}] rows={len(tr_b):4d} -> use GLOBAL XGB+LGB")
        continue

    if 'date' in tr_b.columns and TIME_AWARE:
        tr_b = tr_b.sort_values('date')
        split = int(len(tr_b) * 0.9)
        g_tr, g_val = tr_b.iloc[:split], tr_b.iloc[split:]
    else:
        g_tr, g_val = train_test_split(tr_b, test_size=0.1, random_state=SEED, shuffle=True)

    X_tr, y_tr = g_tr[feature_cols], np.log1p(g_tr[TARGET])
    X_val, y_val = g_val[feature_cols], np.log1p(g_val[TARGET])

    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)

    # XGB
    xgb = XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        eval_metric="mae", early_stopping_rounds=100,
        tree_method="hist", n_jobs=-1, reg_lambda=1.0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # LGB
    lgb = LGBMRegressor(
        n_estimators=4000, learning_rate=0.03,
        num_leaves=256, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)]
    )

    px_log, _, xgb_best = _predict_xgb(xgb, X_val)
    pl_log, _           = _predict_lgb(lgb, X_val)
    val_pred = W_XGB * np.clip(np.expm1(px_log), 0, None) + W_LGB * np.clip(np.expm1(pl_log), 0, None)
    smape = SMAPE(np.expm1(y_val), val_pred)
    build_group_means.append(smape)
    print(f"[BNO {bno}] holdout val SMAPE(ENS)={smape:.3f}  XGB(best_it={xgb_best})  LGB(best_it={getattr(lgb,'best_iteration_',None)})")

    metrics_build.append({
        "group": str(bno), "split": "holdout", "rows": len(tr_b),
        "val_rows": len(X_val), "val_smape": smape, "used_global": 0
    })

    X_te = te_b[feature_cols].fillna(med)
    px_log, _, _ = _predict_xgb(xgb, X_te)
    pl_log, _    = _predict_lgb(lgb, X_te)
    preds = W_XGB * np.clip(np.expm1(px_log), 0, None) + W_LGB * np.clip(np.expm1(pl_log), 0, None)
    test_pred_build[te_b.index] = preds

# ===== 4-C) Global-Full LGBM (train on ALL train, predict ALL test) =====
med_all = train_clean[feature_cols].median()
X_all   = train_clean[feature_cols].fillna(med_all)
y_all   = np.log1p(train_clean[TARGET])

best_iters = getattr(global_lgb_split, "best_iteration_", None)
n_est_full = int(best_iters) if (best_iters is not None and best_iters > 0) else 2000

lgb_full = LGBMRegressor(
    n_estimators=n_est_full, learning_rate=0.03,
    num_leaves=256, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1
)
lgb_full.fit(X_all, y_all)  # full-data fit (no val)

X_te_all = test[feature_cols].fillna(med_all)
preds_log_full = lgb_full.predict(X_te_all)
test_pred_global = np.clip(np.expm1(preds_log_full), 0, None)

# 5) Final 3-way ensemble
w_sum = W_FINAL_TYPE + W_FINAL_BUILD + W_FINAL_GLOBAL
w_type  = W_FINAL_TYPE  / w_sum
w_build = W_FINAL_BUILD / w_sum
w_glob  = W_FINAL_GLOBAL/ w_sum

test_pred_final = w_type*test_pred_type + w_build*test_pred_build + w_glob*test_pred_global

val_smape_type   = float(np.nanmean(type_group_means))  if len(type_group_means)  else np.nan
val_smape_build  = float(np.nanmean(build_group_means)) if len(build_group_means) else np.nan
val_smape_global = global_val_smape_split  # proxy from 90/10 split

val_smape_final = np.nan
if not np.isnan(val_smape_type) and not np.isnan(val_smape_build) and not np.isnan(val_smape_global):
    val_smape_final = w_type*val_smape_type + w_build*val_smape_build + w_glob*val_smape_global

print(f"[VAL] type_mean={val_smape_type:.4f}  build_mean={val_smape_build:.4f}  global_split={val_smape_global:.4f}  final~={val_smape_final:.4f}")

#endregion

# 6) Save
import datetime
save_path = f'./Energy/02/{SEED}_(ver8)submission/'
os.makedirs(save_path , exist_ok=True )

samplesub = pd.read_csv(data_path + 'sample_submission.csv')
samplesub['answer'] = test_pred_final

today = datetime.datetime.now().strftime('%Y%m%d')
score_str = ("nan" if np.isnan(val_smape_final) else f"{val_smape_final:.4f}").replace('.', '_')
filename = f"(ver8){SEED}_02_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

pd.DataFrame(metrics_type ).to_csv(save_path + '(ver8)lgb_by_type_metrics.csv', index=False)
pd.DataFrame(metrics_build).to_csv(save_path + '(ver8)xgb_lgb_by_building_metrics.csv', index=False)

with open(save_path + "(LOG)model.txt", "a") as f:
    f.write(f"<SEED :{SEED}> ver8 Hybrid+Global (holdout only)\n")
    f.write(f"{filename}\n")
    f.write(f"{drop_cols}\n")
    f.write(f"train file name : {train_path}\n")
    f.write(f"test file name : {test_path}\n")
    f.write(f"Add Data SEED : {addSeed}\n")
    f.write(f"[Type ] holdout mean SMAPE={val_smape_type:.4f}\n")
    f.write(f"[BNO  ] holdout mean SMAPE={val_smape_build:.4f}\n")
    f.write(f"[Global(split)] val SMAPE={val_smape_global:.4f}, best_iter={best_iters}\n")
    f.write(f"[Final 3-way] w_type={w_type:.3f}, w_build={w_build:.3f}, w_global={w_glob:.3f}, est SMAPE={val_smape_final}\n")
    f.write("="*40 + "\n")

print("종료")

