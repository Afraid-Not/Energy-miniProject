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

seed_file = "./Energy/02/log/(preprocessing2)SEED_COUNTS.json"

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

def attach_stats_features(
    train, test,
    target_col='target',
    building_col=None,
    hour_col=None,
    dow_col=None,               # 0=월 ~ 6=일
    holiday_col='holidays',
    datetime_col=None,          # 없으면 자동 탐지: ['일시','date_time','datetime','dt']
    ddof=1,                     # 표준편차 자유도(기본 표본 표준편차)
):
    tr = train.copy()
    te = test.copy()

    # ------ 컬럼 자동 감지 ------
    if building_col is None:
        building_col = '건물번호' if '건물번호' in tr.columns else 'building'
    if datetime_col is None:
        for c in ['일시','date_time','datetime','dt']:
            if c in tr.columns:
                datetime_col = c
                break
    if dow_col is None:
        dow_col = 'dow' if 'dow' in tr.columns else '요일'
    if hour_col is None:
        hour_col = 'hour' if 'hour' in tr.columns else '시각'
    if holiday_col not in tr.columns:
        tr[holiday_col] = 0
    if holiday_col not in te.columns:
        te[holiday_col] = 0

    # ------ dow/hour 생성 (없으면) ------
    def _ensure_time_cols(df):
        if dow_col in df.columns and hour_col in df.columns:
            return df
        assert datetime_col is not None, "dow/hour가 없으면 datetime_col이 필요합니다."
        dt = pd.to_datetime(df[datetime_col], errors='coerce')
        if dow_col not in df.columns:
            df[dow_col] = dt.dt.weekday
        if hour_col not in df.columns:
            df[hour_col] = dt.dt.hour
        return df

    tr = _ensure_time_cols(tr)
    te = _ensure_time_cols(te)

    # ------ 전역 평균/표준편차 (백업용) ------
    global_mean = tr[target_col].mean()
    global_std  = tr[target_col].std(ddof=ddof)

    # ------ helper: 평균/표준편차 머지 ------
    def _merge_mean_std(base_df, key_cols, mean_name, std_name):
        m = tr.groupby(key_cols)[target_col].mean().reset_index().rename(columns={target_col: mean_name})
        s = tr.groupby(key_cols)[target_col].std(ddof=ddof).reset_index().rename(columns={target_col: std_name})
        stat = pd.merge(m, s, on=key_cols, how='outer')
        out_base = base_df.merge(stat, on=key_cols, how='left')
        return out_base

    # 1) 건물 × 시각 × 요일
    key_dhy = [building_col, hour_col, dow_col]
    for df in (tr, te):
        df[dow_col] = df[dow_col].astype(int)
        df[hour_col] = df[hour_col].astype(int)
    tr = _merge_mean_std(tr, key_dhy, 'dow_hour_mean', 'dow_hour_std')
    te = _merge_mean_std(te, key_dhy, 'dow_hour_mean', 'dow_hour_std')

    # 2) 건물 × 시각 × holiday
    key_hol = [building_col, hour_col, holiday_col]
    tr = _merge_mean_std(tr, key_hol, 'holiday_mean', 'holiday_std')
    te = _merge_mean_std(te, key_hol, 'holiday_mean', 'holiday_std')

    # 3) 건물 × 시각
    key_h = [building_col, hour_col]
    tr = _merge_mean_std(tr, key_h, 'hour_mean', 'hour_std')
    te = _merge_mean_std(te, key_h, 'hour_mean', 'hour_std')

    # 4) 건물 단위(백업용)
    b_mean = tr.groupby(building_col)[target_col].mean().rename('building_mean')
    b_std  = tr.groupby(building_col)[target_col].std(ddof=ddof).rename('building_std')
    tr = tr.merge(b_mean, on=building_col, how='left').merge(b_std, on=building_col, how='left')
    te = te.merge(b_mean, on=building_col, how='left').merge(b_std, on=building_col, how='left')

    # ------ 결측 백필 규칙 ------
    for df in (tr, te):
        # mean 백필: dow_hour -> holiday -> hour -> building -> global
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
        # 보조 피처도 안전하게 채움
        df['holiday_mean'] = df['holiday_mean'].fillna(df['hour_mean']).fillna(df['building_mean']).fillna(global_mean)
        df['holiday_std']  = df['holiday_std'].fillna(df['hour_std']).fillna(df['building_std']).fillna(global_std)
        df['hour_mean']    = df['hour_mean'].fillna(df['building_mean']).fillna(global_mean)
        df['hour_std']     = df['hour_std'].fillna(df['building_std']).fillna(global_std)

    return tr, te

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

    # 원본 컬럼들 제거
    df = df.drop(['temp_date'], axis=1)
    return df

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

train = peak_holidays(train, is_train=True)
test = peak_holidays(test, is_train=False)

train_path = './Energy/02/trainer/06_train_126_ver3.csv'
test_path = './Energy/02/trainer/06_test_126_ver3.csv'
train_interpolated = pd.read_csv(train_path)
test_interpolated = pd.read_csv(test_path)

train['일사(MJ/m2)'] = train_interpolated['일사(MJ/m2)'].copy()
test['일조(hr)'] = test_interpolated['일조(hr)'].copy()
test['일사(MJ/m2)'] = test_interpolated['일사(MJ/m2)'].copy()


train, test = attach_stats_features(
    train, test,
    target_col='전력소비량(kWh)',      # 네 타깃 이름
    building_col='건물번호',   # or 'building'
    hour_col='hour',          # 없으면 일시에서 자동 생성
    dow_col='dow',            # 없으면 일시에서 자동 생성
    holiday_col='holidays',
    datetime_col='date',       # 'date_time'이면 그걸로
    ddof=1                    # 표본 표준편차
)

train_clean, train_removed = drop_outlier_times(train, intervals, singles, inclusive='both')
# print(train_clean.shape, test.shape)
# (202347, 46) (16800, 45)
# print(train_clean.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
#        '일사(MJ/m2)', '전력소비량(kWh)', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
#        'ESS저장용량(kWh)', 'PCS용량(kW)', '지역그룹ID', 'date', 'dow', 'hour', 'month',
#        'day', 'is_weekend', 'is_workhour', 'SIN_hour', 'COS_hour', 'SIN_day',
#        'COS_day', 'SIN_month', 'COS_month', 'SIN_dow', 'COS_dow', '정오거리',
#        '정오거리_INV', 'day_of_year', 'SIN_day_of_year', 'COS_day_of_year',
#        'holidays', 'peak', 'dow_hour_mean', 'dow_hour_std', 'holiday_mean',
#        'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
#        'building_std'],
#       dtype='object')
# print(test.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형',
#        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
#        '지역그룹ID', 'date', 'dow', 'hour', 'month', 'day', 'is_weekend',
#        'is_workhour', 'SIN_hour', 'COS_hour', 'SIN_day', 'COS_day',
#        'SIN_month', 'COS_month', 'SIN_dow', 'COS_dow', '정오거리', '정오거리_INV',
#        'day_of_year', 'SIN_day_of_year', 'COS_day_of_year', 'holidays', 'peak',
#        '일조(hr)', '일사(MJ/m2)', 'dow_hour_mean', 'dow_hour_std', 'holiday_mean',
#        'holiday_std', 'hour_mean', 'hour_std', 'building_mean',
#        'building_std'],
#       dtype='object')

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


# === XGBoost per-building-type prototype (SMAPE reporting) ===
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

SEED = 73
print(f"[current seed] {SEED}")
TARGET = '전력소비량(kWh)'
GROUP_COL = '건물유형'

def SMAPE(true, pred):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 200

# 1) 공통 피처(숫자형, train/test 교집합)
common = set(train_clean.columns) & set(test.columns)
drop_cols = {TARGET, '일시', 'date'}
feature_cols = [c for c in sorted(common - drop_cols) if pd.api.types.is_numeric_dtype(train_clean[c])]
print(f"[Info] #features: {len(feature_cols)}")

# 2) 결과 저장용
test_pred = np.zeros(len(test), dtype=float)
metrics = []   # 각 건물유형별 val SMAPE 기록
def fit_global_backup():
    df = train_clean.sort_values('date') if 'date' in train_clean.columns else train_clean.copy()
    X = df[feature_cols]; y = df[TARGET]

    # —— 로그 변환
    y = np.log1p(y)

    split = int(len(df) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)

    model = XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        eval_metric="mae", early_stopping_rounds=100,
        tree_method="hist", n_jobs=-1, reg_lambda=1.0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # —— best_iteration 사용
    best_iter = getattr(model, "best_iteration", None)
    used_trees = (best_iter + 1) if best_iter is not None else model.n_estimators

    # —— 검증 예측(로그→원스케일 복원) + SMAPE
    if best_iter is not None:
        val_pred = model.predict(X_val, iteration_range=(0, used_trees))
    else:
        val_pred = model.predict(X_val)
    val_pred_exp = np.expm1(val_pred)
    y_val_exp = np.expm1(y_val)

    smape = SMAPE(y_val_exp, val_pred_exp)
    print(f"[Global] val SMAPE={smape:.3f}  best_iteration={best_iter}  used_trees={used_trees}")

    metrics.append({"group": "GLOBAL", "rows": len(df), "val_rows": len(X_val), "val_smape": smape})
    return model, med

global_model, global_med = fit_global_backup()

all_smape = []
# 4) 유형별 학습→예측
for t in test[GROUP_COL].unique():
    tr_type = train_clean[train_clean[GROUP_COL] == t]
    te_type = test[test[GROUP_COL] == t]

    # 학습 데이터 부족 시 글로벌 모델 사용
    if len(tr_type) < 50:
        X_te = te_type[feature_cols].fillna(global_med)
        best_iter = getattr(global_model, "best_iteration", None)
        used_trees = (best_iter + 1) if best_iter is not None else global_model.n_estimators
        if best_iter is not None:
            preds_log = global_model.predict(X_te, iteration_range=(0, used_trees))
        else:
            preds_log = global_model.predict(X_te)
        test_pred[te_type.index] = np.expm1(preds_log)
        metrics.append({"group": str(t), "rows": len(tr_type), "val_rows": 0, "val_smape": np.nan, "used_global": 1})
        print(f"[{t}] rows={len(tr_type):4d} -> use GLOBAL model")
        continue

    tr_type = tr_type.sort_values('date') if 'date' in tr_type.columns else tr_type
    X = tr_type[feature_cols]; y = tr_type[TARGET]

    # —— 로그 변환
    y = np.log1p(y)

    split = int(len(tr_type) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    med = X_tr.median()
    X_tr = X_tr.fillna(med); X_val = X_val.fillna(med)

    model = XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        eval_metric="mae", early_stopping_rounds=100,
        tree_method="hist", n_jobs=-1, reg_lambda=1.0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # —— best_iteration 사용
    best_iter = getattr(model, "best_iteration", None)
    used_trees = (best_iter + 1) if best_iter is not None else model.n_estimators

    # —— 검증 예측(로그→원스케일 복원) + SMAPE
    if best_iter is not None:
        val_pred = model.predict(X_val, iteration_range=(0, used_trees))
    else:
        val_pred = model.predict(X_val)
    val_pred_exp = np.expm1(val_pred)
    y_val_exp = np.expm1(y_val)

    smape = SMAPE(y_val_exp, val_pred_exp)
    all_smape.append(smape)
    print(f"[{t}] rows={len(tr_type):6d}  val SMAPE={smape:.3f}  best_iteration={best_iter}  used_trees={used_trees}")
    metrics.append({"group": str(t), "rows": len(tr_type), "val_rows": len(X_val), "val_smape": smape, "used_global": 0})

    # —— test 예측(로그→원스케일 복원)
    X_te = te_type[feature_cols].fillna(med)
    if best_iter is not None:
        preds_log = model.predict(X_te, iteration_range=(0, used_trees))
    else:
        preds_log = model.predict(X_te)
    test_pred[te_type.index] = np.expm1(preds_log)

# 전체 평균(SMAPE) - NaN 무시
val_smape = float(np.nanmean(all_smape)) if len(all_smape) else np.nan
print(f"[전체 평균] val SMAPE {val_smape}")

import datetime
save_path = f'./Energy/02/{SEED}_submission/'
os.makedirs(save_path , exist_ok=True )
samplesub = pd.read_csv(data_path + 'sample_submission.csv')
samplesub['answer'] = test_pred
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{val_smape:.4f}".replace('.', '_')

filename = f"{SEED}_02_{today}_SMAPE_{score_str}_proto.csv"
samplesub.to_csv(save_path + filename, index=False)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(save_path + 'xgb_by_type_metrics.csv', index=False)

with open(save_path + "(LOG)model.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"train file name : {train_path}\n")
    f.write(f"test file name : {test_path}\n")
    f.write(f"건물유형별 모델 SMAPE: {val_smape:.4f}\n")
    f.write("="*40 + "\n")

print(f"[4] 종료 ")