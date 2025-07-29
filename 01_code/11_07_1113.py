print(f"[11_06] 시작")
# ========================
# 임포트 및 랜덤 시드 고정
# ========================
import pandas as pd
import numpy as np
import datetime
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

seed_file = "./Energy/11_submission/11_07.json"

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

print(f"[1] 데이터 로드")
# ========================
# 데이터 로드
# ========================

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

data_path = './Energy/'
save_path = './Energy/11_submission/'
os.makedirs(save_path, exist_ok=True)
""" 
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')

# ========================
# building_csv 전처리
# ========================

building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col :
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)

building_csv = building_csv.fillna(0)

train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

# ========================
# train, test 전처리 (Feature Engineering 강화)
# ========================

def CDH(xs):
    cumsum = np.cumsum(xs - 26)
    return np.concatenate((cumsum[:11], cumsum[11:] - cumsum[:-11]))

def calculate_day_values(df, target_col, output_col, aggfunc):
    result_dict = df.groupby(['건물번호', '월', '일'])[target_col].agg(aggfunc).to_dict()
    df[output_col] = [
        result_dict.get((row['건물번호'], row['월'], row['일']), np.nan) for _, row in df.iterrows()
    ]

def feature_engineering(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['일시'] = pd.to_datetime(df['일시'])
    df['시각'] = df['일시'].dt.hour
    df['요일'] = df['일시'].dt.dayofweek
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
    df['근무시간'] = df['시각'].apply(lambda x: 1 if 9 <= x <= 18 else 0)

    # 주기적 패턴 강화
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)
    df['SIN_일'] = np.sin(2 * np.pi * df['일'] / 31)
    df['COS_일'] = np.cos(2 * np.pi * df['일'] / 31)
    df['SIN_월'] = np.sin(2 * np.pi * df['월'] / 12)
    df['COS_월'] = np.cos(2 * np.pi * df['월'] / 12)
    df['SIN_요일'] = np.sin(2 * np.pi * (df['요일'] + 1) / 7)
    df['COS_요일'] = np.cos(2 * np.pi * (df['요일'] + 1) / 7)

    # 요일 One-hot
    df = pd.concat([df, pd.get_dummies(df['요일'], prefix='요일')], axis=1)

    # ======================
    # 건물유형 One-hot
    # ======================
    if '건물유형' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['건물유형'])], axis=1)

    # ======================
    # 기상/에너지 관련 파생 피처
    # ======================
    if '기온(°C)' in df.columns and '습도(%)' in df.columns:
        df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
        df['기온X습도'] = df['기온(°C)'] * df['습도(%)']

    if '강수량(mm)' in df.columns:
        df['강수유무'] = (df['강수량(mm)'] > 0).astype(int)

    # CDH (냉방도일수)
    cdhs = []
    for bno in df['건물번호'].unique():
        temp = df[df['건물번호'] == bno]['기온(°C)'].values
        if len(temp) >= 11:
            cdhs.append(CDH(temp))
        else:
            cdhs.append(np.zeros(len(temp)))
    df['CDH'] = np.concatenate(cdhs)

    # WCT / THI
    if '풍속(m/s)' in df.columns:
        df['WCT'] = (
            13.12 + 0.6215 * df['기온(°C)'] - 11.37 * (df['풍속(m/s)'] ** 0.16)
            + 0.3965 * df['기온(°C)'] * (df['풍속(m/s)'] ** 0.16)
        )
    df['THI'] = 9/5*df['기온(°C)'] - 0.55*(1-df['습도(%)']/100)*(9/5*df['기온(°C)']-26)+32

    # 태양광per냉방면적
    if '태양광용량(kW)' in df.columns and '냉방면적(m2)' in df.columns:
        df['태양광per냉방면적'] = df['태양광용량(kW)'] / (df['냉방면적(m2)'] + 1e-6)

    if 'ESS저장용량(kWh)' in df.columns:
        df['ESS설치여부'] = (df['ESS저장용량(kWh)'].replace('-', 0).astype(float) > 0).astype(int)

    if 'PCS용량(kW)' in df.columns:
        df['PCS설치여부'] = (df['PCS용량(kW)'].replace('-', 0).astype(float) > 0).astype(int)

    if 'ESS저장용량(kWh)' in df.columns and 'PCS용량(kW)' in df.columns and '연면적(m2)' in df.columns:
        ess = df['ESS저장용량(kWh)'].replace('-', 0).astype(float)
        pcs = df['PCS용량(kW)'].replace('-', 0).astype(float)
        area = df['연면적(m2)'] + 1e-6
        df['설비밀도'] = (ess + pcs) / area

    # ======================
    # 일별 온도 통계 피처
    # ======================
    if '기온(°C)' in df.columns:
        calculate_day_values(df, '기온(°C)', 'day_max_temp', 'max')
        calculate_day_values(df, '기온(°C)', 'day_mean_temp', 'mean')
        calculate_day_values(df, '기온(°C)', 'day_min_temp', 'min')
        df['day_temp_range'] = df['day_max_temp'] - df['day_min_temp']

    return df

def impute_train_solar_for_missing_bnos(train_df, zero_bnos, k=5):
    print("  > train 일사(MJ/m2) 이상치 보간 시작")

    df_filled = train_df.copy()
    df_filled['hour'] = pd.to_datetime(df_filled['일시']).dt.hour

    # 후보는 미리 필터링
    candidate_pool = df_filled[
        (~df_filled['건물번호'].isin(zero_bnos)) & (df_filled['일사(MJ/m2)'] > 0)
    ].copy()
    candidate_pool['hour'] = pd.to_datetime(candidate_pool['일시']).dt.hour
    X_pool = candidate_pool[['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']].values

    for bno in zero_bnos:
        print(f"  > [BUILDING {bno}] 보간 중...")

        target_rows = df_filled[(df_filled['건물번호'] == bno) & (df_filled['일사(MJ/m2)'] == 0)]

        for idx, row in tqdm(target_rows.iterrows(), total=len(target_rows)):
            hour = row['hour']
            if hour < 5 or hour > 21:
                df_filled.at[idx, '일사(MJ/m2)'] = 0.0
                continue

            x_target = np.array([[row['기온(°C)'], row['습도(%)'], row['풍속(m/s)'], hour]])

            neigh = NearestNeighbors(n_neighbors=k)
            neigh.fit(X_pool)
            _, indices = neigh.kneighbors(x_target)

            pred_val = candidate_pool.iloc[indices[0]]['일사(MJ/m2)'].mean()
            df_filled.at[idx, '일사(MJ/m2)'] = np.clip(pred_val, 0, None)

    return df_filled.drop(columns='hour')

def impute_test_sun_features(test_df, train_df, k=5):
    print("  > k-NN 보간 시작")
    test_df = test_df.copy()
    train_df = train_df.copy()
    
    test_df['hour'] = pd.to_datetime(test_df['일시']).dt.hour
    train_df['hour'] = pd.to_datetime(train_df['일시']).dt.hour

    # 1. 건물별로 train 데이터 캐싱
    nn_models_sun = {}
    nn_models_rad = {}
    sun_features = ['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']

    for bno in tqdm(train_df['건물번호'].unique(), total=len(train_df['건물번호'].unique())):
        group = train_df[train_df['건물번호'] == bno].copy()

        # 일조 후보
        sun_cand = group[group['일조(hr)'] > 0]
        if len(sun_cand) >= k:
            X_sun = sun_cand[sun_features].values
            nn = NearestNeighbors(n_neighbors=k).fit(X_sun)
            nn_models_sun[bno] = (nn, sun_cand)

        # 일사 후보
        rad_cand = group[group['일사(MJ/m2)'] > 0]
        if len(rad_cand) >= k:
            X_rad = rad_cand[sun_features].values
            nn = NearestNeighbors(n_neighbors=k).fit(X_rad)
            nn_models_rad[bno] = (nn, rad_cand)

    # 2. test 보간
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        hour = row['hour']
        bno = row['건물번호']
        x_target = np.array([[row['기온(°C)'], row['습도(%)'], row['풍속(m/s)'], hour]])

        # 일조
        if hour < 7 or hour > 19:
            test_df.at[idx, '일조(hr)'] = 0.0
        elif bno in nn_models_sun:
            nn, sun_cand = nn_models_sun[bno]
            _, indices = nn.kneighbors(x_target)
            mean_sun = sun_cand.iloc[indices[0]]['일조(hr)'].mean()
            test_df.at[idx, '일조(hr)'] = np.clip(mean_sun, 0, None)

        # 일사
        if hour < 6 or hour > 20:
            test_df.at[idx, '일사(MJ/m2)'] = 0.0
        elif bno in nn_models_rad:
            nn, rad_cand = nn_models_rad[bno]
            _, indices = nn.kneighbors(x_target)
            mean_rad = rad_cand.iloc[indices[0]]['일사(MJ/m2)'].mean()
            test_df.at[idx, '일사(MJ/m2)'] = np.clip(mean_rad, 0, None)

    return test_df.drop(columns='hour')


# train 일사 보간
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
train_all = feature_engineering(train)
train_all_fixed = impute_train_solar_for_missing_bnos(train_all, zero_bnos, k=5)
train_all_fixed.to_csv(save_path + 'preprocessed_train.csv', index=False)
print("    >> train 일사 보간 완료 및 저장")

# test 일조 일사 보간
test_all = feature_engineering(test)
test_all_imputed = impute_test_sun_features(test_all, train_all_fixed, k=5)
test_all_imputed.to_csv(save_path + 'preprocessed_test.csv', index=False)

print("    >> 일조/일사 보간 완료 및 저장")
 """
print(f"  >> 완료")

train = pd.read_csv(save_path + 'preprocessed_train.csv')
test = pd.read_csv(save_path + 'preprocessed_test.csv')
samplesub = pd.read_csv(data_path +'sample_submission.csv')
###############
print("[2] 건물유형별 사전학습")

def one_hot(df) :
    temp = pd.get_dummies(df['건물번호'], prefix='건물')
    df = pd.concat([df, temp], axis=1)
    return df

train = one_hot(train)
test = one_hot(test)
exclude_cols = ['건물번호', '일시', '전력소비량(kWh)', '건물유형', '날짜']

# 최종 피처 정의
features = [col for col in train.columns if col not in exclude_cols]

target = '전력소비량(kWh)'
building_types = train['건물유형'].unique()
pretrained_models = {}
pretrained_scalers = {}

# SMAPE를 Optuna 최적화 함수에 사용할 수 있도록 Scorer로 변환
def smape_score(y_true, y_pred):
    smape_val = np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))
    return smape_val

# 사전 학습을 위한 Optuna 최적화
def optuna_objective(trial, model_name, X, y, kf, scaler):
    if model_name == 'xgb':
        params = {
            'objective': 'reg:squarederror',
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        model = XGBRegressor(**params)
    elif model_name == 'lgb':
        params = {
            'objective': 'mae',
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'verbosity' : -1,
        }
        model = LGBMRegressor(**params)
    elif model_name == 'cat':
        params = {
            'loss_function': 'MAE',
            'random_seed': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        } 
        model = CatBoostRegressor(**params, verbose=0)

    smape_folds = []
    
    for tr_idx, val_idx in kf.split(X):
        x_train, x_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        
        if model_name == 'lgb':
            model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        elif model_name == 'xgb':
            model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)
        elif model_name == 'cat':
            model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=30)
            
        pred_val = model.predict(x_val_scaled)
        smape_folds.append(smape_score(np.expm1(y_val), np.expm1(pred_val)))
        
    return np.mean(smape_folds)

def optuna_objective_retrain(trial, model_name, X, y, kf, scaler):
    # sMAPE가 높은 건물에 대한 재학습용 objective 함수
    if model_name == 'xgb':
        params = {
            'objective': 'reg:squarederror',
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500), # 탐색 범위 확대
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'early_stopping_rounds': 30, # 조기 중단 추가
        }
        model = XGBRegressor(**params)
    elif model_name == 'lgb':
        params = {
            'objective': 'mae',
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'verbosity' : -1,
        }
        model = LGBMRegressor(**params)
    elif model_name == 'cat':
        params = {
            'loss_function': 'MAE',
            'random_seed': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        } 
        model = CatBoostRegressor(**params, verbose=0)
    
    smape_folds = []
    
    for tr_idx, val_idx in kf.split(X):
        x_train, x_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        
        if model_name == 'lgb':
            model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],callbacks=[early_stopping(75, verbose=False), log_evaluation(0)])
        elif model_name == 'xgb':
            model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)
        elif model_name == 'cat':
            model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)
            
        pred_val = model.predict(x_val_scaled)
        smape_folds.append(smape_score(np.expm1(y_val), np.expm1(pred_val)))
        
    return np.mean(smape_folds)


print("[2] 건물유형별 사전학습 (Optuna 최적화)")

N_SPLIT = 5
N_TRIALS = 30
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

for idxb, btype in enumerate(building_types):
    print(f"\n[BUILING TYPE {idxb}: {btype}]")
    train_bt = train[train['건물유형'] == btype].copy()
    x = train_bt[features].reset_index(drop=True)
    y = np.log1p(train_bt[target].reset_index(drop=True))

    best_models_btype = {}
    
    scaler_btype = StandardScaler()
    scaler_btype.fit(x)
    pretrained_scalers[btype] = scaler_btype

    for model_name in ['xgb', 'lgb', 'cat']:
        print(f"  > '{idxb}:{btype}' 유형의 '{model_name}' 모델 최적화...")
        study = optuna.create_study(direction='minimize')
        
        study.optimize(lambda trial: optuna_objective(trial, model_name, x, y, kf, scaler_btype), n_trials=N_TRIALS, show_progress_bar=True)
        
        best_params = study.best_params
        best_smape = study.best_value
        print(f"  > 최적 SMAPE: {best_smape:.4f}, 최적 파라미터: {best_params}\n")
        
        if model_name == 'xgb':
            best_model = XGBRegressor(**best_params, random_state=SEED)
        elif model_name == 'lgb':
            best_model = LGBMRegressor(**best_params, random_state=SEED, verbose=-1)
        elif model_name == 'cat':
            best_model = CatBoostRegressor(**best_params, random_seed=SEED, verbose=0)
            
        best_model.fit(scaler_btype.transform(x), y)
        best_models_btype[model_name] = best_model
        
    pretrained_models[btype] = best_models_btype

print(f"  >> 사전학습 완료")
 
print("[3] 건물별 1차 예측 및 SMAPE 계산")

initial_smapes = {}
initial_test_preds = {}

kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

for bno in train['건물번호'].unique():
    btype = train[train['건물번호'] == bno]['건물유형'].iloc[0]
    print(f"  >> [BUILDING {bno}] 유형: {btype} 1차 예측")
    
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()
    
    x = train_b[features].reset_index(drop=True)
    y = np.log1p(train_b[target].reset_index(drop=True))
    x_test_final = test_b[features].reset_index(drop=True)

    test_pred_folds = np.zeros(len(x_test_final))
    smape_folds = []

    scaler = pretrained_scalers[btype]
    x_scaled = scaler.transform(x)
    x_test_scaled = scaler.transform(x_test_final)

    for fold, (tr_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=N_SPLIT, desc=f"  > 건물 {bno} 1차 CV"):
        x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        scaler_fold = clone(scaler)
        x_train_scaled = scaler_fold.fit_transform(x_train)
        x_val_scaled = scaler_fold.transform(x_val)

        xgb_model = clone(pretrained_models[btype]['xgb'])
        lgb_model = clone(pretrained_models[btype]['lgb'])
        cat_model = clone(pretrained_models[btype]['cat'])

        xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)
        lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

        oof_train_lvl1 = np.vstack([xgb_model.predict(x_train_scaled), lgb_model.predict(x_train_scaled), cat_model.predict(x_train_scaled)]).T
        oof_val_lvl1 = np.vstack([xgb_model.predict(x_val_scaled), lgb_model.predict(x_val_scaled), cat_model.predict(x_val_scaled)]).T
        oof_test_lvl1 = np.vstack([xgb_model.predict(x_test_scaled), lgb_model.predict(x_test_scaled), cat_model.predict(x_test_scaled)]).T

        meta_model = RidgeCV()
        meta_model.fit(oof_train_lvl1, y_train)
        val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
        test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

        final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
        final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)
        
        val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
        smape_folds.append(smape_score(np.expm1(y_val), np.expm1(val_final)))
        test_pred_folds += final_model.predict(test_pred_lvl2.reshape(-1, 1)) / kf.n_splits
    
    initial_smape = np.mean(smape_folds)
    initial_smapes[bno] = initial_smape
    initial_test_preds[bno] = test_pred_folds
    print(f"  >> 1차 예측 SMAPE: {initial_smape:.6f}")

avg_initial_smape = np.mean(list(initial_smapes.values()))
print(f"\n[4] 1차 예측 완료. 모든 건물의 평균 SMAPE: {avg_initial_smape:.6f}")

print("[5] sMAPE가 2.0을 초과하는 건물 선별 및 재학습 시작")
high_smape_buildings = [bno for bno, smape_val in initial_smapes.items() if smape_val > 2.0]
final_preds_list = []

for bno in train['건물번호'].unique():
    btype = train[train['건물번호'] == bno]['건물유형'].iloc[0]
    
    if bno in high_smape_buildings:
        print(f"\n  >> [BUILDING {bno}] SMAPE({initial_smapes[bno]:.4f}) > 2.0. 재학습 시작...")
        
        train_b = train[train['건물번호'] == bno].copy()
        test_b = test[test['건물번호'] == bno].copy()
        x = train_b[features].reset_index(drop=True)
        y = np.log1p(train_b[target].reset_index(drop=True))
        x_test_final = test_b[features].reset_index(drop=True)

        re_tuned_models = {}
        scaler_bno = StandardScaler()
        scaler_bno.fit(x)

        # 재학습 모델을 위한 Optuna 하이퍼파라미터 튜닝
        for model_name in ['xgb', 'lgb', 'cat']:
            print(f"  >> 건물 {bno}에 대한 '{model_name}' 모델 하이퍼파라미터 재튜닝...")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: optuna_objective_retrain(trial, model_name, x, y, kf, scaler_bno), n_trials=N_TRIALS, show_progress_bar=True)
            
            best_params = study.best_params
            best_smape = study.best_value
            print(f"  >> 재튜닝 최적 SMAPE: {best_smape:.4f}, 최적 파라미터: {best_params}\n")

            if model_name == 'xgb':
                best_model = XGBRegressor(**best_params, random_state=SEED)
            elif model_name == 'lgb':
                best_model = LGBMRegressor(**best_params, random_state=SEED, verbose=-1)
            elif model_name == 'cat':
                best_model = CatBoostRegressor(**best_params, random_seed=SEED, verbose=0)
            
            re_tuned_models[model_name] = best_model
        
        # 재튜닝된 모델로 다시 예측 수행
        test_pred_folds_retrain = np.zeros(len(x_test_final))
        for fold, (tr_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=N_SPLIT, desc=f"  > 건물 {bno} 재학습 모델 예측"):
            x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            x_train_scaled = scaler_bno.fit_transform(x_train)
            x_val_scaled = scaler_bno.transform(x_val)
            x_test_scaled = scaler_bno.transform(x_test_final)

            xgb_model = re_tuned_models['xgb']
            lgb_model = re_tuned_models['lgb']
            cat_model = re_tuned_models['cat']

            xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],verbose=False)
            lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
            cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

            oof_train_lvl1 = np.vstack([xgb_model.predict(x_train_scaled), lgb_model.predict(x_train_scaled), cat_model.predict(x_train_scaled)]).T
            oof_val_lvl1 = np.vstack([xgb_model.predict(x_val_scaled), lgb_model.predict(x_val_scaled), cat_model.predict(x_val_scaled)]).T
            oof_test_lvl1 = np.vstack([xgb_model.predict(x_test_scaled), lgb_model.predict(x_test_scaled), cat_model.predict(x_test_scaled)]).T

            meta_model = RidgeCV()
            meta_model.fit(oof_train_lvl1, y_train)
            val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
            test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

            final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
            final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)
            
            test_pred_folds_retrain += final_model.predict(test_pred_lvl2.reshape(-1, 1)) / kf.n_splits
        
        final_preds_list.extend(test_pred_folds_retrain)

    else:
        print(f"  >> [BUILDING {bno}] SMAPE({initial_smapes[bno]:.4f}) <= 2.0. 재학습 없이 진행.")
        final_preds_list.extend(initial_test_preds[bno])


print(f"\n[6] 최종 예측 완료. 최종 평균 SMAPE : {np.mean(list(initial_smapes.values())):.6f}")
print("  >> 최종 예측 결과 저장")

samplesub['answer'] = np.expm1(final_preds_list)
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{np.mean(list(initial_smapes.values())):.4f}".replace('.', '_')

filename = f"11_07_{today}_SMAPE_{score_str}_retrain_conditional_v2.csv"
samplesub.to_csv(save_path + filename, index=False)

with open("./Energy/11_submission/11_07_log.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"SMAPE (final): {np.mean(list(initial_smapes.values()))}\n")
    f.write("="*40 + "\n")

print(f"[7] 종료 ")