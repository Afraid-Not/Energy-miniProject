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

print("[14_01_hyperparameter_tuning] 시작")

seed_file = "./Energy/14_submission/(SEED_COUNT)14_01_hyperparameter_tuning.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 65 #seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

data_path = './Energy/'
save_path = './Energy/14_submission/'
train_save_path = './Energy/14_submission/train_csv/'
test_save_path = './Energy/14_submission/test_csv/'
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
train_call = '13_01_train_SEED65_up.csv'
test_call = '13_01_test_SEED65_up.csv'

train = pd.read_csv(train_save_path + train_call)
test = pd.read_csv(test_save_path + test_call)
samplesub = pd.read_csv(data_path +'sample_submission.csv')

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

# 일조/일사 3시간 변화량 파생 피처 생성
def add_solar_change_features(df):
    """일조와 일사의 이전 3시간 변화량 파생 피처를 추가하는 함수"""
    df = df.copy()
    
    # 데이터프레임을 건물번호와 시간순으로 정렬
    df = df.sort_values(['건물번호', '일시']).reset_index(drop=True)
    
    # 일조와 일사 컬럼이 있는지 확인
    solar_cols = []
    if '일조(hr)' in df.columns:
        solar_cols.append('일조(hr)')
    if '일사(MJ/m2)' in df.columns:
        solar_cols.append('일사(MJ/m2)')
    
    if not solar_cols:
        return df
    
    # 각 건물별로 그룹화하여 처리
    for building_id in df['건물번호'].unique():
        building_mask = df['건물번호'] == building_id
        building_data = df[building_mask].copy()
        
        for col in solar_cols:
            # 1시간, 2시간, 3시간 전 값들 계산
            shift_1 = building_data[col].shift(1)
            shift_2 = building_data[col].shift(2)
            shift_3 = building_data[col].shift(3)
            
            # 변화량 계산 (현재값 - 이전값)
            df.loc[building_mask, f'{col}_1h_변화량'] = building_data[col] - shift_1
            df.loc[building_mask, f'{col}_2h_변화량'] = building_data[col] - shift_2
            df.loc[building_mask, f'{col}_3h_변화량'] = building_data[col] - shift_3
            
            # 변화율 계산 (변화량/이전값 * 100)
            df.loc[building_mask, f'{col}_1h_변화율'] = np.where(
                shift_1 != 0, 
                ((building_data[col] - shift_1) / shift_1) * 100, 
                0
            )
            df.loc[building_mask, f'{col}_2h_변화율'] = np.where(
                shift_2 != 0, 
                ((building_data[col] - shift_2) / shift_2) * 100, 
                0
            )
            df.loc[building_mask, f'{col}_3h_변화율'] = np.where(
                shift_3 != 0, 
                ((building_data[col] - shift_3) / shift_3) * 100, 
                0
            )
            
            # 3시간 평균 대비 현재값 비율
            rolling_3h_mean = building_data[col].rolling(window=3, min_periods=1).mean()
            df.loc[building_mask, f'{col}_3h_평균대비'] = np.where(
                rolling_3h_mean != 0,
                (building_data[col] / rolling_3h_mean) * 100,
                100
            )
            
            # 3시간 최대값/최소값 대비 현재값 위치
            rolling_3h_max = building_data[col].rolling(window=3, min_periods=1).max()
            rolling_3h_min = building_data[col].rolling(window=3, min_periods=1).min()
            
            df.loc[building_mask, f'{col}_3h_최대대비'] = np.where(
                rolling_3h_max != 0,
                (building_data[col] / rolling_3h_max) * 100,
                100
            )
            
            # 정규화된 위치 (0~1 스케일)
            range_diff = rolling_3h_max - rolling_3h_min
            df.loc[building_mask, f'{col}_3h_위치'] = np.where(
                range_diff != 0,
                (building_data[col] - rolling_3h_min) / range_diff,
                0.5
            )
    
    # NaN 값을 0으로 채우기
    change_cols = [col for col in df.columns if any(x in col for x in ['변화량', '변화율', '평균대비', '최대대비', '위치'])]
    df[change_cols] = df[change_cols].fillna(0)
    
    return df

train = add_solar_change_features(train)
test = add_solar_change_features(test)

# 타겟 변수 설정
TARGET = '전력소비량(kWh)'

# 건물번호 Target Encoding 및 건물유형 One-Hot Encoding
def target_encode_with_cv(train_df, test_df, category_col, target_col, cv_folds=5, smoothing=10):
    """Cross-validation을 사용한 Target Encoding (리키지 방지)"""
    
    # 전체 평균값 계산
    global_mean = train_df[target_col].mean()
    
    # Train 데이터용 Target Encoding (CV 방식)
    train_encoded = np.zeros(len(train_df))
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        # Train fold에서 각 카테고리별 평균 계산
        fold_train = train_df.iloc[train_idx]
        category_means = fold_train.groupby(category_col)[target_col].agg(['mean', 'count'])
        
        # 스무딩 적용: (count * mean + smoothing * global_mean) / (count + smoothing)
        category_means['smoothed_mean'] = (
            (category_means['count'] * category_means['mean'] + smoothing * global_mean) /
            (category_means['count'] + smoothing)
        )
        
        # Validation fold에 적용
        for idx in val_idx:
            category_value = train_df.loc[idx, category_col]
            if category_value in category_means.index:
                train_encoded[idx] = category_means.loc[category_value, 'smoothed_mean']
            else:
                train_encoded[idx] = global_mean
    
    # Test 데이터용 Target Encoding (전체 Train 데이터 사용)
    category_means_full = train_df.groupby(category_col)[target_col].agg(['mean', 'count'])
    category_means_full['smoothed_mean'] = (
        (category_means_full['count'] * category_means_full['mean'] + smoothing * global_mean) /
        (category_means_full['count'] + smoothing)
    )
    
    test_encoded = np.zeros(len(test_df))
    for idx in range(len(test_df)):
        category_value = test_df.iloc[idx][category_col]
        if category_value in category_means_full.index:
            test_encoded[idx] = category_means_full.loc[category_value, 'smoothed_mean']
        else:
            test_encoded[idx] = global_mean
    
    return train_encoded, test_encoded, category_means_full

def one_hot_building_type(df):
    """건물유형 One-Hot Encoding"""
    building_type_dummies = pd.get_dummies(df['건물유형'], prefix='건물유형')
    df = pd.concat([df, building_type_dummies], axis=1)
    return df

# 건물번호 Target Encoding
train_building_encoded, test_building_encoded, building_stats = target_encode_with_cv(
    train, test, '건물번호', TARGET, cv_folds=5, smoothing=10
)

train['건물번호_encoded'] = train_building_encoded
test['건물번호_encoded'] = test_building_encoded

# 건물유형 One-Hot Encoding
train = one_hot_building_type(train)
test = one_hot_building_type(test)

# 기존 건물번호, 건물유형 컬럼 제거
train = train.drop(['건물번호', '건물유형'], axis=1)
test = test.drop(['건물번호', '건물유형'], axis=1)

# Target Encoding 결과 저장
encoding_results = {
    'building_stats': building_stats.to_dict(),
    'global_mean': float(train[TARGET].mean()),
    'encoding_column': '건물번호_encoded'
}

encoding_file = f"{save_path}building_target_encoding_SEED{SEED}.json"
with open(encoding_file, 'w') as f:
    json.dump(encoding_results, f, indent=4, default=str)

# 피처와 타겟 분리
feature_cols = [col for col in train.columns if col not in [TARGET, '일시']]
X_train = train[feature_cols]
y_train = train[TARGET]
X_test = test[feature_cols]

print(f"Feature 수: {len(feature_cols)}, Train shape: {X_train.shape}")

def objective(trial):
    """Optuna 목적 함수 - 앙상블 모델 최적화"""
    
    # XGBoost 파라미터
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 10),
        'random_state': SEED,
        'n_jobs': -1
    }
    
    # LightGBM 파라미터
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('lgbm_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 0, 10),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 100),
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # CatBoost 파라미터
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 100, 1000),
        'depth': trial.suggest_int('cat_depth', 3, 10),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('cat_border_count', 32, 255),
        'random_state': SEED,
        'verbose': False
    }
    
    # 앙상블 가중치
    xgb_weight = trial.suggest_float('xgb_weight', 0.1, 0.8)
    lgbm_weight = trial.suggest_float('lgbm_weight', 0.1, 0.8)
    cat_weight = 1.0 - xgb_weight - lgbm_weight
    
    # 가중치 합이 음수가 되지 않도록 조정
    if cat_weight < 0.1:
        return float('inf')
    
    # K-Fold 교차 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 모델 학습
        xgb_model = XGBRegressor(**xgb_params)
        lgbm_model = LGBMRegressor(**lgbm_params)
        cat_model = CatBoostRegressor(**cat_params)
        
        # 개별 모델 학습
        xgb_model.fit(X_tr, y_tr)
        lgbm_model.fit(X_tr, y_tr)
        cat_model.fit(X_tr, y_tr)
        
        # 예측
        xgb_pred = xgb_model.predict(X_val)
        lgbm_pred = lgbm_model.predict(X_val)
        cat_pred = cat_model.predict(X_val)
        
        # 앙상블 예측
        ensemble_pred = (xgb_weight * xgb_pred + 
                        lgbm_weight * lgbm_pred + 
                        cat_weight * cat_pred)
        
        # SMAPE 계산
        fold_score = smape(y_val, ensemble_pred)
        cv_scores.append(fold_score)
    
    return np.mean(cv_scores)

# Optuna 스터디 생성 및 최적화 실행
print("Optuna 최적화 시작...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"최적화 완료! 최적 SMAPE: {study.best_value:.4f}")

# 최적 파라미터 저장
best_params_file = f"{save_path}best_ensemble_params_SEED{SEED}.json"
with open(best_params_file, 'w') as f:
    json.dump({
        'best_value': study.best_value,
        'best_params': study.best_params,
        'seed': SEED
    }, f, indent=4)

# 최적 파라미터로 모델 분리
best_params = study.best_params

# XGBoost 파라미터 추출
xgb_best_params = {
    'n_estimators': best_params['xgb_n_estimators'],
    'max_depth': best_params['xgb_max_depth'],
    'learning_rate': best_params['xgb_learning_rate'],
    'subsample': best_params['xgb_subsample'],
    'colsample_bytree': best_params['xgb_colsample_bytree'],
    'reg_alpha': best_params['xgb_reg_alpha'],
    'reg_lambda': best_params['xgb_reg_lambda'],
    'random_state': SEED,
    'n_jobs': -1
}

# LightGBM 파라미터 추출
lgbm_best_params = {
    'n_estimators': best_params['lgbm_n_estimators'],
    'max_depth': best_params['lgbm_max_depth'],
    'learning_rate': best_params['lgbm_learning_rate'],
    'subsample': best_params['lgbm_subsample'],
    'colsample_bytree': best_params['lgbm_colsample_bytree'],
    'reg_alpha': best_params['lgbm_reg_alpha'],
    'reg_lambda': best_params['lgbm_reg_lambda'],
    'num_leaves': best_params['lgbm_num_leaves'],
    'min_child_samples': best_params['lgbm_min_child_samples'],
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

# CatBoost 파라미터 추출
cat_best_params = {
    'iterations': best_params['cat_iterations'],
    'depth': best_params['cat_depth'],
    'learning_rate': best_params['cat_learning_rate'],
    'l2_leaf_reg': best_params['cat_l2_leaf_reg'],
    'border_count': best_params['cat_border_count'],
    'random_state': SEED,
    'verbose': False
}

# 최적 가중치 추출
best_xgb_weight = best_params['xgb_weight']
best_lgbm_weight = best_params['lgbm_weight']
best_cat_weight = 1.0 - best_xgb_weight - best_lgbm_weight

print(f"최적 앙상블 가중치 - XGB: {best_xgb_weight:.3f}, LGBM: {best_lgbm_weight:.3f}, CAT: {best_cat_weight:.3f}")
print("[14_01_hyperparameter_tuning] 완료")