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
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Optuna 로그 레벨 설정 (WARNING 이상만 표시)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("[02_model] 시작")

seed_file = "./Energy/01/(SEED_COUNT)13_02_model.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 56 # seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

print(f"[1] 데이터 로드 및 전처리 (생략 - 기존 코드에서 이미 수행)")
data_path = './Energy/'
csv_path = './Energy/01/'
trainer = './Energy/01/trainer/'
save_path = f'./Energy/01/{SEED}_submission/'

# Preprocessed data 로드
train_call = '06_train_42_optuna.csv'
test_call = '06_test_42_optuna.csv'

train = pd.read_csv(trainer + train_call)
test = pd.read_csv(trainer + test_call)
print(train.shape)
print(test.shape)
exit()
samplesub = pd.read_csv(data_path +'sample_submission.csv')
os.makedirs(save_path, exist_ok=True)
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

def one_hot_building_id(df):
    temp = pd.get_dummies(df['건물번호'], prefix='건물')
    df = pd.concat([df, temp], axis=1)
    return df

train = one_hot_building_id(train)
test = one_hot_building_id(test)

exclude_cols = ['건물번호', '일시', '전력소비량(kWh)', '건물유형', '날짜']
features = [col for col in train.columns if col not in exclude_cols]
target = '전력소비량(kWh)'

N_SPLIT = 5
KFOLD = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

def sample_data_for_tuning(X, y, sample_ratio=0.2, seed_val=42):
    """데이터의 일부를 샘플링하여 튜닝 속도 향상"""
    if len(X) <= 100:  # 데이터가 너무 작으면 전체 사용
        return X, y
    
    sample_size = max(100, int(len(X) * sample_ratio))  # 최소 100개는 유지
    sample_indices = np.random.RandomState(seed_val).choice(len(X), size=sample_size, replace=False)
    
    if isinstance(X, pd.DataFrame):
        return X.iloc[sample_indices].reset_index(drop=True), y.iloc[sample_indices].reset_index(drop=True)
    else:
        return X[sample_indices], y[sample_indices]

def tune_xgb_with_optuna(X_train, y_train, seed_val, n_trials=20):
    """XGBoost 하이퍼파라미터 튜닝"""
    # 30% 샘플링
    X_sample, y_sample = sample_data_for_tuning(X_train, y_train, seed_val=seed_val)
    
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': seed_val,
            'n_jobs': -1
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=seed_val)
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X_sample, y_sample):
            X_train_fold = X_sample.iloc[train_idx] if isinstance(X_sample, pd.DataFrame) else X_sample[train_idx]
            X_val_fold = X_sample.iloc[val_idx] if isinstance(X_sample, pd.DataFrame) else X_sample[val_idx]
            y_train_fold = y_sample.iloc[train_idx] if isinstance(y_sample, pd.Series) else y_sample[train_idx]
            y_val_fold = y_sample.iloc[val_idx] if isinstance(y_sample, pd.Series) else y_sample[val_idx]
            
            model = XGBRegressor(**param, verbosity=0)  # XGBoost 로그 끄기
            model.fit(X_train_fold, y_train_fold)
            pred = model.predict(X_val_fold)
            mae_scores.append(mean_absolute_error(y_val_fold, pred))
        
        return np.mean(mae_scores)
    
    print("      > XGBoost 튜닝 중...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    
    # Progress bar와 함께 최적화
    with tqdm(total=n_trials, desc="      XGBoost Optuna", leave=False) as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best MAE': f'{study.best_value:.6f}'})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    print(f"      > XGBoost 최적 MAE: {study.best_value:.6f}")
    return study.best_params

def tune_lgb_with_optuna(X_train, y_train, seed_val, n_trials=20):
    """LightGBM 하이퍼파라미터 튜닝"""
    # 30% 샘플링
    X_sample, y_sample = sample_data_for_tuning(X_train, y_train, seed_val=seed_val)
    
    def objective(trial):
        param = {
            'objective': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
            'random_state': seed_val,
            'n_jobs': -1,
            'verbose': -1
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=seed_val)
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X_sample, y_sample):
            X_train_fold = X_sample.iloc[train_idx] if isinstance(X_sample, pd.DataFrame) else X_sample[train_idx]
            X_val_fold = X_sample.iloc[val_idx] if isinstance(X_sample, pd.DataFrame) else X_sample[val_idx]
            y_train_fold = y_sample.iloc[train_idx] if isinstance(y_sample, pd.Series) else y_sample[train_idx]
            y_val_fold = y_sample.iloc[val_idx] if isinstance(y_sample, pd.Series) else y_sample[val_idx]
            
            model = LGBMRegressor(**param)
            model.fit(X_train_fold, y_train_fold)
            pred = model.predict(X_val_fold)
            mae_scores.append(mean_absolute_error(y_val_fold, pred))
        
        return np.mean(mae_scores)
    
    print("      > LightGBM 튜닝 중...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    
    # Progress bar와 함께 최적화
    with tqdm(total=n_trials, desc="      LightGBM Optuna", leave=False) as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best MAE': f'{study.best_value:.6f}'})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    print(f"      > LightGBM 최적 MAE: {study.best_value:.6f}")
    return study.best_params

def peak_model_with_optuna(train_df, test_df, building_type, current_features, current_target, seed_val):
    """피크 모델에 Optuna 적용"""
    peak_hours_apartment_hotel = [18, 19, 20, 21, 22]
    peak_hours_other = [12, 13, 14, 15, 16]

    train_peak_idx_bool = train_df['시각'].isin(peak_hours_apartment_hotel) if building_type in ['아파트', '호텔'] else train_df['시각'].isin(peak_hours_other)
    test_peak_idx_bool = test_df['시각'].isin(peak_hours_apartment_hotel) if building_type in ['아파트', '호텔'] else test_df['시각'].isin(peak_hours_other)

    X_train_peak_subset = train_df[train_peak_idx_bool][current_features].copy()
    y_train_peak_subset = np.log1p(train_df[train_peak_idx_bool][current_target]).copy()
    X_test_peak_subset = test_df[test_peak_idx_bool][current_features].copy()

    if len(X_train_peak_subset) == 0 or len(X_test_peak_subset) == 0:
        return None, None

    # Optuna로 LightGBM 튜닝
    print("      > 피크 모델 Optuna 튜닝 중...")
    best_params = tune_lgb_with_optuna(X_train_peak_subset, y_train_peak_subset, seed_val, n_trials=15)
    
    scaler_peak = StandardScaler()
    X_train_peak_scaled = scaler_peak.fit_transform(X_train_peak_subset)
    X_test_peak_scaled = scaler_peak.transform(X_test_peak_subset)

    peak_model_lgbm = LGBMRegressor(**best_params, random_state=seed_val, n_jobs=-1, verbose=-1)

    oof_preds_for_peak_subset = np.zeros(len(X_train_peak_subset))
    test_preds_for_peak_subset = np.zeros(len(X_test_peak_subset))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_peak_scaled, y_train_peak_subset)):
        X_train_fold, X_val_fold = X_train_peak_scaled[train_idx], X_train_peak_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_peak_subset.iloc[train_idx], y_train_peak_subset.iloc[val_idx]

        peak_model_lgbm.fit(X_train_fold, y_train_fold)
        oof_preds_for_peak_subset[val_idx] = peak_model_lgbm.predict(X_val_fold)
        test_preds_for_peak_subset += peak_model_lgbm.predict(X_test_peak_scaled) / N_SPLIT
    
    return oof_preds_for_peak_subset, test_preds_for_peak_subset

def tune_ensemble_weights_with_optuna(oof_preds_list, y_true, seed_val, n_trials=50):
    """Optuna를 사용하여 앙상블 가중치 최적화"""
    def objective(trial):
        # 가중치 생성 (합이 1이 되도록)
        weights = []
        for i in range(len(oof_preds_list)):
            if i == len(oof_preds_list) - 1:
                # 마지막 가중치는 1에서 나머지를 빼서 합이 1이 되도록
                weights.append(1 - sum(weights))
            else:
                weights.append(trial.suggest_float(f'weight_{i}', 0.0, 1.0))
        
        # 가중치 정규화 (음수 방지)
        weights = np.array(weights)
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        
        # 가중 평균 계산
        ensemble_pred = np.zeros_like(y_true)
        for i, weight in enumerate(weights):
            ensemble_pred += weight * oof_preds_list[i]
        
        # SMAPE 계산 (최소화 목표)
        return smape(np.expm1(y_true), np.expm1(ensemble_pred))
    
    print("      > 앙상블 가중치 최적화 중...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    
    # Progress bar와 함께 최적화
    with tqdm(total=n_trials, desc="      Ensemble Weights", leave=False) as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best SMAPE': f'{study.best_value:.6f}'})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    # 최적 가중치 추출 및 정규화
    best_weights = []
    for i in range(len(oof_preds_list)):
        if i == len(oof_preds_list) - 1:
            best_weights.append(1 - sum(best_weights))
        else:
            best_weights.append(study.best_params[f'weight_{i}'])
    
    best_weights = np.array(best_weights)
    best_weights = np.abs(best_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    print(f"      > 최적 앙상블 가중치: XGB={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}")
    print(f"      > 최적 SMAPE: {study.best_value:.6f}")
    
    return best_weights

def tune_and_predict_with_optuna_ensemble(train_df_subset, test_df_subset, 
                                        current_features, current_target, seed_val, n_trials=15):
    """
    XGB, LGBM 앙상블을 Optuna로 튜닝하고 최적 가중평균을 찾아 예측
    """
    X_train_full = train_df_subset[current_features].reset_index(drop=True)
    y_train_full = np.log1p(train_df_subset[current_target]).reset_index(drop=True)
    X_test_full = test_df_subset[current_features].reset_index(drop=True)

    if len(X_train_full) == 0 or len(X_test_full) == 0:
        return None, None, float('inf')

    # 각 모델 Optuna 튜닝 (30% 샘플링)
    print("    > 재학습용 앙상블 Optuna 튜닝 중 (30% 샘플링)...")
    xgb_best_params = tune_xgb_with_optuna(X_train_full, y_train_full, seed_val, n_trials)
    lgb_best_params = tune_lgb_with_optuna(X_train_full, y_train_full, seed_val, n_trials)

    # 각 모델의 OOF 예측값을 저장할 배열
    oof_preds_xgb = np.zeros(len(X_train_full))
    oof_preds_lgb = np.zeros(len(X_train_full))
    
    test_preds_xgb = np.zeros(len(X_test_full))
    test_preds_lgb = np.zeros(len(X_test_full))

    # K-Fold로 각 모델 학습 및 예측
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    print("    > 앙상블 모델 학습 중...")
    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full_scaled, y_train_full)):
        X_train_fold, X_val_fold = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # XGBoost
        xgb_model = XGBRegressor(**xgb_best_params, random_state=seed_val, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_train_fold, y_train_fold)
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val_fold)
        test_preds_xgb += xgb_model.predict(X_test_full_scaled) / N_SPLIT

        # LightGBM
        lgb_model = LGBMRegressor(**lgb_best_params, random_state=seed_val, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train_fold, y_train_fold)
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
        test_preds_lgb += lgb_model.predict(X_test_full_scaled) / N_SPLIT

    # Optuna로 최적 가중치 찾기
    oof_preds_list = [oof_preds_xgb, oof_preds_lgb]
    test_preds_list = [test_preds_xgb, test_preds_lgb]
    
    best_weights = tune_ensemble_weights_with_optuna(oof_preds_list, y_train_full, seed_val, n_trials=30)
    
    # 최적 가중치로 최종 예측
    oof_final_preds = np.zeros(len(X_train_full))
    test_final_preds = np.zeros(len(X_test_full))
    
    for i, weight in enumerate(best_weights):
        oof_final_preds += weight * oof_preds_list[i]
        test_final_preds += weight * test_preds_list[i]

    final_smape = smape(np.expm1(y_train_full), np.expm1(oof_final_preds))
    print(f"    > Optuna 앙상블 최종 SMAPE: {final_smape:.6f}")

    return oof_final_preds, test_final_preds, final_smape

def train_and_predict_subset_with_optuna(train_subset, test_subset, name="Subset"):
    """모든 모델에 Optuna 적용된 훈련 및 예측 함수"""
    X_train_full = train_subset[features].reset_index(drop=True)
    y_train_full = np.log1p(train_subset[target].reset_index(drop=True))
    X_test = test_subset[features].reset_index(drop=True)

    print(f"    > {name} - Optuna 하이퍼파라미터 튜닝 시작 (30% 샘플링)")
    
    # 각 모델에 대해 Optuna 튜닝 (30% 샘플링)
    print(f"    > XGBoost 튜닝 중...")
    xgb_best_params = tune_xgb_with_optuna(X_train_full, y_train_full, SEED, n_trials=15)
    
    print(f"    > LightGBM 튜닝 중...")
    lgb_best_params = tune_lgb_with_optuna(X_train_full, y_train_full, SEED, n_trials=15)

    oof_preds_lvl1 = np.zeros((len(X_train_full), 2))
    test_preds_lvl1 = np.zeros((len(X_test), 2))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 튜닝된 파라미터로 모델 생성
        xgb_model = XGBRegressor(**xgb_best_params, random_state=SEED, n_jobs=-1)
        lgb_model = LGBMRegressor(**lgb_best_params, random_state=SEED, n_jobs=-1, verbose=-1)
        
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)

        oof_preds_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)

        test_preds_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
    
    meta_model = RidgeCV()
    meta_model.fit(oof_preds_lvl1, y_train_full)

    oof_pred_final = meta_model.predict(oof_preds_lvl1)
    test_pred_final = meta_model.predict(test_preds_lvl1)

    # Peak Model Integration (Optuna 적용)
    building_type_for_peak = train_subset['건물유형'].iloc[0] if not train_subset.empty else None

    oof_pred_final_combined = np.array(oof_pred_final)
    test_pred_final_combined = np.array(test_pred_final)

    if building_type_for_peak:
        peak_oof_pred_subset, peak_test_pred_subset = peak_model_with_optuna(
            train_subset, test_subset, building_type_for_peak, features, target, SEED)

        if peak_oof_pred_subset is not None and peak_test_pred_subset is not None:
            train_peak_idx_bool = train_subset['시각'].isin([18, 19, 20, 21, 22]) if building_type_for_peak in ['아파트', '호텔'] else train_subset['시각'].isin([12, 13, 14, 15, 16])
            test_peak_idx_bool = test_subset['시각'].isin([18, 19, 20, 21, 22]) if building_type_for_peak in ['아파트', '호텔'] else test_subset['시각'].isin([12, 13, 14, 15, 16])

            oof_pred_final_combined[train_peak_idx_bool.values] = peak_oof_pred_subset
            test_pred_final_combined[test_peak_idx_bool.values] = peak_test_pred_subset
    
    current_smape = smape(np.expm1(y_train_full), np.expm1(oof_pred_final_combined))
    print(f"    > {name} SMAPE (Optuna 적용): {current_smape:.6f}")

    return oof_pred_final_combined, test_pred_final_combined, current_smape

# --- 1. 유형별 사전학습 및 예측 (Optuna 적용) ---
print("\n--- [Step 1] 건물유형별 사전학습 및 예측 시작 (Optuna 적용) ---")

type_wise_oof_preds = np.zeros(len(train))
type_wise_test_preds = np.zeros(len(test))
type_wise_smapes = []
type_wise_results = {}

building_types = train['건물유형'].unique()

for btype in building_types:
    print(f"   > 건물유형: {btype} 모델 학습 시작 (KFold 교차 검증 + Optuna 적용)")
    train_bt = train[train['건물유형'] == btype].copy()
    test_bt = test[test['건물유형'] == btype].copy()

    if len(train_bt) == 0 or len(test_bt) == 0:
        continue

    oof_pred, test_pred, current_smape = train_and_predict_subset_with_optuna(
        train_bt, test_bt, name=f"Type {btype}")
    
    type_wise_smapes.append(current_smape)
    type_wise_oof_preds[train_bt.index] = oof_pred
    type_wise_test_preds[test_bt.index] = test_pred
    type_wise_results[btype] = {"smape": current_smape, "train_indices": train_bt.index, "test_indices": test_bt.index}

print("--- [Step 1] 건물유형별 사전학습 및 예측 완료 ---")
t_w_s = np.mean(type_wise_smapes)
print(f"--- 평균 SMAPE (유형별): {t_w_s:.6f} ---")

# Conditional Re-training for Type-wise models
print("\n--- [Step 1.5] 건물유형별 조건부 재학습 시작 ---")
re_trained_type_wise_count = 0
for btype, result in type_wise_results.items():
    if result["smape"] > t_w_s:
        print(f"    > 재학습 필요: 건물유형 {btype} (SMAPE: {result['smape']:.6f} > 평균: {t_w_s:.6f})")
        train_bt = train.loc[result["train_indices"]].copy()
        test_bt = test.loc[result["test_indices"]].copy()

        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna_ensemble(
            train_bt, test_bt, features, target, SEED, n_trials=15
        )
        
        if oof_pred_re is not None:
            type_wise_oof_preds[train_bt.index] = oof_pred_re
            type_wise_test_preds[test_bt.index] = test_pred_re
            re_trained_type_wise_count += 1
            print(f"    > Type {btype} Re-train (Optuna) SMAPE: {current_re_smape:.6f}")
        else:
            print(f"    > Type {btype} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물유형별 조건부 재학습 완료 ({re_trained_type_wise_count}개 유형 재학습) ---")

y_true_all_train = np.log1p(train[target])
final_type_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(type_wise_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (유형별): {final_type_wise_smape_after_retrain:.6f} ---")

# --- 2. 건물별 사전학습 및 예측 (Optuna 적용) ---
print("\n--- [Step 2] 건물별 사전학습 및 예측 시작 (Optuna 적용) ---")

building_wise_oof_preds = np.zeros(len(train))
building_wise_test_preds = np.zeros(len(test))
building_wise_smapes = []
building_wise_results = {}

building_ids = train['건물번호'].unique()

for bno in building_ids:
    print(f"   > 건물번호: {bno} 모델 학습 시작 (KFold 교차 검증 + Optuna 적용)")
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(train_b) == 0 or len(test_b) == 0:
        continue

    oof_pred, test_pred, current_smape = train_and_predict_subset_with_optuna(
        train_b, test_b, name=f"Building {bno}")
    
    building_wise_smapes.append(current_smape)
    building_wise_oof_preds[train_b.index] = oof_pred
    building_wise_test_preds[test_b.index] = test_pred
    building_wise_results[bno] = {"smape": current_smape, "train_indices": train_b.index, "test_indices": test_b.index}

print("--- [Step 2] 건물별 사전학습 및 예측 완료 ---")
b_w_s = np.mean(building_wise_smapes)
print(f"--- 평균 SMAPE (건물별): {b_w_s:.6f} ---")

print("\n--- [Step 2.5] 건물별 조건부 재학습 시작 ---")
re_trained_building_wise_count = 0
for bno, result in building_wise_results.items():
    if result["smape"] > b_w_s:
        print(f"    > 재학습 필요: 건물번호 {bno} (SMAPE: {result['smape']:.6f} > 평균: {b_w_s:.6f})")
        train_b = train.loc[result["train_indices"]].copy()
        test_b = test.loc[result["test_indices"]].copy()

        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna_ensemble(
            train_b, test_b, features, target, SEED, n_trials=15
        )
        
        if oof_pred_re is not None:
            building_wise_oof_preds[train_b.index] = oof_pred_re
            building_wise_test_preds[test_b.index] = test_pred_re
            re_trained_building_wise_count += 1
            print(f"    > Building {bno} Re-train (Optuna) SMAPE: {current_re_smape:.6f}")
        else:
            print(f"    > Building {bno} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물별 조건부 재학습 완료 ({re_trained_building_wise_count}개 건물 재학습) ---")

y_true_all_train = np.log1p(train[target])
final_building_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(building_wise_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (건물별): {final_building_wise_smape_after_retrain:.6f} ---")

building_groups = [
    [28],                                   # 강릉
    [72],                                   # 경주시
    [19,58,75,91],                          # 광주
    [77],                                   # 부안
    [24],                                   # 구미
    [61,74,81],                             # 김해시
    [32,42,65,79,99],                       # 대구
    [11,12,13,41,68,83,88],                 # 대전
    [20,26,44,45,70,100],                   # 부산
    [1,2,3,4,5,6,7,8,27,33,34,35,37,47,67,86,96], # 서울
    [71],                                   # 세종
    [54,84],                                # 속초
    [17,18,29,30,31,40,43,48,49,51,52,53,60,63,64,76,78], # 수원
    [66],                                   # 안동
    [85],                                   # 양산시
    [55,82],                                # 울산
    [15,16,39,59,73,92],                    # 인천
    [80,87],                                # 임실
    [89,90],                                # 전주
    [98],                                   # 제천
    [50],                                   # 진주
    [21,22,23],                             # 창원
    [46,93,94,95],                          # 천안
    [14,69],                                # 청주
    [57],                                   # 춘천
    [97],                                   # 충주
    [36,38,56],                             # 파주
    [25,62],                                # 포항
    [9,10],                                 # 홍천 
]

# --- 3. 건물 그룹별 사전학습 및 예측 (Optuna 적용) ---
print("\n--- [Step 3] 건물 그룹별 예측 시작 (Optuna 적용) ---")

group_oof_preds = np.zeros(len(train))
group_test_preds = np.zeros(len(test))
group_smapes = []
group_results = {}

for group_idx, group in enumerate(building_groups):
    print(f"   > 그룹 {group_idx+1}: 건물들 {group} 모델 학습 시작 (Optuna 적용)")

    train_g = train[train['건물번호'].isin(group)].copy()
    test_g = test[test['건물번호'].isin(group)].copy()

    if len(train_g) < 10 or len(test_g) < 1:
        print(f"   > 그룹 {group_idx+1} (건물: {group}) 데이터 부족으로 건너뜀.")
        continue

    oof_pred, test_pred, current_smape = train_and_predict_subset_with_optuna(
        train_g, test_g, name=f"Group {group_idx+1}")
    
    group_smapes.append(current_smape)
    group_oof_preds[train_g.index] = oof_pred
    group_test_preds[test_g.index] = test_pred
    group_results[group_idx] = {"smape": current_smape, "train_indices": train_g.index, "test_indices": test_g.index, "buildings": group}

print("--- [Step 3] 건물 그룹별 예측 완료 ---")
g_w_s = np.mean(group_smapes)
print(f"--- 평균 SMAPE (그룹별): {g_w_s:.6f} ---")

# Conditional Re-training for Group-wise models
print("\n--- [Step 3.5] 건물 그룹별 조건부 재학습 시작 ---")
re_trained_group_wise_count = 0
for group_idx, result in group_results.items():
    if result["smape"] > g_w_s:
        print(f"    > 재학습 필요: 그룹 {group_idx+1} (건물: {result['buildings']}) (SMAPE: {result['smape']:.6f} > 평균: {g_w_s:.6f})")
        train_g = train.loc[result["train_indices"]].copy()
        test_g = test.loc[result["test_indices"]].copy()

        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna_ensemble(
            train_g, test_g, features, target, SEED, n_trials=15
        )
        
        if oof_pred_re is not None:
            group_oof_preds[train_g.index] = oof_pred_re
            group_test_preds[test_g.index] = test_pred_re
            re_trained_group_wise_count += 1
            print(f"    > Group {group_idx+1} Re-train (Optuna) SMAPE: {current_re_smape:.6f}")
        else:
            print(f"    > Group {group_idx+1} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물 그룹별 조건부 재학습 완료 ({re_trained_group_wise_count}개 그룹 재학습) ---")

y_true_all_train = np.log1p(train[target])
final_group_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(group_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (그룹별): {final_group_wise_smape_after_retrain:.6f} ---")

# --- 4. 잔차 학습 및 최종 예측 (부스팅 앙상블 + Optuna 적용) ---
print("\n--- [Step 4] 잔차 학습 및 최종 예측 시작 (부스팅 앙상블 + Optuna 적용) ---")

# 예측 평균
ensemble_oof_preds = (type_wise_oof_preds + building_wise_oof_preds + group_oof_preds) / 3
ensemble_test_preds = (type_wise_test_preds + building_wise_test_preds + group_test_preds) / 3

# 잔차 계산
y_true_log = np.log1p(train[target])
residuals = y_true_log - ensemble_oof_preds

# 잔차 모델을 위한 각 부스팅 모델 Optuna 튜닝
print("    > 잔차 XGBoost 튜닝...")
xgb_residual_params = tune_xgb_with_optuna(train[features], pd.Series(residuals), SEED, n_trials=20)
print("    > 잔차 LightGBM 튜닝...")
lgb_residual_params = tune_lgb_with_optuna(train[features], pd.Series(residuals), SEED, n_trials=20)

# 잔차 모델 학습을 위한 피처 스케일링
scaler_res = StandardScaler()
X_train_scaled_res = scaler_res.fit_transform(train[features])
X_test_scaled_res = scaler_res.transform(test[features])

# 잔차 예측을 위한 각 모델의 OOF 및 테스트 예측
oof_residual_xgb = np.zeros(len(train))
oof_residual_lgb = np.zeros(len(train))

test_residual_xgb = np.zeros(len(test))
test_residual_lgb = np.zeros(len(test))

# K-Fold로 잔차 모델들 학습
print("    > 잔차 앙상블 모델 학습 중...")
for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_scaled_res, residuals)):
    X_train_fold, X_val_fold = X_train_scaled_res[train_idx], X_train_scaled_res[val_idx]
    y_train_fold, y_val_fold = residuals[train_idx], residuals[val_idx]

    # XGBoost 잔차 모델
    xgb_res_model = XGBRegressor(**xgb_residual_params, random_state=SEED, n_jobs=-1)
    xgb_res_model.fit(X_train_fold, y_train_fold)
    oof_residual_xgb[val_idx] = xgb_res_model.predict(X_val_fold)
    test_residual_xgb += xgb_res_model.predict(X_test_scaled_res) / N_SPLIT

    # LightGBM 잔차 모델
    lgb_res_model = LGBMRegressor(**lgb_residual_params, random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_res_model.fit(X_train_fold, y_train_fold)
    oof_residual_lgb[val_idx] = lgb_res_model.predict(X_val_fold)
    test_residual_lgb += lgb_res_model.predict(X_test_scaled_res) / N_SPLIT

# 잔차 예측을 위한 최적 가중치 찾기
print("    > 잔차 앙상블 가중치 최적화 중...")
residual_oof_list = [oof_residual_xgb, oof_residual_lgb]
residual_test_list = [test_residual_xgb, test_residual_lgb]

residual_best_weights = tune_ensemble_weights_with_optuna(
    residual_oof_list, pd.Series(residuals), SEED, n_trials=30
)

# 최적 가중치로 잔차 예측 결합
predicted_residuals_oof = np.zeros(len(train))
predicted_residuals_test = np.zeros(len(test))

for i, weight in enumerate(residual_best_weights):
    predicted_residuals_oof += weight * residual_oof_list[i]
    predicted_residuals_test += weight * residual_test_list[i]

# 최종 예측: 앙상블 예측 + 예측된 잔차
final_preds_residual = ensemble_test_preds + predicted_residuals_test

# 예측값이 음수가 되지 않도록 하고 원래 스케일로 변환
final_predictions_exp = np.expm1(final_preds_residual)
final_predictions_exp[final_predictions_exp < 0] = 0

# 최종 OOF SMAPE 계산 (모델 성능 평가용)
final_oof_combined_preds = ensemble_oof_preds + predicted_residuals_oof
final_oof_smape = smape(np.expm1(y_true_log), np.expm1(final_oof_combined_preds))

print(f"--- [Step 4] 잔차 학습 및 최종 예측 완료 ---")
print(f"최종 SMAPE (부스팅 앙상블 잔차 모델링 후): {final_oof_smape:.6f}")

print("[3] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions_exp
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_oof_smape:.4f}".replace('.', '_')

filename = f"02_ENSEMBLE_OPTUNA_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

with open(f"./Energy/01/{SEED}_submission/(LOG)model_ensemble_optuna.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"test file name : {test_call}\n")
    f.write(f"부스팅 앙상블 + Optuna 적용 - 모든 모델 30% 샘플링으로 튜닝\n")
    f.write(f"재학습: XGB+LGBM 앙상블 + Optuna 가중평균 최적화\n")
    f.write(f"잔차학습: XGB+LGBM 앙상블 + Optuna 가중평균 최적화\n")
    f.write(f"건물 유형별 학습 SMAPE : {final_type_wise_smape_after_retrain}\n")
    f.write(f"건물 번호별 학습 SMAPE : {final_building_wise_smape_after_retrain}\n")
    f.write(f"건물 그룹별 학습 SMAPE : {final_group_wise_smape_after_retrain}\n") 
    f.write(f"부스팅 앙상블 잔차학습 SMAPE : {final_oof_smape}\n")
    f.write("="*40 + "\n")

print("[02_model] 종료")