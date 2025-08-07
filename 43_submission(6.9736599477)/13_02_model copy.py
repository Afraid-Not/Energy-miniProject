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
# Optuna 로그 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("[02 - model] 시작")

seed_file = "./Energy/02/(SEED_COUNT)02_model.json"

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

print(f"[1] 데이터 로드 및 전처리 (생략 - 기존 코드에서 이미 수행)")
data_path = './Energy/'
csv_path = './Energy/02/'
save_path = f'./Energy/02/{SEED}_submission/'
os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
train = pd.read_csv(csv_path + '06_train_47_2.csv')
test_call = '06_test_47_2.csv'

test = pd.read_csv(csv_path + test_call)
samplesub = pd.read_csv(data_path +'sample_submission.csv')

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

def peak_model(train_df, test_df, building_type, current_features, current_target, seed_val):
    peak_hours_apartment_hotel = [18, 19, 20, 21, 22]
    peak_hours_other = [12, 13, 14, 15, 16]

    train_peak_idx_bool = train_df['시각'].isin(peak_hours_apartment_hotel) if building_type in ['아파트', '호텔'] else train_df['시각'].isin(peak_hours_other)
    test_peak_idx_bool = test_df['시각'].isin(peak_hours_apartment_hotel) if building_type in ['아파트', '호텔'] else test_df['시각'].isin(peak_hours_other)

    X_train_peak_subset = train_df[train_peak_idx_bool][current_features].copy()
    y_train_peak_subset = np.log1p(train_df[train_peak_idx_bool][current_target]).copy()
    X_test_peak_subset = test_df[test_peak_idx_bool][current_features].copy()

    if len(X_train_peak_subset) == 0 or len(X_test_peak_subset) == 0:
        return None, None

    scaler_peak = StandardScaler()
    X_train_peak_scaled = scaler_peak.fit_transform(X_train_peak_subset)
    X_test_peak_scaled = scaler_peak.transform(X_test_peak_subset)

    peak_model_lgbm = LGBMRegressor(objective='mae', random_state=seed_val, n_estimators=500,
                                    learning_rate=0.05, num_leaves=31, max_depth=-1,
                                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                                    n_jobs=-1, verbose=-1)

    oof_preds_for_peak_subset = np.zeros(len(X_train_peak_subset))
    test_preds_for_peak_subset = np.zeros(len(X_test_peak_subset))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_peak_scaled, y_train_peak_subset)):
        X_train_fold, X_val_fold = X_train_peak_scaled[train_idx], X_train_peak_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_peak_subset.iloc[train_idx], y_train_peak_subset.iloc[val_idx]

        peak_model_lgbm.fit(X_train_fold, y_train_fold)
        oof_preds_for_peak_subset[val_idx] = peak_model_lgbm.predict(X_val_fold)
        test_preds_for_peak_subset += peak_model_lgbm.predict(X_test_peak_scaled) / N_SPLIT
    
    return oof_preds_for_peak_subset, test_preds_for_peak_subset

def tune_and_predict_with_optuna(train_df_subset, test_df_subset, 
                                 current_features, current_target, seed_val, n_trials=15):
    """
    Optuna를 사용하여 LGBMRegressor의 하이퍼파라미터를 튜닝하고,
    최적의 모델로 OOF 및 테스트 예측을 수행합니다.
    """
    X_train_full = train_df_subset[current_features].reset_index(drop=True)
    y_train_full = np.log1p(train_df_subset[current_target]).reset_index(drop=True)
    X_test_full = test_df_subset[current_features].reset_index(drop=True)

    if len(X_train_full) == 0 or len(X_test_full) == 0:
        return None, None, float('inf')

    def objective(trial):
        param = {
            'objective': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'random_state': seed_val,
            'n_jobs': -1,
            'verbose': -1,
        }

        kf = KFold(n_splits=2, shuffle=True, random_state=seed_val)
        oof_trial_preds = np.zeros(len(X_train_full))
        
        scaler = StandardScaler()
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full_scaled, y_train_full)):
            X_train_fold, X_val_fold = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
            y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            model = LGBMRegressor(**param)
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      eval_metric='mae',
                      callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)])

            oof_trial_preds[val_idx] = model.predict(X_val_fold)
            
            trial.report(mean_absolute_error(y_val_fold, oof_trial_preds[val_idx]), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return mean_absolute_error(y_train_full, oof_trial_preds)

    # Optuna Study 생성 및 최적화 (로그 억제)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"    > Optuna Best MAE: {study.best_value:.6f}")

    # 최적의 하이퍼파라미터로 최종 모델 학습 및 예측
    best_params = study.best_params
    best_lgbm_model = LGBMRegressor(objective='mae', random_state=seed_val, **best_params, n_jobs=-1, verbose=-1)

    oof_final_preds = np.zeros(len(X_train_full))
    test_final_preds = np.zeros(len(X_test_full))

    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full_scaled, y_train_full)):
        X_train_fold, X_val_fold = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        best_lgbm_model.fit(X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            eval_metric='mae',
                            callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)])

        oof_final_preds[val_idx] = best_lgbm_model.predict(X_val_fold)
        test_final_preds += best_lgbm_model.predict(X_test_full_scaled) / N_SPLIT

    final_smape = smape(np.expm1(y_train_full), np.expm1(oof_final_preds))
    print(f"    > Optuna 최종 모델 SMAPE: {final_smape:.6f}")

    return oof_final_preds, test_final_preds, final_smape

def train_and_predict_subset(train_subset, test_subset, name="Subset"):
    X_train_full = train_subset[features].reset_index(drop=True)
    y_train_full = np.log1p(train_subset[target].reset_index(drop=True))
    X_test = test_subset[features].reset_index(drop=True)

    oof_preds_lvl1 = np.zeros((len(X_train_full), 3))
    test_preds_lvl1 = np.zeros((len(X_test), 3))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        lgb_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1, verbose=-1)
        cat_model = CatBoostRegressor(loss_function='MAE', random_seed=SEED, n_estimators=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0)
        
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)
        cat_model.fit(X_train_scaled, y_train)

        oof_preds_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)
        oof_preds_lvl1[val_idx, 2] = cat_model.predict(X_val_scaled)

        test_preds_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_lvl1[:, 2] += cat_model.predict(X_test_scaled) / N_SPLIT
    
    meta_model = RidgeCV()
    meta_model.fit(oof_preds_lvl1, y_train_full)

    oof_pred_final = meta_model.predict(oof_preds_lvl1)
    test_pred_final = meta_model.predict(test_preds_lvl1)

    # Peak Model Integration
    building_type_for_peak = train_subset['건물유형'].iloc[0] if not train_subset.empty else None

    oof_pred_final_combined = np.array(oof_pred_final)
    test_pred_final_combined = np.array(test_pred_final)

    if building_type_for_peak:
        peak_oof_pred_subset, peak_test_pred_subset = peak_model(train_subset, test_subset, building_type_for_peak, features, target, SEED)

        if peak_oof_pred_subset is not None and peak_test_pred_subset is not None:
            train_peak_idx_bool = train_subset['시각'].isin([18, 19, 20, 21, 22]) if building_type_for_peak in ['아파트', '호텔'] else train_subset['시각'].isin([12, 13, 14, 15, 16])
            test_peak_idx_bool = test_subset['시각'].isin([18, 19, 20, 21, 22]) if building_type_for_peak in ['아파트', '호텔'] else test_subset['시각'].isin([12, 13, 14, 15, 16])

            oof_pred_final_combined[train_peak_idx_bool.values] = peak_oof_pred_subset
            test_pred_final_combined[test_peak_idx_bool.values] = peak_test_pred_subset
    
    current_smape = smape(np.expm1(y_train_full), np.expm1(oof_pred_final_combined))
    print(f"    > {name} SMAPE: {current_smape:.6f}")

    return oof_pred_final_combined, test_pred_final_combined, current_smape

# Optuna 가중치 최적화 함수들
def optimize_ensemble_weights(type_wise_oof_preds, building_wise_oof_preds, 
                            y_true, metric='smape', n_trials=100, random_state=42):
    """
    Optuna를 사용하여 앙상블 가중치 최적화
    """
    print(f"Optuna로 앙상블 가중치 최적화 시작... (메트릭: {metric.upper()}, 시행 횟수: {n_trials})")
    
    def objective(trial):
        w1 = trial.suggest_float('type_wise_weight', 0.0, 1.0)
        w2 = trial.suggest_float('building_wise_weight', 0.0, 1.0)
        
        total_weight = w1 + w2
        if total_weight == 0:
            w1, w2 = 0.5, 0.5
        else:
            w1, w2 = w1/total_weight, w2/total_weight
        
        ensemble_pred = w1 * type_wise_oof_preds + w2 * building_wise_oof_preds
        
        if metric.lower() == 'smape':
            score = smape(np.expm1(y_true), np.expm1(ensemble_pred))
        else:
            score = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            
        return score
    
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    w1 = best_params['type_wise_weight']
    w2 = best_params['building_wise_weight']
    
    total_weight = w1 + w2
    if total_weight == 0:
        w1, w2 = 0.5, 0.5
    else:
        w1, w2 = w1/total_weight, w2/total_weight
    
    best_weights = {
        'type_wise_weight': w1,
        'building_wise_weight': w2
    }
    
    best_score = study.best_value
    
    print(f"최적화 완료!")
    print(f"최적 가중치:")
    print(f"  - Type-wise 모델: {w1:.4f}")
    print(f"  - Building-wise 모델: {w2:.4f}")
    print(f"최적 {metric.upper()} 점수: {best_score:.6f}")
    
    return best_weights, best_score

def apply_optimized_ensemble(type_wise_preds, building_wise_preds, weights):
    """
    최적화된 가중치로 앙상블 예측 수행
    """
    w1 = weights['type_wise_weight']
    w2 = weights['building_wise_weight']
    
    ensemble_preds = w1 * type_wise_preds + w2 * building_wise_preds
    
    print(f"앙상블 예측 완료 (가중치: Type={w1:.4f}, Building={w2:.4f})")
    
    return ensemble_preds

# --- 1. 유형별 사전학습 및 예측 ---
print("\n--- [Step 1] 건물유형별 사전학습 및 예측 시작 ---")

type_wise_oof_preds = np.zeros(len(train))
type_wise_test_preds = np.zeros(len(test))
type_wise_smapes = []
type_wise_results = {}

building_types = train['건물유형'].unique()

for btype in building_types:
    print(f"   > 건물유형: {btype} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_bt = train[train['건물유형'] == btype].copy()
    test_bt = test[test['건물유형'] == btype].copy()

    if len(train_bt) == 0 or len(test_bt) == 0:
        continue

    oof_pred, test_pred, current_smape = train_and_predict_subset(train_bt, test_bt, name=f"Type {btype}")
    
    type_wise_smapes.append(current_smape)
    type_wise_oof_preds[train_bt.index] = oof_pred
    type_wise_test_preds[test_bt.index] = test_pred
    type_wise_results[btype] = {"smape": current_smape, "train_indices": train_bt.index, "test_indices": test_bt.index}

print("--- [Step 1] 건물유형별 사전학습 및 예측 완료 ---")
t_w_s = np.mean(type_wise_smapes)
print(f"--- 평균 SMAPE (유형별): {t_w_s:.6f} ---")

# 조건부 재학습
print("\n--- [Step 1.5] 건물유형별 조건부 재학습 시작 ---")
re_trained_type_wise_count = 0
for btype, result in type_wise_results.items():
    if result["smape"] > t_w_s:
        print(f"    > 재학습 필요: 건물유형 {btype} (SMAPE: {result['smape']:.6f} > 평균: {t_w_s:.6f})")
        train_bt = train.loc[result["train_indices"]].copy()
        test_bt = test.loc[result["test_indices"]].copy()

        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna(
            train_bt, test_bt, features, target, SEED, n_trials=15
        )
        
        if oof_pred_re is not None:
            type_wise_oof_preds[train_bt.index] = oof_pred_re
            type_wise_test_preds[test_bt.index] = test_pred_re
            re_trained_type_wise_count += 1
        else:
            print(f"    > Type {btype} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물유형별 조건부 재학습 완료 ({re_trained_type_wise_count}개 유형 재학습) ---")

y_true_all_train = np.log1p(train[target])
final_type_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(type_wise_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (유형별): {final_type_wise_smape_after_retrain:.6f} ---")

# --- 2. 건물별 사전학습 및 예측 ---
print("\n--- [Step 2] 건물별 사전학습 및 예측 시작 ---")

building_wise_oof_preds = np.zeros(len(train))
building_wise_test_preds = np.zeros(len(test))
building_wise_smapes = []
building_wise_results = {}

building_ids = train['건물번호'].unique()

for bno in building_ids:
    print(f"   > 건물번호: {bno} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(train_b) == 0 or len(test_b) == 0:
        continue

    oof_pred, test_pred, current_smape = train_and_predict_subset(train_b, test_b, name=f"Building {bno}")
    
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

        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna(
            train_b, test_b, features, target, SEED, n_trials=15
        )
        
        if oof_pred_re is not None:
            building_wise_oof_preds[train_b.index] = oof_pred_re
            building_wise_test_preds[test_b.index] = test_pred_re
            re_trained_building_wise_count += 1
        else:
            print(f"    > Building {bno} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물별 조건부 재학습 완료 ({re_trained_building_wise_count}개 건물 재학습) ---")

final_building_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(building_wise_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (건물별): {final_building_wise_smape_after_retrain:.6f} ---")

# --- 3. Optuna 앙상블 가중치 최적화 ---
print("\n--- [Step 3] Optuna 앙상블 가중치 최적화 시작 ---")

best_weights, best_score = optimize_ensemble_weights(
    type_wise_oof_preds, building_wise_oof_preds, y_true_all_train, 
    metric='smape', n_trials=100, random_state=SEED
)

# 최적화된 가중치로 앙상블
ensemble_oof_preds = apply_optimized_ensemble(
    type_wise_oof_preds, building_wise_oof_preds, best_weights
)
ensemble_test_preds = apply_optimized_ensemble(
    type_wise_test_preds, building_wise_test_preds, best_weights
)

print("--- [Step 3] Optuna 앙상블 가중치 최적화 완료 ---")

# --- 4. 잔차 학습 및 최종 예측 ---
print("\n--- [Step 4] 잔차 학습 및 최종 예측 시작 ---")

# 잔차 계산
residuals = y_true_all_train - ensemble_oof_preds

# 잔차 모델
residual_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=300, 
                               learning_rate=0.05, num_leaves=31, max_depth=-1, n_jobs=-1, verbose=-1)

# 잔차 모델 학습을 위한 피처 스케일링
scaler_res = StandardScaler()
X_train_scaled_res = scaler_res.fit_transform(train[features])
X_test_scaled_res = scaler_res.transform(test[features])

# 전체 훈련 데이터로 잔차 모델 학습
residual_model.fit(X_train_scaled_res, residuals)

# 테스트 데이터에 대한 잔차 예측
predicted_residuals = residual_model.predict(X_test_scaled_res)

# 최종 예측: 앙상블 예측 + 예측된 잔차
final_preds_residual = ensemble_test_preds + predicted_residuals

# 예측값이 음수가 되지 않도록 하고 원래 스케일로 변환
final_predictions_exp = np.expm1(final_preds_residual)
final_predictions_exp[final_predictions_exp < 0] = 0

# 최종 OOF SMAPE 계산 (모델 성능 평가용)
final_oof_combined_preds = ensemble_oof_preds + residual_model.predict(X_train_scaled_res)
final_oof_smape = smape(np.expm1(y_true_all_train), np.expm1(final_oof_combined_preds))

print(f"--- [Step 4] 잔차 학습 및 최종 예측 완료 ---")
print(f"최종 SMAPE (잔차 모델링 후): {final_oof_smape:.6f}")

print("[3] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions_exp
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_oof_smape:.4f}".replace('.', '_')

filename = f"02_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

with open(f"./Energy/02/{SEED}_submission/(LOG)model.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"test file name : {test_call}\n")
    f.write(f"건물 유형별 학습 SMAPE : {final_type_wise_smape_after_retrain}\n")
    f.write(f"건물 번호별 학습 SMAPE : {final_building_wise_smape_after_retrain}\n")
    f.write(f"Optuna 앙상블 가중치 : Type={best_weights['type_wise_weight']:.4f}, Building={best_weights['building_wise_weight']:.4f}\n")
    f.write(f"최적화된 앙상블 SMAPE : {best_score:.6f}\n")
    f.write(f"잔차학습 SMAPE : {final_oof_smape}\n")
    f.write("="*40 + "\n")

print(f"[4] 종료")