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

print("[11_06] 시작")

seed_file = "./Energy/11_submission/11_06.json"

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
save_path = './Energy/11_submission/'
os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
train = pd.read_csv(save_path + 'preprocessed_train.csv')
test = pd.read_csv(save_path + 'preprocessed_test.csv')
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

# --- 1. 유형별 사전학습 및 예측 ---
print("\n--- [Step 1] 건물유형별 사전학습 및 예측 시작 ---")

type_wise_oof_preds = np.zeros(len(train))
type_wise_test_preds = np.zeros(len(test))
type_wise_smapes = []

building_types = train['건물유형'].unique()

for btype in building_types:
    print(f"  > 건물유형: {btype} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_bt = train[train['건물유형'] == btype].copy()
    test_bt = test[test['건물유형'] == btype].copy()

    X_train_type_full = train_bt[features].reset_index(drop=True)
    y_train_type_full = np.log1p(train_bt[target].reset_index(drop=True))
    X_test_type = test_bt[features].reset_index(drop=True)

    # KFold OOF 예측 및 테스트 예측을 위한 배열 초기화
    oof_preds_type_fold_lvl1 = np.zeros((len(X_train_type_full), 3)) # xgb, lgb, cat
    test_preds_type_fold_lvl1 = np.zeros((len(X_test_type), 3))

    fold_smape_scores = []

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_type_full, y_train_type_full)):
        X_train, X_val = X_train_type_full.iloc[train_idx], X_train_type_full.iloc[val_idx]
        y_train, y_val = y_train_type_full.iloc[train_idx], y_train_type_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_type)

        # 모델 초기화 (고정된 하이퍼파라미터)
        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        lgb_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1, verbose=-1)
        cat_model = CatBoostRegressor(loss_function='MAE', random_seed=SEED, n_estimators=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0)
        
        # 모델 학습
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)
        cat_model.fit(X_train_scaled, y_train)

        # Level 1 OOF 예측
        oof_preds_type_fold_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_type_fold_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)
        oof_preds_type_fold_lvl1[val_idx, 2] = cat_model.predict(X_val_scaled)

        # Level 1 Test 예측 (폴드별로 누적 평균)
        test_preds_type_fold_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_type_fold_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_type_fold_lvl1[:, 2] += cat_model.predict(X_test_scaled) / N_SPLIT

    # Level 2 Meta model (RidgeCV) 학습 및 예측
    meta_model_type = RidgeCV()
    meta_model_type.fit(oof_preds_type_fold_lvl1, y_train_type_full) # OOF 예측으로 메타 모델 학습

    oof_pred_type_final = meta_model_type.predict(oof_preds_type_fold_lvl1)
    test_pred_type_final = meta_model_type.predict(test_preds_type_fold_lvl1)

    # 최종 SMAPE 계산 및 저장
    current_smape = smape(np.expm1(y_train_type_full), np.expm1(oof_pred_type_final))
    type_wise_smapes.append(current_smape)
    print(f"  > SMAPE: {current_smape:.6f}")

    # 전체 데이터 배열에 저장
    type_wise_oof_preds[train_bt.index] = oof_pred_type_final
    type_wise_test_preds[test_bt.index] = test_pred_type_final

print("--- [Step 1] 건물유형별 사전학습 및 예측 완료 ---")
print(f"--- 평균 SMAPE: {np.mean(type_wise_smapes):.6f} ---")

# --- 2. 건물별 사전학습 및 예측 ---
print("\n--- [Step 2] 건물별 사전학습 및 예측 시작 ---")

building_wise_oof_preds = np.zeros(len(train))
building_wise_test_preds = np.zeros(len(test))
building_wise_smapes = []

building_ids = train['건물번호'].unique()

for bno in building_ids:
    print(f"  > 건물번호: {bno} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    X_train_b_full = train_b[features].reset_index(drop=True)
    y_train_b_full = np.log1p(train_b[target].reset_index(drop=True))
    X_test_b = test_b[features].reset_index(drop=True)

    # KFold OOF 예측 및 테스트 예측을 위한 배열 초기화
    oof_preds_b_fold_lvl1 = np.zeros((len(X_train_b_full), 3)) # xgb, lgb, cat
    test_preds_b_fold_lvl1 = np.zeros((len(X_test_b), 3))

    fold_smape_scores = []

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_b_full, y_train_b_full)):
        X_train, X_val = X_train_b_full.iloc[train_idx], X_train_b_full.iloc[val_idx]
        y_train, y_val = y_train_b_full.iloc[train_idx], y_train_b_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_b)

        # 모델 초기화 (고정된 하이퍼파라미터)
        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        lgb_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1, verbose=-1)
        cat_model = CatBoostRegressor(loss_function='MAE', random_seed=SEED, n_estimators=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0)
        
        # 모델 학습
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)
        cat_model.fit(X_train_scaled, y_train)

        # Level 1 OOF 예측
        oof_preds_b_fold_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_b_fold_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)
        oof_preds_b_fold_lvl1[val_idx, 2] = cat_model.predict(X_val_scaled)

        # Level 1 Test 예측 (폴드별로 누적 평균)
        test_preds_b_fold_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_b_fold_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_b_fold_lvl1[:, 2] += cat_model.predict(X_test_scaled) / N_SPLIT
    
    # Level 2 Meta model (RidgeCV) 학습 및 예측
    meta_model_b = RidgeCV()
    meta_model_b.fit(oof_preds_b_fold_lvl1, y_train_b_full) # OOF 예측으로 메타 모델 학습

    oof_pred_b_final = meta_model_b.predict(oof_preds_b_fold_lvl1)
    test_pred_b_final = meta_model_b.predict(test_preds_b_fold_lvl1)

    # 최종 SMAPE 계산 및 저장
    current_smape = smape(np.expm1(y_train_b_full), np.expm1(oof_pred_b_final))
    building_wise_smapes.append(current_smape)
    print(f"  > SMAPE: {current_smape:.6f}")

    # 전체 데이터 배열에 저장
    building_wise_oof_preds[train_b.index] = oof_pred_b_final
    building_wise_test_preds[test_b.index] = test_pred_b_final

print("--- [Step 2] 건물별 사전학습 및 예측 완료 ---")
print(f"--- 평균 SMAPE: {np.mean(building_wise_smapes):.6f} ---")


# --- 3. 잔차 학습 및 최종 예측 ---
print("\n--- [Step 3] 잔차 학습 및 최종 예측 시작 ---")

# Step 1과 Step 2의 예측을 결합
# 단순 평균 사용 (가중 평균 또는 다른 메타 모델 사용 가능)
ensemble_oof_preds = (type_wise_oof_preds + building_wise_oof_preds) / 2
ensemble_test_preds = (type_wise_test_preds + building_wise_test_preds) / 2

# 잔차 계산
y_true_log = np.log1p(train[target])
residuals = y_true_log - ensemble_oof_preds

# 잔차 모델 (LGBMRegressor)
# 잔차 모델을 위한 고정 하이퍼파라미터
residual_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, n_jobs=-1, verbose=-1)

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
final_predictions_exp[final_predictions_exp < 0] = 0 # 음수 값 0으로 처리

# 최종 OOF SMAPE 계산 (모델 성능 평가용)
final_oof_combined_preds = ensemble_oof_preds + residual_model.predict(X_train_scaled_res)
final_oof_smape = smape(np.expm1(y_true_log), np.expm1(final_oof_combined_preds))

print(f"--- [Step 3] 잔차 학습 및 최종 예측 완료 ---")
print(f"최종 SMAPE (잔차 모델링 후): {final_oof_smape:.6f}")

print("[3] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions_exp
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_oof_smape:.4f}".replace('.', '_')

filename = f"11_06_{today}_SMAPE_{score_str}_residual_refactored_no_optuna_kf_smape_display.csv"
samplesub.to_csv(save_path + filename, index=False)

with open("./Energy/11_submission/11_06_log.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"SMAPE : {final_oof_smape}\n")
    f.write("="*40 + "\n")

print(f"[4] 종료 ")

# <SEED :67>
# 11_06_20250729_SMAPE_1_7729_residual_refactored_no_optuna.csv
# SMAPE : 1.7728904969596897    -> 7.3503754907
# ========================================