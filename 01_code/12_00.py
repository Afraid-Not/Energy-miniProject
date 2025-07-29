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

seed_file = "./Energy/12_submission/12_00.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED =4 # seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

print(f"[1] 데이터 로드 및 전처리 (생략 - 기존 코드에서 이미 수행)")
data_path = './Energy/'
save_path = './Energy/12_submission/'
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

N_SPLIT = 2
KFOLD = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

# --- 1. 유형별 사전학습 및 예측 ---
print("\n--- [Step 1] 건물유형별 사전학습 및 예측 시작 ---")

type_wise_oof_preds_base = np.zeros(len(train))
type_wise_test_preds_base = np.zeros(len(test))
type_wise_smapes_base = {} # 유형별 SMAPE 저장

building_types = train['건물유형'].unique()

for btype in building_types:
    print(f"  > 건물유형: {btype} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_bt = train[train['건물유형'] == btype].copy()
    test_bt = test[test['건물유형'] == btype].copy()

    X_train_type_full = train_bt[features].reset_index(drop=True)
    y_train_type_full = np.log1p(train_bt[target].reset_index(drop=True))
    X_test_type = test_bt[features].reset_index(drop=True)

    oof_preds_type_fold_lvl1 = np.zeros((len(X_train_type_full), 3)) # xgb, lgb, cat
    test_preds_type_fold_lvl1 = np.zeros((len(X_test_type), 3))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_type_full, y_train_type_full)):
        X_train, X_val = X_train_type_full.iloc[train_idx], X_train_type_full.iloc[val_idx]
        y_train, y_val = y_train_type_full.iloc[train_idx], y_train_type_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_type)

        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        lgb_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1, verbose=-1)
        cat_model = CatBoostRegressor(loss_function='MAE', random_seed=SEED, n_estimators=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0)
        
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)
        cat_model.fit(X_train_scaled, y_train)

        oof_preds_type_fold_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_type_fold_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)
        oof_preds_type_fold_lvl1[val_idx, 2] = cat_model.predict(X_val_scaled)

        test_preds_type_fold_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_type_fold_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_type_fold_lvl1[:, 2] += cat_model.predict(X_test_scaled) / N_SPLIT

    meta_model_type = RidgeCV()
    meta_model_type.fit(oof_preds_type_fold_lvl1, y_train_type_full)

    oof_pred_type_final = meta_model_type.predict(oof_preds_type_fold_lvl1)
    test_pred_type_final = meta_model_type.predict(test_preds_type_fold_lvl1)

    current_smape = smape(np.expm1(y_train_type_full), np.expm1(oof_pred_type_final))
    type_wise_smapes_base[btype] = current_smape
    print(f"  > SMAPE: {current_smape:.6f}")

    type_wise_oof_preds_base[train_bt.index] = oof_pred_type_final
    type_wise_test_preds_base[test_bt.index] = test_pred_type_final

print("--- [Step 1] 건물유형별 사전학습 및 예측 완료 ---")
print(f"--- 평균 SMAPE: {np.mean(list(type_wise_smapes_base.values())):.6f} ---")


# --- 2. 건물별 사전학습 및 예측 ---
print("\n--- [Step 2] 건물별 사전학습 및 예측 시작 ---")

building_wise_oof_preds_base = np.zeros(len(train))
building_wise_test_preds_base = np.zeros(len(test))
building_wise_smapes_base = {} # 건물별 SMAPE 저장

building_ids = train['건물번호'].unique()

for bno in building_ids:
    print(f"  > 건물번호: {bno} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    X_train_b_full = train_b[features].reset_index(drop=True)
    y_train_b_full = np.log1p(train_b[target].reset_index(drop=True))
    X_test_b = test_b[features].reset_index(drop=True)

    oof_preds_b_fold_lvl1 = np.zeros((len(X_train_b_full), 3)) # xgb, lgb, cat
    test_preds_b_fold_lvl1 = np.zeros((len(X_test_b), 3))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_b_full, y_train_b_full)):
        X_train, X_val = X_train_b_full.iloc[train_idx], X_train_b_full.iloc[val_idx]
        y_train, y_val = y_train_b_full.iloc[train_idx], y_train_b_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_b)

        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        lgb_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1, verbose=-1)
        cat_model = CatBoostRegressor(loss_function='MAE', random_seed=SEED, n_estimators=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0)
        
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)
        cat_model.fit(X_train_scaled, y_train)

        oof_preds_b_fold_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_b_fold_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)
        oof_preds_b_fold_lvl1[val_idx, 2] = cat_model.predict(X_val_scaled)

        test_preds_b_fold_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_b_fold_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_b_fold_lvl1[:, 2] += cat_model.predict(X_test_scaled) / N_SPLIT
    
    meta_model_b = RidgeCV()
    meta_model_b.fit(oof_preds_b_fold_lvl1, y_train_b_full)

    oof_pred_b_final = meta_model_b.predict(oof_preds_b_fold_lvl1)
    test_pred_b_final = meta_model_b.predict(test_preds_b_fold_lvl1)

    current_smape = smape(np.expm1(y_train_b_full), np.expm1(oof_pred_b_final))
    building_wise_smapes_base[bno] = current_smape
    print(f"  > SMAPE: {current_smape:.6f}")

    building_wise_oof_preds_base[train_b.index] = oof_pred_b_final
    building_wise_test_preds_base[test_b.index] = test_pred_b_final

print("--- [Step 2] 건물별 사전학습 및 예측 완료 ---")
print(f"--- 평균 SMAPE: {np.mean(list(building_wise_smapes_base.values())):.6f} ---")


# --- 3. 유형별 잔차 학습 및 최종 예측 ---
print("\n--- [Step 3] 건물유형별 잔차 학습 및 최종 예측 시작 ---")

type_wise_oof_preds_final = np.copy(type_wise_oof_preds_base)
type_wise_test_preds_final = np.copy(type_wise_test_preds_base)

avg_type_smape = np.mean(list(type_wise_smapes_base.values()))
print(f"  > 건물유형별 평균 SMAPE: {avg_type_smape:.6f}")

y_true_log = np.log1p(train[target])

for btype in building_types:
    if type_wise_smapes_base[btype] > avg_type_smape:
        print(f"  > 건물유형 {btype}의 SMAPE ({type_wise_smapes_base[btype]:.6f})가 평균보다 높아 잔차 학습 수행")
        train_bt = train[train['건물유형'] == btype].copy()
        test_bt = test[test['건물유형'] == btype].copy()

        X_train_type_full = train_bt[features].reset_index(drop=True)
        y_train_type_full = y_true_log.iloc[train_bt.index].reset_index(drop=True)
        X_test_type = test_bt[features].reset_index(drop=True)

        # 1단계 예측과의 잔차 계산
        residuals_type = y_train_type_full - type_wise_oof_preds_base[train_bt.index].copy()

        scaler_res_type = StandardScaler()
        X_train_scaled_res_type = scaler_res_type.fit_transform(X_train_type_full)
        X_test_scaled_res_type = scaler_res_type.transform(X_test_type)

        residual_model_type = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, n_jobs=-1, verbose=-1)
        residual_model_type.fit(X_train_scaled_res_type, residuals_type)

        predicted_residuals_type = residual_model_type.predict(X_test_scaled_res_type)
        
        type_wise_test_preds_final[test_bt.index] = type_wise_test_preds_base[test_bt.index] + predicted_residuals_type
        type_wise_oof_preds_final[train_bt.index] = type_wise_oof_preds_base[train_bt.index] + residual_model_type.predict(X_train_scaled_res_type)

    else:
        print(f"  > 건물유형 {btype}의 SMAPE ({type_wise_smapes_base[btype]:.6f})가 평균보다 낮아 잔차 학습 생략")

final_oof_type_smape = smape(np.expm1(y_true_log), np.expm1(type_wise_oof_preds_final))
print(f"--- [Step 3] 건물유형별 잔차 학습 및 최종 예측 완료 (SMAPE: {final_oof_type_smape:.6f}) ---")


# --- 4. 건물번호별 잔차 학습 및 최종 예측 ---
print("\n--- [Step 4] 건물번호별 잔차 학습 및 최종 예측 시작 ---")

building_wise_oof_preds_final = np.copy(building_wise_oof_preds_base)
building_wise_test_preds_final = np.copy(building_wise_test_preds_base)

avg_building_smape = np.mean(list(building_wise_smapes_base.values()))
print(f"  > 건물번호별 평균 SMAPE: {avg_building_smape:.6f}")

for bno in building_ids:
    if building_wise_smapes_base[bno] > avg_building_smape:
        print(f"  > 건물번호 {bno}의 SMAPE ({building_wise_smapes_base[bno]:.6f})가 평균보다 높아 잔차 학습 수행")
        train_b = train[train['건물번호'] == bno].copy()
        test_b = test[test['건물번호'] == bno].copy()

        X_train_b_full = train_b[features].reset_index(drop=True)
        y_train_b_full = y_true_log.iloc[train_b.index].reset_index(drop=True)
        X_test_b = test_b[features].reset_index(drop=True)

        # 2단계 예측과의 잔차 계산
        residuals_b = y_train_b_full - building_wise_oof_preds_base[train_b.index].copy()

        scaler_res_b = StandardScaler()
        X_train_scaled_res_b = scaler_res_b.fit_transform(X_train_b_full)
        X_test_scaled_res_b = scaler_res_b.transform(X_test_b)

        residual_model_b = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, n_jobs=-1, verbose=-1)
        residual_model_b.fit(X_train_scaled_res_b, residuals_b)

        predicted_residuals_b = residual_model_b.predict(X_test_scaled_res_b)
        
        building_wise_test_preds_final[test_b.index] = building_wise_test_preds_base[test_b.index] + predicted_residuals_b
        building_wise_oof_preds_final[train_b.index] = building_wise_oof_preds_base[train_b.index] + residual_model_b.predict(X_train_scaled_res_b)
    else:
        print(f"  > 건물번호 {bno}의 SMAPE ({building_wise_smapes_base[bno]:.6f})가 평균보다 낮아 잔차 학습 생략")

final_oof_building_smape = smape(np.expm1(y_true_log), np.expm1(building_wise_oof_preds_final))
print(f"--- [Step 4] 건물번호별 잔차 학습 및 최종 예측 완료 (SMAPE: {final_oof_building_smape:.6f}) ---")


# --- 5. 최종 잔차 학습 및 예측 ---
print("\n--- [Step 5] 최종 잔차 학습 및 예측 시작 ---")

# Step 3과 Step 4의 최종 예측을 결합
# 단순 평균 사용
ensemble_oof_preds_final = (type_wise_oof_preds_final + building_wise_oof_preds_final) / 2
ensemble_test_preds_final = (type_wise_test_preds_final + building_wise_test_preds_final) / 2

# 최종 잔차 계산
final_residuals = y_true_log - ensemble_oof_preds_final

# 최종 잔차 모델 (LGBMRegressor)
final_residual_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, n_jobs=-1, verbose=-1)

# 최종 잔차 모델 학습을 위한 피처 스케일링 (전체 데이터 사용)
scaler_final_res = StandardScaler()
X_train_scaled_final_res = scaler_final_res.fit_transform(train[features])
X_test_scaled_final_res = scaler_final_res.transform(test[features])

# 전체 훈련 데이터로 최종 잔차 모델 학습
final_residual_model.fit(X_train_scaled_final_res, final_residuals)

# 테스트 데이터에 대한 최종 잔차 예측
predicted_final_residuals = final_residual_model.predict(X_test_scaled_final_res)

# 최종 예측: 앙상블 예측 + 예측된 최종 잔차
final_predictions_exp = np.expm1(ensemble_test_preds_final + predicted_final_residuals)
final_predictions_exp[final_predictions_exp < 0] = 0 # 음수 값 0으로 처리

# 최종 OOF SMAPE 계산 (모델 성능 평가용)
final_oof_smape = smape(np.expm1(y_true_log), np.expm1(ensemble_oof_preds_final + final_residual_model.predict(X_train_scaled_final_res)))

print(f"--- [Step 5] 최종 잔차 학습 및 최종 예측 완료 ---")
print(f"최종 SMAPE (모든 잔차 모델링 후): {final_oof_smape:.6f}")

print("[3] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions_exp
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_oof_smape:.4f}".replace('.', '_')

print("\n--- [Step 6-SIGNED] 건물별 평균 잔차 기반 예측 보정 시작 ---")

train['예측값'] = np.expm1(ensemble_oof_preds_final + final_residual_model.predict(X_train_scaled_final_res))
train['실제값'] = np.expm1(y_true_log)
train['건물번호'] = train['건물번호'].astype(int)

# signed residual: 예측값 - 실제값
building_residual = train.groupby('건물번호').apply(
    lambda g: np.clip((g['예측값'] - g['실제값']).mean(), -3000, 3000)  # 너무 큰 보정 방지
).to_dict()

# test에 건물번호 붙이기
samplesub['건물번호'] = samplesub['num_date_time'].str.extract(r'^(\d+)_')[0].astype(int)
test['건물번호'] = samplesub['건물번호']
test['predict'] = final_predictions_exp

# 예측값에서 평균 잔차만큼 빼서 보정
test['predict_residual_corrected'] = test.apply(
    lambda row: max(0, row['predict'] - building_residual.get(row['건물번호'], 0)),  # 음수 방지
    axis=1
)

# 최종 저장
samplesub['answer'] = test['predict_residual_corrected']
samplesub = samplesub.drop(columns=['건물번호'])

residual_corrected_filename = f"12_00_{today}_SMAPE_{score_str}_residual_corrected.csv"
samplesub.to_csv(save_path + residual_corrected_filename, index=False)

print(f"✅ Signed Residual 기반 보정 완료 → 저장: {residual_corrected_filename}")

with open("./Energy/12_submission/12_00_log.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{residual_corrected_filename}\n")
    f.write(f"SMAPE : {final_oof_smape}\n")
    f.write("="*40 + "\n")

print(f"[4] 종료 ")


