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

print("[02_model] 시작")

seed_file = "./Energy/_best_code/(SEED_COUNT)02_model.json"

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
save_path = './Energy/_best_code/'
traintest_save_path = './Energy/_best_code/best_train_test/'

os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
test_call = 'best_test_SEED44_up.csv'
train_call = 'best_train_SEED44_up.csv'

train = pd.read_csv(traintest_save_path + train_call)
test = pd.read_csv(traintest_save_path + test_call)
samplesub = pd.read_csv(data_path +'sample_submission.csv')

# Load best parameters from JSON file
with open('./Energy/_best_code/best_params.json', 'r') as f:
    best_params = json.load(f)

best_lgbm_params = best_params['lgbm']
best_xgb_params = best_params['xgb']
best_cat_params = best_params['cat']

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

    peak_model_lgbm = LGBMRegressor(objective='mae', random_state=seed_val, **best_lgbm_params, n_jobs=-1, verbose=-1)

    oof_preds_for_peak_subset = np.zeros(len(X_train_peak_subset))
    test_preds_for_peak_subset = np.zeros(len(X_test_peak_subset))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_peak_scaled, y_train_peak_subset)):
        X_train_fold, X_val_fold = X_train_peak_scaled[train_idx], X_train_peak_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_peak_subset.iloc[train_idx], y_train_peak_subset.iloc[val_idx]

        peak_model_lgbm.fit(X_train_fold, y_train_fold)
        oof_preds_for_peak_subset[val_idx] = peak_model_lgbm.predict(X_val_fold)
        test_preds_for_peak_subset += peak_model_lgbm.predict(X_test_peak_scaled) / N_SPLIT

    return oof_preds_for_peak_subset, test_preds_for_peak_subset

def train_and_predict_subset(train_subset, test_subset, name="Subset"):
    X_train_full = train_subset[features].reset_index(drop=True)
    y_train_full = np.log1p(train_subset[target].reset_index(drop=True))
    X_test = test_subset[features].reset_index(drop=True)

    oof_preds_lvl1 = np.zeros((len(X_train_full), 3)) # xgb, lgb, cat
    test_preds_lvl1 = np.zeros((len(X_test), 3))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_jobs=-1)
        lgb_model = LGBMRegressor(objective='mae', random_state=SEED, n_jobs=-1, verbose=-1)
        cat_model = CatBoostRegressor(loss_function='MAE', random_seed=SEED,  verbose=0)

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

    # --- Peak Model Integration ---
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

y_true_all_train = np.log1p(train[target])
final_type_wise_smape = smape(np.expm1(y_true_all_train), np.expm1(type_wise_oof_preds))
print(f"--- 최종 평균 SMAPE (유형별): {final_type_wise_smape:.6f} ---")


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

y_true_all_train = np.log1p(train[target])
final_building_wise_smape = smape(np.expm1(y_true_all_train), np.expm1(building_wise_oof_preds))
print(f"--- 최종 평균 SMAPE (건물별): {final_building_wise_smape:.6f} ---")


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
print("\n--- [Step 3] 건물 그룹별 예측 시작 ---")

group_oof_preds = np.zeros(len(train))
group_test_preds = np.zeros(len(test))
group_smapes = []
group_results = {}

for group_idx, group in enumerate(building_groups):
    print(f"   > 그룹 {group_idx+1}: 건물들 {group} 모델 학습 시작")

    train_g = train[train['건물번호'].isin(group)].copy()
    test_g = test[test['건물번호'].isin(group)].copy()

    if len(train_g) < 10 or len(test_g) < 1:
        print(f"   > 그룹 {group_idx+1} (건물: {group}) 데이터 부족으로 건너뜀.")
        continue

    oof_pred, test_pred, current_smape = train_and_predict_subset(train_g, test_g, name=f"Group {group_idx+1}")

    group_smapes.append(current_smape)
    group_oof_preds[train_g.index] = oof_pred
    group_test_preds[test_g.index] = test_pred
    group_results[group_idx] = {"smape": current_smape, "train_indices": train_g.index, "test_indices": test_g.index, "buildings": group}


print("--- [Step 3] 건물 그룹별 예측 완료 ---")
g_w_s = np.mean(group_smapes)
print(f"--- 평균 SMAPE (그룹별): {g_w_s:.6f} ---")

y_true_all_train = np.log1p(train[target])
final_group_wise_smape = smape(np.expm1(y_true_all_train), np.expm1(group_oof_preds))
print(f"--- 최종 평균 SMAPE (그룹별): {final_group_wise_smape:.6f} ---")


# --- 4. 잔차 학습 및 최종 예측 ---
print("\n--- [Step 4] 잔차 학습 및 최종 예측 시작 ---")

ensemble_oof_preds = (type_wise_oof_preds + building_wise_oof_preds + group_oof_preds) / 3
ensemble_test_preds = (type_wise_test_preds + building_wise_test_preds + group_test_preds) / 3

y_true_log = np.log1p(train[target])
residuals = y_true_log - ensemble_oof_preds

residual_model = LGBMRegressor(objective='mae', random_state=SEED, **best_lgbm_params, n_jobs=-1, verbose=-1)

scaler_res = StandardScaler()
X_train_scaled_res = scaler_res.fit_transform(train[features])
X_test_scaled_res = scaler_res.transform(test[features])

residual_model.fit(X_train_scaled_res, residuals)

predicted_residuals = residual_model.predict(X_test_scaled_res)

final_preds_residual = ensemble_test_preds + predicted_residuals

final_predictions_exp = np.expm1(final_preds_residual)
final_predictions_exp[final_predictions_exp < 0] = 0

final_oof_combined_preds = ensemble_oof_preds + residual_model.predict(X_train_scaled_res)
final_oof_smape = smape(np.expm1(y_true_log), np.expm1(final_oof_combined_preds))

print(f"--- [Step 4] 잔차 학습 및 최종 예측 완료 ---")
print(f"최종 SMAPE (잔차 모델링 후): {final_oof_smape:.6f}")

print("[3] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions_exp
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_oof_smape:.4f}".replace('.', '_')

filename = f"best_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

with open("./Energy/_best_code/(LOG)02_model.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"test file name : {test_call}\n")
    f.write(f"건물 유형별 학습 SMAPE : {final_type_wise_smape}\n")
    f.write(f"건물 번호별 학습 SMAPE : {final_building_wise_smape}\n")
    f.write(f"건물 그룹별 학습 SMAPE : {final_group_wise_smape}\n")
    f.write(f"잔차학습 SMAPE : {final_oof_smape}\n")
    f.write("="*40 + "\n")

print("[02_model] 종료")