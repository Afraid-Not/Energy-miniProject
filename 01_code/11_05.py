print(f"[11_05] 시작")
# ========================
# 임포트 및 랜덤 시드 고정
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # mean_absolute_error 추가
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import early_stopping, log_evaluation

seed_file = "./Energy/11_submission/11_05.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
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
submission = pd.read_csv(data_path + 'sample_submission.csv')

print(f"    >> 완료")

print(f"[2] 전처리 시작")
"""
# [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
######################## 파일 한번 저장하고 끌 부분 #############################
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

from sklearn.neighbors import NearestNeighbors


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
    print("🔧 train 일사(MJ/m2) 이상치 보간 시작")

    df_filled = train_df.copy()
    df_filled['hour'] = pd.to_datetime(df_filled['일시']).dt.hour

    # 후보는 미리 필터링
    candidate_pool = df_filled[
        (~df_filled['건물번호'].isin(zero_bnos)) & (df_filled['일사(MJ/m2)'] > 0)
    ].copy()
    candidate_pool['hour'] = pd.to_datetime(candidate_pool['일시']).dt.hour
    X_pool = candidate_pool[['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']].values

    for bno in zero_bnos:
        print(f"  > [BUILDING {bno}] 보간 중...")

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


# 저장

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def impute_test_sun_features(test_df, train_df, k=5):
    print("  > k-NN 보간 시작")
    test_df = test_df.copy()
    train_df = train_df.copy()
    
    test_df['hour'] = pd.to_datetime(test_df['일시']).dt.hour
    train_df['hour'] = pd.to_datetime(train_df['일시']).dt.hour

    # 1. 건물별로 train 데이터 캐싱
    nn_models_sun = {}
    nn_models_rad = {}
    sun_features = ['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']

    for bno in train_df['건물번호'].unique():
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
        if hour < 6 or hour > 20:
            test_df.at[idx, '일조(hr)'] = 0.0
        elif bno in nn_models_sun:
            nn, sun_cand = nn_models_sun[bno]
            _, indices = nn.kneighbors(x_target)
            mean_sun = sun_cand.iloc[indices[0]]['일조(hr)'].mean()
            test_df.at[idx, '일조(hr)'] = np.clip(mean_sun, 0, None)

        # 일사
        if hour < 5 or hour > 21:
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
train_all_fixed.to_csv(save_path +  f'R{SEED}_11_05_train.csv', index=False)

# test 일조 일사 보간
test_all = feature_engineering(test)
test_all_imputed = impute_test_sun_features(test_all, train_all_fixed, k=5)
test_all_imputed.to_csv(save_path + f'R{SEED}_11_05_test.csv', index=False)
print("✅ train 일사 보간 완료 및 저장")
print("✅ 일조/일사 보간 완료 및 저장")
"""
print(f"    >> 완료")

print(f"[3] 건물유형별 전이학습 시작")
"""
train = pd.read_csv(save_path + f'R{SEED}_11_05_train.csv')
test = pd.read_csv(save_path + f'R{SEED}_11_05_test.csv')

N_SPLITS = 3

features = [col for col in train.columns if col not in ['일시', '건물번호', '전력소비량(kWh)', '건물유형']]

# 결과 저장 컬럼 초기화
train['사전예측값'] = np.zeros(len(train), dtype='float64')
train['예측오차'] = np.zeros(len(train), dtype='float64')
train['log_사전예측값'] = np.zeros(len(train), dtype='float64')
train['건물유형별 중요도합'] = np.zeros(len(train), dtype='float64')
test['사전예측값'] = np.zeros(len(test), dtype='float64')
test['log_사전예측값'] = np.zeros(len(test), dtype='float64')
test['건물유형별 중요도합'] = np.zeros(len(test), dtype='float64')

# 전이학습
for btype in train['건물유형'].unique():
    print(f"[전이학습] 건물유형 {btype}")

    train_mask = train['건물유형'] == btype
    test_mask = test['건물유형'] == btype

    X = train.loc[train_mask, features]
    y = train.loc[train_mask, '전력소비량(kWh)']
    X_test = test.loc[test_mask, features]

    preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    importances = np.zeros(len(features))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train_idx, val_idx in kf.split(X):
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=SEED,
            early_stopping_rounds=30,
            verbose=-1
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  )
        preds[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / N_SPLITS
        importances += model.feature_importances_

    test.loc[test_mask, '사전예측값'] = test_preds
    train.loc[train_mask, '예측오차'] = y.values - preds
    
    safe_preds = np.clip(preds, a_min=0, a_max=None)
    safe_test_preds = np.clip(test_preds, a_min=0, a_max=None)
    
    train.loc[train_mask, '사전예측값'] = preds
    train.loc[train_mask, 'log_사전예측값'] = np.log1p(safe_preds)
    train.loc[train_mask, '건물유형별 중요도합'] = importances.sum()

    test.loc[test_mask, 'log_사전예측값'] = np.log1p(safe_test_preds)
    test.loc[test_mask, '건물유형별 중요도합'] = importances.sum()

# 건물번호별 사전예측 평균 및 std (train + test 병합 기반)
all_data = pd.concat([train[['건물번호', '사전예측값']], test[['건물번호', '사전예측값']]])
grouped = all_data.groupby('건물번호')['사전예측값'].agg(['mean', 'std']).reset_index()
grouped.columns = ['건물번호', '건물번호별 사전예측평균', '건물번호별 사전예측std']

# 병합
train = pd.merge(train, grouped, on='건물번호', how='left')
test = pd.merge(test, grouped, on='건물번호', how='left')

train.to_csv(save_path+f'R{SEED}_11_05_final_train.csv', index=False)
test.to_csv(save_path+f'R{SEED}_11_05_final_test.csv', index=False)
"""
# ========================
# 전력 사용량 예측
# ========================

# ========== 데이터 로드 ==========
train = pd.read_csv(save_path + f'R{SEED}_11_05_final_train.csv')
test = pd.read_csv(save_path + f'R{SEED}_11_05_final_test.csv')
N_SPLITS = 5
TARGET = '전력소비량(kWh)'
drop_cols = ['일시', '건물번호', '건물유형', '예측오차', TARGET]
features = [col for col in train.columns if col not in drop_cols]

# ========== 평가 함수 ==========
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

# ========== 결과 초기화 ==========
test_preds_by_id = np.zeros(len(test))
test_preds_all = np.zeros(len(test))
oof_preds_all = np.zeros(len(train))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ========== 모델 정의 ==========
def get_models():
    return [
        LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, verbosity=-1, early_stopping_rounds=30),
        XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, verbosity=0, early_stopping_rounds=30,),
        CatBoostRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, logging_level='Silent')
    ]

# ========== [1] 건물번호별 예측 ==========
print("[4] 건물번호별 예측")
for bno in tqdm(train['건물번호'].unique(), total=len(train['건물번호'].unique())):
    mask_train = train['건물번호'] == bno
    mask_test = test['건물번호'] == bno

    X = train.loc[mask_train, features]
    y = train.loc[mask_train, TARGET]
    X_test = test.loc[mask_test, features]

    if len(X) < N_SPLITS:
        continue

    preds_models = np.zeros((len(X_test), 3))  # LGBM, XGB, CB
    for i, model in enumerate(get_models()):
        fold_preds = np.zeros(len(X_test))
        for tr_idx, val_idx in kf.split(X):
            model_name = model.__class__.__name__

            if 'XGB' in model_name:
                model.fit(
                    X.iloc[tr_idx], y.iloc[tr_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    verbose=False
                )
            elif 'CatBoost' in model_name:
                model.fit(
                    X.iloc[tr_idx], y.iloc[tr_idx],
                    eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                    early_stopping_rounds=30,
                    # verbose=False,
                    # logging_level='Silent'
                )
            else:  # LightGBM
                model.fit(
                    X.iloc[tr_idx], y.iloc[tr_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                )

            fold_preds += model.predict(X_test) / N_SPLITS

        preds_models[:, i] = fold_preds

    test_preds_by_id[mask_test] = preds_models.mean(axis=1)

# ========== [2] 전체 모델 예측 ==========
print("[2] 전체 모델 예측")
X = train[features]
y = train[TARGET]
X_test = test[features]

preds_models = np.zeros((len(X_test), 3))
oof_preds_models = np.zeros((len(train), 3))

for i, model in enumerate(get_models()):
    test_fold_preds = np.zeros(len(X_test))
    oof_fold_preds = np.zeros(len(train))

    for tr_idx, val_idx in kf.split(X):
        model_name = model.__class__.__name__

        if 'XGB' in model_name:
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            )
        elif 'CatBoost' in model_name:
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                early_stopping_rounds=30,
                # verbose=False,
                # logging_level='Silent'
            )
        else:  # LightGBM
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[
                    early_stopping(stopping_rounds=30),
                    log_evaluation(0)
                ]
            )
        oof_fold_preds[val_idx] = model.predict(X.iloc[val_idx])
        test_fold_preds += model.predict(X_test) / N_SPLITS

    oof_preds_models[:, i] = oof_fold_preds
    preds_models[:, i] = test_fold_preds

oof_preds_all = oof_preds_models.mean(axis=1)
test_preds_all = preds_models.mean(axis=1)

# ========== [3] 앙상블 및 평가 ==========
print("[5] 앙상블 및 평가")
final_preds = (test_preds_by_id + test_preds_all) / 2
overall_avg_power_smape = smape(y, oof_preds_all)
print(f"✅ 전체 모델 SMAPE: {overall_avg_power_smape:.6f}")

# ========================
# [6] 결과 저장
# ========================
print("[6] 앙상블 및 평가")

submission = pd.read_csv(data_path +'sample_submission.csv')
submission['answer'] = final_preds
import datetime
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{overall_avg_power_smape:.6f}".replace('.', '_')
filename = f"({SEED})_11_05_({today})_({score_str}).csv"
submission.to_csv(save_path + filename, index=False)
print(f"✅ 최종 예측 저장 완료: {filename}")

print(f"  >> {filename}")
# 로그 저장
with open(save_path + "11_05.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"Mean SMAPE : {overall_avg_power_smape:.6f}\n")
    f.write("="*40 + "\n")
    
print(f"[11_05] 종료")




