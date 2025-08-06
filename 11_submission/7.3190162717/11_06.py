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
    print("  > train 일사(MJ/m2) 이상치 보간 시작")

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
train_all_fixed.to_csv(save_path + 'preprocessed_train.csv', index=False)
print("    >> train 일사 보간 완료 및 저장")

# test 일조 일사 보간
test_all = feature_engineering(test)
test_all_imputed = impute_test_sun_features(test_all, train_all_fixed, k=5)
test_all_imputed.to_csv(save_path + 'preprocessed_test.csv', index=False)
print("    >> 일조/일사 보간 완료 및 저장")
print(f"  >> 완료")

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
N_SPLIT = 5

kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

for btype in building_types:
    print(f"[BUILING TYPE {btype}]")
    train_bt = train[train['건물유형'] == btype].copy()
    x = train_bt[features].reset_index(drop=True)
    y = np.log1p(train_bt[target].reset_index(drop=True))

    smape_folds = []
    models_xgb, models_lgb, models_cat, scalers = [], [], [], []

    for fold, (tr_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=len(range(N_SPLIT))):
        x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        # Base models
        xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                 random_state=SEED, early_stopping_rounds=30, objective='reg:squarederror')
        xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

        lgb_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                  random_state=SEED, objective='mae', verbose=-1)
        lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                      callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])

        cat_model = CatBoostRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                      random_seed=SEED, verbose=0, loss_function='MAE')
        cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=30)

        # validation smape 측정 (선택적 로그 출력용)
        val_pred = cat_model.predict(x_val_scaled)
        val_smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(y_val)) /
                            (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(y_val)) + 1e-6))
        smape_folds.append(val_smape)

        # 모델/스케일러 저장
        models_xgb.append(xgb_model)
        models_lgb.append(lgb_model)
        models_cat.append(cat_model)
        scalers.append(scaler)

    best_fold = np.argmin(smape_folds)  # 가장 성능 좋은 fold 선택
    pretrained_models[btype] = {
        'xgb': models_xgb[best_fold],
        'lgb': models_lgb[best_fold],
        'cat': models_cat[best_fold],
        'scaler': scalers[best_fold]
    }

    print(f"  > '{btype}' 평균 SMAPE: {np.mean(smape_folds):.4f}, 선택된 fold: {best_fold}\n")
###############
print(f"  >> 완료")
""" 
print("[2] 건물별 학습 시작")

# 최종 예측 결과 저장용
final_preds = []
val_smapes = []

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

for bno in building_ids:
    btype = train[train['건물번호'] == bno]['건물유형'].iloc[0]
    print(f"\n  >> [BUILDING {bno}] 유형: {btype}")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    x = train_b[features].reset_index(drop=True)
    y = np.log1p(train_b[target].reset_index(drop=True))
    x_test_final = test_b[features].reset_index(drop=True)

    test_pred_folds = np.zeros(len(x_test_final))
    smape_folds = []

    for fold, (tr_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=N_SPLIT):
        x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # 사전학습된 스케일러로 표준화
        scaler = clone(pretrained_models[btype]['scaler'])
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test_final)

        # 사전학습된 모델 clone
        xgb_model = clone(pretrained_models[btype]['xgb'])
        xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

        lgb_model = clone(pretrained_models[btype]['lgb'])
        lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

        cat_model = clone(pretrained_models[btype]['cat'])
        cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

        # Level 1 predictions
        oof_train_lvl1 = np.vstack([
            xgb_model.predict(x_train_scaled),
            lgb_model.predict(x_train_scaled),
            cat_model.predict(x_train_scaled)
        ]).T
        oof_val_lvl1 = np.vstack([
            xgb_model.predict(x_val_scaled),
            lgb_model.predict(x_val_scaled),
            cat_model.predict(x_val_scaled)
        ]).T
        oof_test_lvl1 = np.vstack([
            xgb_model.predict(x_test_scaled),
            lgb_model.predict(x_test_scaled),
            cat_model.predict(x_test_scaled)
        ]).T

        # Level 2 Meta model
        meta_model = RidgeCV()
        meta_model.fit(oof_train_lvl1, y_train)
        val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
        test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

        # Level 3 Final model
        final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
        final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)

        val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
        val_smape = np.mean(200 * np.abs(np.expm1(val_final) - np.expm1(y_val)) /
                            (np.abs(np.expm1(val_final)) + np.abs(np.expm1(y_val)) + 1e-6))
        smape_folds.append(val_smape)
        pred = final_model.predict(test_pred_lvl2.reshape(-1, 1))
        test_pred_folds += pred / kf.n_splits

    # 최종 결과 저장
    building_smape=np.mean(smape_folds)
    val_smapes.append(building_smape)
    print(f"  >> SMAPE : {building_smape}")
    final_preds.extend(test_pred_folds)
    
avg_smape = np.mean(val_smapes)
print(f"  >> 평균 SMAPE : {avg_smape:.6f}")
print("  >> 완료")
print("[3] 저장 시작")

import datetime
# 결과 저장
samplesub['answer'] = np.expm1(final_preds)
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{avg_smape:.4f}".replace('.', '_')

filename = f"11_06_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

with open("./Energy/11_submission/11_06_log.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"SMAPE : {avg_smape}\n")
    f.write("="*40 + "\n")
    
print(f"[4] 종료 ")

# ✅ 최종 SMAPE 점수 : 2.281621
 """
 
print("[2] 건물별 학습 시작")

# 최종 예측 결과 저장용
final_preds_residual = []
val_smapes = []
residual_smapes =[]

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

for bno in building_ids:
    btype = train[train['건물번호'] == bno]['건물유형'].iloc[0]
    print(f"\n  >> [BUILDING {bno}] 유형: {btype}")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    x = train_b[features].reset_index(drop=True)
    y = np.log1p(train_b[target].reset_index(drop=True))
    x_test_final = test_b[features].reset_index(drop=True)

    test_pred_folds_level3 = np.zeros(len(x_test_final))
    oof_preds_level3 = np.zeros(len(x))
    smape_folds = []

    for fold, (tr_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=N_SPLIT):
        x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # 사전학습된 스케일러로 표준화
        scaler = clone(pretrained_models[btype]['scaler'])
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test_final)

        # 사전학습된 모델 clone
        xgb_model = clone(pretrained_models[btype]['xgb'])
        xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

        lgb_model = clone(pretrained_models[btype]['lgb'])
        lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

        cat_model = clone(pretrained_models[btype]['cat'])
        cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

        # Level 1 predictions
        oof_train_lvl1 = np.vstack([
            xgb_model.predict(x_train_scaled),
            lgb_model.predict(x_train_scaled),
            cat_model.predict(x_train_scaled)
        ]).T
        oof_val_lvl1 = np.vstack([
            xgb_model.predict(x_val_scaled),
            lgb_model.predict(x_val_scaled),
            cat_model.predict(x_val_scaled)
        ]).T
        oof_test_lvl1 = np.vstack([
            xgb_model.predict(x_test_scaled),
            lgb_model.predict(x_test_scaled),
            cat_model.predict(x_test_scaled)
        ]).T

        # Level 2 Meta model
        meta_model = RidgeCV()
        meta_model.fit(oof_train_lvl1, y_train)
        val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
        test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

        # Level 3 Final model (GradientBoostingRegressor)
        final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
        final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)

        val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
        
        # OOF 예측값 저장
        oof_preds_level3[val_idx] = val_final
        
        val_smape = np.mean(200 * np.abs(np.expm1(val_final) - np.expm1(y_val)) /
                            (np.abs(np.expm1(val_final)) + np.abs(np.expm1(y_val)) + 1e-6))
        smape_folds.append(val_smape)
        
        pred = final_model.predict(test_pred_lvl2.reshape(-1, 1))
        test_pred_folds_level3 += pred / kf.n_splits

    # ===== 잔차 모델링 추가 시작 =====
    print("  >> 잔차 모델 학습 시작")
    # 1. 잔차 계산
    residuals = y - oof_preds_level3

    # 2. Optuna를 사용한 잔차 모델 최적화
    def residual_objective(trial, X, y):
        # 하이퍼파라미터 탐색 범위 정의
        params = {
            'objective': 'regression_l1', # MAE
            'metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'random_state': SEED,
            'verbose': -1,
            'n_jobs': -1,
        }
        
        # K-Fold 교차 검증으로 모델 학습 및 평가
        kf_optuna = KFold(n_splits=5, shuffle=True, random_state=SEED)
        mae_scores = []
        for train_idx, val_idx in kf_optuna.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_res, y_val_res = y.iloc[train_idx], y.iloc[val_idx]
            
            # 스케일링
            scaler_res = StandardScaler()
            X_train_scaled_res = scaler_res.fit_transform(X_train)
            X_val_scaled_res = scaler_res.transform(X_val)
            
            model = LGBMRegressor(**params)
            model.fit(X_train_scaled_res, y_train_res,
                      eval_set=[(X_val_scaled_res, y_val_res)],
                      callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
            
            pred_val_res = model.predict(X_val_scaled_res)
            mae_scores.append(mean_absolute_error(y_val_res, pred_val_res))
            
        return np.mean(mae_scores)

    # Optuna 스터디 생성 및 최적화
    print("  >> Optuna로 잔차 모델 하이퍼파라미터 최적화...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: residual_objective(trial, x, residuals), n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"  >> 최적 하이퍼파라미터: {best_params}")

    # 3. 최적 하이퍼파라미터로 최종 잔차 모델 학습 및 예측
    test_pred_residual = np.zeros(len(x_test_final))
    
    for fold, (tr_idx, val_idx) in tqdm(enumerate(kf.split(x)), total=N_SPLIT, desc="잔차모델 재학습 및 예측"):
        x_train, y_train_res = x.iloc[tr_idx], residuals.iloc[tr_idx]
        x_val, y_val_res = x.iloc[val_idx], residuals.iloc[val_idx]

        # 스케일링
        scaler_res = StandardScaler()
        x_train_scaled_res = scaler_res.fit_transform(x_train)
        x_test_scaled_res = scaler_res.transform(x_test_final)
        
        # 최적화된 파라미터로 모델 초기화
        residual_model = LGBMRegressor(**best_params, random_state=SEED, verbose=-1)
        residual_model.fit(x_train_scaled_res, y_train_res,
                           eval_set=[(scaler_res.transform(x_val), y_val_res)],
                           callbacks=[early_stopping(50, verbose=False)])
        
        test_pred_residual += residual_model.predict(x_test_scaled_res) / kf.n_splits
    
    # 4. 최종 예측 결합
    final_pred_bno = test_pred_folds_level3 + test_pred_residual
    
    # 최종 결과 저장
    building_smape = np.mean(smape_folds)
    val_smapes.append(building_smape)
    print(f"  >> SMAPE (잔차 모델링 전): {building_smape:.6f}")
    
    # 잔차 모델링 후 SMAPE 계산 (선택적)
    final_oof_pred_bno = oof_preds_level3 + residual_model.predict(scaler_res.transform(x))
    final_smape_bno = np.mean(200 * np.abs(np.expm1(final_oof_pred_bno) - np.expm1(y)) /
                            (np.abs(np.expm1(final_oof_pred_bno)) + np.abs(np.expm1(y)) + 1e-6))
    print(f"  >> SMAPE (잔차 모델링 후): {final_smape_bno:.6f}")
    residual_smapes.append(final_smape_bno)
    final_preds_residual.extend(final_pred_bno)
    
avg_smape = np.mean(val_smapes)
res_smape = np.mean(residual_smapes)
print(f"  >> 평균 SMAPE (잔차 모델링 전) : {avg_smape:.6f}")
print(f"  >> 평균 SMAPE (잔차 모델링 후) : {res_smape:.6f}")
print("  >> 완료")

print("[3] 저장 시작")
samplesub['answer'] = np.expm1(final_preds_residual)
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{res_smape:.4f}".replace('.', '_')

filename = f"11_06_{today}_SMAPE_{score_str}_residual.csv"
samplesub.to_csv(save_path + filename, index=False)

with open("./Energy/11_submission/11_06_log.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"SMAPE : {res_smape}\n")
    f.write("="*40 + "\n")
    
print(f"[4] 종료 ")