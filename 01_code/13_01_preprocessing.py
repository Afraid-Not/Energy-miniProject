# ========================
# 임포트 및 랜덤 시드 고정
# ========================
print(f"[13_01_preprocessing] 시작")
import pandas as pd
import numpy as np
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping

seed_file = "./Energy/13_submission/13_01_preprocessing.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
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

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def peak_weighted_mae(y_true, y_pred):
    weight = y_true / (y_true.max() + 1e-8)
    return np.mean(weight * np.abs(y_true - y_pred))

data_path = './Energy/'
save_path = './Energy/13_submission/'
os.makedirs(save_path, exist_ok=True)

 #(파생피쳐 만들기 / Train 이상치 Nearest Neighbor k=7)

# ========================================================================================================================
# 전처리 1단계
# ========================================================================================================================
""" 
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')

print(f"[2] 전처리 시작")

building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col :
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)

building_csv = building_csv.fillna(0)

train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

def feature_engineering(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['일시'] = pd.to_datetime(df['일시'])
    df['시각'] = df['일시'].dt.hour                       # 시각(0~23)
    df['요일'] = df['일시'].dt.dayofweek               # 요일(0=월 ~ 6=일)
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['근무시간'] = df['시각'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)  # 주기적 패턴
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)
    df['정오거리'] = (df['시각'] - 12).abs()  # 12시(정오) 기준 거리
    df['정오거리_INV'] = 1 / (df['정오거리'] + 1)

    # 요일 One-hot
    dayofweek_ohe = pd.get_dummies(df['요일'], prefix='요일')
    df = pd.concat([df, dayofweek_ohe], axis=1)

    # ======================
    # 기상/에너지 관련 파생 피처
    # ======================

    # 불쾌지수(DI, Discomfort Index)
    if '기온(°C)' in df.columns and '습도(%)' in df.columns:
        df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3

    # 냉방 면적 대비 태양광 용량
    if '태양광용량(kW)' in df.columns and '냉방면적(m2)' in df.columns:
        df['태양광per냉방면적'] = df['태양광용량(kW)'] / (df['냉방면적(m2)'] + 1e-6)

    # ESS/PCS 설치 여부
    if 'ESS저장용량(kWh)' in df.columns:
        df['ESS설치여부'] = df['ESS저장용량(kWh)'].apply(lambda x: 1 if x > 0 else 0)
    if 'PCS용량(kW)' in df.columns:
        df['PCS설치여부'] = df['PCS용량(kW)'].apply(lambda x: 1 if x > 0 else 0)

    # ESS+PCS 총용량 대비 연면적 (설비 밀도)
    if 'ESS저장용량(kWh)' in df.columns and 'PCS용량(kW)' in df.columns and '연면적(m2)' in df.columns:
        df['설비밀도'] = (df['ESS저장용량(kWh)'] + df['PCS용량(kW)']) / (df['연면적(m2)'] + 1e-6)

    return df


def impute_train_solar_for_missing_bnos(train_df, zero_bnos, k=5):
    print("    > train 일사(MJ/m2) 이상치 보간 시작")

    df_filled = train_df.copy()
    df_filled['hour'] = pd.to_datetime(df_filled['일시']).dt.hour

    # 후보는 미리 필터링
    candidate_pool = df_filled[
        (~df_filled['건물번호'].isin(zero_bnos)) & (df_filled['일사(MJ/m2)'] > 0)
    ].copy()
    candidate_pool['hour'] = pd.to_datetime(candidate_pool['일시']).dt.hour
    X_pool = candidate_pool[['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']].values

    for bno in zero_bnos:
        print(f"    > [BUILDING {bno}] 보간 중...")

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

train_all = feature_engineering(train)
test_all = feature_engineering(test)

# train 일사 보간
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
train_all_fixed = impute_train_solar_for_missing_bnos(train_all, zero_bnos, k=5)

test_all.to_csv(save_path + 'pre_test.csv', index=False)
train_all_fixed.to_csv(save_path + 'preprocessing_train.csv', index=False)
 """
# ========================================================================================================================
# 전처리 2단계
# ========================================================================================================================

print(f"[3] 전처리 2단계 시작: 일조(hr) 및 일사(MJ/m2) 예측 범위 조정 및 음수 클리핑")
test_all = pd.read_csv(save_path + 'pre_test.csv')
train_all_fixed = pd.read_csv(save_path + 'preprocessing_train.csv')
""" 
# print(test_all.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형',
#        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '시각',
#        '요일', '월', '일', '주말여부', '근무시간', 'SIN_시', 'COS_시', '정오거리', '정오거리_INV',
#        '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5', '요일_6', '불쾌지수',
#        '태양광per냉방면적', 'ESS설치여부', 'PCS설치여부', '설비밀도'],
#       dtype='object')
# print(train_all_fixed.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
#        '일사(MJ/m2)', '전력소비량(kWh)', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
#        'ESS저장용량(kWh)', 'PCS용량(kW)', '시각', '요일', '월', '일', '주말여부', '근무시간',
#        'SIN_시', 'COS_시', '정오거리', '정오거리_INV', '요일_0', '요일_1', '요일_2', '요일_3',
#        '요일_4', '요일_5', '요일_6', '불쾌지수', '태양광per냉방면적', 'ESS설치여부', 'PCS설치여부',
#        '설비밀도'],
#       dtype='object')
# exit()
 """


for df in [train_all_fixed, test_all]:
    if '건물유형' in df.columns:
        df['건물유형'] = df['건물유형'].astype('category')

def preprocess_solar_data_for_prediction(df):
    df_processed = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_processed['일시']):
        df_processed['일시'] = pd.to_datetime(df_processed['일시'])
    df_processed['시각'] = df_processed['일시'].dt.hour

    if '일조(hr)' in df_processed.columns:
        df_processed.loc[(df_processed['시각'] < 6) | (df_processed['시각'] > 20), '일조(hr)'] = 0
        df_processed['일조(hr)'] = df_processed['일조(hr)'].apply(lambda x: max(0.0, x))
    else:
        print(f"경고: '일조(hr)' 컬럼이 데이터프레임에 없습니다. 이 컬럼은 생성될 것입니다.")

    if '일사(MJ/m2)' in df_processed.columns:
        df_processed.loc[(df_processed['시각'] < 5) | (df_processed['시각'] > 21), '일사(MJ/m2)'] = 0
        df_processed['일사(MJ/m2)'] = df_processed['일사(MJ/m2)'].apply(lambda x: max(0.0, x))
    else:
        print(f"경고: '일사(MJ/m2)' 컬럼이 데이터프레임에 없습니다. 이 컬럼은 생성될 것입니다.")

    return df_processed

if '일조(hr)' not in test_all.columns:
    test_all['일조(hr)'] = 0.0
if '일사(MJ/m2)' not in test_all.columns:
    test_all['일사(MJ/m2)'] = 0.0

test_all_processed = preprocess_solar_data_for_prediction(test_all.copy())
train_all_processed = preprocess_solar_data_for_prediction(train_all_fixed.copy())

print(f"[4] 전처리 완료")
print(f"[5] 모델 학습 및 예측 준비 (일조(hr) 및 일사(MJ/m2) 예측)")

train_df = train_all_processed.drop(columns=['일시'])
test_df = test_all_processed.drop(columns=['일시'])

targets = ['일조(hr)', '일사(MJ/m2)']

building_type_ohe_cols = [col for col in train_df.columns if col.startswith('건물유형_')]

base_features_initial = [col for col in train_df.columns if col not in targets + ['전력소비량(kWh)']]

missing_in_test = set(train_df.columns) - set(test_df.columns)
for col in missing_in_test:
    if col not in targets + ['전력소비량(kWh)']:
        test_df[col] = 0

missing_in_train = set(test_df.columns) - set(train_df.columns)
for col in missing_in_train:
    if col not in targets:
        train_df[col] = 0

common_features = list(set(train_df.columns) & set(test_df.columns))
base_features = [col for col in common_features if col not in targets + ['전력소비량(kWh)']]

lgbm_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'n_jobs': -1,
    'seed': SEED,
    'boosting_type': 'gbdt',
    'verbosity' : -1
}

def train_and_predict_lgbm(train_data, test_data, target_col, features, categorical_features, group_by_col=None):
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    if target_col == '일조(hr)':
        train_filter_mask = (train_data['시각'] >= 6) & (train_data['시각'] <= 20)
        test_zero_mask_base = (test_data['시각'] < 6) | (test_data['시각'] > 20)
    elif target_col == '일사(MJ/m2)':
        train_filter_mask = (train_data['시각'] >= 5) & (train_data['시각'] <= 21)
        test_zero_mask_base = (test_data['시각'] < 5) | (test_data['시각'] > 21)
    else:
        train_filter_mask = pd.Series(True, index=train_data.index)
        test_zero_mask_base = pd.Series(False, index=test_data.index)

    X_train_filtered = train_data.loc[train_filter_mask, features]
    y_train_filtered = train_data.loc[train_filter_mask, target_col]
    
    test_preds_full = np.zeros(len(test_data))
    oof_preds_filtered = np.zeros(len(X_train_filtered))

    if group_by_col:
        print(f"   [{group_by_col}] 그룹별 학습 및 예측 시작...")
        unique_groups = train_data[group_by_col].unique()
        
        for group in tqdm(unique_groups, desc=f"   {group_by_col} Iteration"):
            
            group_train_idx_filtered = X_train_filtered[train_data.loc[train_filter_mask, group_by_col] == group].index
            
            group_test_idx = test_data[test_data[group_by_col] == group].index
            
            if len(group_train_idx_filtered) == 0:
                test_preds_full[group_test_idx] = 0.0
                continue
            
            oof_group_preds = np.zeros(len(group_train_idx_filtered))
            test_group_preds = np.zeros(len(group_test_idx))

            X_train_group = X_train_filtered.loc[group_train_idx_filtered]
            y_train_group = y_train_filtered.loc[group_train_idx_filtered]
            
            X_test_group = test_data.loc[group_test_idx, features]

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_group, y_train_group)):
                X_fold_train, X_fold_val = X_train_group.iloc[train_idx], X_train_group.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_group.iloc[train_idx], y_train_group.iloc[val_idx]

                model = LGBMRegressor(**lgbm_params)
                model.fit(X_fold_train, y_fold_train,
                          eval_set=[(X_fold_val, y_fold_val)],
                          eval_metric='mae',
                          callbacks=[log_evaluation(0), early_stopping(100, verbose=False)],
                          categorical_feature=[f for f in categorical_features if f in features])

                oof_group_preds[val_idx] = model.predict(X_fold_val)
                test_group_preds += model.predict(X_test_group) / n_splits
            
            oof_preds_filtered[X_train_filtered.index.get_indexer(group_train_idx_filtered)] = oof_group_preds
            
            test_preds_full[group_test_idx] = test_group_preds

    else:
        print(f"   [전체] 학습 및 예측 시작...")
        X_test_predict = test_data[features].copy()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_filtered, y_train_filtered)):
            print(f"     [FOLD {fold+1}/{n_splits}] for {target_col}")
            X_fold_train, X_fold_val = X_train_filtered.iloc[train_idx], X_train_filtered.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_filtered.iloc[train_idx], y_train_filtered.iloc[val_idx]

            model = LGBMRegressor(**lgbm_params)
            model.fit(X_fold_train, y_fold_train,
                      eval_set=[(X_fold_val, y_fold_val)],
                      eval_metric='mae',
                      callbacks=[log_evaluation(0), early_stopping(100)],
                      categorical_feature=[f for f in categorical_features if f in features])

            oof_preds_filtered[val_idx] = model.predict(X_fold_val)
            test_preds_full += model.predict(X_test_predict) / n_splits
    
    test_preds_full[test_preds_full < 0] = 0

    test_preds_full[test_zero_mask_base.values] = 0.0
    
    oof_mae_target = mean_absolute_error(y_train_filtered, oof_preds_filtered)
    print(f"   {target_col} OOF 예측 결과 - MAE: {oof_mae_target:.4f}")

    return test_preds_full

predictions_results = {}

for target_col in targets:
    print(f"\n[6] LightGBM 모델 학습 시작: 타겟 = {target_col}")

    print(f"\n[6-1] 건물번호를 피처로 사용한 예측 시작: {target_col}")
    categorical_features_for_building_model = ['건물번호'] + building_type_ohe_cols
    features_for_building_model = list(set(base_features + categorical_features_for_building_model)) 
    
    test_preds_by_building = train_and_predict_lgbm(
        train_df, test_df, target_col, features_for_building_model, categorical_features_for_building_model
    )

    print(f"\n[6-2] 건물유형별 예측 시작: {target_col}")
    features_for_type_model = [f for f in base_features if f != '건물번호'] 
    categorical_features_for_type_model = building_type_ohe_cols 
    
    test_preds_by_type = train_and_predict_lgbm(
        train_df, test_df, target_col, features_for_type_model, categorical_features_for_type_model, group_by_col='건물유형'
    )

    print(f"\n[6-3] 앙상블 수행: {target_col}")
    final_preds = (test_preds_by_building * 0.5) + (test_preds_by_type * 0.5)
    
    if target_col == '일조(hr)':
        final_preds[(test_df['시각'] < 6) | (test_df['시각'] > 20)] = 0.0
    elif target_col == '일사(MJ/m2)':
        final_preds[(test_df['시각'] < 5) | (test_df['시각'] > 21)] = 0.0
    
    final_preds[final_preds < 0] = 0

    predictions_results[target_col] = final_preds

print(f"\n[8] 예측 결과를 test_all_processed에 반영")

test_all_processed['일조(hr)'] = predictions_results['일조(hr)']
test_all_processed['일사(MJ/m2)'] = predictions_results['일사(MJ/m2)']

test_all_processed['일시'] = pd.to_datetime(test_all_processed['일시'])

output_filename = save_path + f'13_01_test_seed_{SEED}_ensemble.csv' 
test_all_processed.to_csv(output_filename, index=False)

print(f"[9] 업데이트된 test_all_processed 저장 완료: {output_filename}")
print(f"[13_01_preprocessing] 종료")