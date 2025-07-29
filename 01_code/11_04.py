# ========================
# 임포트 및 랜덤 시드 고정
print(f"[11_00] 시작")
# ========================
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

seed_file = "./Energy/11_submission/11_03.json"

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

# ========================
# 데이터 로드
print(f"[1] 데이터 로드")
# ========================

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

data_path = './Energy/'
save_path = './Energy/11_submission/'
os.makedirs(save_path, exist_ok=True)

train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')

# ========================
# building_csv 전처리
print(f"[2] 전처리 시작")
# ========================

building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col :
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)

building_csv = building_csv.fillna(0)

train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

# ========================
# train, test 전처리
# ========================

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

    # 주기적 패턴 강화 (시각, 월, 일)
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)
    df['SIN_월'] = np.sin(2 * np.pi * df['월'] / 12)
    df['COS_월'] = np.cos(2 * np.pi * df['월'] / 12)
    df['SIN_일'] = np.sin(2 * np.pi * df['일'] / 31) # 최대 31일
    df['COS_일'] = np.cos(2 * np.pi * df['일'] / 31)

    # 요일 One-hot
    dayofweek_ohe = pd.get_dummies(df['요일'], prefix='요일')
    df = pd.concat([df, dayofweek_ohe], axis=1)

    # ======================
    # 건물유형 One-hot
    # ======================
    if '건물유형' in df.columns:
        building_type_ohe = pd.get_dummies(df['건물유형'])
        df = pd.concat([df, building_type_ohe], axis=1)

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

    # 추가: 기온-습도 상호작용
    if '기온(°C)' in df.columns and '습도(%)' in df.columns:
        df['기온X습도'] = df['기온(°C)'] * df['습도(%)']

    # 추가: 강수량 유무 (이진 변수)
    if '강수량(mm)' in df.columns:
        df['강수유무'] = (df['강수량(mm)'] > 0).astype(int)

    return df

train_all = feature_engineering(train)
test_all = feature_engineering(test)
# ========================
# 건물별 모델 학습 및 예측
# ========================
print(f"[3] 건물별 예측 모델")

# 최종 제출을 위한 DataFrame 초기화
submission_df = pd.read_csv(data_path + 'sample_submission.csv')

# submission_df에 건물번호 컬럼 추가 (num_date_time에서 추출)
# num_date_time 컬럼은 '건물번호_일시' 형식으로 되어 있음.
submission_df['건물번호'] = submission_df['num_date_time'].apply(lambda x: int(x.split('_')[0]))

submission_df['answer'] = 0.0 # 초기값 0으로 설정

# 건물별 예측을 위해 건물 번호 리스트 가져오기
building_ids = sorted(train_all['건물번호'].unique())

# 전력 소비량 예측 모델의 전체 SMAPE 합계를 저장할 변수
total_power_smape = 0
total_evaluated_buildings = 0

# 일조/일사 예측 모델의 전체 MAE 합계를 저장할 변수 추가
total_sunshine_mae = 0
total_insolation_mae = 0

# --- 1단계: 일조/일사 예측 및 전처리 ---
print(f"\n[3-1] 일조/일사 예측 및 전처리")

# 건물 유형 One-hot 컬럼 동적 추가
building_type_cols_all = [col for col in train_all.columns if col.startswith(('IDC', '건물기타', '공공', '백화점', '병원', '상용', '아파트', '연구소', '학교', '호텔'))]
train_feature = [
    '건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '시각', '요일', '월', '일', '주말여부', '근무시간',
    'SIN_시', 'COS_시', 'SIN_월', 'COS_월', 'SIN_일', 'COS_일',
    '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5', '요일_6',
    '불쾌지수', '태양광per냉방면적', 'ESS설치여부', 'PCS설치여부', '설비밀도',
    '기온X습도', '강수유무'
]
train_feature.extend(building_type_cols_all)
train_feature = list(set(train_feature))

for building_id in building_ids:
    print(f"\n> [BUILDING {building_id}] 일조/일사 예측 중...")

    train_building = train_all[train_all['건물번호'] == building_id].copy()
    test_building = test_all[test_all['건물번호'] == building_id].copy()

    X = train_building[train_feature].copy()
    Y1 = train_building['일조(hr)'].copy()
    Y2 = train_building['일사(MJ/m2)'].copy()
    test1 = test_building[train_feature].copy()

    log_col = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '태양광per냉방면적']
    mms_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '불쾌지수']

    for col in log_col:
        if col in X.columns: X.loc[:, col] = np.log1p(X[col])
        if col in test1.columns: test1.loc[:, col] = np.log1p(test1[col])

    for col in mms_col:
        if col in X.columns and col in test1.columns:
            combined_data = pd.concat([X[[col]], test1[[col]]], axis=0)
            mms_scaler = MinMaxScaler()
            mms_scaler.fit(combined_data)
            X.loc[:, col] = mms_scaler.transform(X[[col]])
            test1.loc[:, col] = mms_scaler.transform(test1[[col]])

    n_split = 5
    cv = KFold(n_splits=n_split, random_state=SEED, shuffle=True)

    print(f"  > ['일조(hr)']")

    # 일조 예측
    X_sunshine = X.copy()
    y_sunshine = Y1.copy()
    valid_sunshine_indices = y_sunshine.dropna().index
    X_sunshine = X_sunshine.loc[valid_sunshine_indices]
    y_sunshine = y_sunshine.loc[valid_sunshine_indices]

    if not X_sunshine.empty:
        sunshine_preds_building = np.zeros(test1.shape[0])
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_sunshine, y_sunshine)):
            print(f"    > Fold {fold+1}/{n_split}")
            
            X_train, X_val = X_sunshine.iloc[train_idx], X_sunshine.iloc[val_idx]
            y_train, y_val = y_sunshine.iloc[train_idx], y_sunshine.iloc[val_idx]
            xgb_sunshine = XGBRegressor(random_state=SEED, n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.7, colsample_bytree=0.7, early_stopping_rounds=100, n_jobs=-1, tree_method='hist')
            xgb_sunshine.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            lgbm_sunshine = LGBMRegressor(random_state=SEED, n_estimators=1000, learning_rate=0.05, early_stopping_rounds=100, n_jobs=-1, verbosity=-1)
            lgbm_sunshine.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            sunshine_preds_building += (xgb_sunshine.predict(test1) + lgbm_sunshine.predict(test1)) / (2 * n_split)

            # --- MAE 계산 및 누적 추가 ---
            val_preds_sunshine = (xgb_sunshine.predict(X_val) + lgbm_sunshine.predict(X_val)) / 2
            mae_sunshine = mean_absolute_error(y_val, val_preds_sunshine)
            total_sunshine_mae += mae_sunshine
            # ---------------------------
            print(f"      - MAE : {mae_sunshine:.6f}")
        
        sunshine_preds_building[sunshine_preds_building < 0] = 0
        test_all.loc[test_all['건물번호'] == building_id, '일조(hr)'] = sunshine_preds_building
    else:
        test_all.loc[test_all['건물번호'] == building_id, '일조(hr)'] = 0.0

    print(f"  > ['일사(MJ/m2)']")

    # 일사 예측
    X_insolation = X.copy()
    y_insolation = Y2.copy()
    valid_insolation_indices = y_insolation.dropna().index
    X_insolation = X_insolation.loc[valid_insolation_indices]
    y_insolation = y_insolation.loc[valid_insolation_indices]

    if not X_insolation.empty:
        insolation_preds_building = np.zeros(test1.shape[0])
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_insolation, y_insolation)):
            print(f"    > Fold {fold+1}/{n_split}")
            
            X_train, X_val = X_insolation.iloc[train_idx], X_insolation.iloc[val_idx]
            y_train, y_val = y_insolation.iloc[train_idx], y_insolation.iloc[val_idx]
            xgb_insolation = XGBRegressor(random_state=SEED, n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.7, colsample_bytree=0.7, early_stopping_rounds=100, n_jobs=-1, tree_method='hist')
            xgb_insolation.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            lgbm_insolation = LGBMRegressor(random_state=SEED, n_estimators=1000, learning_rate=0.05, early_stopping_rounds=100, n_jobs=-1, verbosity=-1)
            lgbm_insolation.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            insolation_preds_building += (xgb_insolation.predict(test1) + lgbm_insolation.predict(test1)) / (2 * n_split)
        
            # --- MAE 계산 및 누적 추가 ---
            val_preds_insolation = (xgb_insolation.predict(X_val) + lgbm_insolation.predict(X_val)) / 2
            mae_insolation = mean_absolute_error(y_val, val_preds_insolation)
            total_insolation_mae += mae_insolation
            # ---------------------------
            print(f"      - MAE : {mae_insolation:.6f}")
        
        insolation_preds_building[insolation_preds_building < 0] = 0
        test_all.loc[test_all['건물번호'] == building_id, '일사(MJ/m2)'] = insolation_preds_building
    else:
        test_all.loc[test_all['건물번호'] == building_id, '일사(MJ/m2)'] = 0.0


# --- 2단계: 건물 유형별 사전 학습 (전이 학습) ---
print(f"\n[3-2] 건물 유형별 사전 학습 모델")

# 이제 train_all과 test_all 모두 일조/일사 피처가 존재함
features_base = [
    '건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
    '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '시각', '요일', '월', '일', '주말여부', '근무시간',
    'SIN_시', 'COS_시', 'SIN_월', 'COS_월', 'SIN_일', 'COS_일',
    '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5', '요일_6',
    '불쾌지수', '태양광per냉방면적', 'ESS설치여부', 'PCS설치여부', '설비밀도',
    '기온X습도', '강수유무'
]
features_base.extend(building_type_cols_all)
features_base = list(set(features_base))

# 유형별 예측값을 저장할 컬럼 생성
train_all['type_base_pred'] = np.nan
test_all['type_base_pred'] = np.nan

building_types = train_all['건물유형'].unique()

for b_type in building_types:
    print(f"  > [건물 유형: {b_type}] 유형별 사전학습 훈련 중...")

    train_type = train_all[train_all['건물유형'] == b_type].copy()
    test_type = test_all[test_all['건물유형'] == b_type].copy()

    X_train_type = train_type[features_base].copy()
    y_train_type = train_type['전력소비량(kWh)'].copy()
    X_test_type = test_type[features_base].copy()

    if X_train_type.empty or len(X_train_type) < 50:
        print(f"    - 데이터 부족으로 건너뜀.")
        continue

    # 로그 변환 및 스케일링
    log_col_type = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '태양광per냉방면적']
    mms_col_type = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '불쾌지수', '일조(hr)', '일사(MJ/m2)']

    for col in log_col_type:
        if col in X_train_type.columns: X_train_type.loc[:, col] = np.log1p(X_train_type[col])
        if col in X_test_type.columns: X_test_type.loc[:, col] = np.log1p(X_test_type[col])
    
    for col in mms_col_type:
        if col in X_train_type.columns and col in X_test_type.columns:
            combined_data = pd.concat([X_train_type[[col]], X_test_type[[col]]], axis=0)
            mms_type_scaler = MinMaxScaler()
            mms_type_scaler.fit(combined_data)
            X_train_type.loc[:, col] = mms_type_scaler.transform(X_train_type[[col]])
            X_test_type.loc[:, col] = mms_type_scaler.transform(X_test_type[[col]])

    # LightGBM 사전 학습 모델
    lgbm_type_model = LGBMRegressor(random_state=SEED, n_estimators=1000, learning_rate=0.05, n_jobs=-1, verbosity=-1)
    lgbm_type_model.fit(X_train_type.drop(columns=['건물번호']), y_train_type)

    train_all.loc[train_all['건물유형'] == b_type, 'type_base_pred'] = lgbm_type_model.predict(X_train_type.drop(columns=['건물번호']))
    test_all.loc[test_all['건물유형'] == b_type, 'type_base_pred'] = lgbm_type_model.predict(X_test_type.drop(columns=['건물번호']))

train_all.loc[train_all['type_base_pred'] < 0, 'type_base_pred'] = 0
test_all.loc[test_all['type_base_pred'] < 0, 'type_base_pred'] = 0


# --- 3단계: 최종 건물별 전력소비량 예측 ---
print(f"\n[3-3] 최종 건물별 전력소비량 예측")

for building_id in building_ids:
    print(f"\n> [BUILDING {building_id}]")
    print(f"  > ['전력소비량(kWh)']")

    train_building = train_all[train_all['건물번호'] == building_id].copy()
    test_building = test_all[test_all['건물번호'] == building_id].copy()

    features = features_base.copy()
    features.append('type_base_pred')
    features.remove('건물번호')
    features = list(set(features))

    X_train_final = train_building[features].copy()
    y_train_final = train_building['전력소비량(kWh)'].copy()
    X_test_final = test_building[features].copy()

    log_col_final = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '태양광per냉방면적']
    mms_col_final = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '불쾌지수', '일조(hr)', '일사(MJ/m2)']
    
    for col in log_col_final:
        if col in X_train_final.columns: X_train_final[col] = np.log1p(train_building[col]) 
        if col in X_test_final.columns: X_test_final[col] = np.log1p(test_building[col]) 

    for col in mms_col_final:
        if col in X_train_final.columns and col in X_test_final.columns:
            combined_data = pd.concat([X_train_final[[col]], X_test_final[[col]]], axis=0)
            mms_final_scaler = MinMaxScaler()
            mms_final_scaler.fit(combined_data)
            X_train_final.loc[:, col] = mms_final_scaler.transform(X_train_final[[col]])
            X_test_final.loc[:, col] = mms_final_scaler.transform(X_test_final[[col]])
    
    xgb_preds_building = np.zeros(X_test_final.shape[0])
    lgbm_preds_building = np.zeros(X_test_final.shape[0])

    fold_power_smapes = []
    
    if len(X_train_final) < n_split:
        current_n_split = len(X_train_final)
        if current_n_split == 0:
            submission_df.loc[submission_df['건물번호'] == building_id, 'answer'] = 0.0
            continue
        cv_final = KFold(n_splits=current_n_split, random_state=SEED, shuffle=True)
    else:
        cv_final = cv

    for fold, (train_idx, val_idx) in enumerate(cv_final.split(X_train_final, y_train_final)):
        print(f"    > Fold {fold+1}/{n_split}")
        
        X_train, X_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
        y_train, y_val = y_train_final.iloc[train_idx], y_train_final.iloc[val_idx]

        xgb_model = XGBRegressor(random_state=SEED, n_estimators=1500, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, max_depth=8, n_jobs=-1, early_stopping_rounds=100)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_preds_building += xgb_model.predict(X_test_final) / cv_final.n_splits
        
        lgbm_model = LGBMRegressor(random_state=SEED, n_estimators=1500, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, max_depth=8, n_jobs=-1, early_stopping_rounds=100, verbosity=-1)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgbm_preds_building += lgbm_model.predict(X_test_final) / cv_final.n_splits
        
        ensemble_val_preds = (xgb_model.predict(X_val) + lgbm_model.predict(X_val)) / 2
        ensemble_val_smape = smape(y_val, ensemble_val_preds)
        fold_power_smapes.append(ensemble_val_smape)
        
        print(f"      - SMAPE : {ensemble_val_smape:.6f}")
        

    final_preds_building = (xgb_preds_building + lgbm_preds_building) / 2
    final_preds_building[final_preds_building < 0] = 0
    submission_df.loc[submission_df['건물번호'] == building_id, 'answer'] = final_preds_building

    if fold_power_smapes:
        avg_building_smape = np.mean(fold_power_smapes)
        total_power_smape += avg_building_smape
        total_evaluated_buildings += 1
        print(f">>> [BUILDING {building_id}] 전력소비량 SMAPE: {avg_building_smape:.6f}")
    else:
        print(f">>>>> [BUILDING {building_id}] 전력 소비량 예측 데이터 부족.")


# ========================
# 최종 결과 저장
# ========================

# 전체 건물에 대한 평균 SMAPE 출력 (학습에 사용된 건물만 포함)
if total_evaluated_buildings > 0:
    overall_avg_power_smape = total_power_smape / total_evaluated_buildings
    print(f"\n  >> [FINAL EVALUATION] 전체 건물 평균 SMAPE: {overall_avg_power_smape:.6f}")
else:
    print(f"\n  >> [FINAL EVALUATION] SMAPE를 평가할 건물이 없습니다.")

# 일조/일사 예측의 전체 평균 MAE 계산
# (1단계에서 MAE를 계산하고 누적하는 코드를 추가해야 함)
# 이 부분은 현재 코드에 포함되어 있지 않아 정확한 값은 출력되지 않을 수 있습니다.
# (이하 기존 코드와 동일)
overall_avg_sunshine_mae = total_sunshine_mae / len(building_ids) if len(building_ids) > 0 else 0
overall_avg_insolation_mae = total_insolation_mae / len(building_ids) if len(building_ids) > 0 else 0

print(f"  >> [FINAL EVALUATION] 일조 MAE: {overall_avg_sunshine_mae:.6f}")
print(f"  >> [FINAL EVALUATION] 일사 MAE: {overall_avg_insolation_mae:.6f}")
    
print(f"\n[6] 최종 결과 저장")
submission_df.drop(columns=['건물번호'], inplace=True) # submission_df에서 '건물번호' 컬럼 제거
import datetime
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{overall_avg_power_smape:.6f}".replace('.', '_')
filename = f"({SEED})_11_03_({today})_({score_str}).csv"
submission_df.to_csv(save_path + filename, index=False)

print(f"  >> 저장 완료: {filename}")
# 로그 저장
with open(save_path + "11_03.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"Mean SMAPE : {overall_avg_power_smape:.6f}\n")
    f.write(f"Mean Sunshine MAE : {overall_avg_sunshine_mae:.6f}\n")
    f.write(f"Mean Insolation MAE : {overall_avg_insolation_mae:.6f}\n")
    f.write("="*40 + "\n")
    
print(f"[11_03] 종료")