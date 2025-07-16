print(f"[03_preprocessing] 시작")
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
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping

data_path = './Energy/'
csv_path = './Energy/01/'
save_path = './Energy/01/submission/'
os.makedirs(save_path, exist_ok=True)

seed_file = "./Energy/01/(SEED_COUNT)01_preprocessing.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 51# seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.1,
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': 0
            }

lgbm_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': -1,
                'force_col_wise': True
            }

cat_params = {
                'loss_function': 'MAE',
                'iterations': 500,
                'learning_rate': 0.08,
                'depth': 5,
                'l2_leaf_reg': 2,
                'subsample': 0.9,
                'colsample_bylevel': 0.9,
                'random_strength': 0.5,
                'bagging_temperature': 0.5,
                'border_count': 32,
                'random_seed': SEED,
                'thread_count': -1,
                'verbose': False,
                'allow_writing_files': False
            }
########################### 전처리 함수 정의 ############################

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def feature_engineering(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['일시'] = pd.to_datetime(df['일시'])
    df['시각'] = df['일시'].dt.hour                      # 시각(0~23)
    df['요일'] = df['일시'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['근무시간'] = df['시각'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)  # 주기적 패턴
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)
    df['정오거리'] = (df['시각'] - 12).abs()  # 12시(정오) 기준 거리
    df['정오거리_INV'] = 1 / (df['정오거리'] + 1)
    df['peak_time'] = df['시각'].apply(lambda x: 1 if 11 <= x <= 16 else 0) # Peak time (11 AM to 4 PM)

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

    return df


def add_sunshine_rolling_features(df):
    """일조시간 rolling features 추가"""
    df = df.copy()
    
    # 일조시간 이전 3시간 rolling features
    if '일조(hr)' in df.columns:
        for lag in range(1, 4):
            df[f'일조_변화량_{lag}h'] = df.groupby('건물번호')['일조(hr)'].diff(periods=lag)
            df[f'일조_rolling_mean_{lag}h'] = df.groupby('건물번호')['일조(hr)'].rolling(window=lag, min_periods=1).mean().reset_index(0, drop=True)
            df[f'일조_rolling_sum_{lag}h'] = df.groupby('건물번호')['일조(hr)'].rolling(window=lag, min_periods=1).sum().reset_index(0, drop=True)
    
    df = df.fillna(0)
    return df

def add_insolation_rolling_features(df):
    """일사량 rolling features 추가 (저장 직전)"""
    df = df.copy()
    
    # 일사량 이전 3시간 rolling features
    if '일사(MJ/m2)' in df.columns:
        for lag in range(1, 4):
            df[f'일사_변화량_{lag}h'] = df.groupby('건물번호')['일사(MJ/m2)'].diff(periods=lag)
            df[f'일사_rolling_mean_{lag}h'] = df.groupby('건물번호')['일사(MJ/m2)'].rolling(window=lag, min_periods=1).mean().reset_index(0, drop=True)
            df[f'일사_rolling_sum_{lag}h'] = df.groupby('건물번호')['일사(MJ/m2)'].rolling(window=lag, min_periods=1).sum().reset_index(0, drop=True)
    
    df = df.fillna(0)
    return df

def add_weather_rolling_features(df):
    """기온, 풍속, 강수량, 습도 rolling features 추가"""
    df = df.copy()
    
    # 롤링 피처를 생성할 컬럼 목록
    weather_cols = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)']
    
    for col in weather_cols:
        if col in df.columns:
            for lag in range(1, 4):
                # 변화량: 이전 시점과의 차이
                df[f'{col.split("(")[0]}_변화량_{lag}h'] = df.groupby('건물번호')[col].diff(periods=lag)
                
                # 롤링 평균: 이전 n시간의 평균
                df[f'{col.split("(")[0]}_rolling_mean_{lag}h'] = df.groupby('건물번호')[col].rolling(window=lag, min_periods=1).mean().reset_index(0, drop=True)
                
                # 롤링 합계: 이전 n시간의 합계 (강수량에 특히 유용)
                df[f'{col.split("(")[0]}_rolling_sum_{lag}h'] = df.groupby('건물번호')[col].rolling(window=lag, min_periods=1).sum().reset_index(0, drop=True)
    
    df = df.fillna(0)
    return df


###########################################################################

print(f"[1] 전처리 시작")

########################## building_info 전처리 ############################

building_csv = pd.read_csv(data_path + 'building_info.csv')
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

########################## train 전처리 ############################
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
train_all = feature_engineering(train)
train_all = add_weather_rolling_features(train_all)
train_all = add_sunshine_rolling_features(train_all)
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print(f"[2] train zero_bno 일사 예측 시작")

# 예측에 사용할 피처 선택
insolation_features = ['건물번호', '기온(°C)', '일조(hr)',
                       'SIN_시', 'COS_시',  '정오거리','시각','요일', '월', 
                       '정오거리_INV', 'peak_time']
insolation_features.extend([col for col in train_all.columns if '요일_' in col])
insolation_features.extend([col for col in train_all.columns if '건물유형_' in col])
insolation_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features = list(set(insolation_features))

# '일사(MJ/m2)' 값이 0이 아닌 건물 데이터로 모델 학습 (전체 데이터 사용)
train_insolation_train = train_all[~train_all['건물번호'].isin(zero_bnos) & (train_all['일사(MJ/m2)'] > 0)].copy()
X_train_insolation = train_insolation_train[insolation_features]
y_train_insolation = train_insolation_train['일사(MJ/m2)']

print(f"전체 데이터로 일사량 모델 학습 중...")

# 전체 데이터로 모델 학습
xgb_model_full = XGBRegressor(**xgb_params)
lgbm_model_full = LGBMRegressor(**lgbm_params)

xgb_model_full.fit(X_train_insolation, y_train_insolation)
lgbm_model_full.fit(X_train_insolation, y_train_insolation)

print(f"전체 모델 학습 완료, 이제 건물별로 예측 진행...")

# 건물별로 예측 수행
mae_scores_insolation = {}
for bno in zero_bnos:
    print(f"    > 건물 {bno}번 일사량 예측 시작...")
    
    # 해당 건물의 일사량이 0인 데이터 선택
    building_data = train_all[(train_all['건물번호'] == bno) & (train_all['일사(MJ/m2)'] == 0)].copy()
    
    if len(building_data) == 0:
        print(f"    > 건물 {bno}번: 예측할 데이터가 없음")
        continue
    
    X_building = building_data[insolation_features]
    
    # 전체 모델로 예측
    xgb_pred_building = xgb_model_full.predict(X_building)
    lgbm_pred_building = lgbm_model_full.predict(X_building)
    ensemble_pred_building = (xgb_pred_building + lgbm_pred_building) / 2
    
    # 음수 값 클리핑
    ensemble_pred_building = np.clip(ensemble_pred_building, 0, None)
    
    # 원본 데이터에 예측값 업데이트
    train_all.loc[building_data.index, '일사(MJ/m2)'] = ensemble_pred_building
    
    # validation을 위한 K-Fold (해당 건물만)
    if len(building_data) > 10:  # 충분한 데이터가 있을 때만 validation 수행
        kf = KFold(n_splits=min(5, len(building_data)//2), shuffle=True, random_state=SEED)
        val_mae = 0
        fold_count = 0
        
        for train_idx, val_idx in kf.split(X_building):
            X_train_fold = X_train_insolation
            y_train_fold = y_train_insolation
            X_val_fold = X_building.iloc[val_idx]
            
            # 실제 예측값으로 validation (원본 모델 사용)
            xgb_val_pred = xgb_model_full.predict(X_val_fold)
            lgbm_val_pred = lgbm_model_full.predict(X_val_fold)
            ensemble_val_pred = (xgb_val_pred + lgbm_val_pred) / 2
            ensemble_val_pred = np.clip(ensemble_val_pred, 0, None)
            
            # 실제 정답이 없으므로 예측의 일관성을 평가 (실제로는 의미 없음)
            # 대신 예측값의 통계 출력
            fold_count += 1
        
        print(f"    > 건물 {bno}번: 예측 완료 (예측된 일사량 범위: {ensemble_pred_building.min():.4f} ~ {ensemble_pred_building.max():.4f})")
    else:
        print(f"    > 건물 {bno}번: 예측 완료 (데이터 부족으로 validation 생략)")

# 21시부터 05시까지의 일사량을 0으로 설정
train_all.loc[(train_all['건물번호'].isin(zero_bnos)) & ((train_all['시각'] >= 21) | (train_all['시각'] <= 5)), '일사(MJ/m2)'] = 0

train_all = add_insolation_rolling_features(train_all)
print(f"[3] train zero_bno 일사 예측 완료")


########################## test 전처리 ############################

test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')
test_all = feature_engineering(test)
test_all = add_weather_rolling_features(test_all)

print(f"[4] test 일조시간 지역별 예측 시작")

# 지역별 건물 그룹 정의
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

group_names = ['강릉', '경주시', '광주', '부안', '구미', '김해시', '대구', '대전', '부산', '서울', 
               '세종', '속초', '수원', '안동', '양산시', '울산', '인천', '임실', '전주', '제천',
               '진주', '창원', '천안', '청주', '춘천', '충주', '파주', '포항', '홍천']

# 예측에 사용할 피처 선택
sunshine_features = ['건물번호', '기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)',
                       '시각', '요일', '주말여부', '월', '정오거리', '정오거리_INV', 'peak_time',
                       '태양광용량(kW)', '냉방면적(m2)']

sunshine_features.extend([col for col in train_all.columns if '요일_' in col])
sunshine_features.extend([col for col in train_all.columns if '건물유형_' in col])
sunshine_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
sunshine_features = list(set(sunshine_features))

# 훈련 데이터셋 정의
X_train_sunshine = train_all[sunshine_features]
y_train_sunshine = train_all['일조(hr)']

# 지역별로 예측 수행
mae_scores_sunshine = {}

for group_idx, building_group in enumerate(building_groups):
    group_name = group_names[group_idx]
    print(f"    > {group_name} 지역 (건물: {building_group}) 일조시간 예측 시작...")
    
    # 해당 지역의 테스트 데이터 선택
    group_test_data = test_all[test_all['건물번호'].isin(building_group)].copy()
    
    if len(group_test_data) == 0:
        print(f"    > {group_name} 지역: 테스트 데이터가 없음")
        continue
    
    X_test_group = group_test_data[sunshine_features]
    
    # 해당 지역의 train 데이터만 선택 (validation용)
    group_train_data = train_all[train_all['건물번호'].isin(building_group)].copy()
    
    if len(group_train_data) < 10:  # 데이터가 너무 적으면 전체 데이터 사용
        print(f"    > {group_name} 지역: 훈련 데이터 부족 ({len(group_train_data)}개), 전체 데이터로 validation")
        X_group_train = X_train_sunshine
        y_group_train = y_train_sunshine
    else:
        X_group_train = group_train_data[sunshine_features]
        y_group_train = group_train_data['일조(hr)']
    
    # K-Fold 교차 검증으로 모델 학습 및 예측
    n_splits = min(5, len(X_group_train) // 2) if len(X_group_train) < len(X_train_sunshine) else 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    ensemble_predictions = np.zeros(len(X_test_group))
    val_mae = 0
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_sunshine, y_train_sunshine)):
        # 전체 데이터로 모델 학습 (성능 유지)
        X_train_fold = X_train_sunshine.iloc[train_index]
        y_train_fold = y_train_sunshine.iloc[train_index]
        
        # 해당 지역 데이터로만 validation
        if len(X_group_train) < len(X_train_sunshine):
            # 지역별 데이터가 있는 경우
            val_indices = np.arange(len(X_group_train))
            np.random.seed(SEED + fold)
            np.random.shuffle(val_indices)
            val_size = max(1, len(val_indices) // n_splits)
            start_idx = fold * val_size
            end_idx = min((fold + 1) * val_size, len(val_indices))
            group_val_indices = val_indices[start_idx:end_idx]
            
            X_val_fold = X_group_train.iloc[group_val_indices]
            y_val_fold = y_group_train.iloc[group_val_indices]
        else:
            # 전체 데이터 사용하는 경우
            X_val_fold = X_train_sunshine.iloc[val_index]
            y_val_fold = y_train_sunshine.iloc[val_index]
        
        # 모델 학습
        xgb_model = XGBRegressor(**xgb_params)
        lgbm_model = LGBMRegressor(**lgbm_params)
        
        xgb_model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      verbose=False)
        lgbm_model.fit(X_train_fold, y_train_fold,
                       eval_set=[(X_val_fold, y_val_fold)],
                       callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)])
        
        # 해당 지역 예측
        xgb_pred = xgb_model.predict(X_test_group)
        lgbm_pred = lgbm_model.predict(X_test_group)
        ensemble_predictions += (xgb_pred + lgbm_pred) / 2 / n_splits
        
        # Validation MAE 계산 (해당 지역 데이터로만)
        xgb_val = xgb_model.predict(X_val_fold)
        lgbm_val = lgbm_model.predict(X_val_fold)
        ensemble_val = (xgb_val + lgbm_val) / 2
        fold_mae = mean_absolute_error(y_val_fold, ensemble_val)
        val_mae += fold_mae / n_splits
    
    # 예측값 클리핑 및 저장
    ensemble_predictions = np.clip(ensemble_predictions, 0, 1)
    test_all.loc[test_all['건물번호'].isin(building_group), '일조(hr)'] = ensemble_predictions
    
    mae_scores_sunshine[group_name] = val_mae
    print(f"    > {group_name} 지역 일조시간 예측 완료 - Validation MAE: {val_mae:.6f} (훈련 데이터: {len(X_group_train)}개)")

# 오후 8시부터 오전 6시까지의 일조시간을 0으로 설정
test_all.loc[(test_all['시각'] >= 20) | (test_all['시각'] <= 6), '일조(hr)'] = 0

test_all = add_sunshine_rolling_features(test_all)
print(f"[5] test 일조시간 지역별 예측 완료")
print(f"일조시간 예측 전체 평균 MAE: {np.mean(list(mae_scores_sunshine.values())):.6f}")


print(f"[6] test 일사량 지역별 예측 시작")

# 예측에 사용할 피처 선택
insolation_features_test = sunshine_features + ['일조(hr)']
insolation_features_test.extend([col for col in test_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features_test = list(set(insolation_features_test))

# 훈련 데이터셋 정의
X_train_insolation_test = train_all[insolation_features_test]
y_train_insolation_test = train_all['일사(MJ/m2)']

# 지역별로 예측 수행
mae_scores_insolation_test = {}

for group_idx, building_group in enumerate(building_groups):
    group_name = group_names[group_idx]
    print(f"    > {group_name} 지역 (건물: {building_group}) 일사량 예측 시작...")
    
    # 해당 지역의 테스트 데이터 선택
    group_test_data = test_all[test_all['건물번호'].isin(building_group)].copy()
    
    if len(group_test_data) == 0:
        print(f"    > {group_name} 지역: 테스트 데이터가 없음")
        continue
    
    X_test_group = group_test_data[insolation_features_test]
    
    # 해당 지역의 train 데이터만 선택 (validation용)
    group_train_data = train_all[train_all['건물번호'].isin(building_group)].copy()
    
    if len(group_train_data) < 10:  # 데이터가 너무 적으면 전체 데이터 사용
        print(f"    > {group_name} 지역: 훈련 데이터 부족 ({len(group_train_data)}개), 전체 데이터로 validation")
        X_group_train = X_train_insolation_test
        y_group_train = y_train_insolation_test
    else:
        X_group_train = group_train_data[insolation_features_test]
        y_group_train = group_train_data['일사(MJ/m2)']
    
    # K-Fold 교차 검증으로 모델 학습 및 예측
    n_splits = min(5, len(X_group_train) // 2) if len(X_group_train) < len(X_train_insolation_test) else 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    ensemble_predictions = np.zeros(len(X_test_group))
    val_mae = 0
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_insolation_test, y_train_insolation_test)):
        # 전체 데이터로 모델 학습 (성능 유지)
        X_train_fold = X_train_insolation_test.iloc[train_index]
        y_train_fold = y_train_insolation_test.iloc[train_index]
        
        # 해당 지역 데이터로만 validation
        if len(X_group_train) < len(X_train_insolation_test):
            # 지역별 데이터가 있는 경우
            val_indices = np.arange(len(X_group_train))
            np.random.seed(SEED + fold)
            np.random.shuffle(val_indices)
            val_size = max(1, len(val_indices) // n_splits)
            start_idx = fold * val_size
            end_idx = min((fold + 1) * val_size, len(val_indices))
            group_val_indices = val_indices[start_idx:end_idx]
            
            X_val_fold = X_group_train.iloc[group_val_indices]
            y_val_fold = y_group_train.iloc[group_val_indices]
        else:
            # 전체 데이터 사용하는 경우
            X_val_fold = X_train_insolation_test.iloc[val_index]
            y_val_fold = y_train_insolation_test.iloc[val_index]
        
        # 모델 학습
        xgb_model = XGBRegressor(**xgb_params)
        lgbm_model = LGBMRegressor(**lgbm_params)
        
        xgb_model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      verbose=False)
        lgbm_model.fit(X_train_fold, y_train_fold,
                       eval_set=[(X_val_fold, y_val_fold)],
                       callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)])
        
        # 해당 지역 예측
        xgb_pred = xgb_model.predict(X_test_group)
        lgbm_pred = lgbm_model.predict(X_test_group)
        ensemble_predictions += (xgb_pred + lgbm_pred) / 2 / n_splits
        
        # Validation MAE 계산 (해당 지역 데이터로만)
        xgb_val = xgb_model.predict(X_val_fold)
        lgbm_val = lgbm_model.predict(X_val_fold)
        ensemble_val = (xgb_val + lgbm_val) / 2
        fold_mae = mean_absolute_error(y_val_fold, ensemble_val)
        val_mae += fold_mae / n_splits
    
    # 예측값 클리핑 및 저장
    ensemble_predictions = np.clip(ensemble_predictions, 0, None)
    test_all.loc[test_all['건물번호'].isin(building_group), '일사(MJ/m2)'] = ensemble_predictions
    
    mae_scores_insolation_test[group_name] = val_mae
    print(f"    > {group_name} 지역 일사량 예측 완료 - Validation MAE: {val_mae:.6f} (훈련 데이터: {len(X_group_train)}개)")

# 오후 9시부터 오전 5시까지의 일사량을 0으로 설정
test_all.loc[(test_all['시각'] >= 21) | (test_all['시각'] <= 5), '일사(MJ/m2)'] = 0

test_all = add_insolation_rolling_features(test_all)
print(f"[7] test 일사량 지역별 예측 완료")
print(f"일사량 예측 전체 평균 MAE: {np.mean(list(mae_scores_insolation_test.values())):.6f}")

# 결과 저장
train_all.to_csv(csv_path + 'predict_train.csv', index=False)
test_all.to_csv(csv_path + 'predict_test.csv', index=False)

print(f"[8] 저장 완료")

# 최종 결과 요약
print("\n" + "="*50)
print("최종 결과 요약")
print("="*50)
print(f"Test 일조시간 예측 전체 평균 MAE: {np.mean(list(mae_scores_sunshine.values())):.6f}")
print(f"Test 일사량 예측 전체 평균 MAE: {np.mean(list(mae_scores_insolation_test.values())):.6f}")

print("\n지역별 일조시간 예측 MAE:")
for group_name in sorted(mae_scores_sunshine.keys()):
    print(f"  {group_name} 지역: {mae_scores_sunshine[group_name]:.6f}")

print("\n지역별 일사량 예측 MAE:")
for group_name in sorted(mae_scores_insolation_test.keys()):
    print(f"  {group_name} 지역: {mae_scores_insolation_test[group_name]:.6f}")


with open(csv_path + "(LOG)01_preprocessing.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"Overall Average 일조(hr) MAE : {np.mean(list(mae_scores_sunshine.values())):.6f}\n")
    f.write(f"Overall Average 일사(MJ/m2) MAE : {np.mean(list(mae_scores_insolation_test.values())):.6f}\n")
    f.write("="*40 + "\n")
    
print(f"[03_preprocessing] 종료")