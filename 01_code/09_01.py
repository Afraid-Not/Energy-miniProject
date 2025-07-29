import os
import pandas as pd
import numpy as np
import random
import datetime, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from lightgbm.callback import early_stopping

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
# import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

seed_file = "./Energy/09_00_submission/09_00_seed.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 1 # seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

# 경로 설정


data_path = './Energy/'

# 데이터 로드
buildinginfo = pd.read_csv(data_path + 'building_info.csv')
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
samplesub = pd.read_csv(data_path + 'sample_submission.csv')

print("[1] 데이터 로딩 및 초기 전처리 완료")

# 결측치 처리
for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = np.log1p(buildinginfo[col].replace('-', 0).astype(float))

# Feature Engineering
    def feature_engineering(df):
        df = df.copy()
        df['일시'] = pd.to_datetime(df['일시'])
        df['hour'] = df['일시'].dt.hour
        df['dayofweek'] = df['일시'].dt.dayofweek
        df['month'] = df['일시'].dt.month
        df['day'] = df['일시'].dt.day
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_working_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

        temp = df['기온(°C)']
        humidity = df['습도(%)']
        wind_speed = df['풍속(m/s)'] # 풍속 컬럼 사용

        # 기존 불쾌지수(DI)
        df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32

        # **새로운 체감 온도(AT) 피처 추가 (호주 기상청 공식)**
        e = (humidity / 100) * 6.105 * np.exp((17.27 * temp) / (237.7 + temp))
        
        # 2. AT 계산
        df['AT'] = temp + (0.33 * e) - (0.70 * wind_speed) - 4.00
        
        # 냉방 도일 (CDD) 피처 추가
        # 1. 날짜만 추출
        df['date'] = df['일시'].dt.date
        
        # 2. 일별 평균 기온 계산
        daily_avg_temp = df.groupby(['건물번호', 'date'])['기온(°C)'].mean().reset_index()
        daily_avg_temp.rename(columns={'기온(°C)': 'daily_avg_temp'}, inplace=True)
        
        # 3. 기준 온도 설정 (예: 24도)
        base_temp = 24
        
        # 4. 일별 CDD 계산
        daily_avg_temp['daily_cdd'] = (daily_avg_temp['daily_avg_temp'] - base_temp).apply(lambda x: max(0, x))
        
        # 5. 건물번호, 연도, 월별로 CDD 누적 합계 계산
        daily_avg_temp['month'] = pd.to_datetime(daily_avg_temp['date']).dt.month
        daily_avg_temp['year'] = pd.to_datetime(daily_avg_temp['date']).dt.year
        
        # CDD는 누적 개념이므로, 각 월의 일별 CDD를 합산하여 월별 누적 CDD 생성
        monthly_cdd = daily_avg_temp.groupby(['건물번호', 'year', 'month'])['daily_cdd'].sum().reset_index()
        monthly_cdd.rename(columns={'daily_cdd': 'monthly_cdd'}, inplace=True)
        
        # 원래 데이터프레임에 월별 CDD 병합
        df['year'] = df['일시'].dt.year # year 컬럼 추가
        df = pd.merge(df, monthly_cdd, on=['건물번호', 'year', 'month'], how='left')

        # CDD 계산에 사용된 임시 컬럼 삭제
        df = df.drop(columns=['date', 'year']) 
        
        return df

# 전처리
train = feature_engineering(train)
test = feature_engineering(test)
train = train.merge(buildinginfo, on='건물번호', how='left')
test = test.merge(buildinginfo, on='건물번호', how='left')
train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes

train_day_ohe = pd.get_dummies(train['dayofweek'], prefix='day', dtype=int)
test_day_ohe = pd.get_dummies(test['dayofweek'], prefix='day', dtype=int)
train_day_ohe, test_day_ohe = train_day_ohe.align(test_day_ohe, join='outer', axis=1, fill_value=0)

train = pd.concat([train, train_day_ohe], axis=1)
test = pd.concat([test, test_day_ohe], axis=1)

# print(train.columns)
# print(test.columns)

# Index(['num_date_time', '건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#        '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)', 'hour', 'dayofweek', 'month',
#        'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'AT',
#        'monthly_cdd', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
#        'ESS저장용량(kWh)', 'PCS용량(kW)', 'day_0', 'day_1', 'day_2', 'day_3',
#        'day_4', 'day_5', 'day_6'],

# print(np.unique(train['AT'], return_counts=True))
# print(np.max(train['AT']))
# print(np.min(train['AT']))
# exit()
# --------------------------
# 2. 원핫 인코딩
# --------------------------

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor

# ----------------------------
# 일사 결측치 예측 보조모델 (건물 유형별 훈련)
# ----------------------------
print("[2] 일사 결측치 예측 보조 모델")

empty = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
#0.1316
# 일사 예측에 사용할 컬럼 정의 (이전과 동일)
etr_col = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    'is_working_hours', 'sin_hour', 'cos_hour', 'DI', '일조(hr)', '일사(MJ/m2)'
]

etest_col = etr_col[:-1]

# 결측치가 있는 건물들의 건물 유형을 가져옵니다.
empty_building_types = train[train['건물번호'].isin(empty)]['건물유형'].unique()

# 예측 결과를 저장할 빈 DataFrame 초기화
# pred_e_all의 인덱스를 미리 empty 건물들의 인덱스로 설정하여 최종 반영 시 편리하게 합니다.
pred_e_all = pd.Series(index=train[train['건물번호'].isin(empty)].index, dtype=float)
e_rmse_total = [] # 각 건물 유형별 RMSE를 저장할 리스트

for b_type in empty_building_types:
    print(f"    > 건물유형: {b_type}")
    
    # 해당 건물 유형에 속하는 데이터만 필터링 (훈련 데이터)
    train_e_type = train[~train['건물번호'].isin(empty)][train['건물유형'] == b_type].copy()
    
    # 해당 건물 유형에 속하며 결측치가 있는 데이터 (예측 대상)
    test_e_type = train[train['건물번호'].isin(empty)][train['건물유형'] == b_type].copy()

    # 해당 유형의 데이터가 없으면 건너김
    if train_e_type.empty or test_e_type.empty:
        print(f"    > 건물유형 {b_type}에 대한 훈련/예측 데이터 부족, 건너뜜.")
        # 데이터가 부족하면 해당 유형의 결측치는 0으로 채우도록 합니다.
        if not test_e_type.empty:
            pred_e_all.loc[test_e_type.index] = 0.0
        continue
    x1 = train_e_type[etr_col].drop(['일사(MJ/m2)'], axis=1)
    y1 = train_e_type['일사(MJ/m2)']
    x1_test_final_pred = test_e_type[etest_col].copy() # KFold 후 최종 예측에 사용될 test_e 데이터

    # KFold 적용을 위한 최소 데이터 확인 (KFold n_splits보다 커야 함)
    if len(x1) < 5: # n_splits=5라고 가정했을 때 최소 5개 샘플 필요
        print(f"   > 건물유형 {b_type}: KFold 적용을 위한 데이터 부족 ({len(x1)}개), 단일 train_test_split으로 대체.")
        # 데이터가 너무 적으면 KFold 대신 단일 train_test_split으로 대체
        current_train_size = min(0.8, (len(x1) - 1) / len(x1))
        if current_train_size <= 0.0:
            current_train_size = 0.9 
        
        x1_train_single, x1_val_single, y1_train_single, y1_val_single = train_test_split(
            x1, y1, random_state=SEED, shuffle=True, train_size=current_train_size
        )

        model_e_single = CatBoostRegressor(
            iterations=3000, 
            learning_rate=0.01, 
            depth=6, 
            l2_leaf_reg=5, 
            bagging_temperature=1.0, 
            subsample=0.8, 
            loss_function='Huber:delta=1.0', 
            early_stopping_rounds=100, 
            random_state=SEED,
            verbose=0
        )
        model_e_single.fit(x1_train_single, y1_train_single, eval_set=[(x1_val_single, y1_val_single)], verbose=0)
        
        results_single = model_e_single.predict(x1_val_single)
        rmse_single = np.sqrt(mean_squared_error(y1_val_single, results_single))
        e_rmse_total.append(rmse_single)
        print(f"    > 단일 학습 RMSE : {rmse_single:.4f}")

        pred_e_p = model_e_single.predict(x1_test_final_pred)

    else: # 데이터가 충분하여 KFold 적용
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED) # 5-Fold Cross-Validation
        fold_rmses = []
        fold_preds = [] # 각 폴드의 x1_test_final_pred에 대한 예측을 저장

        for fold, (trn_idx, val_idx) in enumerate(kf.split(x1, y1)):
            # print(f"   > KFold {fold+1}/{kf.n_splits} - Building {b_type}") # 상세 로그는 필요 시 주석 해제

            x1_train_fold, x1_val_fold = x1.iloc[trn_idx], x1.iloc[val_idx]
            y1_train_fold, y1_val_fold = y1.iloc[trn_idx], y1.iloc[val_idx]

            model_e_fold = CatBoostRegressor(
                iterations=3000, 
                learning_rate=0.01, 
                depth=6, 
                l2_leaf_reg=5, 
                bagging_temperature=1.0, 
                subsample=0.8, 
                loss_function='Huber:delta=1.0', 
                early_stopping_rounds=100, 
                random_state=SEED,
                verbose=0
            )

            model_e_fold.fit(x1_train_fold, y1_train_fold, eval_set=[(x1_val_fold, y1_val_fold)], verbose=0)
            
            results_fold = model_e_fold.predict(x1_val_fold)
            rmse_fold = np.sqrt(mean_squared_error(y1_val_fold, results_fold))
            fold_rmses.append(rmse_fold)
            # print(f"     > Fold RMSE: {rmse_fold:.4f}")

            fold_preds.append(model_e_fold.predict(x1_test_final_pred))
        
        e_rmse = np.mean(fold_rmses) # KFold의 평균 RMSE
        e_rmse_total.append(e_rmse)
        print(f"    > KFold 평균 RMSE : {e_rmse:.4f}")

        # 모든 폴드 예측의 평균
        pred_e_p = np.mean(fold_preds, axis=0) 
    
    pred_e_max = np.maximum(pred_e_p, 0)
    pred_e_round = np.round(pred_e_max, 2)
    pred_e_type_result = np.clip(pred_e_round, 0.1, None)

    # 해당 시간 추출 및 야간 처리
    hours = train.loc[test_e_type.index, 'hour']
    night_mask = (hours >= 21) | (hours <= 5)
    
    pred_e_type_result_np = np.array(pred_e_type_result)
    pred_e_type_result_np[night_mask.values] = 0.0
    
    pred_e_all.loc[test_e_type.index] = pred_e_type_result_np


# 모든 유형에 대한 평균 RMSE (선택 사항)
if e_rmse_total:
    print(f"\n    > 일사량 RMSE : {np.mean(e_rmse_total):.4f}")

# 소수점 2자리 반올림 후 원래 위치에 덮어쓰기
# pred_e_all은 이미 올바른 인덱스를 가지고 있으므로 바로 loc로 반영합니다.
train.loc[pred_e_all.index, '일사(MJ/m2)'] = pred_e_all.values

print(f"\n    > ✅ 일사 결측치 {len(pred_e_all)}건 train에 반영 완료")

# 확인
# print(f"    > 일사 SMAPE : {e_rmse}")
# exit


print("[3] 일조/일사 보조모델 학습 및 예측")

print("    > 🌞 [SUN] 건물별 보조모델 학습 시작")

sun_feature_cols = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'DI', 'AT',
    'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour',
]

day_ohe_cols = [col for col in train.columns if col.startswith('day_')]
sun_feature_cols.extend(day_ohe_cols)

sun_target_cols = ['일조(hr)', '일사(MJ/m2)']

sun_test_preds_iljo = []
sun_test_preds_ilsa = []
sun_rmses_iljo = []
sun_rmses_ilsa = []

building_ids = sorted(train['건물번호'].unique())
for bno in building_ids:
    print(f"    > Building {bno}")

    train_b = train[train['건물번호'] == bno].dropna(subset=sun_target_cols).copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(train_b) < 10:
        print(f"    > ⚠️ 학습 데이터 부족 → 0으로 채움 ({len(train_b)}개)")
        sun_test_preds_iljo.append(np.zeros(len(test_b)))
        sun_test_preds_ilsa.append(np.zeros(len(test_b)))
        continue

    current_sun_feature_cols = [col for col in sun_feature_cols if col in train_b.columns and col in test_b.columns]
    
    ss_sun = StandardScaler()
    X_scaled = ss_sun.fit_transform(train_b[current_sun_feature_cols])
    X_test_scaled = ss_sun.transform(test_b[current_sun_feature_cols])

    X = pd.DataFrame(X_scaled, columns=current_sun_feature_cols, index=train_b.index)
    X_test = pd.DataFrame(X_test_scaled, columns=current_sun_feature_cols, index=test_b.index)
    
    y_iljo = train_b['일조(hr)']
    y_ilsa = train_b['일사(MJ/m2)']

    if y_iljo.nunique() < 2:
        print(f"    > ⚠️ 일조(hr) 타겟 값이 모두 동일 → 학습 생략")
        iljo_pred_test = np.zeros(len(test_b))
    else:
        X_trn_iljo, X_val_iljo, y_trn_iljo, y_val_iljo = train_test_split(
            X, y_iljo, train_size=0.8, random_state=SEED
        )
        model_iljo = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            loss_function='RMSE',
            early_stopping_rounds=50,
            random_state=SEED,
            verbose=0
        )
        model_iljo.fit(X_trn_iljo, y_trn_iljo, eval_set=[(X_val_iljo, y_val_iljo)], early_stopping_rounds=50, verbose=0)
        
        y_pred_val_iljo = model_iljo.predict(X_val_iljo)
        rmse_iljo = np.sqrt(mean_squared_error(y_val_iljo, y_pred_val_iljo))
        sun_rmses_iljo.append(rmse_iljo)
        print(f"    > ✅ 일조(hr) 검증 RMSE: {rmse_iljo:.6f}")
        iljo_pred_test = model_iljo.predict(X_test)
        iljo_pred_test = np.round(np.maximum(iljo_pred_test, 0), 1)

    if y_ilsa.nunique() < 2:
        print(f"    > ⚠️ 일사(MJ/m2) 타겟 값이 모두 동일 → 학습 생략")
        ilsa_pred_test = np.zeros(len(test_b))
    else:
        X_trn_ilsa, X_val_ilsa, y_trn_ilsa, y_val_ilsa = train_test_split(
            X, y_ilsa, train_size=0.8, random_state=SEED
        )
        model_ilsa = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=SEED,
            verbose=-1,
            objective='huber'
        )
        model_ilsa.fit(X_trn_ilsa, y_trn_ilsa, eval_set=[(X_val_ilsa, y_val_ilsa)], 
                        callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
        
        y_pred_val_ilsa = model_ilsa.predict(X_val_ilsa)
        rmse_ilsa = np.sqrt(mean_squared_error(y_val_ilsa, y_pred_val_ilsa))
        sun_rmses_ilsa.append(rmse_ilsa)
        print(f"    > ✅ 일사(MJ/m2) 검증 RMSE: {rmse_ilsa:.6f}")
        ilsa_pred_test = model_ilsa.predict(X_test)
        ilsa_pred_test = np.round(np.maximum(ilsa_pred_test, 0), 2)

    sun_test_preds_iljo.append(iljo_pred_test)
    sun_test_preds_ilsa.append(ilsa_pred_test)

iljo_preds_dict = {}
ilsa_preds_dict = {}

for i, bno in enumerate(building_ids):
    if len(train[train['건물번호'] == bno].dropna(subset=sun_target_cols)) < 10:
        test_b_indices = test[test['건물번호'] == bno].index
        for idx in test_b_indices:
            iljo_preds_dict[idx] = 0.0
            ilsa_preds_dict[idx] = 0.0
        continue
    
    iljo_preds_dict.update(pd.Series(sun_test_preds_iljo[i], index=test[test['건물번호'] == bno].index).to_dict())
    ilsa_preds_dict.update(pd.Series(sun_test_preds_ilsa[i], index=test[test['건물번호'] == bno].index).to_dict())

test['일조(hr)'] = pd.Series(iljo_preds_dict)
test['일사(MJ/m2)'] = pd.Series(ilsa_preds_dict)

if '시간' not in test.columns and '일시' in test.columns:
    test['시간'] = pd.to_datetime(test['일시']).dt.hour

test.loc[(test['시간'] < 5) | (test['시간'] > 21), ['일조(hr)', '일사(MJ/m2)']] = 0

final_rmse_iljo = np.mean(sun_rmses_iljo)
final_rmse_ilsa = np.mean(sun_rmses_ilsa)

print(f"\n    > 🌞 건물별 검증 평균 일조(hr) RMSE: {final_rmse_iljo:.6f}")
print(f"    > 🌞 건물별 검증 평균 일사(MJ/m2) RMSE: {final_rmse_ilsa:.6f}")

print("\n[4] 전처리 완료")

# print(train.columns)
# Index(['num_date_time', '건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#        '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)', 'hour', 'dayofweek', 'month',
#        'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'AT',
#        'monthly_cdd', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
#        'ESS저장용량(kWh)', 'PCS용량(kW)', 'day_0', 'day_1', 'day_2', 'day_3',
#        'day_4', 'day_5', 'day_6'],
#       dtype='object')


# print("[4] 전처리 완료")
# print(train.columns)

# exit()
features = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'hour', 'dayofweek', 'month',
       'day', 'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'AT', '건물유형', '연면적(m2)', '냉방면적(m2)', 
       '태양광용량(kW)', 'ESS저장용량(kWh)', '일조(hr)', '일사(MJ/m2)']
# features = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#        'hour', 'month', 'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'AT',
#        'monthly_cdd', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
#        'ESS저장용량(kWh)', 'PCS용량(kW)', 'day_0', 'day_1', 'day_2', 'day_3',
#        'day_4', 'day_5', 'day_6', '일조(hr)', '일사(MJ/m2)']
    # '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    # '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    # 'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    # 'is_working_hours', 'sin_hour', 'cos_hour', 'DI'

# features = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'hour', 'day', 'dayofweek', 'month',
#        'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', 'AT', '건물유형', '연면적(m2)', '냉방면적(m2)', 
#        '태양광용량(kW)', 'ESS저장용량(kWh)', '일조(hr)', '일사(MJ/m2)']


target = '전력소비량(kWh)'

# 최종 예측 결과 저장용
final_preds = []
val_smapes = []

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()

print("[5] 건물별 학습 시작")


for bno in building_ids:
    print(f"    > 🏢 건물번호 {bno} 모델링 중...")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    x = train_b[features]
    y = np.log1p(train_b[target])
    x_test_final = test_b[features]

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=SEED)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_final_scaled = scaler.transform(x_test_final)

    # Base models
    xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                             random_state=SEED, early_stopping_rounds=50, objective='reg:squarederror')
    xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

    lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                              random_state=SEED, objective='huber', verbose=-1)
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                  random_seed=SEED, verbose=0, loss_function='Huber:delta=1.0')
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
        xgb_model.predict(x_test_final_scaled),
        lgb_model.predict(x_test_final_scaled),
        cat_model.predict(x_test_final_scaled)
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
    val_smapes.append(val_smape)

    pred = np.expm1(final_model.predict(test_pred_lvl2.reshape(-1, 1)))
    final_preds.extend(pred)

# 결과 저장
final_preds = np.clip(final_preds, 0.1, None)
samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
score_str = f"{avg_smape:.4f}".replace('.', '_')
filepath = './Energy/09_00_submission/'

os.makedirs(filepath, exist_ok=True)

filename = f"energy_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(filepath + filename, index=False)

print(f"[6] 📁 저장 완료 ")
print(f"✅ 최종 SMAPE 점수 : {avg_smape:.6f}")

# import shutil
# SCORE_THRESHOLD = 10
# if avg_smape > SCORE_THRESHOLD:
#     shutil.rmtree("./Energy/gift_submission")
#     print(f"🚫 Score {avg_smape:.5f} > 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
# else:
#     print(f"🎉 Score {avg_smape:.5f} < 기준 {SCORE_THRESHOLD} → 디렉토리 유지")
    
with open("./Energy/09_00_submission/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: {filename}\n")
    f.write(f"✅ 일사 RMSE : {e_rmse}\n")
    f.write(f"✅ 건물별 검증 평균 일조(hr) RMSE: {final_rmse_iljo:.6f}\n")
    f.write(f"✅ 건물별 검증 평균 일사(MJ/m2) RMSE: {final_rmse_ilsa:.6f}\n")
    f.write(f"✅ 최종 SMAPE 점수 : {avg_smape}\n")
    f.write("="*40 + "\n")
    
print(f"[7] 종료 ")
    