print(f"[01_preprocessing_train_predict] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import math
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping

warnings.filterwarnings('ignore')

data_path = './Energy/'
log_path = './Energy/100/log/'
trainer = './Energy/100/new_csv/'
save_path = './Energy/100/submission/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

seed_file = "./Energy/100/log/(SEED_COUNT)01_preprocessing.json"

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

pd.set_option('max_colwidth', None)

train = pd.read_csv(trainer + '00_train.csv')
test = pd.read_csv(trainer + '00_test.csv')

# print(train.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
#        '일사(MJ/m2)', '전력소비량(kWh)', 'date', 'hour', 'dayofweek', 'month', 'day',
#        'is_weekend', 'is_working', 'SIN_hour', 'COS_hour', 'SIN_day',
#        'COS_day', 'SIN_dayofweek', 'COS_dayofweek', 'peak_time', 'rainy',
#        'high_humidity', 'cloudy_or_rain', 'humid_x_rain', 'discomfort_index',
#        'THI', 'CDH', 'WCT', 'perceived_temperature', 'dew_point',
#        'cloudy_based_on_humidity', 'temp_date', 'day_of_year', 'sunrise_hour',
#        'sunset_hour', 'daylight', 'solar_elevation', 'solar_rel_pos',
#        'extraterrestrial_rad', '기온(°C)_change_1h', '기온(°C)_change_3h',
#        '기온(°C)_change_5h', '기온(°C)_change_12h', '기온(°C)_change_24h',
#        '풍속(m/s)_change_1h', '풍속(m/s)_change_3h', '풍속(m/s)_change_5h',
#        '풍속(m/s)_change_12h', '풍속(m/s)_change_24h', '습도(%)_change_1h',
#        '습도(%)_change_3h', '습도(%)_change_5h', '습도(%)_change_12h',
#        '습도(%)_change_24h', '강수량(mm)_change_1h', '강수량(mm)_change_3h',
#        '강수량(mm)_change_5h', '강수량(mm)_change_12h', '강수량(mm)_change_24h'],
#       dtype='object')
# print(test.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'date', 'hour',
#        'dayofweek', 'month', 'day', 'is_weekend', 'is_working', 'SIN_hour',
#        'COS_hour', 'SIN_day', 'COS_day', 'SIN_dayofweek', 'COS_dayofweek',
#        'peak_time', 'rainy', 'high_humidity', 'cloudy_or_rain', 'humid_x_rain',
#        'discomfort_index', 'THI', 'CDH', 'WCT', 'perceived_temperature',
#        'dew_point', 'cloudy_based_on_humidity', 'temp_date', 'day_of_year',
#        'sunrise_hour', 'sunset_hour', 'daylight', 'solar_elevation',
#        'solar_rel_pos', 'extraterrestrial_rad', '기온(°C)_change_1h',
#        '기온(°C)_change_3h', '기온(°C)_change_5h', '기온(°C)_change_12h',
#        '기온(°C)_change_24h', '풍속(m/s)_change_1h', '풍속(m/s)_change_3h',
#        '풍속(m/s)_change_5h', '풍속(m/s)_change_12h', '풍속(m/s)_change_24h',
#        '습도(%)_change_1h', '습도(%)_change_3h', '습도(%)_change_5h',
#        '습도(%)_change_12h', '습도(%)_change_24h', '강수량(mm)_change_1h',
#        '강수량(mm)_change_3h', '강수량(mm)_change_5h', '강수량(mm)_change_12h',
#        '강수량(mm)_change_24h'],
#       dtype='object')

############################ train 일사 보간 ###################################
############################ train 일사 보간 ###################################
############################ train 일사 보간 ###################################
print(f"[일사 보간] LGBM 기반 시작")

# 일사가 0인 건물들
zero_solar_buildings = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
print(f"일사 보간 대상 건물: {zero_solar_buildings}")

# 정상 일사 데이터를 가진 건물들
normal_buildings = [b for b in train['건물번호'].unique() if b not in zero_solar_buildings]
print(f"정상 일사 데이터 건물 수: {len(normal_buildings)}")

# 정상 건물들의 일사 분포 확인
normal_data = train[train['건물번호'].isin(normal_buildings)].copy()
print(f"정상 일사 데이터 - 평균: {normal_data['일사(MJ/m2)'].mean():.3f}, 최대: {normal_data['일사(MJ/m2)'].max():.3f}")
print(f"0값 비율: {(normal_data['일사(MJ/m2)'] == 0).mean():.3f}")

# 특성 선택 (일사 예측에 중요한 변수들)
feature_cols = [
    # 기상 데이터
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
    # 시간 특성
    'hour', 'month', 'day_of_year', 'dayofweek',
    'is_weekend', 'is_working',
    # 순환 인코딩
    'SIN_hour', 'COS_hour', 'SIN_day', 'COS_day',
    # 태양 관련
    'solar_elevation', 'daylight', 'sunrise_hour', 'sunset_hour',
    # 기상 지수
    'discomfort_index', 'THI', 'perceived_temperature', 'dew_point',
    # 파생 변수들
    'peak_time', 'rainy', 'high_humidity', 'cloudy_or_rain'
]

# extraterrestrial_rad 체크해서 적절한 범위면 추가
if 'extraterrestrial_rad' in train.columns:
    if train['extraterrestrial_rad'].max() < 50:
        feature_cols.append('extraterrestrial_rad')
        print("extraterrestrial_rad 특성 추가")

# 사용 가능한 특성만 필터링
available_features = [col for col in feature_cols if col in train.columns]
print(f"사용할 특성 수: {len(available_features)}")

# 훈련 데이터 준비 (정상 건물들만)
X_train = normal_data[available_features].copy()
y_train = normal_data['일사(MJ/m2)'].copy()

# 결측값 처리
X_train = X_train.fillna(0)

print(f"훈련 데이터 크기: {X_train.shape}")
print(f"타겟 분포 - 평균: {y_train.mean():.3f}, 최대: {y_train.max():.3f}")

# 샘플링 (10%만 사용하여 튜닝)
print("데이터 샘플링 (10%)...")
sample_size = int(len(X_train) * 0.1)
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train.iloc[sample_indices].copy()
y_sample = y_train.iloc[sample_indices].copy()
print(f"샘플 데이터 크기: {X_sample.shape}")

# 샘플 데이터를 train/val로 분할
X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=SEED
)

# Optuna 하이퍼파라미터 튜닝
print("Optuna 하이퍼파라미터 튜닝 시작...")
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'n_jobs': 1,  # 병렬 처리를 위해 1로 설정
        'verbose': -1
    }
    
    model = LGBMRegressor(**params, n_estimators=100)  # 빠른 튜닝을 위해 적은 estimators
    model.fit(X_train_sample, y_train_sample)
    
    preds = model.predict(X_val_sample)
    preds = np.maximum(preds, 0)  # 음수 제거
    
    # RMSE와 MAE 계산
    rmse = np.sqrt(mean_squared_error(y_val_sample, preds))
    mae = mean_absolute_error(y_val_sample, preds)
    
    # RMSE를 주요 지표로 사용하되 MAE도 기록
    trial.set_user_attr('mae', mae)
    return rmse

# Optuna 스터디 실행
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n=== Optuna 튜닝 결과 ===")
print(f"Best RMSE: {study.best_value:.4f}")
print(f"Best MAE: {study.best_trial.user_attrs['mae']:.4f}")
print(f"Best params: {study.best_params}")

# 상위 5개 trial의 성능 확인
print(f"\n=== 상위 5개 Trial 성능 ===")
top_trials = sorted(study.trials, key=lambda t: t.value)[:5]
for i, trial in enumerate(top_trials):
    rmse = trial.value
    mae = trial.user_attrs.get('mae', 'N/A')
    print(f"Trial {i+1}: RMSE={rmse:.4f}, MAE={mae:.4f}")

# 최적 파라미터로 전체 데이터에 대해 모델 훈련
print("\n최적 파라미터로 전체 데이터 훈련...")
best_params = study.best_params.copy()
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
})

# 최적 estimators 수 찾기 (early stopping 사용)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED
)

lgbm_model = LGBMRegressor(**best_params, n_estimators=1000)
lgbm_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val_split, y_val_split)],
    callbacks=[early_stopping(50), log_evaluation(0)]
)

print(f"최적 반복 횟수: {lgbm_model._best_iteration}")

# 전체 데이터로 최종 모델 훈련
best_params['n_estimators'] = lgbm_model._best_iteration
lgbm_final = LGBMRegressor(**best_params)
lgbm_final.fit(X_train, y_train)

# 최종 모델의 검증 성능 확인
print(f"\n=== 최종 모델 성능 ===")
val_preds = lgbm_model.predict(X_val_split)
val_preds = np.maximum(val_preds, 0)
final_rmse = np.sqrt(mean_squared_error(y_val_split, val_preds))
final_mae = mean_absolute_error(y_val_split, val_preds)
final_r2 = r2_score(y_val_split, val_preds)

print(f"Validation RMSE: {final_rmse:.4f}")
print(f"Validation MAE: {final_mae:.4f}")
print(f"Validation R²: {final_r2:.4f}")

# 특성 중요도 확인
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': lgbm_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 상위 10개 중요 특성 ===")
print(feature_importance.head(10))

# 예측 수행
print("\n일사 예측 중...")
train_imputed = train.copy()

for building in zero_solar_buildings:
    print(f"건물 {building} 예측 중...")
    building_data = train[train['건물번호'] == building][available_features].copy()
    building_data = building_data.fillna(0)
    
    # 예측
    predictions = lgbm_final.predict(building_data)
    
    # 음수 제거
    predictions = np.maximum(predictions, 0)
    
    # 야간 시간대는 강제로 0 (안전장치)
    building_hours = train[train['건물번호'] == building]['hour'].values
    night_mask = (building_hours < 6) | (building_hours > 19)
    predictions[night_mask] = 0
    
    # 예측값 적용
    building_mask = train_imputed['건물번호'] == building
    train_imputed.loc[building_mask, '일사(MJ/m2)'] = predictions

# 결과 확인
print("\n=== 보간 결과 확인 ===")
all_imputed_preds = []
all_true_solar = []

for building in zero_solar_buildings:
    building_data = train_imputed[train_imputed['건물번호'] == building]
    solar_stats = building_data['일사(MJ/m2)'].describe()
    zero_ratio = (building_data['일사(MJ/m2)'] == 0).mean()
    print(f"건물 {building}: 평균={solar_stats['mean']:.3f}, 최대={solar_stats['max']:.3f}, 0비율={zero_ratio:.3f}")
    
    all_imputed_preds.extend(building_data['일사(MJ/m2)'].values)

# 전체 보간 결과와 정상 데이터 비교 (MAE 포함)
print(f"\n=== 정상 데이터와 비교 ===")
normal_mean = normal_data['일사(MJ/m2)'].mean()
normal_max = normal_data['일사(MJ/m2)'].max()
normal_zero_ratio = (normal_data['일사(MJ/m2)'] == 0).mean()
normal_std = normal_data['일사(MJ/m2)'].std()

imputed_data = train_imputed[train_imputed['건물번호'].isin(zero_solar_buildings)]
imputed_mean = imputed_data['일사(MJ/m2)'].mean()
imputed_max = imputed_data['일사(MJ/m2)'].max()
imputed_zero_ratio = (imputed_data['일사(MJ/m2)'] == 0).mean()
imputed_std = imputed_data['일사(MJ/m2)'].std()

print(f"평균 - 정상: {normal_mean:.3f}, 보간: {imputed_mean:.3f} (차이: {abs(normal_mean-imputed_mean):.3f})")
print(f"최대 - 정상: {normal_max:.3f}, 보간: {imputed_max:.3f}")
print(f"표준편차 - 정상: {normal_std:.3f}, 보간: {imputed_std:.3f}")
print(f"0비율 - 정상: {normal_zero_ratio:.3f}, 보간: {imputed_zero_ratio:.3f}")

# 분포 유사성 평가
mean_diff_ratio = abs(normal_mean - imputed_mean) / normal_mean * 100
print(f"\n=== 분포 유사성 평가 ===")
print(f"평균값 차이율: {mean_diff_ratio:.1f}%")
if mean_diff_ratio < 10:
    print("✅ 평균값 분포가 매우 유사합니다")
elif mean_diff_ratio < 20:
    print("🟡 평균값 분포가 어느정도 유사합니다") 
else:
    print("❌ 평균값 분포에 차이가 있습니다")

# 보간된 데이터 저장
train_imputed.to_csv(trainer + f'01_{SEED}_train.csv', index=False)
print(f"\n보간된 데이터 저장 완료: {trainer}01_{SEED}_train.csv")

print(f"[일사 보간] 완료")