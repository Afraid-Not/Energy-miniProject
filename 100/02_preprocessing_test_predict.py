print(f"[02_preprocessing_test_predict] 시작")
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

seed_file = "./Energy/100/log/(SEED_COUNT)02_preprocessing.json"

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

train = pd.read_csv(trainer + '01_45_train.csv')
test = pd.read_csv(trainer + '00_test.csv')

print(f"[일조 예측] Test 데이터 일조(hr) 예측 시작 (XGB + LGBM 앙상블)")

# 일조(hr) 컬럼이 train에만 있고 test에는 없는 상황
print(f"Train 컬럼에 일조(hr) 존재: {'일조(hr)' in train.columns}")
print(f"Test 컬럼에 일조(hr) 존재: {'일조(hr)' in test.columns}")

if '일조(hr)' not in train.columns:
    print("❌ Train 데이터에 일조(hr) 컬럼이 없습니다!")
else:
    # Train 데이터의 일조 분포 확인
    sunshine_data = train['일조(hr)'].copy()
    print(f"일조 데이터 - 평균: {sunshine_data.mean():.3f}, 최대: {sunshine_data.max():.3f}, 최소: {sunshine_data.min():.3f}")
    print(f"0값 비율: {(sunshine_data == 0).mean():.3f}")
    print(f"데이터 개수: {len(sunshine_data)}")

    # 시간대별 0값 분포 확인
    sunshine_by_hour = train.groupby('hour')['일조(hr)'].agg(['mean', 'count', lambda x: (x==0).sum()])
    sunshine_by_hour.columns = ['mean', 'count', 'zero_count']
    sunshine_by_hour['zero_ratio'] = sunshine_by_hour['zero_count'] / sunshine_by_hour['count']
    print(f"\n시간대별 일조 패턴 (상위 5시간):")
    print(sunshine_by_hour.sort_values('mean', ascending=False).head())

    # 공통 특성 찾기 (train과 test 둘 다 있는 컬럼들)
    train_cols = set(train.columns) - {'일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'}  # 예측 대상 제외
    test_cols = set(test.columns)
    common_cols = list(train_cols.intersection(test_cols))
    
    # 일조 예측에 중요한 특성들 선택
    feature_cols = [
        # 기상 데이터
        '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
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
        'peak_time', 'rainy', 'high_humidity', 'cloudy_or_rain',
        'extraterrestrial_rad'
    ]
    
    # 사용 가능한 특성만 필터링 (공통 컬럼 중에서)
    available_features = [col for col in feature_cols if col in common_cols]
    print(f"\n사용할 특성 수: {len(available_features)}")
    print(f"사용할 특성들: {available_features[:10]}...")  # 처음 10개만 출력

    # 훈련 데이터 준비
    X_train = train[available_features].copy()
    y_train = train['일조(hr)'].copy()
    
    # 테스트 데이터 준비
    X_test = test[available_features].copy()

    # 결측값 처리
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"훈련 데이터 크기: {X_train.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    print(f"타겟 분포 - 평균: {y_train.mean():.3f}, 최대: {y_train.max():.3f}")

    # 샘플링 (50%만 사용하여 튜닝)
    print("데이터 샘플링 (50%)...")
    sample_size = int(len(X_train) * 0.5)
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train.iloc[sample_indices].copy()
    y_sample = y_train.iloc[sample_indices].copy()
    print(f"샘플 데이터 크기: {X_sample.shape}")

    # 샘플 데이터를 train/val로 분할
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=SEED
    )

    # Optuna 하이퍼파라미터 튜닝 (LGBM)
    print("LGBM Optuna 하이퍼파라미터 튜닝 시작...")
    import optuna
    from optuna.samplers import TPESampler

    def objective_lgbm(trial):
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
            'n_jobs': 1,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params, n_estimators=100)
        model.fit(X_train_sample, y_train_sample)
        
        preds = model.predict(X_val_sample)
        preds = np.maximum(preds, 0)  # 음수 제거
        
        rmse = np.sqrt(mean_squared_error(y_val_sample, preds))
        mae = mean_absolute_error(y_val_sample, preds)
        
        trial.set_user_attr('mae', mae)
        return rmse

    # LGBM 튜닝
    sampler_lgbm = TPESampler(seed=SEED)
    study_lgbm = optuna.create_study(direction='minimize', sampler=sampler_lgbm)
    study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=True)

    print(f"\nLGBM 튜닝 결과:")
    print(f"Best RMSE: {study_lgbm.best_value:.4f}")
    print(f"Best MAE: {study_lgbm.best_trial.user_attrs['mae']:.4f}")

    # XGBoost 하이퍼파라미터 튜닝
    print("\nXGBoost Optuna 하이퍼파라미터 튜닝 시작...")
    
    def objective_xgb(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': SEED,
            'n_jobs': 1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train_sample, y_train_sample)
        
        preds = model.predict(X_val_sample)
        preds = np.maximum(preds, 0)  # 음수 제거
        
        rmse = np.sqrt(mean_squared_error(y_val_sample, preds))
        mae = mean_absolute_error(y_val_sample, preds)
        
        trial.set_user_attr('mae', mae)
        return rmse

    # XGBoost 튜닝
    sampler_xgb = TPESampler(seed=SEED+1)
    study_xgb = optuna.create_study(direction='minimize', sampler=sampler_xgb)
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

    print(f"\nXGBoost 튜닝 결과:")
    print(f"Best RMSE: {study_xgb.best_value:.4f}")
    print(f"Best MAE: {study_xgb.best_trial.user_attrs['mae']:.4f}")

    # 최적 파라미터로 전체 데이터에 대해 모델 훈련
    print("\n최적 파라미터로 전체 데이터 훈련...")
    
    # LGBM 최적 파라미터 설정
    best_params_lgbm = study_lgbm.best_params.copy()
    best_params_lgbm.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    })

    # XGBoost 최적 파라미터 설정
    best_params_xgb = study_xgb.best_params.copy()
    best_params_xgb.update({
        'objective': 'reg:squarederror',
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': 0
    })

    # Early stopping으로 최적 반복 수 찾기
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED
    )

    # LGBM Early stopping
    lgbm_model = LGBMRegressor(**best_params_lgbm, n_estimators=1000)
    lgbm_model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    # XGBoost Early stopping
    xgb_model = XGBRegressor(**best_params_xgb, early_stopping_rounds=50, n_estimators=1000)
    xgb_model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        
        verbose=False
    )

    print(f"LGBM 최적 반복 횟수: {lgbm_model._best_iteration}")
    print(f"XGBoost 최적 반복 횟수: {xgb_model.best_iteration}")

    # 전체 데이터로 최종 모델 훈련
    best_params_lgbm['n_estimators'] = lgbm_model._best_iteration
    best_params_xgb['n_estimators'] = xgb_model.best_iteration

    lgbm_final = LGBMRegressor(**best_params_lgbm)
    lgbm_final.fit(X_train, y_train)

    xgb_final = XGBRegressor(**best_params_xgb)
    xgb_final.fit(X_train, y_train)

    # 개별 모델 성능 확인
    print(f"\n=== 개별 모델 성능 (일조) ===")
    
    lgbm_val_preds = lgbm_model.predict(X_val_split)
    lgbm_val_preds = np.maximum(lgbm_val_preds, 0)
    lgbm_rmse = np.sqrt(mean_squared_error(y_val_split, lgbm_val_preds))
    lgbm_mae = mean_absolute_error(y_val_split, lgbm_val_preds)
    lgbm_r2 = r2_score(y_val_split, lgbm_val_preds)

    xgb_val_preds = xgb_model.predict(X_val_split)
    xgb_val_preds = np.maximum(xgb_val_preds, 0)
    xgb_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_val_preds))
    xgb_mae = mean_absolute_error(y_val_split, xgb_val_preds)
    xgb_r2 = r2_score(y_val_split, xgb_val_preds)

    print(f"LGBM - RMSE: {lgbm_rmse:.4f}, MAE: {lgbm_mae:.4f}, R²: {lgbm_r2:.4f}")
    print(f"XGBoost - RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}, R²: {xgb_r2:.4f}")

    # 앙상블 성능 확인 (가중평균)
    ensemble_weights = [0.6, 0.4]  # LGBM 60%, XGBoost 40%
    ensemble_val_preds = ensemble_weights[0] * lgbm_val_preds + ensemble_weights[1] * xgb_val_preds
    ensemble_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_val_preds))
    ensemble_mae = mean_absolute_error(y_val_split, ensemble_val_preds)
    ensemble_r2 = r2_score(y_val_split, ensemble_val_preds)

    print(f"앙상블 - RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")
    print(f"앙상블 가중치: LGBM {ensemble_weights[0]}, XGBoost {ensemble_weights[1]}")

    # Test 데이터 일조 예측 (앙상블)
    print("\n테스트 데이터 일조 예측 중 (앙상블)...")
    
    lgbm_test_preds = lgbm_final.predict(X_test)
    xgb_test_preds = xgb_final.predict(X_test)
    
    # 앙상블 예측
    test_sunshine_preds = ensemble_weights[0] * lgbm_test_preds + ensemble_weights[1] * xgb_test_preds
    
    # 음수 제거
    test_sunshine_preds = np.maximum(test_sunshine_preds, 0)
    
    # 더 엄격한 야간 시간대 처리
    test_hours = test['hour'].values
    night_mask = (test_hours < 6) | (test_hours > 19)
    test_sunshine_preds[night_mask] = 0
    
    # 강수량이 많은 경우 일조량 감소
    if '강수량(mm)' in test.columns:
        heavy_rain_mask = test['강수량(mm)'].values > 3
        test_sunshine_preds[heavy_rain_mask] *= 0.1  # 90% 감소
        
        rain_mask = (test['강수량(mm)'].values > 0.5) & (test['강수량(mm)'].values <= 3)
        test_sunshine_preds[rain_mask] *= 0.3  # 70% 감소
        
        light_rain_mask = (test['강수량(mm)'].values > 0.1) & (test['강수량(mm)'].values <= 0.5)
        test_sunshine_preds[light_rain_mask] *= 0.6  # 40% 감소
    
    # 습도가 매우 높은 경우
    if '습도(%)' in test.columns:
        very_high_humidity_mask = test['습도(%)'].values > 90
        test_sunshine_preds[very_high_humidity_mask] *= 0.5  # 50% 감소
    
    # rainy, cloudy_or_rain 변수 활용
    if 'rainy' in test.columns:
        rainy_mask = test['rainy'].values == 1
        test_sunshine_preds[rainy_mask] *= 0.2  # 80% 감소
        
    if 'cloudy_or_rain' in test.columns:
        cloudy_mask = test['cloudy_or_rain'].values == 1
        test_sunshine_preds[cloudy_mask] *= 0.4  # 60% 감소
    
    # 0에 가까운 매우 작은 값들을 0으로 처리
    very_small_mask = test_sunshine_preds < 0.05
    test_sunshine_preds[very_small_mask] = 0
    
    # Test 데이터에 일조 컬럼 추가
    test_with_sunshine = test.copy()
    test_with_sunshine['일조(hr)'] = test_sunshine_preds

    # 결과 확인
    print("\n=== 예측 결과 확인 (일조 앙상블) ===")
    pred_stats = pd.Series(test_sunshine_preds).describe()
    print(f"예측 일조 - 평균: {pred_stats['mean']:.3f}, 최대: {pred_stats['max']:.3f}, 최소: {pred_stats['min']:.3f}")
    print(f"예측 0값 비율: {(test_sunshine_preds == 0).mean():.3f}")
    
    # Train 데이터와 비교
    print(f"\n=== Train 데이터와 비교 (일조) ===")
    train_mean = y_train.mean()
    train_max = y_train.max()
    train_zero_ratio = (y_train == 0).mean()
    train_std = y_train.std()
    
    pred_mean = test_sunshine_preds.mean()
    pred_max = test_sunshine_preds.max()
    pred_zero_ratio = (test_sunshine_preds == 0).mean()
    pred_std = test_sunshine_preds.std()

    print(f"평균 - Train: {train_mean:.3f}, Test 예측: {pred_mean:.3f} (차이: {abs(train_mean-pred_mean):.3f})")
    print(f"최대 - Train: {train_max:.3f}, Test 예측: {pred_max:.3f}")
    print(f"표준편차 - Train: {train_std:.3f}, Test 예측: {pred_std:.3f}")
    print(f"0비율 - Train: {train_zero_ratio:.3f}, Test 예측: {pred_zero_ratio:.3f}")

    # 분포 유사성 평가
    mean_diff_ratio = abs(train_mean - pred_mean) / train_mean * 100
    print(f"\n=== 분포 유사성 평가 (일조) ===")
    print(f"평균값 차이율: {mean_diff_ratio:.1f}%")
    if mean_diff_ratio < 10:
        print("✅ 평균값 분포가 매우 유사합니다")
    elif mean_diff_ratio < 20:
        print("🟡 평균값 분포가 어느정도 유사합니다") 
    else:
        print("❌ 평균값 분포에 차이가 있습니다")

print(f"[일조 예측 앙상블] 완료")

# ===============================================================================
# 일조 변화량 피쳐 생성
# ===============================================================================

print(f"\n[일조 변화량 피쳐] 생성 시작")

def create_change_features(train, test):
    # 변화량을 계산할 컬럼들
    change_columns = ['일조(hr)']  # 수정: 일조(hr) 컬럼명 맞춤
    # 변화량 시간 간격 (시간 단위)
    time_intervals = [1, 3, 5, 12, 24]
    
    # 건물번호별로 처리
    buildings = train['건물번호'].unique()
    
    train_with_changes = []
    test_with_changes = []
    
    for building_num in tqdm(buildings, desc="Processing buildings"):
        # 해당 건물의 train과 test 데이터 추출
        building_train = train[train['건물번호'] == building_num].copy()
        building_test = test[test['건물번호'] == building_num].copy()
        
        # train과 test를 시간순으로 연결 (train 뒤에 test)
        combined_df = pd.concat([building_train, building_test], ignore_index=True)
        combined_df = combined_df.sort_values('일시').reset_index(drop=True)  # 수정: 'date' -> '일시'
        
        # 각 컬럼별, 각 시간 간격별로 변화량 계산
        for col in change_columns:
            # 해당 컬럼의 평균값 계산 (빈 값 채우기용)
            col_mean = combined_df[col].mean()
            
            for interval in time_intervals:
                change_col_name = f'{col}_change_{interval}h'
                
                # 변화량 계산: 현재값 - interval시간 전 값
                combined_df[change_col_name] = combined_df[col] - combined_df[col].shift(interval)
        
        # train과 test 부분으로 다시 분리
        train_len = len(building_train)
        building_train_updated = combined_df.iloc[:train_len].copy()
        building_test_updated = combined_df.iloc[train_len:].copy()
        
        train_with_changes.append(building_train_updated)
        test_with_changes.append(building_test_updated)
    
    # 모든 건물의 데이터를 다시 합치기
    train_final = pd.concat(train_with_changes, ignore_index=True)
    test_final = pd.concat(test_with_changes, ignore_index=True)
    
    # 전체 데이터에서 변화량 피쳐들의 NaN 값을 해당 피쳐의 평균값으로 채우기
    change_features = [col for col in train_final.columns if '_change_' in col]
    
    # train과 test를 합쳐서 전체 변화량 통계 계산
    combined_all = pd.concat([train_final, test_final], ignore_index=True)
    
    for change_col in change_features:
        # 전체 데이터의 변화량 평균 계산
        change_mean = combined_all[change_col].mean()
        
        # train과 test 모두에서 NaN 값을 평균값으로 채우기
        train_final[change_col] = train_final[change_col].fillna(change_mean)
        test_final[change_col] = test_final[change_col].fillna(change_mean)
    
    # 원래 순서로 정렬
    train_final = train_final.sort_values(['건물번호', '일시']).reset_index(drop=True)  # 수정: 'date' -> '일시'
    test_final = test_final.sort_values(['건물번호', '일시']).reset_index(drop=True)    # 수정: 'date' -> '일시'
    
    return train_final, test_final

# Train에 일조 변화량 피쳐가 없다면 생성
sunshine_change_cols = ['일조(hr)_change_1h', '일조(hr)_change_3h', '일조(hr)_change_5h', 
                       '일조(hr)_change_12h', '일조(hr)_change_24h']

missing_cols_train = [col for col in sunshine_change_cols if col not in train.columns]
if missing_cols_train:
    print(f"Train과 Test 데이터에 일조 변화량 피쳐 추가: {missing_cols_train}")
    train_with_changes, test_with_sunshine_change = create_change_features(train, test_with_sunshine)
    
    # 원본 train 업데이트
    for col in sunshine_change_cols:
        if col in train_with_changes.columns:
            train[col] = train_with_changes[col]
else:
    print("Train 데이터에 일조 변화량 피쳐가 이미 존재합니다.")
    # Test 데이터에만 변화량 피쳐 추가
    _, test_with_sunshine_change = create_change_features(train, test_with_sunshine)

# 변화량 피쳐 통계 확인
print(f"\n=== 일조 변화량 피쳐 통계 ===")
for col in sunshine_change_cols:
    if col in train.columns and col in test_with_sunshine_change.columns:
        train_stats = train[col].describe()
        test_stats = test_with_sunshine_change[col].describe()
        print(f"{col}:")
        print(f"  Train - 평균: {train_stats['mean']:.3f}, 표준편차: {train_stats['std']:.3f}")
        print(f"  Test  - 평균: {test_stats['mean']:.3f}, 표준편차: {test_stats['std']:.3f}")

print(f"[일조 변화량 피쳐] 완료")

# ===============================================================================
# ===============================================================================
# Test 데이터 일사(MJ/m2) 예측 (일조 변화량 피쳐 포함 + XGB + LGBM 앙상블)
# ===============================================================================
# ===============================================================================

print(f"\n[일사 예측] Test 데이터 일사(MJ/m2) 예측 시작 (일조 변화량 포함 + XGB + LGBM 앙상블)")

# 일사(MJ/m2) 컬럼이 train에만 있고 test에는 없는 상황
print(f"Train 컬럼에 일사(MJ/m2) 존재: {'일사(MJ/m2)' in train.columns}")
print(f"Test 컬럼에 일사(MJ/m2) 존재: {'일사(MJ/m2)' in test_with_sunshine_change.columns}")

if '일사(MJ/m2)' not in train.columns:
    print("❌ Train 데이터에 일사(MJ/m2) 컬럼이 없습니다!")
else:
    # Train 데이터의 일사 분포 확인
    solar_data = train['일사(MJ/m2)'].copy()
    print(f"일사 데이터 - 평균: {solar_data.mean():.3f}, 최대: {solar_data.max():.3f}, 최소: {solar_data.min():.3f}")
    print(f"0값 비율: {(solar_data == 0).mean():.3f}")
    print(f"데이터 개수: {len(solar_data)}")

    # 시간대별 0값 분포 확인
    solar_by_hour = train.groupby('hour')['일사(MJ/m2)'].agg(['mean', 'count', lambda x: (x==0).sum()])
    solar_by_hour.columns = ['mean', 'count', 'zero_count']
    solar_by_hour['zero_ratio'] = solar_by_hour['zero_count'] / solar_by_hour['count']
    print(f"\n시간대별 일사 패턴 (상위 5시간):")
    print(solar_by_hour.sort_values('mean', ascending=False).head())

    # 공통 특성 찾기 (train과 test_with_sunshine_change 둘 다 있는 컬럼들)
    train_cols = set(train.columns) - {'일사(MJ/m2)', '전력소비량(kWh)'}  # 예측 대상 제외
    test_cols = set(test_with_sunshine_change.columns)
    common_cols = list(train_cols.intersection(test_cols))
    
    # 일사 예측에 중요한 특성들 선택 (일조 변화량 피쳐 포함)
    feature_cols = [
        # 기상 데이터 (일조를 가장 앞에)
        '일조(hr)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
        # 일조 변화량 피쳐들 (새로 추가!)
        # '일조(hr)_change_1h', '일조(hr)_change_3h', '일조(hr)_change_5h', 
        # '일조(hr)_change_12h', '일조(hr)_change_24h',
        # 시간 특성
        'hour', 'month', 'day_of_year', 'dayofweek',
        'is_weekend', 'is_working',
        # 순환 인코딩
        'SIN_hour', 'COS_hour', 'SIN_day', 'COS_day',
        # 태양 관련
        'solar_elevation', 'daylight', 'sunrise_hour', 'sunset_hour',
        # 기상 지수 (습도/강수 관련 우선)
        'high_humidity', 'rainy', 'cloudy_or_rain', 'humid_x_rain',
        'discomfort_index', 'THI', 'perceived_temperature', 'dew_point',
        # 파생 변수들
        # 'peak_time', 'extraterrestrial_rad'
    ]
    
    # 사용 가능한 특성만 필터링 (공통 컬럼 중에서)
    available_features = [col for col in feature_cols if col in common_cols]
    print(f"\n사용할 특성 수: {len(available_features)}")
    
    # 일조 관련 피쳐가 포함되었는지 확인
    sunshine_features = [col for col in available_features if '일조' in col]
    weather_features = [col for col in available_features if col in ['강수량(mm)', '습도(%)', 'high_humidity', 'rainy', 'cloudy_or_rain']]
    print(f"일조 관련 피쳐 개수: {len(sunshine_features)} - {sunshine_features}")
    print(f"날씨 관련 피쳐 개수: {len(weather_features)} - {weather_features}")
    print(f"기타 특성들: {[col for col in available_features if '일조' not in col and col not in weather_features][:5]}...")  # 처음 5개만 출력

    # 훈련 데이터 준비
    X_train_solar = train[available_features].copy()
    y_train_solar = train['일사(MJ/m2)'].copy()
    
    # 테스트 데이터 준비
    X_test_solar = test_with_sunshine_change[available_features].copy()

    # 결측값 처리
    X_train_solar = X_train_solar.fillna(0)
    X_test_solar = X_test_solar.fillna(0)

    print(f"훈련 데이터 크기: {X_train_solar.shape}")
    print(f"테스트 데이터 크기: {X_test_solar.shape}")
    print(f"타겟 분포 - 평균: {y_train_solar.mean():.3f}, 최대: {y_train_solar.max():.3f}")

    # 샘플링 (50%만 사용하여 튜닝)
    print("데이터 샘플링 (50%)...")
    sample_size = int(len(X_train_solar) * 0.5)
    sample_indices = np.random.choice(len(X_train_solar), sample_size, replace=False)
    X_sample_solar = X_train_solar.iloc[sample_indices].copy()
    y_sample_solar = y_train_solar.iloc[sample_indices].copy()
    print(f"샘플 데이터 크기: {X_sample_solar.shape}")

    # 샘플 데이터를 train/val로 분할
    X_train_sample_solar, X_val_sample_solar, y_train_sample_solar, y_val_sample_solar = train_test_split(
        X_sample_solar, y_sample_solar, test_size=0.3, random_state=SEED
    )

    # Optuna 하이퍼파라미터 튜닝 (LGBM for 일사)
    print("일사 예측 - LGBM Optuna 하이퍼파라미터 튜닝 시작...")

    def objective_solar_lgbm(trial):
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
            'n_jobs': 1,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params, n_estimators=100)
        model.fit(X_train_sample_solar, y_train_sample_solar)
        
        preds = model.predict(X_val_sample_solar)
        preds = np.maximum(preds, 0)  # 음수 제거
        
        rmse = np.sqrt(mean_squared_error(y_val_sample_solar, preds))
        mae = mean_absolute_error(y_val_sample_solar, preds)
        
        trial.set_user_attr('mae', mae)
        return rmse

    # LGBM 튜닝 (일사)
    sampler_solar_lgbm = TPESampler(seed=SEED+10)
    study_solar_lgbm = optuna.create_study(direction='minimize', sampler=sampler_solar_lgbm)
    study_solar_lgbm.optimize(objective_solar_lgbm, n_trials=30, show_progress_bar=True)

    print(f"\n일사 예측 - LGBM 튜닝 결과:")
    print(f"Best RMSE: {study_solar_lgbm.best_value:.4f}")
    print(f"Best MAE: {study_solar_lgbm.best_trial.user_attrs['mae']:.4f}")

    # XGBoost 하이퍼파라미터 튜닝 (일사)
    print("\n일사 예측 - XGBoost Optuna 하이퍼파라미터 튜닝 시작...")
    
    def objective_solar_xgb(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': SEED,
            'n_jobs': 1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train_sample_solar, y_train_sample_solar)
        
        preds = model.predict(X_val_sample_solar)
        preds = np.maximum(preds, 0)  # 음수 제거
        
        rmse = np.sqrt(mean_squared_error(y_val_sample_solar, preds))
        mae = mean_absolute_error(y_val_sample_solar, preds)
        
        trial.set_user_attr('mae', mae)
        return rmse

    # XGBoost 튜닝 (일사)
    sampler_solar_xgb = TPESampler(seed=SEED+11)
    study_solar_xgb = optuna.create_study(direction='minimize', sampler=sampler_solar_xgb)
    study_solar_xgb.optimize(objective_solar_xgb, n_trials=30, show_progress_bar=True)

    print(f"\n일사 예측 - XGBoost 튜닝 결과:")
    print(f"Best RMSE: {study_solar_xgb.best_value:.4f}")
    print(f"Best MAE: {study_solar_xgb.best_trial.user_attrs['mae']:.4f}")

    # 최적 파라미터로 전체 데이터에 대해 모델 훈련
    print("\n일사 예측 - 최적 파라미터로 전체 데이터 훈련...")
    
    # LGBM 최적 파라미터 설정 (일사)
    best_params_solar_lgbm = study_solar_lgbm.best_params.copy()
    best_params_solar_lgbm.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    })

    # XGBoost 최적 파라미터 설정 (일사)
    best_params_solar_xgb = study_solar_xgb.best_params.copy()
    best_params_solar_xgb.update({
        'objective': 'reg:squarederror',
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': 0
    })

    # Early stopping으로 최적 반복 수 찾기 (일사)
    X_train_split_solar, X_val_split_solar, y_train_split_solar, y_val_split_solar = train_test_split(
        X_train_solar, y_train_solar, test_size=0.2, random_state=SEED
    )

    # LGBM Early stopping (일사)
    lgbm_model_solar = LGBMRegressor(**best_params_solar_lgbm, n_estimators=1000)
    lgbm_model_solar.fit(
        X_train_split_solar, y_train_split_solar,
        eval_set=[(X_val_split_solar, y_val_split_solar)],
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    # XGBoost Early stopping (일사)
    xgb_model_solar = XGBRegressor(**best_params_solar_xgb, early_stopping_rounds=50, n_estimators=1000)
    xgb_model_solar.fit(
        X_train_split_solar, y_train_split_solar,
        eval_set=[(X_val_split_solar, y_val_split_solar)],
        verbose=False
    )

    print(f"일사 예측 - LGBM 최적 반복 횟수: {lgbm_model_solar._best_iteration}")
    print(f"일사 예측 - XGBoost 최적 반복 횟수: {xgb_model_solar.best_iteration}")

    # 전체 데이터로 최종 모델 훈련 (일사)
    best_params_solar_lgbm['n_estimators'] = lgbm_model_solar._best_iteration
    best_params_solar_xgb['n_estimators'] = xgb_model_solar.best_iteration

    lgbm_final_solar = LGBMRegressor(**best_params_solar_lgbm)
    lgbm_final_solar.fit(X_train_solar, y_train_solar)

    xgb_final_solar = XGBRegressor(**best_params_solar_xgb)
    xgb_final_solar.fit(X_train_solar, y_train_solar)

    # 개별 모델 성능 확인 (일사)
    print(f"\n=== 개별 모델 성능 (일사) ===")
    
    lgbm_val_preds_solar = lgbm_model_solar.predict(X_val_split_solar)
    lgbm_val_preds_solar = np.maximum(lgbm_val_preds_solar, 0)
    lgbm_rmse_solar = np.sqrt(mean_squared_error(y_val_split_solar, lgbm_val_preds_solar))
    lgbm_mae_solar = mean_absolute_error(y_val_split_solar, lgbm_val_preds_solar)
    lgbm_r2_solar = r2_score(y_val_split_solar, lgbm_val_preds_solar)

    xgb_val_preds_solar = xgb_model_solar.predict(X_val_split_solar)
    xgb_val_preds_solar = np.maximum(xgb_val_preds_solar, 0)
    xgb_rmse_solar = np.sqrt(mean_squared_error(y_val_split_solar, xgb_val_preds_solar))
    xgb_mae_solar = mean_absolute_error(y_val_split_solar, xgb_val_preds_solar)
    xgb_r2_solar = r2_score(y_val_split_solar, xgb_val_preds_solar)

    print(f"LGBM - RMSE: {lgbm_rmse_solar:.4f}, MAE: {lgbm_mae_solar:.4f}, R²: {lgbm_r2_solar:.4f}")
    print(f"XGBoost - RMSE: {xgb_rmse_solar:.4f}, MAE: {xgb_mae_solar:.4f}, R²: {xgb_r2_solar:.4f}")

    # 앙상블 성능 확인 (일사)
    ensemble_weights_solar = [0.6, 0.4]  # LGBM 60%, XGBoost 40%
    ensemble_val_preds_solar = ensemble_weights_solar[0] * lgbm_val_preds_solar + ensemble_weights_solar[1] * xgb_val_preds_solar
    ensemble_rmse_solar = np.sqrt(mean_squared_error(y_val_split_solar, ensemble_val_preds_solar))
    ensemble_mae_solar = mean_absolute_error(y_val_split_solar, ensemble_val_preds_solar)
    ensemble_r2_solar = r2_score(y_val_split_solar, ensemble_val_preds_solar)

    print(f"앙상블 - RMSE: {ensemble_rmse_solar:.4f}, MAE: {ensemble_mae_solar:.4f}, R²: {ensemble_r2_solar:.4f}")
    print(f"앙상블 가중치: LGBM {ensemble_weights_solar[0]}, XGBoost {ensemble_weights_solar[1]}")

    # 특성 중요도 확인 (앙상블 평균)
    feature_importance_lgbm = pd.DataFrame({
        'feature': available_features,
        'importance_lgbm': lgbm_final_solar.feature_importances_
    })
    
    feature_importance_xgb = pd.DataFrame({
        'feature': available_features,
        'importance_xgb': xgb_final_solar.feature_importances_
    })

    feature_importance_solar = feature_importance_lgbm.merge(feature_importance_xgb, on='feature')
    feature_importance_solar['importance_ensemble'] = (
        ensemble_weights_solar[0] * feature_importance_solar['importance_lgbm'] + 
        ensemble_weights_solar[1] * feature_importance_solar['importance_xgb']
    )
    feature_importance_solar = feature_importance_solar.sort_values('importance_ensemble', ascending=False)

    print("\n=== 상위 10개 중요 특성 (일사 앙상블) ===")
    print(feature_importance_solar[['feature', 'importance_ensemble']].head(10))

    # Test 데이터 일사 예측 (앙상블)
    print("\n테스트 데이터 일사 예측 중 (앙상블)...")
    
    lgbm_test_preds_solar = lgbm_final_solar.predict(X_test_solar)
    xgb_test_preds_solar = xgb_final_solar.predict(X_test_solar)
    
    # 앙상블 예측
    test_solar_preds = ensemble_weights_solar[0] * lgbm_test_preds_solar + ensemble_weights_solar[1] * xgb_test_preds_solar
    
    # 음수 제거
    test_solar_preds = np.maximum(test_solar_preds, 0)
    
    # 더 엄격한 야간/악천후 조건 적용
    test_hours = test_with_sunshine_change['hour'].values
    
    # 1. 완전 야간 시간대 (일출 전, 일몰 후)
    night_mask = (test_hours < 6) | (test_hours > 18)
    test_solar_preds[night_mask] = 0
    
    # 2. 일조량이 0인 경우 일사량도 0 (강한 상관관계)
    sunshine_zero_mask = test_with_sunshine_change['일조(hr)'].values == 0
    test_solar_preds[sunshine_zero_mask] = 0
    
    # 3. 강수량이 많은 경우 일사량 대폭 감소
    if '강수량(mm)' in test_with_sunshine_change.columns:
        heavy_rain_mask = test_with_sunshine_change['강수량(mm)'].values > 5
        test_solar_preds[heavy_rain_mask] *= 0.2  # 80% 감소
        
        light_rain_mask = (test_with_sunshine_change['강수량(mm)'].values > 1) & (test_with_sunshine_change['강수량(mm)'].values <= 5)
        test_solar_preds[light_rain_mask] *= 0.5  # 50% 감소
        
        drizzle_mask = (test_with_sunshine_change['강수량(mm)'].values > 0.1) & (test_with_sunshine_change['강수량(mm)'].values <= 1)
        test_solar_preds[drizzle_mask] *= 0.7  # 30% 감소
    
    # 4. 습도가 매우 높은 경우 (구름 많음)
    if '습도(%)' in test_with_sunshine_change.columns:
        very_high_humidity_mask = test_with_sunshine_change['습도(%)'].values > 95
        test_solar_preds[very_high_humidity_mask] *= 0.3  # 70% 감소
        
        high_humidity_mask = (test_with_sunshine_change['습도(%)'].values > 85) & (test_with_sunshine_change['습도(%)'].values <= 95)
        test_solar_preds[high_humidity_mask] *= 0.6  # 40% 감소
    
    # 5. 일출/일몰 경계 시간대 추가 감소
    dawn_dusk_mask = (test_hours == 6) | (test_hours == 7) | (test_hours == 17) | (test_hours == 18)
    test_solar_preds[dawn_dusk_mask] *= 0.3  # 70% 감소
    
    # 6. 매우 작은 일조량에 대한 일사량 조정
    very_low_sunshine_mask = (test_with_sunshine_change['일조(hr)'].values > 0) & (test_with_sunshine_change['일조(hr)'].values < 0.1)
    test_solar_preds[very_low_sunshine_mask] *= 0.4  # 60% 감소
    
    # 7. 0에 가까운 매우 작은 값들을 0으로 처리
    very_small_mask = test_solar_preds < 0.05
    test_solar_preds[very_small_mask] = 0
    
    # 8. rainy, cloudy_or_rain 변수 활용
    if 'rainy' in test_with_sunshine_change.columns:
        rainy_mask = test_with_sunshine_change['rainy'].values == 1
        test_solar_preds[rainy_mask] *= 0.4  # 60% 감소
        
    if 'cloudy_or_rain' in test_with_sunshine_change.columns:
        cloudy_mask = test_with_sunshine_change['cloudy_or_rain'].values == 1
        test_solar_preds[cloudy_mask] *= 0.5  # 50% 감소
    
    # Test 데이터에 일사 컬럼 추가
    test_complete = test_with_sunshine_change.copy()
    test_complete['일사(MJ/m2)'] = test_solar_preds

    # 결과 확인
    print("\n=== 예측 결과 확인 (일사 앙상블) ===")
    pred_stats_solar = pd.Series(test_solar_preds).describe()
    print(f"예측 일사 - 평균: {pred_stats_solar['mean']:.3f}, 최대: {pred_stats_solar['max']:.3f}, 최소: {pred_stats_solar['min']:.3f}")
    print(f"예측 0값 비율: {(test_solar_preds == 0).mean():.3f}")
    
    # Train 데이터와 비교
    print(f"\n=== Train 데이터와 비교 (일사) ===")
    train_mean_solar = y_train_solar.mean()
    train_max_solar = y_train_solar.max()
    train_zero_ratio_solar = (y_train_solar == 0).mean()
    train_std_solar = y_train_solar.std()
    
    pred_mean_solar = test_solar_preds.mean()
    pred_max_solar = test_solar_preds.max()
    pred_zero_ratio_solar = (test_solar_preds == 0).mean()
    pred_std_solar = test_solar_preds.std()

    print(f"평균 - Train: {train_mean_solar:.3f}, Test 예측: {pred_mean_solar:.3f} (차이: {abs(train_mean_solar-pred_mean_solar):.3f})")
    print(f"최대 - Train: {train_max_solar:.3f}, Test 예측: {pred_max_solar:.3f}")
    print(f"표준편차 - Train: {train_std_solar:.3f}, Test 예측: {pred_std_solar:.3f}")
    print(f"0비율 - Train: {train_zero_ratio_solar:.3f}, Test 예측: {pred_zero_ratio_solar:.3f}")

    # 분포 유사성 평가
    mean_diff_ratio_solar = abs(train_mean_solar - pred_mean_solar) / train_mean_solar * 100
    print(f"\n=== 분포 유사성 평가 (일사) ===")
    print(f"평균값 차이율: {mean_diff_ratio_solar:.1f}%")
    if mean_diff_ratio_solar < 10:
        print("✅ 평균값 분포가 매우 유사합니다")
    elif mean_diff_ratio_solar < 20:
        print("🟡 평균값 분포가 어느정도 유사합니다") 
    else:
        print("❌ 평균값 분포에 차이가 있습니다")

    # 일조 변화량 피쳐의 중요도 확인
    sunshine_change_importance = feature_importance_solar[
        feature_importance_solar['feature'].str.contains('일조.*_change', regex=True)
    ]
    if len(sunshine_change_importance) > 0:
        print(f"\n=== 일조 변화량 피쳐 중요도 (앙상블) ===")
        print(sunshine_change_importance[['feature', 'importance_ensemble']])
        total_importance = feature_importance_solar['importance_ensemble'].sum()
        sunshine_change_total = sunshine_change_importance['importance_ensemble'].sum()
        print(f"일조 변화량 피쳐들의 총 중요도 비율: {(sunshine_change_total/total_importance)*100:.1f}%")

    # 최종 앙상블 모델 성능 요약
    print(f"\n=== 최종 앙상블 모델 성능 요약 ===")
    print(f"일조 예측 앙상블 - RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")
    print(f"일사 예측 앙상블 - RMSE: {ensemble_rmse_solar:.4f}, MAE: {ensemble_mae_solar:.4f}, R²: {ensemble_r2_solar:.4f}")

    # 최종 완성된 테스트 데이터 저장
    test_complete.to_csv(trainer + f'02_{SEED}_test.csv', index=False)
    print(f"\n일조+일조변화량+일사가 모두 추가된 테스트 데이터 저장 완료 (앙상블): {trainer}02_{SEED}_test.csv")
    
    print(f"\n=== 최종 Test 데이터 컬럼 구성 (앙상블) ===")
    print(f"총 컬럼 수: {len(test_complete.columns)}")
    print(f"새로 추가된 컬럼: 일조(hr), 일조(hr) 변화량 5개, 일사(MJ/m2)")
    print(f"사용된 모델: XGBoost + LightGBM 앙상블")

print(f"[일사 예측 앙상블] 완료")

# ===============================================================================
# 최종 결과 요약 및 모델 저장
# ===============================================================================

print(f"\n{'='*60}")
print(f"{'='*60}")
print(f"[최종 앙상블 모델 요약]")
print(f"{'='*60}")

print(f"\n🎯 일조(hr) 예측 앙상블:")
print(f"  - LGBM 가중치: {ensemble_weights[0]}")
print(f"  - XGBoost 가중치: {ensemble_weights[1]}")
print(f"  - 성능: RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")

print(f"\n🎯 일사(MJ/m2) 예측 앙상블:")
print(f"  - LGBM 가중치: {ensemble_weights_solar[0]}")
print(f"  - XGBoost 가중치: {ensemble_weights_solar[1]}")
print(f"  - 성능: RMSE={ensemble_rmse_solar:.4f}, MAE={ensemble_mae_solar:.4f}, R²={ensemble_r2_solar:.4f}")





""" 
print(f"\n📊 특성 중요도 요약:")
print(f"  - 일조 예측 모델에서 상위 3개 특성:")
if 'lgbm_final' in locals() and 'xgb_final' in locals():
    # 일조 예측 앙상블 특성 중요도
    sunshine_lgbm_importance = lgbm_final.feature_importances_
    sunshine_xgb_importance = xgb_final.feature_importances_
    sunshine_ensemble_importance = ensemble_weights[0] * sunshine_lgbm_importance + ensemble_weights[1] * sunshine_xgb_importance
    sunshine_feature_df = pd.DataFrame({
        'feature': available_features,
        'importance': sunshine_ensemble_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in sunshine_feature_df.head(3).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")

print(f"  - 일사 예측 모델에서 상위 3개 특성:")
if 'feature_importance_solar' in locals():
    for i, row in feature_importance_solar.head(3).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance_ensemble']:.4f}")

print(f"\n📈 데이터 분포 비교:")
print(f"  - 일조: Train 평균 {train_mean:.3f} vs Test 예측 {pred_mean:.3f} (차이율: {mean_diff_ratio:.1f}%)")
print(f"  - 일사: Train 평균 {train_mean_solar:.3f} vs Test 예측 {pred_mean_solar:.3f} (차이율: {mean_diff_ratio_solar:.1f}%)")

print(f"\n💾 저장된 파일:")
print(f"  - 최종 테스트 데이터: {trainer}02_{SEED}_test.csv")
print(f"  - 포함된 새로운 컬럼: 일조(hr), 일조 변화량 피쳐 5개, 일사(MJ/m2)")

print(f"\n🔧 사용된 기술:")
print(f"  - 하이퍼파라미터 최적화: Optuna TPESampler")
print(f"  - 교차검증: Early Stopping")
print(f"  - 앙상블 기법: 가중평균 (LGBM + XGBoost)")
print(f"  - 피쳐 엔지니어링: 시간별 변화량 계산")
print(f"  - 후처리: 도메인 지식 기반 규칙 적용")

print(f"\n⚡ 성능 개선 포인트:")
print(f"  - 일조 변화량 피쳐 추가로 시간적 패턴 학습")
print(f"  - XGBoost + LightGBM 앙상블로 과적합 방지")
print(f"  - 기상 조건별 후처리로 현실적 예측")

print(f"\n{'='*60}")
print(f"[02_preprocessing_test_predict 앙상블] 완료 ✅")
print(f"SEED: {SEED}")
print(f"{'='*60}")
print(f"{'='*60}") """