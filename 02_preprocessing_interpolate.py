print(f"[02_preprocessing] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
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
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

data_path = './Energy/'
csv_path = './Energy/02/'
trainer = './Energy/02/new_csv/'  # 01에서 생성된 파일들 경로
log_path = './Energy/02/log/'
os.makedirs(trainer, exist_ok=True)

seed_file = "./Energy/02/log/(02_SEED_COUNT)preprocessing.json"

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

print(f"[1] 전처리된 데이터 로드")

train_all = pd.read_csv(trainer + 'train_with_weather_features.csv', index_col=0)
test_all = pd.read_csv(trainer + 'test_with_weather_features.csv', index_col=0)

def add_sunshine_rolling_features(df, target_col='일조(hr)'):
    """일조시간 rolling 피처 추가"""
    df = df.copy()
    
    # 건물별로 rolling 피처 생성
    building_groups = df.groupby('건물번호')
    
    for window in [3, 6, 12, 24]:
        df[f'{target_col}_rolling_mean_{window}h'] = building_groups[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'{target_col}_rolling_std_{window}h'] = building_groups[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )
    
    # 변화량 피처
    df[f'{target_col}_변화량_1h'] = building_groups[target_col].transform(
        lambda x: x.diff().fillna(0)
    )
    df[f'{target_col}_변화량_3h'] = building_groups[target_col].transform(
        lambda x: x.diff(3).fillna(0)
    )
    
    # NaN 값을 각 피처의 평균값으로 채우기
    feature_cols = [col for col in df.columns if target_col in col and ('rolling' in col or '변화량' in col)]
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def add_insolation_rolling_features(df, target_col='일사(MJ/m2)'):
    """일사량 rolling 피처 추가"""
    df = df.copy()
    
    # 건물별로 rolling 피처 생성
    building_groups = df.groupby('건물번호')
    
    for window in [3, 6, 12, 24]:
        df[f'{target_col}_rolling_mean_{window}h'] = building_groups[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'{target_col}_rolling_std_{window}h'] = building_groups[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )
    
    # 변화량 피처
    df[f'{target_col}_변화량_1h'] = building_groups[target_col].transform(
        lambda x: x.diff().fillna(0)
    )
    df[f'{target_col}_변화량_3h'] = building_groups[target_col].transform(
        lambda x: x.diff(3).fillna(0)
    )
    
    # NaN 값을 각 피처의 평균값으로 채우기
    feature_cols = [col for col in df.columns if target_col in col and ('rolling' in col or '변화량' in col)]
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def add_lag_features_with_continuity(train_df, test_df, feature_cols, lag_hours=[24, 48]):
    """시계열 연속성을 고려한 lag 피처 추가"""
    test_df = test_df.copy()
    
    all_buildings = sorted(test_df['건물번호'].unique())
    
    processed_test_list = []
    
    for building_num in tqdm(all_buildings, desc="Adding lag features"):
        # 해당 건물의 train, test 데이터 추출
        building_train = train_df[train_df['건물번호'] == building_num].copy()
        building_test = test_df[test_df['건물번호'] == building_num].copy()
        
        if len(building_train) == 0:
            # train 데이터가 없는 경우 0으로 채우기
            for col in feature_cols:
                for lag in lag_hours:
                    building_test[f'{col}_lag{lag}'] = 0
            processed_test_list.append(building_test)
            continue
        
        # date 컬럼이 datetime이 아닌 경우 변환
        if building_train['date'].dtype == 'object':
            building_train['date'] = pd.to_datetime(building_train['date'])
        if building_test['date'].dtype == 'object':
            building_test['date'] = pd.to_datetime(building_test['date'])
        
        # train과 test를 시간순으로 합치기
        building_combined = pd.concat([building_train, building_test], ignore_index=True)
        building_combined = building_combined.sort_values('date').reset_index(drop=True)
        
        # lag 피처 생성
        for col in feature_cols:
            if col in building_combined.columns:
                for lag in lag_hours:
                    building_combined[f'{col}_lag{lag}'] = building_combined[col].shift(lag)
        
        # test 부분만 추출
        train_len = len(building_train)
        processed_building_test = building_combined.iloc[train_len:].copy()
        
        # lag 피처의 NaN 값을 0으로 채우기
        for col in feature_cols:
            if col in building_combined.columns:
                for lag in lag_hours:
                    lag_col = f'{col}_lag{lag}'
                    if lag_col in processed_building_test.columns:
                        processed_building_test[lag_col] = processed_building_test[lag_col].fillna(0)
        
        processed_test_list.append(processed_building_test)
    
    # 모든 건물의 데이터를 다시 합치기
    final_test = pd.concat(processed_test_list, ignore_index=True)
    
    return final_test

# Optuna 최적화를 위한 기본 파라미터 (초기값)
def get_default_params():
    return {
        'xgb': {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
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
        },
        'lgbm': {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 500,
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
        },
        'cat': {
            'loss_function': 'MAE',
            'iterations': 300,
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
    }

# Optuna 최적화 함수들
def create_xgb_objective(X_train, y_train, X_val, y_val, sample_weight=None):
    """XGBoost 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
            'gamma': trial.suggest_float('gamma', 0.01, 0.3),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, 
                  sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], 
                  verbose=False)
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        return rmse
    
    return objective

def create_lgbm_objective(X_train, y_train, X_val, y_val, sample_weight=None):
    """LightGBM 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'num_leaves': trial.suggest_int('num_leaves', 15, 100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.5),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(stopping_rounds=30, verbose=False), log_evaluation(0)])
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        return rmse
    
    return objective

def create_catboost_objective(X_train, y_train, X_val, y_val, sample_weight=None):
    """CatBoost 하이퍼파라미터 최적화"""
    def objective(trial):
        params = {
            'loss_function': 'MAE',
            'iterations': trial.suggest_int('iterations', 200, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 5.0),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.95),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),
            'border_count': trial.suggest_int('border_count', 16, 64),
            'random_seed': SEED,
            'thread_count': -1,
            'verbose': False,
            'allow_writing_files': False
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train,
                  sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)])
        
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        return rmse
    
    return objective

def optimize_hyperparameters(X_train_sample, y_train_sample, X_val_sample, y_val_sample, 
                           model_type, sample_weight=None, n_trials=50):
    """
    하이퍼파라미터 최적화 실행
    """
    print(f"    > {model_type.upper()} 하이퍼파라미터 최적화 시작 ({n_trials} trials)...")
    
    study = optuna.create_study(
        direction='minimize', 
        sampler=TPESampler(seed=SEED),
        study_name=f'{model_type}_optimization'
    )
    
    if model_type == 'xgb':
        objective = create_xgb_objective(X_train_sample, y_train_sample, X_val_sample, y_val_sample, sample_weight)
    elif model_type == 'lgbm':
        objective = create_lgbm_objective(X_train_sample, y_train_sample, X_val_sample, y_val_sample, sample_weight)
    elif model_type == 'catboost':
        objective = create_catboost_objective(X_train_sample, y_train_sample, X_val_sample, y_val_sample, sample_weight)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    study.optimize(objective, n_trials=n_trials)
    
    print(f"    > {model_type.upper()} 최적화 완료. Best RMSE: {study.best_value:.6f}")
    
    # 기본 파라미터와 병합 (키 매핑 처리)
    model_key_mapping = {'catboost': 'cat'}  # catboost -> cat 매핑
    param_key = model_key_mapping.get(model_type, model_type)
    
    default_params = get_default_params()[param_key]
    optimized_params = default_params.copy()
    optimized_params.update(study.best_params)
    
    return optimized_params, study.best_value

def sample_data_for_optimization(X_data, y_data, sample_ratio=0.2, min_samples=1000):
    """
    최적화를 위한 데이터 샘플링
    """
    total_samples = len(X_data)
    sample_size = max(int(total_samples * sample_ratio), min_samples)
    sample_size = min(sample_size, total_samples)  # 전체 데이터 크기를 넘지 않도록
    
    # 무작위 샘플링
    sample_indices = np.random.choice(total_samples, size=sample_size, replace=False)
    
    X_sample = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
    y_sample = y_data.iloc[sample_indices] if hasattr(y_data, 'iloc') else y_data[sample_indices]
    
    print(f"    > 데이터 샘플링: {total_samples} -> {sample_size} ({sample_ratio*100:.1f}%)")
    
    return X_sample, y_sample, sample_indices

def identify_transition_periods(X_data):
    """새벽/저녁 전환 구간 식별"""
    if 'hour' not in X_data.columns:
        return np.zeros(len(X_data))
    
    hour = X_data['hour'].values
    transition_mask = np.zeros(len(hour))
    
    # 새벽 전환 구간: 5-8시 (일출 전후)
    morning_transition = (hour >= 5) & (hour <= 8)
    
    # 저녁 전환 구간: 17-20시 (일몰 전후) 
    evening_transition = (hour >= 17) & (hour <= 20)
    
    # 전환 구간을 더 세분화
    # 5-6시, 19-20시: 급격한 변화 구간 (가중치 3.0)
    # 6-8시, 17-19시: 완만한 변화 구간 (가중치 2.0)
    
    transition_mask = np.where(
        ((hour >= 5) & (hour <= 6)) | ((hour >= 19) & (hour <= 20)), 3.0,  # 급격한 전환
        np.where(
            ((hour >= 6) & (hour <= 8)) | ((hour >= 17) & (hour <= 19)), 2.0,  # 완만한 전환
            1.0  # 일반 구간
        )
    )
    
    return transition_mask

def add_transition_features(df):
    """전환 구간 특화 피처 추가"""
    df = df.copy()
    
    if 'hour' not in df.columns:
        return df
    
    # 새벽/저녁 전환 구간 식별
    df['morning_transition'] = df['hour'].apply(lambda x: 1 if 5 <= x <= 8 else 0)
    df['evening_transition'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)
    df['transition_intensity'] = df['hour'].apply(lambda x: 
        3.0 if (5 <= x <= 6) or (19 <= x <= 20) else  # 급격한 전환
        2.0 if (6 <= x <= 8) or (17 <= x <= 19) else  # 완만한 전환
        0.0  # 일반
    )
    
    # 일출/일몰 시간과의 거리 (계절별 변화 고려)
    month = df['month'] if 'month' in df.columns else 6  # 기본값
    
    # 간단한 일출/일몰 시간 추정 (위도 37도 기준)
    if isinstance(month, pd.Series):
        sunrise_hour = 6.5 - 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
        sunset_hour = 18.5 + 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
    else:
        sunrise_hour = 6.5 - 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
        sunset_hour = 18.5 + 1.5 * np.cos(2 * np.pi * (month - 6) / 12)
    
    df['sunrise_distance'] = np.abs(df['hour'] - sunrise_hour)
    df['sunset_distance'] = np.abs(df['hour'] - sunset_hour)
    df['min_transition_distance'] = np.minimum(df['sunrise_distance'], df['sunset_distance'])
    
    # 전환 구간에서의 기울기 변화 감지
    if '기온(°C)' in df.columns:
        df['temp_transition_signal'] = df['기온(°C)'] * df['transition_intensity']
    
    return df

def train_ensemble_models_enhanced(X_train, y_train, X_val, y_val, is_insolation=False, use_log_scale=False, optimize_params=False):
    """Peak + Transition 모델이 추가된 강화된 앙상블 모델 (Optuna 최적화 포함)"""
    
    # 전환 구간 피처 추가
    X_train_enhanced = add_transition_features(X_train)
    X_val_enhanced = add_transition_features(X_val)
    
    # 하락 구간 가중치 계산
    y_train_reset = y_train.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    
    if len(y_train_reset) > 1:
        decline_mask_train = y_train_reset.iloc[1:].values < y_train_reset.iloc[:-1].values
        decline_weights = np.ones(len(y_train_reset))
        decline_weights[1:] = np.where(decline_mask_train, 4.0, 1.0)
    else:
        decline_weights = np.ones(len(y_train_reset))
    
    # Peak 구간 가중치 계산
    if 'peak_time' in X_train_enhanced.columns:
        peak_mask_train = X_train_enhanced['peak_time'].values == 1
        peak_weights = np.where(peak_mask_train, 4.0, 1.0)
        print("    > Peak 시간대 가중치 적용")
    else:
        peak_weights = np.ones(len(y_train_reset))
        print("    > Peak_time 피처 없음, 일반 가중치 사용")
    
    # Transition 구간 가중치 계산 (새로 추가)
    transition_weights = identify_transition_periods(X_train_enhanced)
    print(f"    > Transition 구간 가중치 적용 (평균 가중치: {transition_weights.mean():.2f})")
    
    # log1p 스케일링 적용
    if use_log_scale:
        y_train_scaled = np.log1p(y_train_reset)
        y_val_scaled = np.log1p(y_val_reset)
        print("    > log1p 스케일링 적용")
    else:
        y_train_scaled = y_train_reset
        y_val_scaled = y_val_reset
    
    print("    > 6개 모델 앙상블 훈련... (하락 + Peak + Transition 가중치 적용)")
    
    # Optuna 하이퍼파라미터 최적화 수행
    optimized_params = {}
    if optimize_params:
        print("    > Optuna 하이퍼파라미터 최적화 시작...")
        
        # 20% 샘플링된 데이터로 최적화
        X_train_sample, y_train_sample, train_sample_indices = sample_data_for_optimization(X_train_enhanced, y_train_scaled)
        X_val_sample, y_val_sample, val_sample_indices = sample_data_for_optimization(X_val_enhanced, y_val_scaled)
        
        # 샘플링된 가중치
        decline_weights_sample = decline_weights[train_sample_indices]
        peak_weights_sample = peak_weights[train_sample_indices]
        transition_weights_sample = transition_weights[train_sample_indices]
        
        # XGBoost 최적화
        xgb_params_opt, xgb_score = optimize_hyperparameters(
            X_train_sample, y_train_sample, X_val_sample, y_val_sample,
            'xgb', sample_weight=decline_weights_sample, n_trials=30
        )
        optimized_params['xgb'] = xgb_params_opt
        
        # LightGBM 최적화
        lgbm_params_opt, lgbm_score = optimize_hyperparameters(
            X_train_sample, y_train_sample, X_val_sample, y_val_sample,
            'lgbm', sample_weight=decline_weights_sample, n_trials=30
        )
        optimized_params['lgbm'] = lgbm_params_opt
        
        # CatBoost 최적화
        cat_params_opt, cat_score = optimize_hyperparameters(
            X_train_sample, y_train_sample, X_val_sample, y_val_sample,
            'catboost', sample_weight=decline_weights_sample, n_trials=20
        )
        optimized_params['cat'] = cat_params_opt
        
        print(f"    > 최적화 완료: XGB({xgb_score:.6f}), LGBM({lgbm_score:.6f}), CAT({cat_score:.6f})")
        
    else:
        # 기본 파라미터 사용
        default_params = get_default_params()
        optimized_params['xgb'] = default_params['xgb'].copy()
        optimized_params['lgbm'] = default_params['lgbm'].copy()
        optimized_params['cat'] = default_params['cat'].copy()
    
    # 특화 모델용 파라미터 생성
    decline_params = optimized_params['xgb'].copy()
    decline_params.update({
        'max_depth': min(optimized_params['xgb'].get('max_depth', 6) + 2, 12),
        'learning_rate': optimized_params['xgb'].get('learning_rate', 0.05) * 0.6,
        'n_estimators': int(optimized_params['xgb'].get('n_estimators', 500) * 1.2),
        'gamma': optimized_params['xgb'].get('gamma', 0.1) * 0.5,
    })
    
    peak_params = optimized_params['xgb'].copy()
    peak_params.update({
        'max_depth': min(optimized_params['xgb'].get('max_depth', 6) + 1, 10),
        'learning_rate': optimized_params['xgb'].get('learning_rate', 0.05) * 0.8,
        'reg_alpha': optimized_params['xgb'].get('reg_alpha', 0.1) * 0.5,
    })
    
    transition_params = optimized_params['xgb'].copy()
    transition_params.update({
        'max_depth': min(optimized_params['xgb'].get('max_depth', 6) + 3, 15),
        'learning_rate': optimized_params['xgb'].get('learning_rate', 0.05) * 0.5,
        'n_estimators': int(optimized_params['xgb'].get('n_estimators', 500) * 1.5),
        'gamma': optimized_params['xgb'].get('gamma', 0.1) * 0.1,
        'min_child_weight': 1,
    })
    
    # 1. 기본 XGBoost
    xgb_model = XGBRegressor(**optimized_params['xgb'])
    xgb_model.fit(X_train_enhanced, y_train_scaled, 
                  sample_weight=decline_weights,
                  eval_set=[(X_val_enhanced, y_val_scaled)], 
                  verbose=False)
    
    # 2. 기본 LightGBM
    lgbm_model = LGBMRegressor(**optimized_params['lgbm'])
    lgbm_model.fit(X_train_enhanced, y_train_scaled,
                   sample_weight=decline_weights,
                   eval_set=[(X_val_enhanced, y_val_scaled)],
                   callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)])
    
    # 3. 하락 특화 모델
    decline_model = XGBRegressor(**decline_params)
    extreme_decline_weights = np.where(decline_weights > 1, 5.0, 1.0)
    decline_model.fit(X_train_enhanced, y_train_scaled,
                     sample_weight=extreme_decline_weights,
                     eval_set=[(X_val_enhanced, y_val_scaled)],
                     verbose=False)
    
    # 4. Peak 특화 모델
    peak_model = XGBRegressor(**peak_params)
    combined_peak_weights = decline_weights * peak_weights
    peak_model.fit(X_train_enhanced, y_train_scaled,
                  sample_weight=combined_peak_weights,
                  eval_set=[(X_val_enhanced, y_val_scaled)],
                  verbose=False)
    
    # 5. Transition 특화 모델 (새로 추가)
    transition_model = XGBRegressor(**transition_params)
    combined_transition_weights = decline_weights * transition_weights
    transition_model.fit(X_train_enhanced, y_train_scaled,
                        sample_weight=combined_transition_weights,
                        eval_set=[(X_val_enhanced, y_val_scaled)],
                        verbose=False)
    
    # 6. CatBoost 모델
    cat_model = CatBoostRegressor(**optimized_params['cat'])
    cat_model.fit(X_train_enhanced, y_train_scaled,
                  sample_weight=decline_weights,
                  eval_set=[(X_val_enhanced, y_val_scaled)])
    
    # 각 모델의 예측
    xgb_pred_val = xgb_model.predict(X_val_enhanced)
    lgbm_pred_val = lgbm_model.predict(X_val_enhanced)
    decline_pred_val = decline_model.predict(X_val_enhanced)
    peak_pred_val = peak_model.predict(X_val_enhanced)
    transition_pred_val = transition_model.predict(X_val_enhanced)
    cat_pred_val = cat_model.predict(X_val_enhanced)
    
    # 동적 가중치 계산
    val_decline_mask = y_val_reset.iloc[1:].values < y_val_reset.iloc[:-1].values
    val_decline_indicators = np.zeros(len(y_val_reset))
    val_decline_indicators[1:] = val_decline_mask.astype(float)
    
    # Peak 구간 감지
    if 'peak_time' in X_val_enhanced.columns:
        val_peak_indicators = X_val_enhanced['peak_time'].values
    else:
        val_peak_indicators = np.zeros(len(y_val_reset))
    
    # Transition 구간 감지
    val_transition_weights = identify_transition_periods(X_val_enhanced)
    val_transition_indicators = val_transition_weights > 1.5  # transition 가중치가 1.5 이상인 구간
    
    # 앙상블 예측 (6개 모델)
    ensemble_pred_val = np.zeros_like(xgb_pred_val)
    
    for i in range(len(ensemble_pred_val)):
        is_decline = val_decline_indicators[i] > 0
        is_peak = val_peak_indicators[i] > 0
        is_transition = val_transition_indicators[i]
        
        if is_transition and is_decline:  # 전환 + 하락
            # Transition과 Decline 모델에 최우선 가중치
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.08 + 
                                   lgbm_pred_val[i] * 0.12 + 
                                   decline_pred_val[i] * 0.35 + 
                                   peak_pred_val[i] * 0.1 + 
                                   transition_pred_val[i] * 0.25 + 
                                   cat_pred_val[i] * 0.1)
        elif is_transition:  # 전환 구간만
            # Transition 모델 최우선
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.1 + 
                                   lgbm_pred_val[i] * 0.15 + 
                                   decline_pred_val[i] * 0.15 + 
                                   peak_pred_val[i] * 0.15 + 
                                   transition_pred_val[i] * 0.35 + 
                                   cat_pred_val[i] * 0.1)
        elif is_decline and is_peak:  # 하락 + Peak
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.1 + 
                                   lgbm_pred_val[i] * 0.15 + 
                                   decline_pred_val[i] * 0.3 + 
                                   peak_pred_val[i] * 0.25 + 
                                   transition_pred_val[i] * 0.1 + 
                                   cat_pred_val[i] * 0.1)
        elif is_decline:  # 하락 구간만
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.15 + 
                                   lgbm_pred_val[i] * 0.2 + 
                                   decline_pred_val[i] * 0.35 + 
                                   peak_pred_val[i] * 0.1 + 
                                   transition_pred_val[i] * 0.1 + 
                                   cat_pred_val[i] * 0.1)
        elif is_peak:  # Peak 구간만
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.2 + 
                                   lgbm_pred_val[i] * 0.2 + 
                                   decline_pred_val[i] * 0.1 + 
                                   peak_pred_val[i] * 0.3 + 
                                   transition_pred_val[i] * 0.1 + 
                                   cat_pred_val[i] * 0.1)
        else:  # 일반 구간
            ensemble_pred_val[i] = (xgb_pred_val[i] * 0.2 + 
                                   lgbm_pred_val[i] * 0.2 + 
                                   decline_pred_val[i] * 0.15 + 
                                   peak_pred_val[i] * 0.15 + 
                                   transition_pred_val[i] * 0.15 + 
                                   cat_pred_val[i] * 0.15)
    
    # log 스케일 역변환 및 클리핑
    if use_log_scale:
        ensemble_pred_val = np.expm1(ensemble_pred_val)
        ensemble_pred_val = np.clip(ensemble_pred_val, -0.1 if not is_insolation else 0, None)
        if not is_insolation:
            ensemble_pred_val = np.maximum(ensemble_pred_val, 0)
    else:
        if is_insolation:
            ensemble_pred_val = np.clip(ensemble_pred_val, 0, None)
        else:
            # 일조시간: 전환 구간에서 더 관대한 처리
            ensemble_pred_val = np.clip(ensemble_pred_val, 0, 1.3)
            ensemble_pred_val = np.minimum(ensemble_pred_val, 1.0)
    
    final_mae = np.sqrt(mean_squared_error(y_val_reset, ensemble_pred_val))
    
    print(f"    > 최종 RMSE: {final_mae:.6f}")
    
    return {
        'models': (xgb_model, lgbm_model, decline_model, peak_model, transition_model, cat_model),
        'final_mae': final_mae,
        'use_log_scale': use_log_scale,
        'decline_indicators': val_decline_indicators,
        'peak_indicators': val_peak_indicators,
        'transition_indicators': val_transition_indicators,
        'optimized_params': optimized_params  # 최적화된 파라미터 저장
    }

def predict_with_ensemble_enhanced(models_dict, X_test, is_insolation=False):
    """Peak + Transition 모델이 포함된 6개 모델 앙상블 예측"""
    xgb_model, lgbm_model, decline_model, peak_model, transition_model, cat_model = models_dict['models']
    use_log_scale = models_dict.get('use_log_scale', False)
    
    # 전환 구간 피처 추가
    X_test_enhanced = add_transition_features(X_test)
    
    # 각 모델 예측
    xgb_pred = xgb_model.predict(X_test_enhanced)
    lgbm_pred = lgbm_model.predict(X_test_enhanced)
    decline_pred = decline_model.predict(X_test_enhanced)
    peak_pred = peak_model.predict(X_test_enhanced)
    transition_pred = transition_model.predict(X_test_enhanced)
    cat_pred = cat_model.predict(X_test_enhanced)
    
    # 구간 감지
    if 'peak_time' in X_test_enhanced.columns:
        peak_indicators = X_test_enhanced['peak_time'].values
    else:
        peak_indicators = np.zeros(len(xgb_pred))
    
    transition_weights = identify_transition_periods(X_test_enhanced)
    transition_indicators = transition_weights > 1.5
    
    # 동적 가중치 앙상블
    ensemble_pred = np.zeros_like(xgb_pred)
    
    for i in range(len(ensemble_pred)):
        is_peak = peak_indicators[i] > 0
        is_transition = transition_indicators[i]
        
        if is_transition:  # 전환 구간 (최우선)
            ensemble_pred[i] = (xgb_pred[i] * 0.1 + lgbm_pred[i] * 0.15 + 
                               decline_pred[i] * 0.15 + peak_pred[i] * 0.15 + 
                               transition_pred[i] * 0.35 + cat_pred[i] * 0.1)
        elif is_peak:  # Peak 구간
            ensemble_pred[i] = (xgb_pred[i] * 0.2 + lgbm_pred[i] * 0.2 + 
                               decline_pred[i] * 0.1 + peak_pred[i] * 0.3 + 
                               transition_pred[i] * 0.1 + cat_pred[i] * 0.1)
        else:  # 일반 구간
            ensemble_pred[i] = (xgb_pred[i] * 0.2 + lgbm_pred[i] * 0.2 + 
                               decline_pred[i] * 0.15 + peak_pred[i] * 0.15 + 
                               transition_pred[i] * 0.15 + cat_pred[i] * 0.15)
    
    # log 스케일 역변환 및 클리핑
    if use_log_scale:
        ensemble_pred = np.expm1(ensemble_pred)
        ensemble_pred = np.clip(ensemble_pred, -0.1 if not is_insolation else 0, None)
        if not is_insolation:
            ensemble_pred = np.maximum(ensemble_pred, 0)
    else:
        if is_insolation:
            ensemble_pred = np.clip(ensemble_pred, 0, None)
        else:
            ensemble_pred = np.clip(ensemble_pred, 0, 1.3)
            ensemble_pred = np.minimum(ensemble_pred, 1.0)
    
    return ensemble_pred

def add_sunshine_bin_onehot(df, col='일조(hr)', prefix='sun_bin'):
    """
    일조량을 구간화한 후, 구간별로 원-핫 인코딩된 피처를 추가합니다.
    """
    def sunshine_bin(x):
        if pd.isna(x): 
            return -1
        elif x == 0: 
            return 0
        elif x < 0.2: 
            return 1     # 매우 낮음
        elif x < 0.5: 
            return 2     # 약간 흐림
        elif x < 0.8: 
            return 3
        elif x < 1: 
            return 4     # 중간
        elif x == 1: 
            return 5
        else:
            return 5     # 1 초과하는 경우도 처리

    bin_col = f"{prefix}_cat"
    
    # 안전한 방식으로 구간화 적용
    df[bin_col] = df[col].apply(sunshine_bin)
    
    # NaN 처리 후 정수형 변환
    df[bin_col] = pd.to_numeric(df[bin_col], errors='coerce').fillna(-1).astype(int)

    # 원핫 인코딩
    onehot_df = pd.get_dummies(df[bin_col], prefix=prefix)
    df = pd.concat([df, onehot_df], axis=1)
    df.drop(columns=[bin_col], inplace=True)

    return df

def identify_feature_types(columns):
    """
    피처 타입을 식별하여 스케일링 여부 결정
    """
    # One-hot 인코딩된 피처들 (스케일링 X)
    onehot_patterns = ['요일_', 'sun_bin_', '_이상치', '_설치여부', '강수시작', '강수끝', '강수강도', 
                       '주말여부', '근무시간', 'peak_time', '건물번호_encoded', 'dayoff', 'hot_day']
    
    # 건물 유형 (스케일링 X)
    building_type_patterns = ['공공업무시설', '교육연구시설', '기타', '노유자시설', '다세대주택',
                             '단독주택', '상업시설', '아파트', '업무시설', '의료시설', '종교시설']
    
    categorical_features = []
    continuous_features = []
    
    for col in columns:
        # One-hot 인코딩된 피처들
        if any(pattern in col for pattern in onehot_patterns):
            categorical_features.append(col)
        # 건물 유형
        elif any(pattern in col for pattern in building_type_patterns):
            categorical_features.append(col)
        # 연속형 피처들
        else:
            continuous_features.append(col)
    
    return categorical_features, continuous_features

def apply_feature_scaling(X_train, X_val, feature_cols, scaler_type='standard'):
    """
    연속형 피처에만 스케일링 적용
    """
    print(f"    > {scaler_type} 스케일링 적용 중...")
    
    # 피처 타입 식별
    categorical_features, continuous_features = identify_feature_types(feature_cols)
    
    print(f"    > 카테고리 피처 수: {len(categorical_features)}")
    print(f"    > 연속형 피처 수: {len(continuous_features)}")
    
    # 스케일러 선택
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    # 연속형 피처만 추출
    continuous_features_in_data = [col for col in continuous_features if col in X_train.columns]
    
    if len(continuous_features_in_data) > 0:
        # 연속형 피처에만 스케일링 적용
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        
        # 스케일러 학습 및 변환
        X_train_scaled[continuous_features_in_data] = scaler.fit_transform(X_train[continuous_features_in_data])
        X_val_scaled[continuous_features_in_data] = scaler.transform(X_val[continuous_features_in_data])
        
        print(f"    > {len(continuous_features_in_data)}개 연속형 피처에 스케일링 적용 완료")
        return X_train_scaled, X_val_scaled, scaler, continuous_features_in_data
    else:
        print(f"    > 스케일링할 연속형 피처가 없음")
        return X_train, X_val, None, []

def apply_test_scaling(X_test, scaler, continuous_features_in_data):
    """
    테스트 데이터에 동일한 스케일링 적용
    """
    if scaler is not None and len(continuous_features_in_data) > 0:
        X_test_scaled = X_test.copy()
        X_test_scaled[continuous_features_in_data] = scaler.transform(X_test[continuous_features_in_data])
        return X_test_scaled
    else:
        return X_test

def train_global_model_with_building_features(train_data, features, target_col, building_col='건물번호', 
                                            scaler_type='standard', optimize_params=False):
    """
    전체 데이터로 학습하되 건물별 특성을 반영하는 모델 훈련 (잔차학습 없음 + 스케일링 + Optuna 최적화)
    """
    print(f"전체 데이터로 {target_col} 모델 학습 시작... (건물번호 피처 포함, 잔차학습 없음, {scaler_type} 스케일링)")
    if optimize_params:
        print("    > Optuna 하이퍼파라미터 최적화 활성화")
    
    # 건물 번호를 라벨 인코딩으로 추가
    le = LabelEncoder()
    building_features = features + [building_col]
    X_full = train_data[building_features].copy()
    y_full = train_data[target_col]
    
    # 건물번호를 라벨 인코딩
    X_full[f'{building_col}_encoded'] = le.fit_transform(X_full[building_col])
    
    # 건물번호 원본은 제거하고 인코딩된 버전 사용
    feature_cols = [col for col in X_full.columns if col != building_col]
    X_full_final = X_full[feature_cols]
    
    # Train-Validation 분할
    X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
        X_full_final, y_full, test_size=0.2, random_state=SEED, shuffle=True
    )
    
    # 피처 스케일링 적용
    X_train_scaled, X_val_scaled, scaler, continuous_features_in_data = apply_feature_scaling(
        X_train_global, X_val_global, feature_cols, scaler_type
    )
    
    # 잔차학습 없는 앙상블 모델 학습 (Optuna 최적화 포함)
    global_models = train_ensemble_models_enhanced(
        X_train_scaled, y_train_global, X_val_scaled, y_val_global, 
        is_insolation=(target_col == '일사(MJ/m2)'), use_log_scale=True, optimize_params=optimize_params
    )
    
    return global_models, le, feature_cols, scaler, continuous_features_in_data

def predict_with_global_model(test_data, global_models, label_encoder, 
                            feature_cols, target_col, scaler, continuous_features_in_data, building_col='건물번호'):
    """
    전체 모델로 건물별 예측 수행 (잔차학습 없음 + 스케일링)
    """
    predictions = {}
    
    # 테스트 데이터에서 훈련에 사용된 건물들만 필터링
    train_buildings = set(label_encoder.classes_)
    test_buildings = set(test_data[building_col].unique())
    valid_buildings = train_buildings.intersection(test_buildings)
    
    # 테스트 데이터 준비
    test_encoded = test_data.copy()
    
    # 훈련에 없었던 건물은 평균값으로 대체
    unknown_buildings = test_buildings - train_buildings
    if unknown_buildings:
        # 가장 많은 건물의 인코딩값으로 대체 (보통 0)
        default_encoding = 0
        test_encoded.loc[test_encoded[building_col].isin(unknown_buildings), building_col] = label_encoder.classes_[default_encoding]
    
    # 라벨 인코딩 적용
    test_encoded[f'{building_col}_encoded'] = label_encoder.transform(test_encoded[building_col])
    X_test_final = test_encoded[feature_cols]
    
    # 테스트 데이터에 스케일링 적용
    X_test_scaled = apply_test_scaling(X_test_final, scaler, continuous_features_in_data)
    
    # 전체 모델로 예측 (잔차학습 없음)
    all_predictions = predict_with_ensemble_enhanced(
        global_models, X_test_scaled, 
        is_insolation=(target_col == '일사(MJ/m2)')
    )
    
    # 건물별로 예측 결과 저장
    for i, building_num in enumerate(test_data[building_col]):
        if building_num not in predictions:
            predictions[building_num] = []
        predictions[building_num].append(all_predictions[i])
    
    return predictions, all_predictions

# 01에서 만든 피쳐에서 일조(hr) 관련 피쳐가 없다면 추가
if '일조(hr)' not in train_all.columns:
    print("경고: train 데이터에 일조(hr) 컬럼이 없습니다.")
    # 기본값으로 0 설정 또는 다른 처리 방법 사용
    train_all['일조(hr)'] = 0

# 일조시간 rolling 피처 추가
train_all = add_sunshine_rolling_features(train_all)
train_all = add_sunshine_bin_onehot(train_all)

# 일사량 rolling 피처 추가  
train_all = add_insolation_rolling_features(train_all)

# zero_bno 정의
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print(f"[2] train zero_bno 일사 예측 시작 (시계열 연속성 + Optuna 최적화 앙상블 모델)")

# 예측에 사용할 피처 선택 (01에서 만든 피쳐들 사용)
insolation_features = []

# 기본 피쳐들
basic_features = ['peak_time', 'hour', 'dayofweek', 'month', 'day', 'is_working']
basic_features.extend(['SIN_hour', 'COS_hour', 'SIN_day', 'COS_day', 'SIN_dayofweek', 'COS_dayofweek'])
basic_features.extend(['dew_point_temperature', '기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)'])

# 01에서 생성된 피쳐들 중 사용 가능한 것들 추가
for feat in basic_features:
    if feat in train_all.columns:
        insolation_features.append(feat)

# 일조(hr)가 있다면 추가
if '일조(hr)' in train_all.columns:
    insolation_features.append('일조(hr)')

# rolling 및 변화량 피처들 추가
feature_patterns = ['_rolling_', '_변화량_', '_lag', 'sun_bin', 'dayoff', 'hot_day']
for pattern in feature_patterns:
    pattern_features = [col for col in train_all.columns if pattern in col]
    insolation_features.extend(pattern_features)

# 중복 제거
insolation_features = list(set(insolation_features))
print(f"사용할 일사량 예측 피쳐 수: {len(insolation_features)}")

# train zero_bno 일사량 예측 (Optuna 최적화 활성화)
train_insolation_data = train_all[~train_all['건물번호'].isin(zero_bnos) & (train_all['일사(MJ/m2)'] > 0)].copy()

insolation_models, insolation_label_encoder, insolation_feature_cols, insolation_scaler, insolation_continuous_features = train_global_model_with_building_features(
    train_insolation_data, insolation_features, '일사(MJ/m2)', '건물번호', scaler_type='standard', optimize_params=True
)

print(f"전체 모델 학습 완료, 이제 건물별로 예측 진행...")

# 건물별 예측 수행
for bno in zero_bnos:
    building_data = train_all[(train_all['건물번호'] == bno) & (train_all['일사(MJ/m2)'] == 0)].copy()
    
    if len(building_data) == 0:
        continue
    
    _, building_predictions = predict_with_global_model(
        building_data, insolation_models, insolation_label_encoder, 
        insolation_feature_cols, '일사(MJ/m2)', insolation_scaler, insolation_continuous_features, '건물번호'
    )
    
    train_all.loc[building_data.index, '일사(MJ/m2)'] = building_predictions
    print(f"    > 건물 {bno}번: 예측 완료 (예측된 일사량 범위: {building_predictions.min():.4f} ~ {building_predictions.max():.4f})")

# 야간 일사량 0으로 설정
train_all.loc[(train_all['건물번호'].isin(zero_bnos)) & ((train_all['hour'] >= 21) | (train_all['hour'] <= 5)), '일사(MJ/m2)'] = 0
train_all = add_insolation_rolling_features(train_all)

print(f"[3] train zero_bno 일사 예측 완료")

########################## test 일조시간 예측 ############################
print(f"[4] test 일조시간 예측 시작 (시계열 연속성 고려 + Optuna 최적화)")

# test에 일조시간 컬럼이 없다면 추가
if '일조(hr)' not in test_all.columns:
    test_all['일조(hr)'] = 0

# 일조시간 예측용 피쳐 선택
sunshine_features = []

# 기본 피쳐들 (일조 제외)
sunshine_basic_features = ['hour', 'dayofweek', 'month', 'day', 'is_working', 'peak_time']
sunshine_basic_features.extend(['SIN_hour', 'COS_hour', 'SIN_day', 'COS_day', 'SIN_dayofweek', 'COS_dayofweek'])
sunshine_basic_features.extend(['dew_point_temperature', '기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)'])

for feat in sunshine_basic_features:
    if feat in test_all.columns:
        sunshine_features.append(feat)

# rolling 및 변화량 피쳐들 추가 (일조 제외)
sunshine_patterns = ['기온_rolling', '습도_rolling', '풍속_rolling', '강수량_rolling', 
                    '기온_변화량', '습도_변화량', '풍속_변화량', '강수량_변화량',
                    '기온_lag', '습도_lag', '풍속_lag', '강수량_lag', 'dayoff', 'hot_day']
for pattern in sunshine_patterns:
    pattern_features = [col for col in test_all.columns if pattern in col]
    sunshine_features.extend(pattern_features)

# train과 test에 공통으로 존재하는 피쳐만 사용
sunshine_features = [col for col in sunshine_features if col in train_all.columns and col in test_all.columns]
sunshine_features = list(set(sunshine_features))
print(f"사용할 일조시간 예측 피쳐 수: {len(sunshine_features)}")

sunshine_models, sunshine_label_encoder, sunshine_feature_cols, sunshine_scaler, sunshine_continuous_features = train_global_model_with_building_features(
    train_all, sunshine_features, '일조(hr)', '건물번호', scaler_type='standard', optimize_params=True
)

sunshine_predictions_dict, all_sunshine_predictions = predict_with_global_model(
    test_all, sunshine_models, sunshine_label_encoder, 
    sunshine_feature_cols, '일조(hr)', sunshine_scaler, sunshine_continuous_features, '건물번호'
)

test_all['일조(hr)'] = all_sunshine_predictions
test_all.loc[(test_all['hour'] >= 20) | (test_all['hour'] <= 6), '일조(hr)'] = 0
test_all = add_sunshine_rolling_features(test_all)
test_all = add_sunshine_bin_onehot(test_all)

print(f"[5] test 일조시간 예측 완료 - Validation RMSE: {sunshine_models['final_mae']:.6f}")

########################## test 일사량 예측 ############################
print(f"[6] test 일사량 예측 시작 (시계열 연속성 고려 + Optuna 최적화)")

# 일사량 예측용 피쳐 선택 (일조시간 포함)
insolation_features_test = sunshine_features + ['일조(hr)']

# 일조 관련 피쳐들 추가
sunshine_related_features = [col for col in test_all.columns if '일조' in col or 'sun_bin' in col]
insolation_features_test.extend(sunshine_related_features)

# train과 test에 공통으로 존재하는 피쳐만 사용
insolation_features_test = [col for col in insolation_features_test if col in train_all.columns and col in test_all.columns]
insolation_features_test = list(set(insolation_features_test))
print(f"사용할 일사량 예측 피쳐 수: {len(insolation_features_test)}")

insolation_test_models, insolation_test_label_encoder, insolation_test_feature_cols, insolation_test_scaler, insolation_test_continuous_features = train_global_model_with_building_features(
    train_all, insolation_features_test, '일사(MJ/m2)', '건물번호', scaler_type='standard', optimize_params=True
)

insolation_predictions_dict, all_insolation_predictions = predict_with_global_model(
    test_all, insolation_test_models, insolation_test_label_encoder, 
    insolation_test_feature_cols, '일사(MJ/m2)', insolation_test_scaler, insolation_test_continuous_features, '건물번호'
)

test_all['일사(MJ/m2)'] = all_insolation_predictions
test_all.loc[(test_all['hour'] >= 21) | (test_all['hour'] <= 5), '일사(MJ/m2)'] = 0
print(f"[7] test 일사량 예측 완료 - Validation RMSE: {insolation_test_models['final_mae']:.6f}")
test_all = add_insolation_rolling_features(test_all)

# 일조/일사량 예측 완료 후 해당 lag 피처들을 test 데이터에 추가
print(f"[7.5] test 데이터에 일조/일사량 lag 피처 추가...")
weather_lag_cols_sunshine_insolation = ['일조(hr)', '일사(MJ/m2)']
test_all = add_lag_features_with_continuity(train_all, test_all, weather_lag_cols_sunshine_insolation, lag_hours=[24, 48])

# 결과 저장
train_all.to_csv(trainer + f'06_train_{SEED}_optuna.csv', index=False)
test_all.to_csv(trainer + f'06_test_{SEED}_optuna.csv', index=False)

print(f"[8] 저장 완료 (시계열 연속성 반영 + Optuna 최적화)")

# 결과 요약
print("\n" + "="*70)
print("최종 결과 요약 (시계열 연속성 + 전체 학습 + Optuna 최적화 앙상블)")
print("="*70)
print(f"Train 일사량 예측 RMSE: {insolation_models['final_mae']:.6f}")
print(f"Test 일조시간 예측 Validation RMSE: {sunshine_models['final_mae']:.6f}")
print(f"Test 일사량 예측 Validation RMSE: {insolation_test_models['final_mae']:.6f}")

# Optuna 최적화 결과 출력
if 'optimized_params' in insolation_models:
    print("\n[Optuna 최적화 결과]")
    for model_name, params in insolation_models['optimized_params'].items():
        print(f"{model_name.upper()} 최적 파라미터:")
        for key, value in params.items():
            if key not in ['random_state', 'n_jobs', 'verbosity', 'thread_count', 'verbose', 'allow_writing_files', 'force_col_wise']:
                print(f"  - {key}: {value}")

# 로그 파일에 결과 저장
log_file = log_path + "(LOG)02_preprocessing_optuna.txt"
os.makedirs(log_path, exist_ok=True)
with open(log_file, "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"Train 일사량 예측 RMSE: {insolation_models['final_mae']:.6f}\n")
    f.write(f"Test 일조(hr) RMSE : {sunshine_models['final_mae']:.6f}\n")
    f.write(f"Test 일사(MJ/m2) RMSE : {insolation_test_models['final_mae']:.6f}\n")
    f.write("="*50 + "\n")

print(f"[02_preprocessing] 종료 (시계열 연속성 + Optuna 최적화 + 하락 폭 보존 최적화 완료)")