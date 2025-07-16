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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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
SEED = seed_state["seed"]
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

# 잔차학습용 파라미터 (더 작은 learning rate와 적은 estimators)
xgb_residual_params = xgb_params.copy()
xgb_residual_params.update({
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 4
})

lgbm_residual_params = lgbm_params.copy()
lgbm_residual_params.update({
    'n_estimators': 500,
    'learning_rate': 0.03,
    'num_leaves': 20
})

cat_residual_params = cat_params.copy()
cat_residual_params.update({
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 4
})

########################### 전처리 함수 정의 ############################

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def calculate_dewpoint(temp, humidity):
    """이슬점 온도 계산 (Magnus 공식 사용) - 안전한 버전"""
    # 습도가 0이거나 음수인 경우 처리
    humidity = np.clip(humidity, 0.01, 100)  # 0.01% ~ 100% 범위로 제한
    
    a = 17.27
    b = 237.7
    
    # log 계산 시 0으로 나누기 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        dewpoint = (b * alpha) / (a - alpha)
    
    # 이상값 처리
    dewpoint = np.where(np.isfinite(dewpoint), dewpoint, temp - 10)  # 이상값은 기온-10도로 대체
    return dewpoint

def calculate_apparent_temp(temp, humidity, wind_speed):
    """체감온도 계산"""
    return temp + 0.33 * (6.105 * np.exp(17.27 * temp / (237.7 + temp)) * humidity / 100) - 0.70 * wind_speed - 4.00

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
    df['SIN_일'] = np.sin(2 * np.pi * df['일'] / 31)  # 일의 주기적 패턴 (31일 기준)
    df['COS_일'] = np.cos(2 * np.pi * df['일'] / 31)
    df['SIN_요일'] = np.sin(2 * np.pi * df['요일'] / 7)  # 요일의 주기적 패턴 (7일 기준)
    df['COS_요일'] = np.cos(2 * np.pi * df['요일'] / 7)
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

    # 이슬점 온도 계산
    if '기온(°C)' in df.columns and '습도(%)' in df.columns:
        df['이슬점온도'] = calculate_dewpoint(df['기온(°C)'], df['습도(%)'])
        df['이슬점차이'] = df['기온(°C)'] - df['이슬점온도']  # 기온과 이슬점의 차이 (수증기 압력 관련)
    
    # 체감온도 계산
    if '기온(°C)' in df.columns and '습도(%)' in df.columns and '풍속(m/s)' in df.columns:
        df['체감온도'] = calculate_apparent_temp(df['기온(°C)'], df['습도(%)'], df['풍속(m/s)'])

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

def add_lag_features(df, target_cols, lag_hours=[24, 48]):
    """전날(또는 며칠 전) 날씨 정보 추가"""
    df = df.copy()
    df = df.sort_values(['건물번호', '일시']).reset_index(drop=True)
    
    for col in target_cols:
        if col in df.columns:
            for lag in lag_hours:
                df[f'{col.split("(")[0]}_lag_{lag}h'] = df.groupby('건물번호')[col].shift(lag)
    
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
    weather_cols = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '이슬점온도', '체감온도']
    
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

def add_weather_volatility_features(df):
    """날씨 급변 감지 피처 추가"""
    df = df.copy()
    
    # 기온/습도/풍속 급변 감지
    df['기온_급변'] = df.groupby('건물번호')['기온(°C)'].diff().abs().fillna(0)
    df['습도_급변'] = df.groupby('건물번호')['습도(%)'].diff().abs().fillna(0)
    df['풍속_급변'] = df.groupby('건물번호')['풍속(m/s)'].diff().abs().fillna(0)
    
    # 강수 시작/끝 감지
    prev_rain = df.groupby('건물번호')['강수량(mm)'].shift(1).fillna(0)
    df['강수시작'] = ((df['강수량(mm)'] > 0) & (prev_rain == 0)).astype(int)
    df['강수끝'] = ((df['강수량(mm)'] == 0) & (prev_rain > 0)).astype(int)
    
    # 강수 강도 분류 (NaN 처리 개선)
    df['강수강도'] = pd.cut(df['강수량(mm)'], 
                         bins=[-0.001, 0.1, 1, 5, 10, 100], 
                         labels=[0, 1, 2, 3, 4], 
                         include_lowest=True).fillna(0).astype(int)
    
    # 종합 날씨 불안정 지수
    df['날씨불안정지수'] = (df['기온_급변'] + df['습도_급변'] * 0.5 + df['풍속_급변'] * 2)
    
    # 시간대별 이상치 감지 (안전한 방식으로 수정)
    for col in ['기온(°C)', '습도(%)', '풍속(m/s)']:
        if col in df.columns:
            # 시간대별 통계 계산
            hourly_stats = df.groupby('시각')[col].agg(['mean', 'std']).reset_index()
            hourly_stats.columns = ['시각', f'{col}_hourly_mean', f'{col}_hourly_std']
            
            # 병합
            df = df.merge(hourly_stats, on='시각', how='left')
            
            # Z-score 계산 (0으로 나누기 방지)
            std_col = f'{col}_hourly_std'
            mean_col = f'{col}_hourly_mean'
            df[std_col] = df[std_col].fillna(1).replace(0, 1)  # 표준편차 0인 경우 1로 대체
            
            df[f'{col}_zscore'] = (df[col] - df[mean_col]) / df[std_col]
            df[f'{col}_이상치'] = (df[f'{col}_zscore'].abs() > 2).astype(int)
            
            # 임시 컬럼 제거
            df = df.drop([mean_col, std_col], axis=1)
    
    # 모든 NaN 값을 0으로 채우기
    df = df.fillna(0)
    return df

def train_models_with_residual_enhanced(X_train, y_train, X_val, y_val, is_insolation=False, use_log_scale=False):
    """급변 대응이 강화된 잔차학습 모델"""
    
    # 하락 구간 가중치 계산 (인덱스 리셋 후 계산)
    y_train_reset = y_train.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    
    if len(y_train_reset) > 1:
        decline_mask_train = y_train_reset.iloc[1:].values < y_train_reset.iloc[:-1].values
        decline_weights = np.ones(len(y_train_reset))
        decline_weights[1:] = np.where(decline_mask_train, 2.0, 1.0)  # 하락시 2배 가중치
    else:
        decline_weights = np.ones(len(y_train_reset))
    
    # log1p 스케일링 적용
    if use_log_scale:
        y_train_scaled = np.log1p(y_train_reset)
        y_val_scaled = np.log1p(y_val_reset)
        print("    > log1p 스케일링 적용")
    else:
        y_train_scaled = y_train_reset
        y_val_scaled = y_val_reset
    
    # 1단계: 기본 모델 훈련 (가중치 적용)
    print("    > 1단계: 기본 모델 훈련... (하락 가중치 적용)")
    
    # XGBoost with sample weights
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_scaled, 
                  sample_weight=decline_weights,
                  eval_set=[(X_val, y_val_scaled)], 
                  verbose=False)
    
    # LightGBM with sample weights  
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train_scaled,
                   sample_weight=decline_weights,
                   eval_set=[(X_val, y_val_scaled)],
                   callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)])
    
    # 하락 특화 모델 추가
    decline_params = xgb_params.copy()
    decline_params.update({
        'max_depth': 8,  # 더 깊은 트리
        'learning_rate': 0.03,  # 더 세밀한 학습
        'subsample': 0.7,
        'colsample_bytree': 0.7
    })
    
    decline_model = XGBRegressor(**decline_params)
    # 하락 가중치를 더 강하게
    heavy_decline_weights = np.where(decline_weights > 1, 3.0, 1.0)
    decline_model.fit(X_train, y_train_scaled,
                     sample_weight=heavy_decline_weights,
                     eval_set=[(X_val, y_val_scaled)],
                     verbose=False)
    
    # 1단계 예측 (3개 모델 앙상블)
    xgb_pred_val = xgb_model.predict(X_val)
    lgbm_pred_val = lgbm_model.predict(X_val)
    decline_pred_val = decline_model.predict(X_val)
    
    # 가중 평균 (하락 특화 모델에 더 높은 가중치)
    ensemble_pred_val = (xgb_pred_val * 0.4 + lgbm_pred_val * 0.4 + decline_pred_val * 0.2)
    
    # log 스케일인 경우 역변환
    if use_log_scale:
        ensemble_pred_val = np.expm1(ensemble_pred_val)
        ensemble_pred_val = np.clip(ensemble_pred_val, 0, None)
    else:
        if is_insolation:
            ensemble_pred_val = np.clip(ensemble_pred_val, 0, None)
        else:
            ensemble_pred_val = np.clip(ensemble_pred_val, 0, 1)
    
    # 1단계 잔차 계산 (원본 스케일에서)
    residual_val = y_val_reset - ensemble_pred_val
    
    # 훈련 데이터에 대한 잔차도 원본 스케일에서 계산
    train_pred = (xgb_model.predict(X_train) * 0.4 + 
                  lgbm_model.predict(X_train) * 0.4 + 
                  decline_model.predict(X_train) * 0.2)
    if use_log_scale:
        train_pred = np.expm1(train_pred)
        train_pred = np.clip(train_pred, 0, None)
    else:
        if is_insolation:
            train_pred = np.clip(train_pred, 0, None)
        else:
            train_pred = np.clip(train_pred, 0, 1)
    
    residual_train = y_train_reset - train_pred
    
    stage1_mae = mean_absolute_error(y_val_reset, ensemble_pred_val)
    print(f"    > 1단계 Validation MAE: {stage1_mae:.6f}")
    
    # 2단계: 잔차 학습 모델 훈련
    print("    > 2단계: 잔차 학습 모델 훈련...")
    xgb_residual = XGBRegressor(**xgb_residual_params)
    lgbm_residual = LGBMRegressor(**lgbm_residual_params)
    
    xgb_residual.fit(X_train, residual_train, eval_set=[(X_val, residual_val)], verbose=False)
    lgbm_residual.fit(X_train, residual_train, eval_set=[(X_val, residual_val)],
                      callbacks=[early_stopping(stopping_rounds=30, verbose=False), log_evaluation(0)])
    
    # 2단계 잔차 예측
    xgb_residual_pred = xgb_residual.predict(X_val)
    lgbm_residual_pred = lgbm_residual.predict(X_val)
    ensemble_residual_pred = (xgb_residual_pred + lgbm_residual_pred) / 2
    
    # 최종 예측 (1단계 + 2단계)
    final_pred = ensemble_pred_val + ensemble_residual_pred
    
    if is_insolation:
        final_pred = np.clip(final_pred, 0, None)
    else:
        final_pred = np.clip(final_pred, 0, 1)
    
    final_mae = mean_absolute_error(y_val_reset, final_pred)
    improvement = stage1_mae - final_mae
    
    print(f"    > 2단계 (최종) Validation MAE: {final_mae:.6f} (개선: {improvement:.6f})")
    
    return {
        'stage1_models': (xgb_model, lgbm_model, decline_model),
        'stage2_models': (xgb_residual, lgbm_residual),
        'final_mae': final_mae,
        'improvement': improvement,
        'use_log_scale': use_log_scale
    }

def predict_with_residual_enhanced(models_dict, X_test, is_insolation=False):
    """강화된 잔차학습 예측"""
    xgb_model, lgbm_model, decline_model = models_dict['stage1_models']
    xgb_residual, lgbm_residual = models_dict['stage2_models']
    use_log_scale = models_dict.get('use_log_scale', False)
    
    # 1단계 예측 (3개 모델 앙상블)
    xgb_pred = xgb_model.predict(X_test)
    lgbm_pred = lgbm_model.predict(X_test)
    decline_pred = decline_model.predict(X_test)
    
    stage1_pred = (xgb_pred * 0.4 + lgbm_pred * 0.4 + decline_pred * 0.2)
    
    # log 스케일인 경우 역변환
    if use_log_scale:
        stage1_pred = np.expm1(stage1_pred)
        stage1_pred = np.clip(stage1_pred, 0, None)
    else:
        if is_insolation:
            stage1_pred = np.clip(stage1_pred, 0, None)
        else:
            stage1_pred = np.clip(stage1_pred, 0, 1)
    
    # 2단계 잔차 예측
    xgb_residual_pred = xgb_residual.predict(X_test)
    lgbm_residual_pred = lgbm_residual.predict(X_test)
    stage2_pred = (xgb_residual_pred + lgbm_residual_pred) / 2
    
    # 최종 예측
    final_pred = stage1_pred + stage2_pred
    
    if is_insolation:
        final_pred = np.clip(final_pred, 0, None)
    else:
        final_pred = np.clip(final_pred, 0, 1)
    
    return final_pred

def add_sunshine_bin_onehot(df, col='일조(hr)', prefix='sun_bin'):
    """
    일조량을 구간화한 후, 구간별로 원-핫 인코딩된 피처를 추가합니다.
    
    Parameters:
    - df: 입력 DataFrame
    - col: 일조량 컬럼 이름 (기본: '일조(hr)')
    - prefix: 원핫 피처 prefix 이름 (기본: 'sun_bin')

    반환값:
    - df: 구간 힌트 피처들이 추가된 DataFrame
    """
    def sunshine_bin(x):
        if pd.isna(x): return -1
        elif x < 0.2: return 0     # 매우 낮음
        elif x < 0.5: return 1     # 약간 흐림
        elif x < 0.8: return 2     # 중간
        else: return 3             # 맑음

    bin_col = f"{prefix}_cat"
    df[bin_col] = df[col].apply(sunshine_bin).astype(int)

    # 원핫 인코딩
    onehot_df = pd.get_dummies(df[bin_col], prefix=prefix)
    df = pd.concat([df, onehot_df], axis=1)
    df.drop(columns=[bin_col], inplace=True)

    return df

# 전체 학습 방식 함수 추가
def train_global_model_with_building_features(train_data, features, target_col, building_col='건물번호'):
    """
    전체 데이터로 학습하되 건물별 특성을 반영하는 모델 훈련
    """
    print(f"전체 데이터로 {target_col} 모델 학습 시작... (건물번호 피처 포함)")
    
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
    
    # 전체 모델 학습
    global_models = train_models_with_residual_enhanced(
        X_train_global, y_train_global, X_val_global, y_val_global, 
        is_insolation=(target_col == '일사(MJ/m2)'), use_log_scale=True
    )
    
    return global_models, le, feature_cols

def predict_with_global_model(test_data, global_models, label_encoder, 
                            feature_cols, target_col, building_col='건물번호'):
    """
    전체 모델로 건물별 예측 수행
    """
    predictions = {}
    
    # 테스트 데이터에서 훈련에 사용된 건물들만 필터링
    train_buildings = set(label_encoder.classes_)
    test_buildings = set(test_data[building_col].unique())
    valid_buildings = train_buildings.intersection(test_buildings)
    
    print(f"    > 예측 가능한 건물 수: {len(valid_buildings)}/{len(test_buildings)}")
    
    # 테스트 데이터 준비
    test_encoded = test_data.copy()
    
    # 훈련에 없었던 건물은 평균값으로 대체
    unknown_buildings = test_buildings - train_buildings
    if unknown_buildings:
        print(f"    > 훈련에 없던 건물들: {sorted(unknown_buildings)} -> 평균 건물로 대체")
        # 가장 많은 건물의 인코딩값으로 대체 (보통 0)
        default_encoding = 0
        test_encoded.loc[test_encoded[building_col].isin(unknown_buildings), building_col] = label_encoder.classes_[default_encoding]
    
    # 라벨 인코딩 적용
    test_encoded[f'{building_col}_encoded'] = label_encoder.transform(test_encoded[building_col])
    X_test_final = test_encoded[feature_cols]
    
    # 전체 모델로 예측
    all_predictions = predict_with_residual_enhanced(
        global_models, X_test_final, 
        is_insolation=(target_col == '일사(MJ/m2)')
    )
    
    # 건물별로 예측 결과 저장
    for i, building_num in enumerate(test_data[building_col]):
        if building_num not in predictions:
            predictions[building_num] = []
        predictions[building_num].append(all_predictions[i])
    
    return predictions, all_predictions


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

# 전날 날씨 정보 추가
weather_lag_cols = ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)', '일조(hr)', '일사(MJ/m2)']
train_all = add_lag_features(train_all, weather_lag_cols, lag_hours=[24, 48])

train_all = add_weather_rolling_features(train_all)
train_all = add_sunshine_rolling_features(train_all)
train_all = add_sunshine_bin_onehot(train_all)
train_all = add_weather_volatility_features(train_all)  # 날씨 급변 피처 추가
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

print(f"[2] train zero_bno 일사 예측 시작 (잔차학습 + log1p 스케일링 적용)")

# 예측에 사용할 피처 선택
insolation_features = ['기온(°C)', '일조(hr)', '이슬점온도', '이슬점차이', '체감온도',
                       'SIN_시', 'COS_시', '정오거리', '시각', '요일', '월', 'SIN_일', 'COS_일', 'SIN_요일', 'COS_요일',
                       '정오거리_INV', 'peak_time']

insolation_features.extend([col for col in train_all.columns if '요일_' in col])
insolation_features.extend([col for col in train_all.columns if 'sun_bin' in col])
insolation_features.extend([col for col in train_all.columns if '건물유형_' in col])
insolation_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '이슬점' in col and ('변화량' in col or 'rolling' in col)])
insolation_features.extend([col for col in train_all.columns if '체감' in col and ('변화량' in col or 'rolling' in col)])
# 전날 날씨 정보 추가
insolation_features.extend([col for col in train_all.columns if 'lag_' in col])
insolation_features = list(set(insolation_features))

# '일사(MJ/m2)' 값이 0이 아닌 건물 데이터로 모델 학습 (전체 데이터 + 건물번호 사용)
train_insolation_data = train_all[~train_all['건물번호'].isin(zero_bnos) & (train_all['일사(MJ/m2)'] > 0)].copy()

# 전체 학습 방식 적용
insolation_models, insolation_label_encoder, insolation_feature_cols = train_global_model_with_building_features(
    train_insolation_data, insolation_features, '일사(MJ/m2)', '건물번호'
)

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
    
    # 전체 모델로 예측 (강화된 버전)
    _, building_predictions = predict_with_global_model(
        building_data, insolation_models, insolation_label_encoder, 
        insolation_feature_cols, '일사(MJ/m2)', '건물번호'
    )
    
    # 원본 데이터에 예측값 업데이트
    train_all.loc[building_data.index, '일사(MJ/m2)'] = building_predictions
    
    print(f"    > 건물 {bno}번: 예측 완료 (예측된 일사량 범위: {building_predictions.min():.4f} ~ {building_predictions.max():.4f})")

# 21시부터 05시까지의 일사량을 0으로 설정
train_all.loc[(train_all['건물번호'].isin(zero_bnos)) & ((train_all['시각'] >= 21) | (train_all['시각'] <= 5)), '일사(MJ/m2)'] = 0

train_all = add_insolation_rolling_features(train_all)
print(f"[3] train zero_bno 일사 예측 완료 (전체 학습 + 잔차학습 + log1p 스케일링 적용)")


########################## test 전처리 ############################

test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')
test_all = feature_engineering(test)

# 전날 날씨 정보 추가 (test 데이터에서는 기상 데이터만)
weather_lag_TEST_cols = ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)']
test_all = add_lag_features(test_all, weather_lag_TEST_cols, lag_hours=[24, 48])

test_all = add_weather_rolling_features(test_all)
test_all = add_weather_volatility_features(test_all)  # 날씨 급변 피처 추가

print(f"[4] test 일조시간 전체 학습 방식 예측 시작 (잔차학습 + log1p 스케일링 + 건물번호 피처 적용)")

# 예측에 사용할 피처 선택 (개선된 피처 포함)
sunshine_features = ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)', 
                     '이슬점온도', '이슬점차이', '체감온도',
                     'SIN_시', 'COS_시', '정오거리', '시각', '요일', '월', 'SIN_일', 'COS_일', 'SIN_요일', 'COS_요일',
                     '정오거리_INV', 'peak_time', '태양광용량(kW)', '냉방면적(m2)']

sunshine_features.extend([col for col in train_all.columns if '요일_' in col])
sunshine_features.extend([col for col in train_all.columns if 'sun_bin' in col])
sunshine_features.extend([col for col in train_all.columns if '건물유형_' in col])
sunshine_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
sunshine_features.extend([col for col in train_all.columns if '이슬점' in col and ('변화량' in col or 'rolling' in col)])
sunshine_features.extend([col for col in train_all.columns if '체감' in col and ('변화량' in col or 'rolling' in col)])
# 전날 기상 정보만 추가 (일조/일사량 lag는 test에 없으므로 제외)
sunshine_features.extend([col for col in train_all.columns if 'lag_' in col and ('기온' in col or '습도' in col or '풍속' in col or '강수량' in col)])
sunshine_features = list(set(sunshine_features))

# train과 test에 공통으로 존재하는 피처만 사용
sunshine_features = [col for col in sunshine_features if col in test_all.columns]

# 전체 학습 방식으로 일조시간 모델 훈련
sunshine_models, sunshine_label_encoder, sunshine_feature_cols = train_global_model_with_building_features(
    train_all, sunshine_features, '일조(hr)', '건물번호'
)

# 전체 테스트 데이터에 대해 예측 수행
print(f"    > 전체 테스트 데이터에 대해 일조시간 예측 수행...")
sunshine_predictions_dict, all_sunshine_predictions = predict_with_global_model(
    test_all, sunshine_models, sunshine_label_encoder, 
    sunshine_feature_cols, '일조(hr)', '건물번호'
)

# 예측값 저장
test_all['일조(hr)'] = all_sunshine_predictions

# 오후 8시부터 오전 6시까지의 일조시간을 0으로 설정
test_all.loc[(test_all['시각'] >= 20) | (test_all['시각'] <= 6), '일조(hr)'] = 0

test_all = add_sunshine_rolling_features(test_all)
print(f"[5] test 일조시간 전체 학습 방식 예측 완료 - Validation MAE: {sunshine_models['final_mae']:.6f} (개선: {sunshine_models['improvement']:.6f})")


print(f"[6] test 일사량 전체 학습 방식 예측 시작 (잔차학습 + log1p 스케일링 + 건물번호 피처 적용)")

# 예측에 사용할 피처 선택
insolation_features_test = sunshine_features + ['일조(hr)']
insolation_features_test.extend([col for col in test_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features_test = list(set(insolation_features_test))

# train과 test에 공통으로 존재하는 피처만 사용
insolation_features_test = [col for col in insolation_features_test if col in test_all.columns]

# 전체 학습 방식으로 일사량 모델 훈련
insolation_test_models, insolation_test_label_encoder, insolation_test_feature_cols = train_global_model_with_building_features(
    train_all, insolation_features_test, '일사(MJ/m2)', '건물번호'
)

# 전체 테스트 데이터에 대해 예측 수행
print(f"    > 전체 테스트 데이터에 대해 일사량 예측 수행...")
insolation_predictions_dict, all_insolation_predictions = predict_with_global_model(
    test_all, insolation_test_models, insolation_test_label_encoder, 
    insolation_test_feature_cols, '일사(MJ/m2)', '건물번호'
)

# 예측값 저장
test_all['일사(MJ/m2)'] = all_insolation_predictions

# 오후 9시부터 오전 5시까지의 일사량을 0으로 설정
test_all.loc[(test_all['시각'] >= 21) | (test_all['시각'] <= 5), '일사(MJ/m2)'] = 0

test_all = add_insolation_rolling_features(test_all)
print(f"[7] test 일사량 전체 학습 방식 예측 완료 - Validation MAE: {insolation_test_models['final_mae']:.6f} (개선: {insolation_test_models['improvement']:.6f})")

# 결과 저장
train_all.to_csv(csv_path + 'predict_train.csv', index=False)
test_all.to_csv(csv_path + 'predict_test.csv', index=False)

print(f"[8] 저장 완료")

# 최종 결과 요약
print("\n" + "="*50)
print("최종 결과 요약 (전체 학습 + 잔차학습 + log1p 스케일링 + 건물번호 피처)")
print("="*50)
print(f"Train 일사량 예측 개선량: {insolation_models['improvement']:.6f}")
print(f"Test 일조시간 예측 Validation MAE: {sunshine_models['final_mae']:.6f} (개선: {sunshine_models['improvement']:.6f})")
print(f"Test 일사량 예측 Validation MAE: {insolation_test_models['final_mae']:.6f} (개선: {insolation_test_models['improvement']:.6f})")

# 개선사항 요약
print("\n" + "="*30)
print("주요 개선사항:")
print("="*30)
print("1. ⭐ 전체 학습 방식 도입: 지역별 학습 → 전체 학습 + 건물번호 피처")
print("2. 일사량 예측에 log1p 스케일링 적용")
print("3. 일조시간 예측에도 log1p 스케일링 적용")
print("4. 이슬점온도 및 체감온도 피처 추가")
print("5. 전날(24h, 48h) 날씨 정보 lag 피처 추가")
print("6. 이슬점차이 (기온-이슬점) 피처 추가")
print("7. 개선된 피처 호환성 처리 (train/test 공통 피처 사용)")
print("8. 더욱 안정적인 lag 피처 생성")
print("9. Sin/Cos 주기적 패턴 피처 수정 (올바른 주기 적용)")
print("10. 날씨 급변 감지 피처 추가 (기온/습도/풍속 급변, 강수 패턴)")
print("11. 하락 구간 가중치 학습 및 하락 특화 모델 추가")
print("12. 시간대별 이상치 감지 Z-score 피처")
print("13. ⭐ 건물번호를 라벨 인코딩하여 건물별 특성 학습")
print("14. ⭐ 더 많은 데이터로 robust한 모델 학습")

# 로그 파일에 결과 저장
with open(csv_path + "(LOG)01_preprocessing.txt", "a") as f:
    f.write(f"<SEED : {SEED}> - 전체 학습 + 잔차학습 + log1p + 건물번호 피처\n")
    f.write(f"Train 일사량 예측 개선량: {insolation_models['improvement']:.6f}\n")
    f.write(f"Test 일조(hr) MAE : {sunshine_models['final_mae']:.6f} (개선: {sunshine_models['improvement']:.6f})\n")
    f.write(f"Test 일사(MJ/m2) MAE : {insolation_test_models['final_mae']:.6f} (개선: {insolation_test_models['improvement']:.6f})\n")
    f.write("개선사항: 전체 학습 방식, 건물번호 피처, log1p, 이슬점온도, 체감온도, 전날 기상 lag, 날씨 급변 감지, 하락 가중치\n")
    f.write("="*40 + "\n")
    
print(f"[03_preprocessing] 종료 (전체 학습 방식 적용 완료)")