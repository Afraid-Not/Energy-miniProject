import numpy as np
import random
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SEED = 333
np.random.seed(SEED)
random.seed(SEED)

data_path = './Energy/'
csv_path = './Energy/final/train_test/'

train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)

def feature_engineering(df, latitude=37.5665):
    """
    기상 데이터와 시간 정보를 활용한 피처 엔지니어링
    """
    df = df.copy()
    
    # 기본 시간 피처 생성
    df['Date'] = pd.to_datetime(df['일시'])
    
    # 시간 관련 피처
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekday'] = df['Date'].dt.weekday
    
    # 주말 여부
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 순환 피처 (Cyclical features)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 계절 구분
    df['season'] = ((df['month'] % 12) // 3).astype(int)  # 0:겨울, 1:봄, 2:여름, 3:가을
    
    # 근무시간대 구분
    df['is_work_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & (df['dayofweek'] < 5)).astype(int)
    df['is_peak_hours'] = ((df['hour'].isin([8, 9, 18, 19, 20])) & (df['dayofweek'] < 5)).astype(int)
    
    ################################# 태양 관련 피처 #############################################
    
    # 태양 적위각 (Solar declination)
    df['solar_declination'] = 23.45 * np.sin(np.radians(360 * (284 + df['dayofyear']) / 365.25))
    
    # 시간각 계산 (hour angle)
    decimal_hour = df['hour']  # 분 정보가 없으므로 시간만 사용
    df['hour_angle'] = 15 * (decimal_hour - 12)  # degrees
    
    # 태양 고도각 계산 (solar elevation angle)
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(df['solar_declination'])
    hour_rad = np.radians(df['hour_angle'])
    
    sin_elevation = (np.sin(lat_rad) * np.sin(dec_rad) + 
                    np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad))
    df['solar_elevation'] = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
    
    # 태양 방위각 계산 (solar azimuth angle) - 0으로 나누기 방지
    cos_elevation = np.cos(np.radians(df['solar_elevation']))
    cos_elevation = np.where(cos_elevation == 0, 1e-10, cos_elevation)  # 0 방지
    
    cos_azimuth = ((np.sin(dec_rad) * np.cos(lat_rad) - 
                   np.cos(dec_rad) * np.sin(lat_rad) * np.cos(hour_rad)) / cos_elevation)
    cos_azimuth = np.clip(cos_azimuth, -1, 1)
    azimuth = np.degrees(np.arccos(cos_azimuth))
    df['solar_azimuth'] = np.where(df['hour_angle'] > 0, azimuth, 360 - azimuth)
    
    # 일출/일몰 시간 계산
    cos_sunrise = -np.tan(lat_rad) * np.tan(dec_rad)
    cos_sunrise = np.clip(cos_sunrise, -1, 1)
    sunrise_hour_angle = np.degrees(np.arccos(cos_sunrise))
    df['sunrise_time'] = 12 - sunrise_hour_angle / 15
    df['sunset_time'] = 12 + sunrise_hour_angle / 15
    df['daylight_hours'] = df['sunset_time'] - df['sunrise_time']
    
    # 태양 관련 시간 피처
    df['time_to_solar_noon'] = np.abs(decimal_hour - 12)
    df['time_from_sunrise'] = decimal_hour - df['sunrise_time']
    df['time_to_sunset'] = df['sunset_time'] - decimal_hour
    
    # 태양 위치 기반 플래그
    df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)
    df['is_sunlight_hours'] = ((decimal_hour >= df['sunrise_time']) & 
                               (decimal_hour <= df['sunset_time'])).astype(int)
    df['is_prime_solar_hours'] = ((df['solar_elevation'] > 30) & 
                                  (df['time_to_solar_noon'] < 3)).astype(int)
    
    # 대기권 외부 일사량 (extraterrestrial solar radiation)
    solar_constant = 1367  # W/m²
    earth_sun_distance = 1 + 0.033 * np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['extraterrestrial_radiation'] = (solar_constant * earth_sun_distance * 
                                       np.maximum(0, np.sin(np.radians(df['solar_elevation']))))
    
    # 대기 질량 계산 (Air mass)
    elevation_positive = np.maximum(df['solar_elevation'], 0.1)
    air_mass = 1 / (np.sin(np.radians(elevation_positive)) + 
                    0.50572 * (elevation_positive + 6.07995)**(-1.6364))
    df['air_mass'] = np.where(df['solar_elevation'] > 0, air_mass, 10)  # inf 대신 큰 값
    
    # 청명도 지수 근사
    df['clearness_index'] = np.exp(-df['습도(%)'] / 100) * (1 - np.minimum(df['강수량(mm)'] / 10, 1))
    df['clearness_index'] = np.clip(df['clearness_index'], 0, 1)
    
    # 계절별 태양 각도 보정
    df['seasonal_solar_factor'] = np.cos(2 * np.pi * (df['dayofyear'] - 172) / 365.25)
    
    # 구름량 추정
    df['estimated_cloud_cover'] = np.clip(
        (df['습도(%)'] / 100) * 0.7 + 
        np.clip(df['강수량(mm)'] / 5, 0, 1) * 0.3, 0, 1
    )
    
    # 태양복사 감쇠 인자
    df['solar_attenuation'] = (1 - df['estimated_cloud_cover']) * df['clearness_index']
    
    # 예상 직달 일사량
    df['estimated_direct_radiation'] = (df['extraterrestrial_radiation'] * 
                                       df['solar_attenuation'] * 
                                       np.maximum(0, np.sin(np.radians(df['solar_elevation']))))
    
    # 하루 중 태양 에너지 비율
    daylight_hours_safe = np.maximum(df['daylight_hours'], 1e-6)  # 0으로 나누기 방지
    daylight_progress = np.clip((decimal_hour - df['sunrise_time']) / daylight_hours_safe, 0, 1)
    df['solar_day_progress'] = daylight_progress
    df['solar_energy_potential'] = np.sin(np.pi * daylight_progress) * df['is_daytime']
    
    ########################################## 기온 관련 피처 #######################################
    
    # 절대 온도
    df['temp_kelvin'] = df['기온(°C)'] + 273.15
    
    # 온도 범주화
    df['temp_category'] = pd.cut(df['기온(°C)'], 
                                bins=[-50, 0, 10, 20, 30, 50], 
                                labels=[0, 1, 2, 3, 4], include_lowest=True).astype(float)
    
    # 냉난방도일 (Cooling/Heating Degree Days) - 기준온도 18°C
    df['heating_degree_days'] = np.maximum(0, 18 - df['기온(°C)'])
    df['cooling_degree_days'] = np.maximum(0, df['기온(°C)'] - 24)
    
    # 온도 범위별 플래그
    df['is_freezing'] = (df['기온(°C)'] <= 0).astype(int)
    df['is_mild'] = ((df['기온(°C)'] > 10) & (df['기온(°C)'] <= 25)).astype(int)
    df['is_hot'] = (df['기온(°C)'] > 30).astype(int)
    df['is_extreme_cold'] = (df['기온(°C)'] < -10).astype(int)
    df['is_extreme_hot'] = (df['기온(°C)'] > 35).astype(int)
    
    # 체감온도 (Wind Chill / Heat Index)
    # 체감온도 계산 (간단한 모델)
    df['apparent_temp'] = (df['기온(°C)'] + 
                          0.33 * (df['습도(%)'] / 100 * 6.105 * 
                                  np.exp(17.27 * df['기온(°C)'] / (237.7 + df['기온(°C)']))) - 
                          0.7 * df['풍속(m/s)'] - 4.0)
    
    # 불쾌지수 (Discomfort Index)
    df['discomfort_index'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    
    ########################################## 기상 변수 상호작용 #######################################
    
    # 기상 변수간 상호작용
    df['temp_humidity_interaction'] = df['기온(°C)'] * df['습도(%)'] / 100
    df['temp_wind_interaction'] = df['기온(°C)'] * df['풍속(m/s)']
    df['wind_humidity_interaction'] = df['풍속(m/s)'] * df['습도(%)'] / 100
    
    # 강수량 관련 피처
    df['has_precipitation'] = (df['강수량(mm)'] > 0).astype(int)
    df['precipitation_category'] = pd.cut(df['강수량(mm)'], 
                                         bins=[0, 0.1, 5, 20, 100], 
                                         labels=[0, 1, 2, 3], include_lowest=True).astype(float)
    
    # 습도 관련 피처
    df['humidity_category'] = pd.cut(df['습도(%)'], 
                                    bins=[0, 30, 60, 80, 100], 
                                    labels=[0, 1, 2, 3], include_lowest=True).astype(float)
    df['is_humid'] = (df['습도(%)'] > 70).astype(int)
    df['is_dry'] = (df['습도(%)'] < 40).astype(int)
    
    # 풍속 관련 피처
    df['wind_category'] = pd.cut(df['풍속(m/s)'], 
                                bins=[0, 2, 5, 10, 50], 
                                labels=[0, 1, 2, 3], include_lowest=True).astype(float)
    df['is_windy'] = (df['풍속(m/s)'] > 5).astype(int)
    
    ########################################## 시계열 피처 (정렬 기반) #######################################
    
    # 데이터를 건물별, 시간순으로 정렬
    df_sorted = df.sort_values(['건물번호', 'Date']).copy()
    
    # 온도 변화율 (lag features)
    for col in ['기온(°C)', '습도(%)', '풍속(m/s)']:
        df_sorted[f'{col}_lag1'] = df_sorted.groupby('건물번호')[col].shift(1)
        df_sorted[f'{col}_change'] = df_sorted[col] - df_sorted[f'{col}_lag1']
        df_sorted[f'{col}_change'] = df_sorted[f'{col}_change'].fillna(0)
    
    # 롤링 통계 (3시간, 6시간, 12시간, 24시간)
    for window in [3, 6, 12, 24]:
        for col in ['기온(°C)', '습도(%)', '풍속(m/s)']:
            df_sorted[f'{col}_mean_{window}h'] = df_sorted.groupby('건물번호')[col].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            df_sorted[f'{col}_std_{window}h'] = df_sorted.groupby('건물번호')[col].rolling(
                window=window, min_periods=1).std().reset_index(0, drop=True).fillna(0)
            df_sorted[f'{col}_max_{window}h'] = df_sorted.groupby('건물번호')[col].rolling(
                window=window, min_periods=1).max().reset_index(0, drop=True)
            df_sorted[f'{col}_min_{window}h'] = df_sorted.groupby('건물번호')[col].rolling(
                window=window, min_periods=1).min().reset_index(0, drop=True)
    
    # 일교차 (24시간 내 최대-최소)
    df_sorted['temp_daily_range'] = (df_sorted['기온(°C)_max_24h'] - 
                                    df_sorted['기온(°C)_min_24h'])
    
    # 안정성 지표
    df_sorted['temp_stability_3h'] = 1 / (1 + df_sorted['기온(°C)_std_3h'])
    df_sorted['temp_stability_12h'] = 1 / (1 + df_sorted['기온(°C)_std_12h'])
    
    ########################################## 건물별 피처 #######################################
    
    # 건물별 통계 (전체 기간 기준)
    building_stats = df_sorted.groupby('건물번호').agg({
        '기온(°C)': ['mean', 'std'],
        '습도(%)': ['mean', 'std'],
        '풍속(m/s)': ['mean', 'std']
    }).round(3)
    
    # 컬럼명 평탄화
    building_stats.columns = ['_'.join(col).strip() for col in building_stats.columns]
    building_stats = building_stats.add_prefix('building_')
    
    # 건물별 통계를 원본 데이터에 병합
    df_sorted = df_sorted.merge(building_stats, left_on='건물번호', right_index=True, how='left')
    
    # 건물별 편차
    df_sorted['temp_dev_from_building'] = (df_sorted['기온(°C)'] - 
                                          df_sorted['building_기온(°C)_mean'])
    df_sorted['humidity_dev_from_building'] = (df_sorted['습도(%)'] - 
                                              df_sorted['building_습도(%)_mean'])
    
    ########################################## 최종 정리 #######################################
    
    # 원본 인덱스 순서로 복원
    df_result = df_sorted.sort_index()
    
    # 불필요한 중간 컬럼 제거
    cols_to_drop = [col for col in df_result.columns if col.endswith('_lag1')]
    df_result = df_result.drop(columns=cols_to_drop, errors='ignore')
    
    # NaN 값 처리
    df_result = df_result.fillna(0)
    
    return df_result

# 피처 엔지니어링 실행
print("피처 엔지니어링 시작...")
train = feature_engineering(train_csv)
test = feature_engineering(test_csv)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def solar_imputation(train_df, zero_bnos, n_splits=5):
    """
    KNN + LightGBM 앙상블을 사용한 일사량 보간
    
    Parameters:
    -----------
    train_df : DataFrame
        피처 엔지니어링이 완료된 훈련 데이터
    zero_bnos : list
        일사량이 0인 건물 번호들
    n_splits : int
        교차검증 폴드 수
    
    Returns:
    --------
    train_imputed : DataFrame
        일사량이 보간된 훈련 데이터
    """
    
    print("=== 일사량 보간 시작 ===")
    train_imputed = train_df.copy()
    
    # 일사량 피처가 있는지 확인
    if '일사(MJ/m2)' not in train_imputed.columns:
        print("Error: '일사(MJ/m2)' 컬럼이 없습니다.")
        return train_imputed
    
    # 전체 데이터에서 일사량이 있는 건물과 없는 건물 분리
    has_solar_mask = ~train_imputed['건물번호'].isin(zero_bnos)
    no_solar_mask = train_imputed['건물번호'].isin(zero_bnos)
    
    print(f"일사량 데이터가 있는 건물: {len(train_imputed[has_solar_mask]['건물번호'].unique())}개")
    print(f"일사량 데이터가 없는 건물: {len(zero_bnos)}개")
    print(f"보간할 데이터 포인트: {no_solar_mask.sum():,}개")
    
    # 일사량이 있는 데이터로 학습용 데이터셋 구성
    train_solar = train_imputed[has_solar_mask].copy()
    predict_solar = train_imputed[no_solar_mask].copy()
    
    # 피처 선택 (일사량 예측에 유용한 피처들)
    solar_features = [
        # 기본 시간 피처
        'month', 'day', 'hour', 'dayofweek', 'dayofyear', 'quarter',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'day_sin', 'day_cos', 'month_sin', 'month_cos', 'season',
        
        # 태양 관련 피처 (핵심)
        'solar_declination', 'hour_angle', 'solar_elevation', 'solar_azimuth',
        'sunrise_time', 'sunset_time', 'daylight_hours', 'time_to_solar_noon',
        'time_from_sunrise', 'time_to_sunset', 'is_daytime', 'is_sunlight_hours',
        'is_prime_solar_hours', 'extraterrestrial_radiation', 'air_mass',
        'clearness_index', 'seasonal_solar_factor', 'estimated_cloud_cover',
        'solar_attenuation', 'estimated_direct_radiation', 'solar_day_progress',
        'solar_energy_potential',
        
        # 기상 피처
        '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
        'temp_kelvin', 'heating_degree_days', 'cooling_degree_days',
        'apparent_temp', 'discomfort_index',
        
        # 기상 상호작용
        'temp_humidity_interaction', 'temp_wind_interaction', 'wind_humidity_interaction',
        'has_precipitation', 'precipitation_category', 'humidity_category', 'wind_category',
        'is_humid', 'is_dry', 'is_windy',
        
        # 시계열 피처 (변화율)
        '기온(°C)_change', '습도(%)_change', '풍속(m/s)_change',
        
        # 롤링 통계 (일부만 선택)
        '기온(°C)_mean_3h', '기온(°C)_std_3h', '습도(%)_mean_3h', '습도(%)_std_3h',
        '기온(°C)_mean_6h', '기온(°C)_std_6h', 'temp_daily_range',
        'temp_stability_3h', 'temp_stability_12h',
        
        # 건물 관련
        'temp_dev_from_building', 'humidity_dev_from_building'
    ]
    
    # 실제로 존재하는 피처만 선택
    available_features = [f for f in solar_features if f in train_solar.columns]
    print(f"사용할 피처 수: {len(available_features)}")
    
    # 학습/예측 데이터 준비
    X_train = train_solar[available_features].fillna(0)
    y_train = train_solar['일사(MJ/m2)'].fillna(0)
    X_predict = predict_solar[available_features].fillna(0)
    
    print(f"학습 데이터 크기: {X_train.shape}")
    print(f"예측 데이터 크기: {X_predict.shape}")
    
    # 1. KNN 보간
    print("\n--- KNN 보간 진행 ---")
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # KNN을 위한 데이터 준비 (일사량이 0인 값을 NaN으로 변경)
    y_train_knn = y_train.copy()
    
    # KNN으로 예측
    X_combined = pd.concat([X_train, X_predict], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train_knn, pd.Series([np.nan] * len(X_predict))], axis=0, ignore_index=True)
    
    # KNN 보간 수행
    combined_data = pd.concat([X_combined, y_combined.rename('solar')], axis=1)
    imputed_data = pd.DataFrame(knn_imputer.fit_transform(combined_data), columns=combined_data.columns)
    
    knn_predictions = imputed_data['solar'].iloc[len(X_train):].values
    
    print(f"KNN 예측 완료: {len(knn_predictions)}개")
    print(f"KNN 예측 범위: {knn_predictions.min():.3f} ~ {knn_predictions.max():.3f}")
    
    # 2. LightGBM 교차검증 예측
    print("\n--- LightGBM 교차검증 진행 ---")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    lgb_predictions = np.zeros(len(X_predict))
    oof_predictions = np.zeros(len(X_train))
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    feature_importance = np.zeros(len(available_features))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}/{n_splits} 진행중...")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 모델 훈련
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # 검증 예측
        oof_predictions[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # 테스트 예측
        lgb_predictions += model.predict(X_predict, num_iteration=model.best_iteration) / n_splits
        
        # 피처 중요도 누적
        feature_importance += model.feature_importance(importance_type='gain') / n_splits
    
    # 교차검증 성능 출력
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    oof_mae = mean_absolute_error(y_train, oof_predictions)
    print(f"\nLightGBM OOF RMSE: {oof_rmse:.4f}")
    print(f"LightGBM OOF MAE: {oof_mae:.4f}")
    print(f"LightGBM 예측 범위: {lgb_predictions.min():.3f} ~ {lgb_predictions.max():.3f}")
    
    # 3. 앙상블 (KNN + LightGBM)
    print("\n--- 앙상블 예측 ---")
    ensemble_weight_knn = 0.3
    ensemble_weight_lgb = 0.7
    
    ensemble_predictions = (ensemble_weight_knn * knn_predictions + 
                           ensemble_weight_lgb * lgb_predictions)
    
    print(f"앙상블 가중치: KNN={ensemble_weight_knn}, LightGBM={ensemble_weight_lgb}")
    print(f"앙상블 예측 범위: {ensemble_predictions.min():.3f} ~ {ensemble_predictions.max():.3f}")
    
    # 4. 후처리 및 제약 조건 적용
    print("\n--- 후처리 및 제약 조건 적용 ---")
    
    final_predictions = ensemble_predictions.copy()
    
    # 음수를 0으로 클리핑
    negative_count = (final_predictions < 0).sum()
    final_predictions = np.maximum(final_predictions, 0)
    print(f"음수 값 0으로 클리핑: {negative_count}개")
    
    # 최대값을 3.5로 클리핑
    max_count = (final_predictions > 3.5).sum()
    final_predictions = np.minimum(final_predictions, 3.5)
    print(f"최대값 3.5로 클리핑: {max_count}개")
    
    # 일출 이전, 일몰 이후 시간은 0으로 설정
    predict_solar_copy = predict_solar.copy()
    predict_solar_copy['ensemble_solar'] = final_predictions
    
    # 일출/일몰 기반 필터링
    night_mask = (predict_solar_copy['is_sunlight_hours'] == 0) | (predict_solar_copy['solar_elevation'] <= 0)
    night_count = night_mask.sum()
    final_predictions[night_mask] = 0
    print(f"야간 시간대 0으로 설정: {night_count}개")
    
    # 급변하는 일사량 스무딩 (건물별로 처리)
    print("\n--- 급변 일사량 스무딩 ---")
    final_predictions_smoothed = final_predictions.copy()
    
    for bno in zero_bnos:
        building_mask = predict_solar['건물번호'] == bno
        building_predictions = final_predictions[building_mask]
        building_hours = predict_solar[building_mask]['hour'].values
        
        if len(building_predictions) > 2:
            # 시간순 정렬
            sorted_indices = np.argsort(building_hours)
            sorted_predictions = building_predictions[sorted_indices]
            
            # 3-point 이동평균으로 스무딩 (야간 제외)
            smoothed = sorted_predictions.copy()
            for i in range(1, len(smoothed) - 1):
                if sorted_predictions[i] > 0:  # 0이 아닌 값만 스무딩
                    window = sorted_predictions[max(0, i-1):min(len(sorted_predictions), i+2)]
                    if len(window) >= 2:
                        smoothed[i] = np.mean(window)
            
            # 원래 순서로 복원
            restore_indices = np.argsort(sorted_indices)
            final_predictions_smoothed[building_mask] = smoothed[restore_indices]
    
    print("급변 일사량 스무딩 완료")
    
    # 최종 결과 적용
    train_imputed.loc[no_solar_mask, '일사(MJ/m2)'] = final_predictions_smoothed
    
    # 결과 요약
    print(f"\n=== 보간 결과 요약 ===")
    print(f"최종 예측 범위: {final_predictions_smoothed.min():.3f} ~ {final_predictions_smoothed.max():.3f}")
    print(f"0값 비율: {(final_predictions_smoothed == 0).mean():.1%}")
    print(f"최대값(3.5) 비율: {(final_predictions_smoothed == 3.5).mean():.1%}")
    
    # 건물별 일사량 통계
    print(f"\n=== 건물별 일사량 통계 ===")
    for bno in zero_bnos[:5]:  # 처음 5개 건물만 출력
        building_solar = final_predictions_smoothed[predict_solar['건물번호'] == bno]
        print(f"건물 {bno}: 평균={building_solar.mean():.3f}, "
              f"최대={building_solar.max():.3f}, "
              f"0값비율={((building_solar == 0).mean())*100:.1f}%")
    
    # 피처 중요도 상위 10개 출력
    print(f"\n=== 주요 피처 중요도 (상위 10개) ===")
    feature_importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.1f}")
    
    return train_imputed

# 일사량 보간 실행
train_with_solar = solar_imputation(train, zero_bnos, n_splits=5)

# 보간 전후 비교
print(f"\n=== 보간 전후 비교 ===")
print(f"보간 전 일사량 결측값: {(train['일사(MJ/m2)'] == 0).sum():,}개")
print(f"보간 후 일사량 결측값: {(train_with_solar['일사(MJ/m2)'] == 0).sum():,}개")

# 최종 데이터 저장 (선택사항)
train_with_solar.to_csv(csv_path + 'train_with_imputed_solar.csv')
print(f"\n일사량 보간 완료!")