print(f"[preprocessing] 시작")
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
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import log_evaluation, early_stopping
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# 경로 설정
data_path = './Energy/'
log_path = './Energy/100/log/'
trainer = './Energy/100/new_csv/'
save_path = './Energy/100/submission/'
os.makedirs(log_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)

# SEED 관리
seed_file = "./Energy/100/log/(SEED_COUNT)preprocessing.json"

if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

SEED = seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

print(f"[1] 전처리 시작")

# ========================== 1. 데이터 로딩 ==========================
print("[1-1] 데이터 로딩 중...")

# Building info 전처리
building_csv = pd.read_csv(data_path + 'building_info.csv')
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

# Train 데이터 로딩 및 머지
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')

# Test 데이터 로딩 및 머지
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ========================== 2. 특성 엔지니어링 함수 ==========================
def create_time_features(df):
    """시간 관련 특성 생성"""
    df = df.copy()
    
    # 날짜/시간 파싱
    df['datetime_str'] = df['일시'].astype(str)
    df['year'] = df['datetime_str'].str[:4].astype(int)
    df['month'] = df['datetime_str'].str[4:6].astype(int)
    df['day'] = df['datetime_str'].str[6:8].astype(int)
    df['hour'] = df['datetime_str'].str.split(' ').str[1].fillna('0').astype(int)
    
    # 계절 특성
    df['season'] = df['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                   6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
    
    # 주기적 특성 (sin, cos 변환)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # 날짜 순서 특성
    df['date_ordinal'] = pd.to_datetime(df['datetime_str'].str[:8], format='%Y%m%d').map(pd.Timestamp.toordinal)
    
    return df

def create_weather_features(df):
    """기상 관련 특성 생성"""
    df = df.copy()
    
    # 체감온도 (Heat Index 근사)
    df['heat_index'] = df['기온(°C)'] + 0.5 * (df['습도(%)'] / 100) * (df['기온(°C)'] - 14.5)
    
    # 바람 냉각 지수
    df['wind_chill'] = 13.12 + 0.6215 * df['기온(°C)'] - 11.37 * (df['풍속(m/s)'] ** 0.16) + 0.3965 * df['기온(°C)'] * (df['풍속(m/s)'] ** 0.16)
    
    # 불쾌지수
    df['discomfort_index'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    
    # 기상 조합 특성
    df['temp_humidity'] = df['기온(°C)'] * df['습도(%)'] / 100
    df['temp_wind'] = df['기온(°C)'] * df['풍속(m/s)']
    df['rain_wind'] = df['강수량(mm)'] * df['풍속(m/s)']
    
    # 기상 상태 분류
    df['is_rainy'] = (df['강수량(mm)'] > 0).astype(int)
    df['is_windy'] = (df['풍속(m/s)'] > 3).astype(int)
    df['is_hot'] = (df['기온(°C)'] > 25).astype(int)
    df['is_cold'] = (df['기온(°C)'] < 10).astype(int)
    df['is_humid'] = (df['습도(%)'] > 70).astype(int)
    
    return df

# ========================== 3. 앙상블 보간 클래스 ==========================
class EnsembleImputer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        
    def _prepare_features(self, df, target_col):
        """보간을 위한 특성 준비 - 날씨 관련 피쳐만 사용"""
        # 기본 날씨 특성만 사용
        feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
        
        # 시간 특성 추가 (태양 관련이므로 날씨와 연관)
        time_cols = ['hour', 'month', 'day', 'season', 'hour_sin', 'hour_cos', 
                    'month_sin', 'month_cos', 'day_sin', 'day_cos']
        
        # 기상 조합 특성 추가  
        weather_cols = ['heat_index', 'wind_chill', 'discomfort_index', 'temp_humidity',
                       'temp_wind', 'rain_wind', 'is_rainy', 'is_windy', 'is_hot', 'is_cold', 'is_humid']
        
        # 날씨 관련 특성만 결합
        all_features = feature_cols + time_cols + weather_cols
        
        # 존재하는 특성만 선택
        available_features = [col for col in all_features if col in df.columns]
        
        return df[available_features].fillna(0), available_features
    
    def fit(self, df, target_col, sample_ratio=0.3):
        """앙상블 모델 학습"""
        print(f"    - {target_col} 보간 모델 학습 중...")
        
        # 완전한 데이터만 사용 (target이 결측이 아닌)
        complete_data = df[df[target_col].notna()].copy()
        
        # 샘플링 (메모리 절약)
        if len(complete_data) > 50000:
            complete_data = complete_data.sample(n=int(len(complete_data) * sample_ratio), 
                                               random_state=self.random_state)
        
        X, feature_cols = self._prepare_features(complete_data, target_col)
        y = complete_data[target_col]
        
        print(f"      학습 데이터: {len(X)} 행, {len(feature_cols)} 특성")
        
        self.feature_columns = feature_cols
        
        # 데이터 스케일링 (KNN을 위해)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[target_col] = scaler
        
        # 모델들 정의
        models = {
            'knn': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
            'lgbm': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            'xgb': XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0
            )
        }
        
        # 모델별 학습
        fitted_models = {}
        
        # KNN (스케일된 데이터 사용)
        print("      KNN 학습 중...")
        models['knn'].fit(X_scaled, y)
        fitted_models['knn'] = models['knn']
        
        # LightGBM (원본 데이터 사용)
        print("      LightGBM 학습 중...")
        models['lgbm'].fit(X, y)
        fitted_models['lgbm'] = models['lgbm']
        
        # XGBoost (원본 데이터 사용)
        print("      XGBoost 학습 중...")
        models['xgb'].fit(X, y)
        fitted_models['xgb'] = models['xgb']
        
        self.models[target_col] = fitted_models
        
        # 모델 성능 평가 (Cross-validation)
        from sklearn.model_selection import cross_val_score
        print("      모델 성능 평가 (CV R²):")
        
        cv_scores = cross_val_score(fitted_models['knn'], X_scaled, y, cv=3, scoring='r2', n_jobs=-1)
        print(f"        KNN: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        cv_scores = cross_val_score(fitted_models['lgbm'], X, y, cv=3, scoring='r2', n_jobs=-1)
        print(f"        LGBM: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        cv_scores = cross_val_score(fitted_models['xgb'], X, y, cv=3, scoring='r2', n_jobs=-1)
        print(f"        XGB: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
    def predict(self, df, target_col, weights={'knn': 0.2, 'lgbm': 0.4, 'xgb': 0.4}):
        """앙상블 예측"""
        if target_col not in self.models:
            raise ValueError(f"모델이 {target_col}에 대해 학습되지 않았습니다.")
        
        X, _ = self._prepare_features(df, target_col)
        X = X[self.feature_columns]  # 학습시 사용한 특성만 선택
        
        # 각 모델별 예측
        predictions = {}
        
        # KNN 예측 (스케일된 데이터)
        X_scaled = self.scalers[target_col].transform(X.fillna(0))
        predictions['knn'] = self.models[target_col]['knn'].predict(X_scaled)
        
        # LGBM, XGB 예측 (원본 데이터)
        X_filled = X.fillna(0)
        predictions['lgbm'] = self.models[target_col]['lgbm'].predict(X_filled)
        predictions['xgb'] = self.models[target_col]['xgb'].predict(X_filled)
        
        # 가중 평균 앙상블
        ensemble_pred = (weights['knn'] * predictions['knn'] + 
                        weights['lgbm'] * predictions['lgbm'] + 
                        weights['xgb'] * predictions['xgb'])
        
        # 음수 값 처리 (일조/일사는 음수가 될 수 없음)
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred

# ========================== 4. 특성 엔지니어링 적용 ==========================
print("[2] 특성 엔지니어링")

train_fe = create_time_features(train)
train_fe = create_weather_features(train_fe)

test_fe = create_time_features(test)
test_fe = create_weather_features(test_fe)

print("특성 엔지니어링 완료")

# ========================== 5. 문제 건물 일사량 보간 ==========================
problem_buildings = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
print(f"\n[3] 문제 건물들의 일사량 앙상블 보간")
print(f"문제 건물 번호: {problem_buildings}")

# 문제 건물 확인
for building_num in problem_buildings:
    building_data = train_fe[train_fe['건물번호'] == building_num]
    solar_unique = building_data['일사(MJ/m2)'].unique()
    print(f"건물 {building_num}: 일사량 고유값 = {solar_unique}")

# 일사량 앙상블 보간
solar_imputer = EnsembleImputer(random_state=SEED)

# 정상 건물 데이터로 모델 학습
normal_train_data = train_fe[~train_fe['건물번호'].isin(problem_buildings)].copy()
solar_imputer.fit(normal_train_data, '일사(MJ/m2)')

# 문제 건물들의 일사량 예측
train_updated = train_fe.copy()

for building_num in tqdm(problem_buildings, desc="일사량 보간"):
    building_mask = train_updated['건물번호'] == building_num
    building_data = train_updated[building_mask].copy()
    
    # 일조가 있는 경우에만 일사량 예측
    sunshine_mask = building_data['일조(hr)'] > 0
    if sunshine_mask.sum() > 0:
        predicted_solar = solar_imputer.predict(building_data[sunshine_mask], '일사(MJ/m2)')
        
        # 예측 결과 적용
        building_indices = building_data[sunshine_mask].index
        train_updated.loc[building_indices, '일사(MJ/m2)'] = predicted_solar

print("문제 건물 일사량 보간 완료")

# 보간 결과 확인
print("보간 결과:")
for building_num in problem_buildings:
    building_data = train_updated[train_updated['건물번호'] == building_num]
    solar_positive = building_data[building_data['일사(MJ/m2)'] > 0]
    if len(solar_positive) > 0:
        print(f"  건물 {building_num}: 양수 일사량 {len(solar_positive)}개, 최대 {solar_positive['일사(MJ/m2)'].max():.2f}, 평균 {solar_positive['일사(MJ/m2)'].mean():.2f}")

# ========================== 6. Test 데이터 일조 예측 ==========================
print(f"\n[4] Test 데이터 일조 앙상블 예측")

# 일조 예측을 위한 분류기 (일조 유무)
sunshine_classifier = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)

# 분류 모델 학습 데이터 준비 - 날씨 피쳐만 사용
train_clf_data = train_updated[train_updated['일조(hr)'].notna()].copy()
X_clf, clf_features = solar_imputer._prepare_features(train_clf_data, '일조(hr)')
y_clf = (train_clf_data['일조(hr)'] > 0).astype(int)

print("일조 분류 모델 학습...")
sunshine_classifier.fit(X_clf.fillna(0), y_clf)

# 일조량 회귀 예측 (일조가 있는 경우의 양 예측)
sunshine_imputer = EnsembleImputer(random_state=SEED)
positive_sunshine_data = train_updated[train_updated['일조(hr)'] > 0].copy()
sunshine_imputer.fit(positive_sunshine_data, '일조(hr)')

# Test 데이터 예측
test_updated = test_fe.copy()

print("Test 일조량 예측 중...")

# 1단계: 일조 유무 분류
X_test_clf, _ = sunshine_imputer._prepare_features(test_updated, '일조(hr)')
sunshine_proba = sunshine_classifier.predict_proba(X_test_clf.fillna(0))[:, 1]

# 2단계: 일조량 회귀 예측
sunshine_pred = sunshine_imputer.predict(test_updated, '일조(hr)')

# 3단계: 분류와 회귀 결합 (확률 임계값: 0.3)
final_sunshine_pred = np.where(sunshine_proba > 0.3, sunshine_pred, 0)
test_updated['일조(hr)'] = final_sunshine_pred

positive_count = (final_sunshine_pred > 0).sum()
print(f"Test 일조 예측 완료: 양수 개수 {positive_count}/{len(final_sunshine_pred)} ({positive_count/len(final_sunshine_pred)*100:.1f}%)")

# ========================== 7. Test 데이터 일사 예측 ==========================
print(f"\n[5] Test 데이터 일사 앙상블 예측")

# 일사량 예측 모델 (전체 train 데이터 사용)
solar_test_imputer = EnsembleImputer(random_state=SEED)
solar_test_imputer.fit(train_updated, '일사(MJ/m2)')

print("Test 일사량 예측 중...")
test_solar_pred = solar_test_imputer.predict(test_updated, '일사(MJ/m2)')

# 일조가 0인 경우 일사도 0으로 설정
test_solar_pred = np.where(test_updated['일조(hr)'] > 0, test_solar_pred, 0)
test_updated['일사(MJ/m2)'] = test_solar_pred

positive_solar_count = (test_solar_pred > 0).sum()
print(f"Test 일사 예측 완료: 양수 개수 {positive_solar_count}/{len(test_solar_pred)} ({positive_solar_count/len(test_solar_pred)*100:.1f}%)")

# ========================== 8. 최종 데이터 정리 및 저장 ==========================
print(f"\n[6] 최종 데이터 저장")

# 임시 특성 제거
columns_to_remove = ['datetime_str', 'year', 'month', 'day', 'hour', 'season',
                     'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                     'date_ordinal', 'heat_index', 'wind_chill', 'discomfort_index',
                     'temp_humidity', 'temp_wind', 'rain_wind', 'is_rainy', 'is_windy',
                     'is_hot', 'is_cold', 'is_humid']

# 존재하는 컬럼만 제거
train_final = train_updated.drop([col for col in columns_to_remove if col in train_updated.columns], axis=1)
test_final = test_updated.drop([col for col in columns_to_remove if col in test_updated.columns], axis=1)

# 건물유형 인코딩 컬럼 제거 (있다면)
if '건물유형_encoded' in train_final.columns:
    train_final = train_final.drop(['건물유형_encoded'], axis=1)
if '건물유형_encoded' in test_final.columns:
    test_final = test_final.drop(['건물유형_encoded'], axis=1)

# CSV 파일로 저장
train_save_path = os.path.join(trainer, 'train_preprocessed_ensemble.csv')
test_save_path = os.path.join(trainer, 'test_preprocessed_ensemble.csv')

train_final.to_csv(train_save_path)
test_final.to_csv(test_save_path)

print(f"Train 데이터 저장: {train_save_path}")
print(f"Test 데이터 저장: {test_save_path}")

# ========================== 9. 최종 결과 요약 ==========================
print(f"\n[7] 앙상블 보간 결과 요약")
print(f"=" * 60)
print(f"Train 데이터:")
print(f"  - Shape: {train_final.shape}")
print(f"  - 일조(hr) 결측치: {train_final['일조(hr)'].isna().sum()}")
print(f"  - 일사(MJ/m2) 결측치: {train_final['일사(MJ/m2)'].isna().sum()}")
print(f"  - 일사량 > 0인 데이터: {(train_final['일사(MJ/m2)'] > 0).sum()}")

print(f"\nTest 데이터:")
print(f"  - Shape: {test_final.shape}")
print(f"  - 일조(hr) > 0인 데이터: {(test_final['일조(hr)'] > 0).sum()}")
print(f"  - 일사(MJ/m2) > 0인 데이터: {(test_final['일사(MJ/m2)'] > 0).sum()}")

print(f"\n문제 건물들의 앙상블 일사량 보간 완료:")
for building_num in problem_buildings:
    building_data = train_final[train_final['건물번호'] == building_num]
    positive_solar = (building_data['일사(MJ/m2)'] > 0).sum()
    max_solar = building_data['일사(MJ/m2)'].max()
    mean_solar = building_data[building_data['일사(MJ/m2)'] > 0]['일사(MJ/m2)'].mean()
    print(f"  - 건물 {building_num}: {positive_solar}/{len(building_data)} 개, 최대 {max_solar:.2f}, 평균 {mean_solar:.2f}")

# 예측 품질 통계
print(f"\n예측 품질 통계:")
train_sunshine_stats = train_final[train_final['일조(hr)'] > 0]['일조(hr)'].describe()
test_sunshine_stats = test_final[test_final['일조(hr)'] > 0]['일조(hr)'].describe()

print(f"일조량 분포:")
print(f"  Train: 평균 {train_sunshine_stats['mean']:.2f}, 최대 {train_sunshine_stats['max']:.2f}")
print(f"  Test:  평균 {test_sunshine_stats['mean']:.2f}, 최대 {test_sunshine_stats['max']:.2f}")

train_solar_stats = train_final[train_final['일사(MJ/m2)'] > 0]['일사(MJ/m2)'].describe()
test_solar_stats = test_final[test_final['일사(MJ/m2)'] > 0]['일사(MJ/m2)'].describe()

print(f"일사량 분포:")
print(f"  Train: 평균 {train_solar_stats['mean']:.2f}, 최대 {train_solar_stats['max']:.2f}")
print(f"  Test:  평균 {test_solar_stats['mean']:.2f}, 최대 {test_solar_stats['max']:.2f}")

# ========================== 10. 모델 성능 및 보간 품질 검증 ==========================
print(f"\n[8] 보간 품질 검증")

# 정상 건물에서 일부 데이터를 마스킹하여 보간 성능 테스트
def validate_imputation_quality():
    print("보간 품질 검증 중...")
    
    # 정상 건물 중 하나 선택 (건물 1)
    test_building = 1
    building_data = train_final[train_final['건물번호'] == test_building].copy()
    
    # 일사량이 있는 데이터 중 20% 마스킹
    positive_mask = building_data['일사(MJ/m2)'] > 0
    positive_indices = building_data[positive_mask].index
    
    if len(positive_indices) > 20:
        # 20% 랜덤 선택
        np.random.seed(SEED)
        masked_indices = np.random.choice(positive_indices, 
                                        size=int(len(positive_indices) * 0.2), 
                                        replace=False)
        
        # 원본 값 저장
        original_values = building_data.loc[masked_indices, '일사(MJ/m2)'].copy()
        
        # 마스킹
        building_data_masked = building_data.copy()
        building_data_masked.loc[masked_indices, '일사(MJ/m2)'] = 0
        
        # 임시 특성 추가
        building_data_masked = create_time_features(building_data_masked)
        building_data_masked = create_weather_features(building_data_masked)
        
        # 보간 실행
        temp_imputer = EnsembleImputer(random_state=SEED)
        other_data = train_updated[train_updated['건물번호'] != test_building].copy()
        other_data = create_time_features(other_data)
        other_data = create_weather_features(other_data)
        
        temp_imputer.fit(other_data, '일사(MJ/m2)', sample_ratio=0.1)  # 빠른 학습
        
        predicted_values = temp_imputer.predict(
            building_data_masked.loc[masked_indices], '일사(MJ/m2)'
        )
        
        # 성능 평가
        mae = mean_absolute_error(original_values, predicted_values)
        mse = mean_squared_error(original_values, predicted_values)
        rmse = np.sqrt(mse)
        
        if len(original_values) > 1:
            r2 = r2_score(original_values, predicted_values)
            print(f"    보간 품질 (건물 {test_building}):")
            print(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            print(f"      원본 평균: {original_values.mean():.2f}, 예측 평균: {predicted_values.mean():.2f}")
        else:
            print(f"    보간 품질 (건물 {test_building}):")
            print(f"      MAE: {mae:.4f}, RMSE: {rmse:.4f}")

try:
    validate_imputation_quality()
except Exception as e:
    print(f"    보간 품질 검증 실패: {str(e)}")

# ========================== 11. 추가 데이터 검증 및 후처리 ==========================
print(f"\n[9] 데이터 일관성 검증 및 후처리")

# 1. 일조-일사 관계 일관성 검증
def check_sunshine_solar_consistency(df, data_name):
    print(f"  {data_name} 일조-일사 통계:")
    
    # 일조=0인데 일사>0인 경우 (정상적일 수 있음)
    case_1 = df[(df['일조(hr)'] == 0) & (df['일사(MJ/m2)'] > 0)]
    print(f"    일조=0, 일사>0: {len(case_1)}개 (구름 산란광)")
    
    # 일조>0인데 일사=0인 경우
    case_2 = df[(df['일조(hr)'] > 0) & (df['일사(MJ/m2)'] == 0)]
    print(f"    일조>0, 일사=0: {len(case_2)}개")
    
    # 양수 데이터 통계
    positive_sunshine = (df['일조(hr)'] > 0).sum()
    positive_solar = (df['일사(MJ/m2)'] > 0).sum()
    print(f"    양수 일조: {positive_sunshine}개, 양수 일사: {positive_solar}개")
    
    # 비정상적으로 높은 비율 (일사/일조 > 5)
    positive_both = df[(df['일조(hr)'] > 0) & (df['일사(MJ/m2)'] > 0)]
    if len(positive_both) > 0:
        ratios = positive_both['일사(MJ/m2)'] / positive_both['일조(hr)']
        high_ratio = ratios > 5
        print(f"    고비율 (>5): {high_ratio.sum()}개")
        
        if high_ratio.sum() > 0:
            print(f"      최대 비율: {ratios.max():.2f}")

check_sunshine_solar_consistency(train_final, "Train")
check_sunshine_solar_consistency(test_final, "Test")

# 2. 일출/일몰 시간 기반 일조/일사 클리핑
print("\n  일출/일몰 시간 기반 클리핑:")

import math
from datetime import datetime, date

def calculate_sunrise_sunset(day_of_year, latitude=37.5665):  # 서울 기준
    """일출/일몰 시간 계산 (한국 서울 기준)"""
    # 태양의 적위각 계산
    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    
    # 시간각 계산
    hour_angle = math.degrees(math.acos(-math.tan(math.radians(latitude)) * math.tan(math.radians(declination))))
    
    # 일출/일몰 시간 (시간 단위)
    sunrise = 12 - hour_angle / 15
    sunset = 12 + hour_angle / 15
    
    return max(0, sunrise), min(24, sunset)

def apply_daylight_clipping(df, data_name):
    """일출/일몰 시간 외에는 일조/일사를 0으로 클리핑"""
    df = df.copy()
    
    # 날짜에서 day of year 계산
    df['temp_date'] = pd.to_datetime(df['일시'].str[:8], format='%Y%m%d')
    df['day_of_year'] = df['temp_date'].dt.dayofyear
    df['temp_hour'] = df['일시'].str.split(' ').str[1].fillna('0').astype(int)
    
    clipped_sunshine = 0
    clipped_solar = 0
    
    for idx, row in df.iterrows():
        sunrise, sunset = calculate_sunrise_sunset(row['day_of_year'])
        
        # 일출 전이나 일몰 후 시간 확인 (여유시간 30분 추가)
        if row['temp_hour'] < (sunrise - 0.5) or row['temp_hour'] > (sunset + 0.5):
            if df.at[idx, '일조(hr)'] > 0:
                df.at[idx, '일조(hr)'] = 0
                clipped_sunshine += 1
            if df.at[idx, '일사(MJ/m2)'] > 0:
                df.at[idx, '일사(MJ/m2)'] = 0
                clipped_solar += 1
    
    # 임시 컬럼 제거
    df = df.drop(['temp_date', 'day_of_year', 'temp_hour'], axis=1)
    
    print(f"    {data_name}: 일조 {clipped_sunshine}개, 일사 {clipped_solar}개 클리핑")
    return df

train_final = apply_daylight_clipping(train_final, "Train")
test_final = apply_daylight_clipping(test_final, "Test")

# 3. 극값 처리 (99.9% 분위수 기준 클리핑)
for col in ['일조(hr)', '일사(MJ/m2)']:
    if col in train_final.columns:
        q999 = train_final[col].quantile(0.999)
        extreme_count = (train_final[col] > q999).sum()
        if extreme_count > 0:
            print(f"    {col} 극값 처리: {extreme_count}개 → {q999:.2f}로 클리핑")
            train_final.loc[train_final[col] > q999, col] = q999
    
    if col in test_final.columns:
        q999 = test_final[col].quantile(0.999)
        extreme_count = (test_final[col] > q999).sum()
        if extreme_count > 0:
            print(f"    Test {col} 극값 처리: {extreme_count}개 → {q999:.2f}로 클리핑")
            test_final.loc[test_final[col] > q999, col] = q999

# ========================== 12. 최종 데이터 저장 ==========================
print(f"\n[10] 최종 데이터 저장")

# CSV 파일로 저장 (2개 파일만)
train_save_path = os.path.join(trainer, 'train.csv')
test_save_path = os.path.join(trainer, 'test.csv')

# 최종 데이터 저장
train_final.to_csv(train_save_path, index=False)
test_final.to_csv(test_save_path, index=False)

print(f"Train 데이터 저장: {train_save_path}")
print(f"Test 데이터 저장: {test_save_path}")

print(f"\n{'='*60}")
print(f"🎉 앙상블 전처리 파이프라인 완료!")
print(f"{'='*60}")
print(f"📊 최종 결과:")
print(f"   • Train: {train_final.shape[0]:,}행 × {train_final.shape[1]}열")
print(f"   • Test:  {test_final.shape[0]:,}행 × {test_final.shape[1]}열")
print(f"   • 사용 피쳐: 날씨 관련 피쳐만 사용 (기온, 강수량, 풍속, 습도, 시간)")
print(f"   • 일출/일몰 기반 물리적 일관성 보장")
print(f"   • 앙상블 모델: KNN + LightGBM + XGBoost")
print(f"{'='*60}")

# 최종 통계 출력
print(f"\n=== 최종 처리 결과 ===")
print(f"Train 데이터:")
print(f"  - 총 행수: {train_final.shape[0]:,}")
print(f"  - 일조(hr) > 0: {(train_final['일조(hr)'] > 0).sum():,}개 ({(train_final['일조(hr)'] > 0).sum()/len(train_final)*100:.1f}%)")
print(f"  - 일사(MJ/m2) > 0: {(train_final['일사(MJ/m2)'] > 0).sum():,}개 ({(train_final['일사(MJ/m2)'] > 0).sum()/len(train_final)*100:.1f}%)")

print(f"\nTest 데이터:")
print(f"  - 총 행수: {test_final.shape[0]:,}")
print(f"  - 일조(hr) > 0: {(test_final['일조(hr)'] > 0).sum():,}개 ({(test_final['일조(hr)'] > 0).sum()/len(test_final)*100:.1f}%)")
print(f"  - 일사(MJ/m2) > 0: {(test_final['일사(MJ/m2)'] > 0).sum():,}개 ({(test_final['일사(MJ/m2)'] > 0).sum()/len(test_final)*100:.1f}%)")

print(f"\n문제 건물별 보간 결과:")
for building_num in problem_buildings:
    building_data = train_final[train_final['건물번호'] == building_num]
    positive_solar = (building_data['일사(MJ/m2)'] > 0).sum()
    max_solar = building_data['일사(MJ/m2)'].max()
    mean_positive_solar = building_data[building_data['일사(MJ/m2)'] > 0]['일사(MJ/m2)'].mean() if positive_solar > 0 else 0
    print(f"  건물 {building_num}: 양수 일사량 {positive_solar}/{len(building_data)}개, 최대 {max_solar:.2f}, 평균 {mean_positive_solar:.2f}")

# 메모리 정리
import gc
gc.collect()

print("\n✅ 전체 전처리 과정이 성공적으로 완료되었습니다!")
print("   저장된 파일:")
print(f"   - {train_save_path}")
print(f"   - {test_save_path}")
print("   사용된 피쳐: 날씨 관련 피쳐만 (기온, 강수량, 풍속, 습도, 시간 정보)")