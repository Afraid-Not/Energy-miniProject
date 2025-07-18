import matplotlib.pyplot as plt
import pandas as pd
print(f"[01_preprocessing] 시작")
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
SEED = seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

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
insolation_features = ['건물번호', '기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)', '일조(hr)',
                       '시각', '요일', '주말여부', '월', '정오거리', '정오거리_INV', 'peak_time',
                       '태양광용량(kW)', '냉방면적(m2)']
insolation_features.extend([col for col in train_all.columns if '요일_' in col])
insolation_features.extend([col for col in train_all.columns if '건물유형_' in col])
insolation_features.extend([col for col in train_all.columns if '기온_변화량' in col or '기온_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '습도_변화량' in col or '습도_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '풍속_변화량' in col or '풍속_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '강수량_변화량' in col or '강수량_rolling' in col])
insolation_features.extend([col for col in train_all.columns if '일조_변화량' in col or '일조_rolling' in col])
insolation_features = list(set(insolation_features))

# '일사(MJ/m2)' 값이 0이 아닌 건물 데이터로 모델 학습
train_insolation_train = train_all[~train_all['건물번호'].isin(zero_bnos) & (train_all['일사(MJ/m2)'] > 0)].copy()
X_train_insolation = train_insolation_train[insolation_features]
y_train_insolation = train_insolation_train['일사(MJ/m2)']

# print(f"사용되는 Column : \n{insolation_features}")

# '일사(MJ/m2)' 값이 0인 건물 데이터 예측
train_insolation_test = train_all[train_all['건물번호'].isin(zero_bnos) & (train_all['일사(MJ/m2)'] == 0)].copy()
X_test_insolation = train_insolation_test[insolation_features]

# 모델 초기화 (앙상블)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=SEED, early_stopping_rounds=50)
lgbm_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=SEED, verbosity=-1)

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
# 앙상블 예측값 저장
ensemble_predictions = np.zeros(len(X_test_insolation))
mae_1 = 0

for fold, (train_index, val_index) in enumerate(kf.split(X_train_insolation, y_train_insolation)):
    print(f"    > Fold {fold+1} train_insolation_model 학습 시작...")
    X_train, y_train = X_train_insolation.iloc[train_index], y_train_insolation.iloc[train_index]
    X_val, y_val = X_train_insolation.iloc[val_index], y_train_insolation.iloc[val_index]

    # 모델 학습
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    lgbm_model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)])

    # 예측 및 앙상블
    xgb_pred = xgb_model.predict(X_test_insolation)
    lgbm_pred = lgbm_model.predict(X_test_insolation)
    ensemble_predictions += (xgb_pred + lgbm_pred) / 2 / kf.n_splits

    xgb_val = xgb_model.predict(X_val)
    lgbm_val = lgbm_model.predict(X_val)
    ensemble_val = (xgb_val + lgbm_val) / 2
    trn_mae = mean_absolute_error(y_val, ensemble_val)
    mae_1 += trn_mae / kf.n_splits  # <-- Corrected MAE calculation

print(f"Train 일사 예측 : {mae_1:.6f}")

# 예측값으로 원본 데이터 업데이트
train_all.loc[train_insolation_test.index, '일사(MJ/m2)'] = np.clip(ensemble_predictions, 0, None)

# 21시부터 05시까지의 일사량을 0으로 설정
train_all.loc[(train_all['건물번호'].isin(zero_bnos)) & ((train_all['시각'] >= 21) | (train_all['시각'] <= 5)), '일사(MJ/m2)'] = 0

train_all = add_insolation_rolling_features(train_all)
print(f"[3] train zero_bno 일사 예측 완료")


# Get feature importances from both models
xgb_importance = xgb_model.feature_importances_
lgbm_importance = lgbm_model.feature_importances_

# Combine feature importances by averaging
feature_importance = (xgb_importance + lgbm_importance) / 2

# Create a DataFrame for better visualization and sorting
feature_importance_df = pd.DataFrame({
    'feature': X_train_insolation.columns,
    'importance': feature_importance
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
# Print the feature importance
print("Feature Importance:")
print(feature_importance_df)

# Plot the top N features
n = 20 # Number of features to plot
plt.figure(figsize=(12, 10))
plt.barh(feature_importance_df['feature'].iloc[:n], feature_importance_df['importance'].iloc[:n])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 20 Feature Importance for Insolation Prediction (Ensemble)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Suggest which columns to remove based on low importance
low_importance_features = feature_importance_df[feature_importance_df['importance'] < 0.001]['feature']
print("\nSuggested columns to remove due to low importance:")
if not low_importance_features.empty:
    print(low_importance_features.tolist())
else:
    print("No features with very low importance were found based on the threshold.")
    
print(insolation_features)
    
    
    
