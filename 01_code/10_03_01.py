print(f"#0. 시작")
import pandas as pd
import numpy as np
import time
import os
import json
import random
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

seed_file = "./Energy/05_submission/05_02.json"

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
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

data_path = './Energy/'
save_path = './Energy/10_00_submission/'
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')
submit = pd.read_csv(data_path + 'sample_submission.csv')

print(f"[1] 전처리")
for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_csv[col] = building_csv[col].replace('-', 0).astype(float)

building_csv.loc[building_csv['PCS용량(kW)'] == 0, '태양광용량(kW)'] = 0
    
train = pd.merge(train_csv, building_csv, on=['건물번호'], how='left')
test = pd.merge(test_csv, building_csv, on=['건물번호'], how='left')

def add_datetime_features(df):
    df['일시'] = pd.to_datetime(df['일시'])
    df['연'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['요일'] = df['일시'].dt.weekday
    df['hour'] = df['일시'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = df['요일'].isin([5, 6]).astype(int)
    df['is_working_hours'] = df['hour'].between(9, 18).astype(int)
    
    # 공휴일: 6월 6일, 8월 15일
    df['is_holiday'] = df['일시'].dt.strftime('%m-%d').isin(['06-06', '08-15']).astype(int)

    return df

def add_weather_features(df):
    # 체감온도(°C) = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
    T = df['기온(°C)']
    V = df['풍속(m/s)']
    df['feels_like'] = 13.12 + 0.6215*T - 11.37*V**0.16 + 0.3965*T*V**0.16

    # 불쾌지수 = 1.8*T - 0.55*(1-H)*1.8*T + 32
    H = df['습도(%)'] / 100
    df['discomfort_index'] = 1.8*T - 0.55*(1 - H)*1.8*T + 32

    # 냉방도일: 24도 이상일 때만
    df['cooling_degree_day'] = (T - 24).clip(lower=0)

    return df

def add_rainfall_features(df):
    df['강수량(mm)'] = df['강수량(mm)'].replace('NaN', 0).astype(float)
    df['rainfall_change_rate'] = df['강수량(mm)'].diff().fillna(0)
    df['is_rain_x_working_hours'] = ((df['강수량(mm)'] > 0) & (df['is_working_hours'] == 1)).astype(int)
    return df

def apply_all_feature_engineering(df):
    df = add_datetime_features(df)
    df = add_weather_features(df)
    df = add_rainfall_features(df)
    return df

train = apply_all_feature_engineering(train)
test = apply_all_feature_engineering(test)

# 낮시간 조건
def mark_solar_as_nan(df, building_ids, col='일사(MJ/m2)'):
    cond = (
        (df['건물번호'].isin(building_ids)) &
        (df['hour'].between(6, 21)) &
        (df[col] == 0)
    )
    df.loc[cond, col] = np.nan
    return df

empty_buildings = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
train = mark_solar_as_nan(train, empty_buildings)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_solar_model(train, seed=1):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import VotingRegressor

    # 예측 대상: 일사(MJ/m2)가 결측인 행
    target_col = '일사(MJ/m2)'
    feature_cols = [
        '기온(°C)', '강수량(mm)', '풍속(m/s)', 'sin_hour', 'cos_hour', 
        '일조(hr)', '태양광용량(kW)', 'ESS저장용량(kWh)', 
        'PCS용량(kW)', '전력소비량(kWh)', '건물유형'
    ]

    # 건물유형 인코딩
    le = LabelEncoder()
    train['건물유형'] = le.fit_transform(train['건물유형'])

    # 예측 대상(결측치가 있는 건물만 필터)
    target_idx = train[train[target_col].isna()].index
    building_types = train.loc[target_idx, '건물유형'].unique()

    # 결과 담기
    filled = []

    for btype in building_types:
        data_btype = train[train['건물유형'] == btype]

        train_data = data_btype[data_btype[target_col].notna()]
        test_data = data_btype[data_btype[target_col].isna()]

        X = train_data[feature_cols]
        y = train_data[target_col]
        X_test = test_data[feature_cols]

        xgb = XGBRegressor(random_state=seed, n_estimators=200, learning_rate=0.05, verbosity=0)
        lgb = LGBMRegressor(random_state=seed, n_estimators=200, learning_rate=0.05, verbosity=-1)
        model = VotingRegressor([('xgb', xgb), ('lgb', lgb)])

        x_temp, x_test_eval, y_temp, y_test_eval = train_test_split(X, y, test_size=0.1, random_state=seed)
        model.fit(x_temp, y_temp)

        preds = model.predict(x_test_eval)
        rmse = np.sqrt(mean_squared_error(y_test_eval, preds))
        print(f"🏠 건물유형 {btype} | RMSE = {rmse:.4f}")

        # 결측값 예측
        pred_missing = model.predict(X_test)
        test_data[target_col] = pred_missing
        filled.append(test_data)

    # 채워진 데이터로 train 갱신
    filled_df = pd.concat(filled)
    train.loc[filled_df.index, target_col] = filled_df[target_col]

    return train


def zero_night_solar(df):
    df.loc[~df['hour'].between(6, 21), '일사(MJ/m2)'] = 0
    return df

train = mark_solar_as_nan(train, empty_buildings)
train = train_solar_model(train, seed=SEED)
train = zero_night_solar(train)


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

def predict_sun_columns(train, test, target_col, seed=1):
    feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
                    'sin_hour', 'cos_hour', '건물유형', '연면적(m2)']

    # 범주형 인코딩
    train['건물유형'] = train['건물유형'].astype(str)
    test['건물유형'] = test['건물유형'].astype(str)

    all_types = pd.concat([train['건물유형'], test['건물유형']])
    le = LabelEncoder()
    le.fit(all_types)

    train['건물유형'] = le.transform(train['건물유형'])
    test['건물유형'] = le.transform(test['건물유형'])
    # 결과 저장
    pred_all = pd.Series(index=test.index, dtype=np.float32)

    building_ids = test['건물번호'].unique()
    for bno in building_ids:
        print(f"🌞 건물 {bno} - {target_col} 예측 중...")

        train_b = train[train['건물번호'] == bno]
        test_b = test[test['건물번호'] == bno]

        # 낮시간만 예측
        train_b = train_b[train_b['hour'].between(6, 21)]
        test_b = test_b[test_b['hour'].between(6, 21)]

        if train_b[target_col].isna().sum() > 0 or len(train_b) < 50:
            print(f"⚠️ 건물 {bno} - 학습 데이터 부족 또는 결측 있음, 스킵")
            continue

        x = train_b[feature_cols]
        y = train_b[target_col]
        x_pred = test_b[feature_cols]

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.1, random_state=seed
        )

        model = VotingRegressor([
            ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=seed)),
            ('lgb', LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=seed)),
            ('cat', CatBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=seed, verbose=0))
        ])
        model.fit(x_train, y_train)

        preds_val = model.predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        print(f"✅ 건물 {bno} | {target_col} RMSE: {rmse:.4f}")

        preds_test = model.predict(x_pred)
        pred_all.loc[test_b.index] = preds_test

    # 나머지 밤 시간은 0으로 설정
    pred_all.fillna(0, inplace=True)
    return pred_all


test['일조(hr)'] = predict_sun_columns(train, test, target_col='일조(hr)', seed=SEED)
test['일사(MJ/m2)'] = predict_sun_columns(train, test, target_col='일사(MJ/m2)', seed=SEED) 

def add_solar_features(df):
    df = df.copy()
    df['태양광사용량'] = df['일사(MJ/m2)'] * df['태양광용량(kW)']
    df['sun_x_temp'] = df['일조(hr)'] * df['기온(°C)']
    df['sun_x_cooling_area'] = df['일조(hr)'] * df['냉방면적(m2)']
    df['sun_x_working_hours'] = df['일조(hr)'] * df['is_working_hours']
    df['sun_x_solar_capacity'] = df['일조(hr)'] * df['태양광용량(kW)']

    df['solar_x_cooling_area'] = df['일사(MJ/m2)'] * df['냉방면적(m2)']
    df['solar_x_working_hours'] = df['일사(MJ/m2)'] * df['is_working_hours']
    df['solar_x_solar_capacity'] = df['일사(MJ/m2)'] * df['태양광용량(kW)']

    # 일자 단위 총 일조량
    df['date'] = df['일시'].dt.date
    df['daily_total_sunshine'] = df.groupby(['건물번호', 'date'])['일조(hr)'].transform('sum')

    # 시간 단위 rolling 평균 (3시간)
    df.sort_values(['건물번호', '일시'], inplace=True)
    df['rolling_hourly_sunshine'] = df.groupby('건물번호')['일조(hr)'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['rolling_hourly_solar_radiation'] = df.groupby('건물번호')['일사(MJ/m2)'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # 일사량 변화율
    df['solar_radiation_change'] = df.groupby('건물번호')['일사(MJ/m2)'].diff().fillna(0)

    return df

train = add_solar_features(train)
test = add_solar_features(test)

train.to_csv(save_path + f'{SEED}_new_train.csv')
test.to_csv(save_path + f'{SEED}_new_test.csv')

print(train.columns)
print(test.columns)



