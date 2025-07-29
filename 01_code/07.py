import os
import pandas as pd
import numpy as np
import random
import datetime, json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Seed 설정
seed_file = "./Energy/seed_count/gift_seed.json"
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
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

# 경로 설정 및 데이터 로딩
data_path = './Energy/'
buildinginfo = pd.read_csv(data_path + 'building_info.csv')
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
samplesub = pd.read_csv(data_path + 'sample_submission.csv')
print("[1] 데이터 로딩 및 초기 전처리 완료")

# 결측치 처리
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)


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
    for col in ['일조(hr)', '일사(MJ/m2)']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
    return df

def add_lag_features(df, lags=[1, 24, 168]):
    for lag in lags:
        df[f'lag_{lag}h'] = df.groupby('건물번호')['전력소비량(kWh)'].shift(lag)
    return df

def add_rolling_features(df, windows=[3, 24, 168]):
    for window in windows:
        df[f'rolling_mean_{window}h'] = df.groupby('건물번호')['전력소비량(kWh)'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}h'] = df.groupby('건물번호')['전력소비량(kWh)'].shift(1).rolling(window).std()
    return df

def add_lag_rolling_to_test(train, test):
    test_with_lag = []
    for bno in test['건물번호'].unique():
        train_b = train[train['건물번호'] == bno].copy()
        test_b = test[test['건물번호'] == bno].copy()

        # Get last 168 rows of train for this building
        train_tail = train_b.sort_values('일시').iloc[-168:].copy()
        test_combined = pd.concat([train_tail, test_b], axis=0)

        # Add lag/rolling features
        test_combined = add_lag_features(test_combined)
        test_combined = add_rolling_features(test_combined)

        # Keep only the test part
        test_b_processed = test_combined.loc[test_b.index]
        test_with_lag.append(test_b_processed)

    test = pd.concat(test_with_lag).sort_index()
    return test

# Apply lag/rolling to test using train tail


# 전처리
train = feature_engineering(train)
train = add_lag_features(train)
train = add_rolling_features(train)
train = train.merge(buildinginfo, on='건물번호', how='left')
test = feature_engineering(test)
# test = add_lag_features(test)
# test = add_rolling_features(test)
test = test.merge(buildinginfo, on='건물번호', how='left')
train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes
train = train.dropna(subset=['lag_1h', 'lag_24h', 'lag_168h'])
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Feature 목록
features = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    'is_working_hours', 'sin_hour', 'cos_hour', 'DI',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_mean_3h', 'rolling_mean_24h', 'rolling_mean_168h',
    'rolling_std_24h', 'rolling_std_168h'
]
# exit()
target = '전력소비량(kWh)'
print("[2] 전처리 완료")



# 모델링
final_preds = []
val_smapes = []
building_ids = train['건물번호'].unique()
print("[3] 건물별 학습 시작")

ts_split = TimeSeriesSplit(n_splits=5)

test = add_lag_rolling_to_test(train, test)
# print(train.columns)
# print(test.columns)
# exit()

# [2] 모델링
for bno in building_ids:
    print(f"    > 🏢 건물번호 {bno} 모델링 중...")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    x = train_b[features].reset_index(drop=True)
    y = np.log1p(train_b[target].reset_index(drop=True))
    x_test_final = test_b[features]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_test_scaled = scaler.transform(x_test_final)

    oof_train = []
    oof_target = []
    oof_test_all = []

    for fold, (tr_idx, val_idx) in enumerate(ts_split.split(x_scaled)):
        x_tr, x_val = x_scaled[tr_idx], x_scaled[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                             random_state=SEED, early_stopping_rounds=50, objective='reg:squarederror')
        xgb_model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

        lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                random_state=SEED, objective='mae', verbose=-1)
        lgb_model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

        cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                    random_seed=SEED, verbose=0, loss_function='MAE')
        cat_model.fit(x_tr, y_tr, eval_set=(x_val, y_val), early_stopping_rounds=50)
        
        pred_stack = np.vstack([
            xgb_model.predict(x_val),
            lgb_model.predict(x_val),
            cat_model.predict(x_val)
        ]).T
        oof_train.append(pred_stack)
        oof_target.append(y_val)

        test_stack = np.vstack([
            xgb_model.predict(x_test_scaled),
            lgb_model.predict(x_test_scaled),
            cat_model.predict(x_test_scaled)
        ]).T
        oof_test_all.append(test_stack)

    oof_train_all = np.vstack(oof_train)
    oof_target_all = np.hstack(oof_target)
    oof_test_mean = np.mean(np.stack(oof_test_all), axis=0)

    meta = RidgeCV()
    meta.fit(oof_train_all, oof_target_all)
    val_pred = meta.predict(oof_train_all)

    val_smape = np.mean(200 * np.abs(np.expm1(val_pred) - np.expm1(oof_target_all)) /
                        (np.abs(np.expm1(val_pred)) + np.abs(np.expm1(oof_target_all)) + 1e-6))
    val_smapes.append(val_smape)

    final_pred = np.expm1(meta.predict(oof_test_mean))
    final_preds.extend(final_pred)

print("[4] 건물별 학습 완료")
print("[5] 저장 시작")

samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
score_str = f"{avg_smape:.4f}".replace('.', '_')
filepath = './Energy/test_submission/'
os.makedirs(filepath, exist_ok=True)
filename = f"energy_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(filepath + filename, index=False)

print(f"[6] 저장 완료 ")
print(f"최종 SMAPE 점수 : {avg_smape:.6f}")

# 로그 저장
with open("./Energy/gift_submission/test_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: {filename}\n")
    f.write(f"최종 SMAPE 점수 : {avg_smape}\n")
    f.write("="*40 + "\n")
