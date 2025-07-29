import os
import numpy as np
import pandas as pd
import random
import warnings
import json
from datetime import datetime
import shutil

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# Seed 관리
seed_file = "./Energy/seed_count/seed_state.json"
os.makedirs(os.path.dirname(seed_file), exist_ok=True)
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

# SMAPE 계산
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))

# 전처리 함수
def preprocess_all(train, test, building):
    building.replace('-', np.nan, inplace=True)
    for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
        building[col] = pd.to_numeric(building[col], errors='coerce')
    building.fillna(0, inplace=True)
    building['면적당_냉방비율'] = building['냉방면적(m2)'] / building['연면적(m2)']
    building['태양광_설비_유무'] = (building['태양광용량(kW)'] > 0).astype(int)
    building['ESS_설비_유무'] = (building['ESS저장용량(kWh)'] > 0).astype(int)
    building['PCS_설비_유무'] = (building['PCS용량(kW)'] > 0).astype(int)
    building['설비_총용량'] = building['태양광용량(kW)'] + building['ESS저장용량(kWh)'] + building['PCS용량(kW)']
    building = pd.get_dummies(building, columns=['건물유형'])
    train = pd.merge(train, building, on='건물번호', how='left')
    test = pd.merge(test, building, on='건물번호', how='left')

    for df in [train, test]:
        df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
        df['월'] = df['일시'].dt.month
        df['일'] = df['일시'].dt.day
        df['시간'] = df['일시'].dt.hour
        df['요일'] = df['일시'].dt.weekday
        df['주말여부'] = (df['요일'] >= 5).astype(int)
        df['기온×습도'] = df['기온(°C)'] * df['습도(%)']
        df['풍속×기온'] = df['풍속(m/s)'] * df['기온(°C)']
        df['강수여부'] = (df['강수량(mm)'] > 0).astype(int)

    predictors = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
    for target in ['일조(hr)', '일사(MJ/m2)']:
        df_temp = train[predictors + [target]].dropna()
        model = LinearRegression().fit(df_temp[predictors], df_temp[target])
        test[target] = model.predict(test[predictors])

    test['일조×일사'] = test['일조(hr)'] * test['일사(MJ/m2)']
    test['풍속×일사'] = test['풍속(m/s)'] * test['일사(MJ/m2)']
    test['연면적당_전력소비량'] = np.nan
    test['냉방면적당_전력소비량'] = np.nan

    for col in [c for c in train.columns if c.startswith('건물유형_')]:
        type_mean = train[train[col] == 1]['전력소비량(kWh)'].mean()
        test[f'{col}_평균전력'] = type_mean if col in test.columns else 0
        train[f'{col}_평균전력'] = type_mean

    return train, test

# 데이터 로딩
train_all = pd.read_csv("./Energy/train.csv")
test_all = pd.read_csv("./Energy/test.csv")
building_info = pd.read_csv("./Energy/building_info.csv")
submission_all = pd.read_csv("./Energy/sample_submission.csv")
submission_all['answer'] = 0

# 상위 100개 건물만 처리
top_buildings = train_all['건물번호'].value_counts().index[:100]
all_oof, all_y = [], []

# 앙상블 예측
for i, b_no in enumerate(top_buildings):
    print(f"\n🏢 {i+1}/100 - 건물번호: {b_no}")
    train = train_all[train_all['건물번호'] == b_no].copy()
    test = test_all[test_all['건물번호'] == b_no].copy()
    building = building_info[building_info['건물번호'] == b_no].copy()
    if train.empty or test.empty:
        print("⚠️ 데이터 없음, 스킵")
        continue

    train, test = preprocess_all(train, test, building)

    remove_features = ['PCS_설비_유무', '습도(%)', '풍속(m/s)', '설비_총용량',
                       '일사(MJ/m2)', '풍속×일사', '건물유형_건물기타', '요일', '일']
    target_col = '전력소비량(kWh)'
    drop_cols = ['num_date_time', '일시', target_col]
    X = train.drop(columns=drop_cols + remove_features, errors='ignore')
    y = train[target_col]
    X_test = test[X.columns]

    x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=SEED)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(X_test)

    xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=7,
                             random_state=SEED, early_stopping_rounds=50, objective='reg:squarederror')
    lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=7,
                              random_state=SEED, objective='mae', verbose=-1, early_stopping_rounds=50)
    cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=7,
                                  random_seed=SEED, verbose=0, loss_function='MAE', early_stopping_rounds=50)

    xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=0)
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)])
    cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val))

    oof_val_lvl1 = np.vstack([
        xgb_model.predict(x_val_scaled),
        lgb_model.predict(x_val_scaled),
        cat_model.predict(x_val_scaled)
    ]).T
    oof_test_lvl1 = np.vstack([
        xgb_model.predict(x_test_scaled),
        lgb_model.predict(x_test_scaled),
        cat_model.predict(x_test_scaled)
    ]).T

    final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
    final_model.fit(oof_val_lvl1, y_val)

    val_final = final_model.predict(oof_val_lvl1)
    test_final = final_model.predict(oof_test_lvl1)

    all_oof.extend(val_final)
    all_y.extend(y_val)

    sub_idx = submission_all['num_date_time'].isin(test['num_date_time'])
    submission_all.loc[sub_idx, 'answer'] = test_final

# 평가 및 저장
all_oof = np.array(all_oof)
all_y = np.array(all_y)
SMAPE = smape(all_y, all_oof)
os.makedirs('./Energy/submission', exist_ok=True)
DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_all.to_csv(f'./Energy/submission/Energy_{SEED}_{DATE}.csv', index=False)
print(f"\n<{SEED} 회차> 저장 완료")
print(f"✅ 최종 SMAPE 점수 : {SMAPE:.6f}")

if SMAPE > 10:
    shutil.rmtree("./Energy/submission")
    print(f"🚫 Score {SMAPE:.5f} > 기준 10 → 전체 디렉토리 삭제 완료")
else:
    print(f"🎉 Score {SMAPE:.5f} < 기준 10 → 디렉토리 유지")

with open("./Energy/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: Energy_{SEED}_{DATE}.csv\n")
    f.write(f"최종 SMAPE 점수 : {SMAPE:.6f}\n")
    f.write("="*40 + "\n")
