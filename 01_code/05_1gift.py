import os
import pandas as pd
import numpy as np
import random
import datetime, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
# import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

seed_file = "./Energy/seed_count/gift_seed.json"

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

# 경로 설정


data_path = './Energy/'

# 데이터 로드
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

# 전처리
train = feature_engineering(train)
test = feature_engineering(test)
train = train.merge(buildinginfo, on='건물번호', how='left')
test = test.merge(buildinginfo, on='건물번호', how='left')

train['건물유형'] = train['건물유형'].astype('category').cat.codes
test['건물유형'] = test['건물유형'].astype('category').cat.codes

features = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', 
    '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    'hour', 'dayofweek', 'month', 'day', 'is_weekend',
    'is_working_hours', 'sin_hour', 'cos_hour', 'DI'
]

target = '전력소비량(kWh)'
print("[2] 전처리 완료")

# 최종 예측 결과 저장용
final_preds = []
val_smapes = []

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()

print("[3] 건물별 학습 시작")


for bno in building_ids:
    print(f"    > 🏢 건물번호 {bno} 모델링 중...")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    x = train_b[features]
    y = np.log1p(train_b[target])
    x_test_final = test_b[features]

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=SEED)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_final_scaled = scaler.transform(x_test_final)

    # Base models
    xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                             random_state=SEED, early_stopping_rounds=50, objective='reg:squarederror')
    xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

    lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                              random_state=SEED, objective='mae', verbose=-1)
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                  random_seed=SEED, verbose=0, loss_function='MAE')
    cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

    # Level 1 predictions
    oof_train_lvl1 = np.vstack([
        xgb_model.predict(x_train_scaled),
        lgb_model.predict(x_train_scaled),
        cat_model.predict(x_train_scaled)
    ]).T
    oof_val_lvl1 = np.vstack([
        xgb_model.predict(x_val_scaled),
        lgb_model.predict(x_val_scaled),
        cat_model.predict(x_val_scaled)
    ]).T
    oof_test_lvl1 = np.vstack([
        xgb_model.predict(x_test_final_scaled),
        lgb_model.predict(x_test_final_scaled),
        cat_model.predict(x_test_final_scaled)
    ]).T

    # Level 2 Meta model
    meta_model = RidgeCV()
    meta_model.fit(oof_train_lvl1, y_train)
    val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
    test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

    # Level 3 Final model
    final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
    final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)

    val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
    val_smape = np.mean(200 * np.abs(np.expm1(val_final) - np.expm1(y_val)) /
                        (np.abs(np.expm1(val_final)) + np.abs(np.expm1(y_val)) + 1e-6))
    val_smapes.append(val_smape)

    pred = final_model.predict(test_pred_lvl2.reshape(-1, 1))
    final_preds.extend(pred)
    
print("[4] 건물별 학습 완료")
print("[5] 저장 시작")

# 결과 저장
samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
score_str = f"{avg_smape:.4f}".replace('.', '_')
filepath = './Energy/gift_submission/'

os.makedirs(filepath, exist_ok=True)

filename = f"energy_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(filepath + filename, index=False)

print(f"[6] 📁 저장 완료 ")
print(f"✅ 최종 SMAPE 점수 : {score_str:.6f}")

# import shutil
# SCORE_THRESHOLD = 10
# if avg_smape > SCORE_THRESHOLD:
#     shutil.rmtree("./Energy/gift_submission")
#     print(f"🚫 Score {avg_smape:.5f} > 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
# else:
#     print(f"🎉 Score {avg_smape:.5f} < 기준 {SCORE_THRESHOLD} → 디렉토리 유지")
    
with open("./Energy/gift_submission/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: {filename}\n")
    f.write(f"최종 SMAPE 점수 : {avg_smape}\n")
    f.write("="*40 + "\n")