import pandas as pd
import numpy as np
import optuna
import os
import json
import datetime
import shutil
import random
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
# seed 상태 저장용 파일
seed_file = "./Energy/seed_count/seed_state.json"

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

def smape(y_true, y_pred):
    return 100 * (2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)).mean()

random.seed(SEED)
np.random.seed(SEED)

# Load data
train = pd.read_csv("./Energy/train.csv")
test = pd.read_csv("./Energy/test.csv")
building = pd.read_csv("./Energy/building_info.csv")
submission = pd.read_csv("./Energy/sample_submission.csv")

# Clean building info
building_cleaned = building.copy()
numeric_cols = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for col in numeric_cols:
    building_cleaned[col] = pd.to_numeric(building_cleaned[col].replace('-', 0), errors='coerce').fillna(0)

# Merge
train = pd.merge(train, building_cleaned, on='건물번호', how='left')
test = pd.merge(test, building_cleaned, on='건물번호', how='left')

# Time features
for df in [train, test]:
    df['일시'] = pd.to_datetime(df['일시'], format="%Y%m%d %H")
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['시간'] = df['일시'].dt.hour
    df['요일'] = df['일시'].dt.weekday
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)

# Encoding
le = LabelEncoder()
train['건물유형_encoded'] = le.fit_transform(train['건물유형'])
test['건물유형_encoded'] = le.transform(test['건물유형'])

# Solar features
train['예상_태양광발전량'] = train['일사(MJ/m2)'] * train['태양광용량(kW)']
train['충전가능량(kWh)'] = train[['예상_태양광발전량', 'ESS저장용량(kWh)']].min(axis=1)
test['일사(MJ/m2)'] = test.get('일사(MJ/m2)', 0)
test['일조(hr)'] = test.get('일조(hr)', 0)
test['예상_태양광발전량'] = test['일사(MJ/m2)'] * test['태양광용량(kW)']
test['충전가능량(kWh)'] = test[['예상_태양광발전량', 'ESS저장용량(kWh)']].min(axis=1)

# 파생 변수
train['냉방면적비율'] = train['냉방면적(m2)'] / train['연면적(m2)']
test['냉방면적비율'] = test['냉방면적(m2)'] / test['연면적(m2)']
building_avg = train.groupby('건물유형_encoded')['전력소비량(kWh)'].mean().rename("건물유형별_평균소비량")
train = train.merge(building_avg, on='건물유형_encoded', how='left')
# test = test.merge(building_avg, on='건물유형_encoded', how='left')

# Features
feature_cols = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
    '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '예상_태양광발전량', '충전가능량(kWh)', '월', '일', '시간', '요일', '주말여부',
    '건물유형_encoded', '냉방면적비율'
]
target_col = '전력소비량(kWh)'
X = train[feature_cols]
y = train[target_col]
X_test = test[feature_cols]

# ===== XGBoost 튜닝 =====
def objective_xgb(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": SEED,
        "n_jobs": 1,
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    smape_list = []
    best_iterations = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBRegressor(**params, n_estimators=400 , early_stopping_rounds=20,)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        pred = model.predict(X_val)
        smape_val = smape(y_val, pred)
        smape_list.append(smape_val)
        best_iterations.append(model.best_iteration)

    trial.set_user_attr("best_iter", int(np.mean(best_iterations)))
    return np.mean(smape_list)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)

# === best_iteration 적용 ===
best_iter_xgb = int(study_xgb.best_trial.user_attrs["best_iter"])
xgb_model = XGBRegressor(**study_xgb.best_params, n_estimators=best_iter_xgb)
xgb_model.fit(X, y)
xgb_val_pred = xgb_model.predict(X)
xgb_test_pred = xgb_model.predict(X_test)

# ===== LGBM 튜닝 =====
def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": SEED,
        "n_jobs": 1,
        
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    smape_list = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMRegressor(**params, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred = model.predict(X_val)
        smape_val = smape(y_val, pred)
        smape_list.append(smape_val)
        
    return np.mean(smape_list)

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=30)
lgb_model = LGBMRegressor(**study_lgb.best_params)
lgb_model.fit(X, y)
lgb_val_pred = lgb_model.predict(X)
lgb_test_pred = lgb_model.predict(X_test)

# ===== 앙상블 가중치 튜닝 =====
alpha = 0.5  # 가중치 0.5 : 0.5

# 최종 예측
final_pred = 0.5 * xgb_test_pred + 0.5 * lgb_test_pred
submission['answer'] = final_pred

# SMAPE 계산
val_blended = 0.5 * xgb_val_pred + 0.5 * lgb_val_pred
val_smape = smape(y, val_blended)
val_smape_rounded = round(val_smape, 2)

# === 날짜 생성 ===
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# === 저장 경로 및 이름 생성 ===
os.makedirs("./Energy/submission", exist_ok=True)
save_path = f"./Energy/submission/energy_{SEED}_{val_smape_rounded}_{now}.csv"
submission.to_csv(save_path, index=False)
print(f"[✔] Saved: {save_path}")


print(f"{SEED}회차")
print(f"[XGB best SMAPE]: {study_xgb.best_value:.4f}")
print(f"[LGB best SMAPE]: {study_lgb.best_value:.4f}")
print(f"[Best ensemble alpha]: {alpha:.1f}")
print(f"[Ensemble SMAPE]: {val_smape:.4f}")

# 🎯 기준 점수 설정
SCORE_THRESHOLD = 7.48062  # 원하는 기준값으로 설정

# 점수가 기준보다 높으면 전체 디렉토리 삭제
if val_smape > SCORE_THRESHOLD:
    shutil.rmtree("./Energy/submission")
    print(f"🚫 Score {val_smape:.5f} > 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
else:
    print(f"🎉 Score {val_smape:.5f} < 기준 {SCORE_THRESHOLD} → 디렉토리 유지")
    

# [✔] Saved: ./Energy/submission/energy_53_5.09_20250716_214417.csv
# 53회차
# [XGB best SMAPE]: 7.2012
# [LGB best SMAPE]: 12.0962
# [Best ensemble alpha]: 1.0
# [Ensemble SMAPE]: 5.0940
# 12.7966642893