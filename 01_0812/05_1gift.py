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

seed_file = "./Energy/01/(SEED_COUNT)Gift_model.json"

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
csv_path = './Energy/01/'
trainer = './Energy/01/(GIFT)trainer/'
save_path = './Energy/01/(GIFT)submissions/'
os.makedirs(trainer, exist_ok=True)

train_call = '06_train_42_optuna'
test_call = '06_test_42_optuna'

train = pd.read_csv(trainer + train_call)
test = pd.read_csv(trainer + test_call)
samplesub = pd.read_csv(data_path + 'sample_submission.csv')

print("[1] 데이터 로딩 및 초기 전처리 완료")
exclude_cols = ['건물번호', '일시', '전력소비량(kWh)', '건물유형', '날짜']
features = [col for col in train.columns if col not in exclude_cols]
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

filename = f"{SEED}_energy_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

print(f"[6] 📁 저장 완료 ")
print(f"✅ 최종 SMAPE 점수 : {score_str:.6f}")

with open("./Energy/01/(LOG)Gift_model.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"<05_1gift.py>\n")
    f.write(f"✅ 저장 완료: {filename}\n")
    f.write(f"최종 SMAPE 점수 : {avg_smape}\n")
    f.write("="*40 + "\n")