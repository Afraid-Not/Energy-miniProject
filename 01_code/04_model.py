import os
import numpy as np
import pandas as pd
import random
import warnings
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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

random.seed(SEED)
np.random.seed(SEED)

# ------------------------
# 공통 함수
# ------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

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

# ------------------------
# 실행
# ------------------------
train_all = pd.read_csv("./Energy/train.csv")
test_all = pd.read_csv("./Energy/test.csv")
building_info = pd.read_csv("./Energy/building_info.csv")
submission_all = pd.read_csv("./Energy/sample_submission.csv")
submission_all['answer'] = 0

top_buildings = train_all['건물번호'].value_counts().index[:100]

all_oof = []
all_y = []
for i, b_no in enumerate(top_buildings):
    print(f"\n🏢 {i+1}/100 - 건물번호: {b_no}")
    train = train_all[train_all['건물번호'] == b_no].copy()
    test = test_all[test_all['건물번호'] == b_no].copy()
    building = building_info[building_info['건물번호'] == b_no].copy()
    if train.empty or test.empty:
        print("⚠️ 데이터 없음, 스킵")
        continue

    train, test = preprocess_all(train, test, building)

    remove_features = [
        'PCS_설비_유무', '습도(%)', '풍속(m/s)', '설비_총용량',
        '일사(MJ/m2)', '풍속×일사', '건물유형_건물기타', '요일', '일'
    ]
    target_col = '전력소비량(kWh)'
    drop_cols = ['num_date_time', '일시', target_col]
    X = train.drop(columns=drop_cols + remove_features, errors='ignore')
    y = train[target_col]
    X_test = test[X.columns]

    # 하드코딩된 파라미터
    params_lgb = {'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 32, 'subsample': 0.8}
    params_xgb = {'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8}
    params_cat = {'learning_rate': 0.05, 'depth': 7}

    oof_lgbm = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    pred_lgbm = np.zeros(len(X_test))
    pred_xgb = np.zeros(len(X_test))
    pred_cat = np.zeros(len(X_test))

    gkf = GroupKFold(n_splits=5)
    groups = X['시간']

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        lgbm = LGBMRegressor(**params_lgb, n_estimators=1000, random_state=42, verbosity=-1)
        xgb = XGBRegressor(**params_xgb, n_estimators=1000, random_state=42, verbosity=0)
        cat = CatBoostRegressor(**params_cat, n_estimators=1000, random_state=42, verbose=0)

        lgbm.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        cat.fit(X_train, y_train)

        oof_lgbm[val_idx] = lgbm.predict(X_val)
        oof_xgb[val_idx] = xgb.predict(X_val)
        oof_cat[val_idx] = cat.predict(X_val)

        pred_lgbm += lgbm.predict(X_test) / 5
        pred_xgb += xgb.predict(X_test) / 5
        pred_cat += cat.predict(X_test) / 5

    final_pred = 0.4 * pred_lgbm + 0.3 * pred_xgb + 0.3 * pred_cat
    final_pred = np.where(final_pred < 0, 0, final_pred)
    final_oof = 0.4 * oof_lgbm + 0.3 * oof_xgb + 0.3 * oof_cat
    print(f"📊 SMAPE: {smape(y, final_oof):.4f}")
    all_oof.extend(final_oof)
    all_y.extend(y)

    sub_idx = submission_all['num_date_time'].isin(test['num_date_time'])
    submission_all.loc[sub_idx, 'answer'] = final_pred

all_oof = np.array(all_oof)
all_y = np.array(all_y)
SMAPE = smape(all_y, all_oof)

# 저장
from datetime import datetime
os.makedirs('./Energy/submission', exist_ok=True)
DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
submission_all.to_csv(f'./Energy/submission/Energy_{SEED}_{DATE}.csv', index=False)
print(f"\n<{SEED} 회차> ")
print("✅ 저장 완료: ensemble_433_per_buildingNo_manual.csv")
print(f"최종 SMAPE 점수 : {SMAPE:.6f}")

# # 삭제 여부
import shutil
SCORE_THRESHOLD = 10
if SMAPE > SCORE_THRESHOLD:
    shutil.rmtree("./Energy/submission")
    print(f"🚫 Score {SMAPE:.5f} > 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
else:
    print(f"🎉 Score {SMAPE:.5f} < 기준 {SCORE_THRESHOLD} → 디렉토리 유지")
    
with open("./Energy/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write("✅ 저장 완료: ensemble_433_per_buildingNo_manual.csv\n")
    f.write(f"최종 SMAPE 점수 : {SMAPE:.6f}\n")
    f.write("="*40 + "\n")