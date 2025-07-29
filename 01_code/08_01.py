import numpy as np
import pandas as pd
import random
import json
import warnings
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore')

# --------------------------
# 고정 SEED 설정
# --------------------------
seed_file = "./Energy/seed_count/08_seed.json"
os.makedirs(os.path.dirname(seed_file), exist_ok=True)

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

# --------------------------
# 평가 함수
# --------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def get_models():
    return {
        'xgb': XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1),
        'lgb': LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=0),
        'cat': CatBoostRegressor(random_state=SEED, verbose=0)
    }

# --------------------------
# 데이터 로딩 및 전처리
# --------------------------

print("[1] 데이터 로딩 및 초기 전처리 완료")

data_path = './Energy/'
subm_path = './Energy/08_submission/'

train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)

# 필요없는 컬럼 제거
train = train.drop(['일조(hr)', '일사(MJ/m2)'], axis=1)

train['일시'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
test['일시'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')

for df in [train, test]:
    df['시간'] = df['일시'].dt.hour
    df['요일'] = df['일시'].dt.weekday
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['기온×습도'] = df['기온(°C)'] * df['습도(%)']
    df['풍속×기온'] = df['풍속(m/s)'] * df['기온(°C)']
    df['강수여부'] = (df['강수량(mm)'] > 0).astype(int)
    df['근무시간'] = df['시간'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['시간'] / 24)

# 요일 더미변수
for i in ['요일']:
    train = pd.concat([train, pd.get_dummies(train[i], prefix=i)], axis=1)
    test = pd.concat([test, pd.get_dummies(test[i], prefix=i)], axis=1)
    
print(train.columns)
print(test.columns)

# 불필요한 컬럼 제거
drop_cols = ['일시', '시간', '요일', ]
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)
print("[2] 전처리 완료")
# exit()
# --------------------------
# Train / Validation Split
# --------------------------
train['target'] = np.log1p(train['전력소비량(kWh)'])
X_all = train.drop(columns=['전력소비량(kWh)', 'target'])
y_all = train['target']

X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, shuffle=True)
val_df = X_val.copy()
val_df['target'] = y_val
val_df['pred'] = np.nan  # 예측값 저장용

# --------------------------
# Building별 학습 및 예측
# --------------------------
print("[3] Building별 학습 및 예측")

N_SPLITS = 5
pred_list = []
building_ids = sorted(train['건물번호'].unique())

for bno in building_ids:
    print(f"\n   > Building {bno} - Modeling 시작")

    # 개별 데이터 분리
    tr_b = X_tr[X_tr['건물번호'] == bno].copy()
    val_b = val_df[val_df['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(tr_b) < N_SPLITS:
        print(f"  ⚠️ Skipped due to insufficient samples: {len(tr_b)}")
        pred_list.append(np.zeros(len(test_b)))
        continue

    y_b = y_tr[tr_b.index]

    # 스케일링
    ss = StandardScaler()
    x = ss.fit_transform(tr_b)
    x_test_b = ss.transform(test_b)

    # KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = {name: np.zeros(len(x)) for name in get_models()}
    test_preds_all = {name: np.zeros(len(x_test_b)) for name in get_models()}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(x)):
        print(f"     > 🔁 Fold {fold+1}/{N_SPLITS}")
        x_trn, x_val = x[tr_idx], x[val_idx]
        y_trn, y_val = y_b.iloc[tr_idx], y_b.iloc[val_idx]

        models = get_models()
        for name, model in models.items():
            model.fit(x_trn, y_trn)
            oof_preds[name][val_idx] = model.predict(x_val)
            test_preds_all[name] += model.predict(x_test_b) / N_SPLITS

    # Meta 모델 학습
    meta_X = pd.DataFrame(oof_preds)
    meta_y = y_b.reset_index(drop=True)
    meta_model = RidgeCV()
    meta_model.fit(meta_X, meta_y)

    # Test 예측
    meta_test_X = pd.DataFrame(test_preds_all)
    final_test_pred = np.expm1(meta_model.predict(meta_test_X))
    pred_list.append(final_test_pred)

    # Hold-out 예측 및 평가
    if len(val_b) > 0:
        val_scaled = ss.transform(val_b.drop(columns=['target', 'pred']))
        val_preds = {name: model.predict(val_scaled) for name, model in models.items()}
        meta_val_X = pd.DataFrame(val_preds)
        val_pred_log = meta_model.predict(meta_val_X)
        val_df.loc[val_b.index, 'pred'] = np.expm1(val_pred_log)

# --------------------------
# 결과 저장 및 평가
# --------------------------
print("[4] 결과 저장 및 평가")

submission_df = pd.read_csv(data_path + 'sample_submission.csv')
submission_df['answer'] = np.concatenate(pred_list)
os.makedirs(subm_path, exist_ok=True)

val_df = val_df.dropna(subset=['pred'])
final_smape = smape(np.expm1(val_df['target']), val_df['pred'])
save_path = subm_path + f'Energy_08_01_{SEED}_{final_smape:.4f}.csv'
submission_df.to_csv(save_path, index=False)

print(f"\n✅ 저장 완료: {save_path}")
print(f"📊 전체 Hold-out SMAPE: {final_smape:.4f}")

print("[5] 결과 기록")

with open("./Energy/08_submission/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: Energy_08_01_{SEED}_{final_smape:.4f}.csv\n")
    f.write(f"최종 SMAPE 점수 : {final_smape}\n")
    f.write("="*40 + "\n")
    
print("[6] 완료")
    