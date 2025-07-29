import numpy as np
import pandas as pd
import random
import json
import warnings
import os

warnings.filterwarnings('ignore')

# SEED 고정
seed_file = "./Energy/seed_count/08_seed.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
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

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

#1. 데이터
data_path = './Energy/'
subm_path = './Energy/08_submission/'
train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)
building = pd.read_csv(data_path + 'building_info.csv')

# print(train.shape, train.columns)
# (204000, 9) Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
#        '일사(MJ/m2)', '전력소비량(kWh)'],
#       dtype='object')
# print(test.shape, test.columns)
# (16800, 6) Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)'], dtype='object')
# print(building.shape, building.columns)
# (100, 7) Index(['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
#        'PCS용량(kW)'],
#       dtype='object')

train = train.drop(['일조(hr)', '일사(MJ/m2)'], axis = 1)

train['일시'] = pd.to_datetime(train['일시'])
test['일시'] = pd.to_datetime(test['일시'])

for df in [train, test]:
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    # df['월'] = df['일시'].dt.month
    # df['일'] = df['일시'].dt.day
    df['시간'] = df['일시'].dt.hour
    df['요일'] = df['일시'].dt.weekday
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['기온×습도'] = df['기온(°C)'] * df['습도(%)']
    df['풍속×기온'] = df['풍속(m/s)'] * df['기온(°C)']
    df['강수여부'] = (df['강수량(mm)'] > 0).astype(int)
    df['근무시간'] = df['시간'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['시간'] / 24)

for i in ['요일'] :
    train_temp = pd.get_dummies(train[i], prefix=i)
    test_temp = pd.get_dummies(test[i], prefix=i)
    train = pd.concat([train, train_temp], axis=1)
    test = pd.concat([test, test_temp], axis=1)

for i in ['일시', '시간', '요일']:
    train = train.drop(i, axis=1)
    test = test.drop(i, axis=1)
# print(train.columns)
# print(test.columns)
    
# exit()
x = train.drop(['전력소비량(kWh)'], axis=1)
y = train['전력소비량(kWh)']
y = np.log1p(y)

print(x['sin_hour'].head())
print(y.shape)

from sklearn.model_selection import KFold, train_test_split
n = 5
cv = KFold(n_splits=n, random_state=SEED, shuffle=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=SEED, shuffle=True, train_size=0.8,
)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)
test = ss.transform(test)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV  # 메타 모델
N_SPLITS = 5

# 데이터 준비 (x, y는 기존 학습용 데이터)
# x: (N, F) / y: (N,)
# x_train = x_train.reset_index(drop=True)
# x_test = x_test.reset_index(drop=True)
# y_train = y_train.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# OOF 예측 저장용
oof_preds = {
    'xgb': np.zeros(len(x)),
    'lgb': np.zeros(len(x)),
    'cat': np.zeros(len(x))
}

# Test 세트 예측 누적용 (스태킹 시 활용 시 사용 가능)
# test_preds = {name: np.zeros(len(test)) for name in oof_preds.keys()}

# 각 모델 정의 함수
def get_models():
    return {
        'xgb': XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1),
        'lgb': LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=0),
        'cat': CatBoostRegressor(random_state=SEED, verbose=0)
    }

# KFold 루프
for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    print(f"\n🔁 Fold {fold+1}/{N_SPLITS}")
    x_trn, x_val = x_train[train_idx], x_train[val_idx]
    y_trn, y_val = y_train[train_idx], y_train[val_idx]
    
    models = get_models()

    for name, model in models.items():
        model.fit(x_trn, y_trn)
        preds = model.predict(x_val)
        preds_exp = np.expm1(preds)
        y_val_exp = np.expm1(y_val)
        oof_preds[name][val_idx] = preds
        print(f"✅ {name.upper()} Fold {fold+1} SMAPE: {smape(y_val_exp, preds_exp):.4f}")

# OOF prediction으로 meta dataset 생성
meta_X = pd.DataFrame(oof_preds)
meta_y = y.copy()

print("\n🎯 Meta model training...")
meta_model = RidgeCV()
meta_model.fit(meta_X, meta_y)

# OOF로 예측한 전체 스태킹 성능
final_oof_pred = meta_model.predict(meta_X)

meta_y_exp = np.expm1(meta_y)
final_oof_pred_exp = np.expm1(final_oof_pred)

smape_meta = smape(meta_y_exp, final_oof_pred_exp)
print(smape_meta)






