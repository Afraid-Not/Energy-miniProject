import numpy as np
import pandas as pd
import random 
import keras.backend as K
import warnings
import json
import os

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


def smape_tf(y_true, y_pred):
    numerator = K.abs(y_pred - y_true)
    denominator = (K.abs(y_pred) + K.abs(y_true)) + K.epsilon()
    return 200.0 * K.mean(numerator / denominator)

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    Parameters:
    - y_true: 정답값 (array-like)
    - y_pred: 예측값 (array-like)

    Returns:
    - SMAPE score (%)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-8  # 분모 0 방지
    diff = np.abs(y_pred - y_true)
    smape = np.mean(2.0 * diff / denominator) * 100
    return smape
    
def create_sequences_stride(data, target_cols, time_steps=24, stride=24):
    X, Y = [], []
    feature_cols = [col for col in data.columns if col not in target_cols]

    for i in range(0, len(data) - time_steps, stride):
        x_seq = data[feature_cols].iloc[i:i+time_steps].values
        y_seq = data[target_cols].iloc[i+time_steps:i+time_steps+time_steps].values
        X.append(x_seq)
        Y.append(y_seq)

    return np.array(X), np.array(Y)

data_path = './Energy/data/'

train = pd.read_csv(data_path + 'train_new.csv')
test = pd.read_csv(data_path + 'test_new.csv')


# print(train.columns)
# print(test.columns)
# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
#        '건물유형', '예상_태양광_발전량(kWh)', '순수_전력소비량(kWh)', '요일', '시간', '주말여부'],
#       dtype='object')
# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형', '요일', '시간',
#        '주말여부'],
#       dtype='object')

# exit()

X1 = train.drop(['일조(hr)', '일사(MJ/m2)', '예상_태양광_발전량(kWh)', '순수_전력소비량(kWh)'], axis=1)
Y1 = train[['일조(hr)', '일사(MJ/m2)']]
X1_test = test.copy()

# print(X1.shape) (204000, 8)
# print(Y1.shape) (204000, 2)
# print(X1_test.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    X1, Y1, random_state=SEED, train_size=0.8
)

ss_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x1_train[ss_col])
x1_train[ss_col] = ss.transform(x1_train[ss_col])
x1_test[ss_col] = ss.transform(x1_test[ss_col])
X1_test[ss_col] = ss.transform(X1_test[ss_col])

from xgboost import XGBRegressor
xgb = XGBRegressor(random_state=SEED, early_stopping_rounds=50)
xgb.fit(x1_train, y1_train, eval_set=[(x1_test, y1_test)])
score = xgb.score(x1_test, y1_test)
y1_pred = xgb.predict(X1_test)

# print(y1_pred)

pred_df = pd.DataFrame(y1_pred, columns=['일조(hr)', '일사(MJ/m2)'])

# 2. test 데이터와 예측값 결합 (axis=1 방향으로)
test_sun = pd.concat([test.reset_index(drop=True), pred_df], axis=1)

need = pd.read_csv('./Energy/building_info.csv')
# print(need.columns)
# Index(['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
#        'PCS용량(kW)'],
need_col = need[['건물번호','태양광용량(kW)']]
need_col['태양광용량(kW)'] = pd.to_numeric(need_col['태양광용량(kW)'], errors='coerce')
need_col = need_col.fillna(0)
test_sun = pd.merge(test_sun, need_col, on=['건물번호'], how='left')

# 3. 발전량 계산
test_sun['예상_태양광_발전량(kWh)'] = test_sun['일사(MJ/m2)'] * 0.2778 * test_sun['태양광용량(kW)']

# 4. 결과 확인
# print(np.min(test_sun['예상_태양광_발전량(kWh)']))
# print(np.max(test_sun['예상_태양광_발전량(kWh)']))

# print(test_sun.shape)
# (16800, 12)

# print(train.columns)
# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
#        '건물유형', '예상_태양광_발전량(kWh)', '순수_전력소비량(kWh)', '요일', '주말여부'],
# print(test_sun.columns)
# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형', '요일', '주말여부',
#        '일조(hr)', '일사(MJ/m2)', '태양광용량(kW)', '예상_태양광_발전량(kWh)'],
#       dtype='object')
# exit()

train2 = train.drop(['일조(hr)', '일사(MJ/m2)', '예상_태양광_발전량(kWh)'], axis=1)
test2 = test_sun.drop(['일조(hr)', '일사(MJ/m2)', '태양광용량(kW)', '예상_태양광_발전량(kWh)'], axis=1)
target = ['순수_전력소비량(kWh)']

first_test = train2.iloc[-24:].drop(['순수_전력소비량(kWh)'], axis=1)
first_test = np.array(first_test).reshape(1, 24, 9)
# print(first_test)
# print(first_test.shape)
# exit()
# print(train2.shape)
# (204000, 9)
# print(test2.shape)
# (16800, 8)
print('2번 모델')
# exit()
X2, Y2 = create_sequences_stride(train2, target, time_steps=24, stride=24)
# print(X2.shape)         # (8499, 24, 9)
# print(Y2.shape)         # (8499, 24, 1)
X2_test, _ = create_sequences_stride(test2, target_cols=[], time_steps=24, stride=24)
# print(X2_test.shape)    # (699, 24, 9)


# exit()
x2_train, x2_test, y2_train, y2_test = train_test_split(
    X2, Y2, random_state=SEED, train_size=0.8
)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x2_train_re = x2_train.reshape(-1, x2_train.shape[1] * x2_train.shape[2])
X2_test_re = X2_test.reshape(-1, X2_test.shape[1] * X2_test.shape[2])
first_test_re = first_test.reshape(-1, first_test.shape[1] * first_test.shape[2])
ss.fit(x2_train_re)
x2_train_re = ss.transform(x2_train_re)
X2_test_re = ss.transform(X2_test_re)
first_test_re = ss.transform(first_test_re)
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ==========================
# 1. 데이터 리쉐이핑
# ==========================
x2_train_re = x2_train.reshape(x2_train.shape[0], -1)  # (N, 24*9)
x2_test_re = x2_test.reshape(x2_test.shape[0], -1)
X2_test_re = X2_test.reshape(X2_test.shape[0], -1)
first_test_re = first_test.reshape(first_test.shape[0], -1)

y2_train_re = y2_train.reshape(y2_train.shape[0], -1)  # (N, 24)
y2_test_re = y2_test.reshape(y2_test.shape[0], -1)

# ==========================
# 2. 개별 모델 정의
# ==========================
lgbm = LGBMRegressor(random_state=SEED, n_estimators=300)
xgb = XGBRegressor(random_state=SEED, n_estimators=300)
cat = CatBoostRegressor(random_seed=SEED, iterations=300, verbose=0)

# ==========================
# 3. 스태킹 모델 정의
# ==========================
from sklearn.multioutput import MultiOutputRegressor

base_stack = StackingRegressor(
    estimators=[
        ('lgbm', lgbm),
        ('xgb', xgb),
        ('cat', cat)
    ],
    final_estimator=BayesianRidge()
)

multi_output_model = MultiOutputRegressor(base_stack)
multi_output_model.fit(x2_train_re, y2_train_re)
y2_pred = multi_output_model.predict(x2_test_re)

smape2 = smape(y2_test_re, y2_pred)
print(f'SMAPE : {smape2:.6f}')

# ==========================
# 6. 최종 예측
# ==========================
y_pred_test = multi_output_model.predict(X2_test_re)
y_pred_first = multi_output_model.predict(first_test_re)

# (699*24, 1)
y_pred_test = y_pred_test.reshape(-1, 1)
y_pred_first = y_pred_first.reshape(-1, 1)
total_y_pred = np.concatenate([y_pred_first, y_pred_test], axis=0)

# test_sun에 결과 반영
test_sun['순수_전력소비량(kWh)'] = total_y_pred
test_sun['전력소비량(kWh)'] = test_sun['순수_전력소비량(kWh)'] + test_sun['예상_태양광_발전량(kWh)']

submit = pd.read_csv('./Energy/sample_submission.csv')
submit['answer'] = test_sun['전력소비량(kWh)']



submitpath = './Energy/submission/'
filename = f"{submitpath}{SEED}_submit.csv"
submit.to_csv(filename, index=False)

print(f'SMAPE : {smape2:.6f}')
print(f'Random: {SEED}')
print('#### 저장 완료 ####')
