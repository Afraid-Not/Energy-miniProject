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

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def smape_eval(preds, dtrain):
    y_true = dtrain.get_label()
    denominator = (np.abs(y_true) + np.abs(preds)) + 1e-6  # 분모 0 방지
    diff = np.abs(preds - y_true) / denominator
    smape = 200 * np.mean(diff)
    return 'SMAPE', smape, False 

seed_file = "./Energy/09_00_submission/09_00_seed.json"

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
for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = np.log1p(buildinginfo[col].replace('-', 0).astype(float))

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
    # for col in ['일조(hr)', '일사(MJ/m2)']:
    #     if col in df.columns:
    #         df[col] = df[col].fillna(0)
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

print(train.columns, test.columns)

# Index(['num_date_time', '건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
#        '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)', 'hour', 'dayofweek', 'month',
#        'day', 'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI',
#        '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
#        'PCS용량(kW)'],
exit()
# --------------------------
# 2. 원핫 인코딩
# --------------------------

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor

# ----------------------------
# 일사 결측치 예측 보조모델
# ----------------------------
print("[2] 일사 결측치 예측 보조 모델")

empty = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]
train_e = train[~train['건물번호'].isin(empty)]
test_e = train[train['건물번호'].isin(empty)]

etr_col = ['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
       '일조(hr)',  'hour', 'dayofweek', 'month',
       'day', 'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI',
       '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
       'PCS용량(kW)', '일사(MJ/m2)']
etest_col = etr_col[:-1]
train_e_1= train_e[etr_col].copy()
test_e_1 = test_e[etest_col].copy()

x1 = train_e_1.drop(['일사(MJ/m2)'], axis=1)
y1 = train_e_1['일사(MJ/m2)']

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=SEED, shuffle=True, train_size=0.8
)

model_e = CatBoostRegressor(
    iterations=3000,             # 충분히 긴 학습
    learning_rate=0.01,          # 낮은 러닝레이트 (정확도 ↑, 학습시간 ↑)
    depth=6,                     # 트리 깊이 (보통 6~10)
    l2_leaf_reg=5,               # 정규화 (5~10 사이 튜닝 가능)
    bagging_temperature=1.0,     # 과적합 방지 (0~1: 낮을수록 랜덤샘플 다양성 ↑)
    subsample=0.8,               # 데이터 일부 샘플링 (overfitting 방지)
    loss_function='Huber:delta=1.0',       # 회귀는 일반적으로 RMSE
    early_stopping_rounds=100,   # 조기 종료
    random_state=SEED,
    verbose=0                  # 100 step마다 로그 출력
)

model_e.fit(x1_train, y1_train, eval_set=[(x1_test, y1_test)])
results = model_e.predict(x1_test)
e_rmse = smape(y1_test, results)

pred_e_p = model_e.predict(test_e_1)
pred_e_max = np.maximum(pred_e_p, 0)
pred_e_round = np.round(pred_e_max, 2)
pred_e = np.clip(pred_e_round, 0.1, None)

# 해당 시간 추출
hours = train.loc[test_e.index, 'hour']
night_mask = (hours >= 21) | (hours <= 5)
pred_e[night_mask.values] = 0.0

# 소수점 2자리 반올림 후 원래 위치에 덮어쓰기
train.loc[test_e.index, '일사(MJ/m2)'] = np.round(pred_e, 2)

# 확인
print(f"    > 일사 SMAPE : {e_rmse}")
# print(f"    > ✅ 일사 결측치 {len(pred_e)}건 train에 반영 완료")

# exit()
print("[3] 일조/일사 보조모델 학습 및 예측")


print("    > 🌞 [SUN] 건물별 보조모델 학습 시작")
sun_target_cols = ['일조(hr)', '일사(MJ/m2)']
sun_feature_cols = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'hour', 'dayofweek', 'month',
       'day', 'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', '건물유형', '연면적(m2)', '냉방면적(m2)', 
       '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']

sun_test_preds = []
sun_rmses = []

building_ids = sorted(train['건물번호'].unique())
for bno in building_ids:
    print(f"    > Building {bno} - 일조/일사 예측 모델")

    train_b = train[train['건물번호'] == bno].dropna(subset=sun_target_cols).copy()
    test_b = test[test['건물번호'] == bno].copy()

    # if len(train_b) < 10:
    #     print(f"    > ⚠️ 학습 데이터 부족 → 0으로 채움 ({len(train_b)}개)")
    #     sun_test_preds.append(np.zeros((len(test_b), 2)))
    #     continue

    X = train_b[sun_feature_cols]
    y = train_b[sun_target_cols]
    X_test = test_b[sun_feature_cols]

    ss_sun = StandardScaler()
    X = ss_sun.fit_transform(X)
    X_test = ss_sun.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=SEED)
    
    X_train = pd.DataFrame(X_train, columns=sun_feature_cols)
    X_val = pd.DataFrame(X_val, columns=sun_feature_cols)
    X_test = pd.DataFrame(X_test, columns=sun_feature_cols)
    
    if (pd.DataFrame(y_train).nunique(axis=0) < 2).any():
        print(f"    > ⚠️ 타겟 값이 모두 동일 → 모델 학습 생략")
        sun_test_preds.append(np.zeros((len(test_b), 2)))
        continue

    stacking_model = LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=8,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=SEED,
                verbose=-1,
                objective='huber'
            )

    # 2. 멀티아웃풋으로 감싸기 (일조, 일사 예측 동시에)
    stack_model = MultiOutputRegressor(stacking_model)

    stack_model.fit(X_train, y_train)

    y_pred_val = stack_model.predict(X_val)
    rmse = smape(y_val, y_pred_val)
    sun_rmses.append(rmse)
    print(f"    - ✅ 일조 SMAPE: {rmse[0]:.6f}")
    print(f"    - ✅ 일사 SMAPE: {rmse[1]:.6f}")

    pred_test = stack_model.predict(X_test)
    pred_test = np.maximum(pred_test, 0)
    pred_test[:,0] = np.round(pred_test[:,0], 1)
    pred_test[:,1] = np.round(pred_test[:,1], 2)
    sun_test_preds.append(pred_test)

# 최종 결합 및 저장
sun_final_pred = np.vstack(sun_test_preds)
sun_final_pred = np.clip(sun_final_pred, 0.1, None)
test[['일조(hr)', '일사(MJ/m2)']] = sun_final_pred

# 🌙 야간 시간대 처리: 해가 없으면 일조/일사는 0
if '시간' not in test.columns and '일시' in test.columns:
    test['시간'] = pd.to_datetime(test['일시']).dt.hour

test.loc[(test['시간'] < 6) | (test['시간'] > 18), ['일조(hr)', '일사(MJ/m2)']] = 0

final_rmse = np.mean(sun_rmses)
print(f"\n    > 🌞 건물별 검증 평균 SMAPE: {final_rmse:.6f}")

# print(train.columns)
# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
#        '전력소비량(kWh)', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
#        'PCS용량(kW)', 'has_solar', 'has_ess', 'has_pcs', '주말여부', '강수여부', '근무시간',
#        'sin_hour', 'cos_hour', '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5',
#        '요일_6', '건물유형_IDC(전화국)', '건물유형_건물기타', '건물유형_공공', '건물유형_백화점', '건물유형_병원',
#        '건물유형_상용', '건물유형_아파트', '건물유형_연구소', '건물유형_학교', '건물유형_호텔'],
#       dtype='object')

# test.to_csv('./Energy/sunlight_prediction.csv', index=False)

print("[4] 전처리 완료")

features = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'hour', 'dayofweek', 'month',
       'day', 'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI', '건물유형', '연면적(m2)', '냉방면적(m2)', 
       '태양광용량(kW)', 'ESS저장용량(kWh)', '일조(hr)', '일사(MJ/m2)']

target = '전력소비량(kWh)'

# 최종 예측 결과 저장용
final_preds = []
val_smapes = []

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()

print("[5] 건물별 학습 시작")


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
                              random_state=SEED, objective='huber', verbose=-1)
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                  random_seed=SEED, verbose=0, loss_function='Huber:delta=1.0')
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

    pred = np.expm1(final_model.predict(test_pred_lvl2.reshape(-1, 1)))
    final_preds.extend(pred)

# 결과 저장
final_preds = np.clip(final_preds, 0.1, None)
samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
score_str = f"{avg_smape:.4f}".replace('.', '_')
filepath = './Energy/09_00_submission/'

os.makedirs(filepath, exist_ok=True)

filename = f"energy_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(filepath + filename, index=False)

print(f"[6] 📁 저장 완료 ")
print(f"✅ 최종 SMAPE 점수 : {avg_smape:.6f}")

# import shutil
# SCORE_THRESHOLD = 10
# if avg_smape > SCORE_THRESHOLD:
#     shutil.rmtree("./Energy/gift_submission")
#     print(f"🚫 Score {avg_smape:.5f} > 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
# else:
#     print(f"🎉 Score {avg_smape:.5f} < 기준 {SCORE_THRESHOLD} → 디렉토리 유지")
    
with open("./Energy/09_00_submission/result_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"✅ 저장 완료: {filename}\n")
    f.write(f"✅ 일사 SMAPE : {e_rmse}\n")
    f.write(f"✅ 일조/일사 SMAPE: {final_rmse:.6f}\n")
    f.write(f"✅ 최종 SMAPE 점수 : {avg_smape}\n")
    f.write("="*40 + "\n")
    
print(f"[7] 종료 ")
    