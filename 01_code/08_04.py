import numpy as np
import pandas as pd
import random
import json
import warnings
import os
import time
from tqdm import tqdm
import time
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

s_time = time.time()
warnings.filterwarnings('ignore')

# --------------------------
# 고정 SEED 설정
# --------------------------
seed_file = "./Energy/seed_count/08_04_seed.json"
os.makedirs(os.path.dirname(seed_file), exist_ok=True)

if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

SEED = 1 #seed_state["seed"]
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

# --------------------------
# 데이터 로딩 및 전처리
# --------------------------

print("[1] 데이터 로딩 및 초기 전처리 완료")

data_path = './Energy/'
subm_path = './Energy/08_submission/'

train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)
building = pd.read_csv(data_path + 'building_info.csv')

# print(building.columns)
# Index(['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
#        'PCS용량(kW)'], dtype='object')

building['태양광용량(kW)'] = pd.to_numeric(building['태양광용량(kW)'], errors='coerce')
building['ESS저장용량(kWh)'] = pd.to_numeric(building['ESS저장용량(kWh)'], errors='coerce')
building['PCS용량(kW)'] = pd.to_numeric(building['PCS용량(kW)'], errors='coerce')

log_col = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
building[log_col] = np.log1p(building[log_col])

# NaN → 0으로 채우기
building.fillna(0, inplace=True)

# 설비 존재 여부: 1 if 용량 > 0 else 0
building['has_solar'] = (building['태양광용량(kW)'] > 0).astype(int)
building['has_ess'] = (building['ESS저장용량(kWh)'] > 0).astype(int)
building['has_pcs'] = (building['PCS용량(kW)'] > 0).astype(int)

# 확인
# print(building[['연면적(m2)', '냉방면적(m2)']].head())

train = pd.merge(train, building, on='건물번호', how='left')
test = pd.merge(test, building, on='건물번호', how='left')

train['일시'] = pd.to_datetime(train['일시'], format='%Y%m%d %H')
test['일시'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')

for df in [train, test]:
    df['시간'] = df['일시'].dt.hour
    df['요일'] = df['일시'].dt.weekday
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['습도(%)'] = df['습도(%)'] / 100
    df['강수여부'] = (df['강수량(mm)'] > 0).astype(int)
    df['근무시간'] = df['시간'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['시간'] / 24)
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    df['DI'] = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3

# 요일 더미변수
for i in ['요일', '건물유형']:
    train = pd.concat([train, pd.get_dummies(train[i], prefix=i)], axis=1)
    test = pd.concat([test, pd.get_dummies(test[i], prefix=i)], axis=1)
    
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

etr_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형_IDC(전화국)', '건물유형_건물기타', '건물유형_공공',
       '건물유형_백화점', '건물유형_병원', '건물유형_상용', '건물유형_아파트', '건물유형_연구소', '건물유형_학교',
       '건물유형_호텔', '일조(hr)', 'DI', '일사(MJ/m2)']
etest_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형_IDC(전화국)', '건물유형_건물기타', '건물유형_공공',
       '건물유형_백화점', '건물유형_병원', '건물유형_상용', '건물유형_아파트', '건물유형_연구소', '건물유형_학교',
       '건물유형_호텔', 'DI', '일조(hr)']
train_e_1= train_e[etr_col].copy()
test_e_1 = test_e[etest_col].copy()

x1 = train_e_1.drop(['일사(MJ/m2)'], axis=1)
y1 = train_e_1['일사(MJ/m2)']

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=SEED, shuffle=True, train_size=0.8
)

model_e = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.01, 
    depth=6,
    l2_leaf_reg=5,
    bagging_temperature=1.0,
    subsample=0.8, 
    loss_function='RMSE',
    early_stopping_rounds=50,
    random_state=SEED,
    verbose=0 
)

model_e.fit(x1_train, y1_train, eval_set=[(x1_test, y1_test)])
results = model_e.predict(x1_test)
e_rmse = np.sqrt(mean_squared_error(y1_test, results))

pred_e = model_e.predict(test_e_1)
pred_e = np.maximum(pred_e, 0)
pred_e = np.round(pred_e, 2)

# 해당 시간 추출
hours = train.loc[test_e.index, '시간']
night_mask = (hours >= 21) | (hours <= 5)
pred_e[night_mask.values] = 0.0

# 소수점 2자리 반올림 후 원래 위치에 덮어쓰기
train.loc[test_e.index, '일사(MJ/m2)'] = np.round(pred_e, 2)

# 확인
print(f"    > train 일사 예측 RMSE : {e_rmse:.6f}")
# print(f"    > ✅ 일사 결측치 {len(pred_e)}건 train에 반영 완료")

# exit()
# ----------------------------
# 일조/일사 예측 보조모델
# ----------------------------
print("[3] 일조/일사 보조모델 학습 및 예측")

# -----------------------------
# 🔧 변수 정의
# -----------------------------
sun_target_cols = ['일조(hr)', '일사(MJ/m2)']
sun_feature_cols = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 
                    'sin_hour', 'cos_hour', 'DI']
input_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'sin_hour', 'cos_hour', 'DI']
building_ids = train['건물번호'].unique()
N_SPLITS = 5

# -----------------------------
# 데이터 복사
# -----------------------------
X_all = train[sun_feature_cols].copy()
y_all = train[sun_target_cols].copy()
test_all = test[sun_feature_cols].copy()

# 최종 결과 저장
sun_pred_test = pd.DataFrame(index=test.index, columns=sun_target_cols)
sun_oof_preds = pd.DataFrame(index=train.index, columns=sun_target_cols)

# -----------------------------
# 모델 학습 및 예측
# -----------------------------
# print("🌞 [SUN] 건물별 보조모델 KFold 학습 시작")

for col in sun_target_cols:
    print(f"    > 🎯 예측 대상: {col}")
    oof_col = np.zeros(len(X_all))
    test_col_pred = np.zeros(len(test_all))

    for bno in tqdm(building_ids):
        bld_idx = (X_all['건물번호'] == bno)
        bld_test_idx = (test_all['건물번호'] == bno)

        x_bld = X_all.loc[bld_idx, input_cols].reset_index(drop=True)
        y_bld = y_all.loc[bld_idx, col].reset_index(drop=True)
        x_test_bld = test_all.loc[bld_test_idx, input_cols].reset_index(drop=True)

        if len(x_bld) < N_SPLITS:
            continue

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        bld_oof = np.zeros(len(x_bld))
        bld_test_preds = []

        for train_idx, val_idx in kf.split(x_bld):
            x_tr, x_val = x_bld.iloc[train_idx], x_bld.iloc[val_idx]
            y_tr, y_val = y_bld.iloc[train_idx], y_bld.iloc[val_idx]

            ss = StandardScaler()
            x_tr = ss.fit_transform(x_tr)
            x_val = ss.transform(x_val)
            x_test_bld = ss.transform(x_test_bld)

            model_lgb = LGBMRegressor(
                n_estimators=1000,
                random_state=SEED,
                early_stopping_rounds=50,
                verbosity=-1
            )
            model_xgb = XGBRegressor(
                n_estimators=1000,
                random_state=SEED,
                eval_metric='mae',  # 또는 'rmse'
                early_stopping_rounds=50,
            )
            
            model_lgb.fit(
                x_tr, y_tr,
                eval_set=[(x_val, y_val)],
            )
            model_xgb.fit(
                x_tr, y_tr,
                eval_set=[(x_val, y_val)],
                verbose=0,
            )

            pred_val = (model_lgb.predict(x_val) + model_xgb.predict(x_val)) / 2
            bld_oof[val_idx] = pred_val

            pred_test = (model_lgb.predict(x_test_bld) + model_xgb.predict(x_test_bld)) / 2
            bld_test_preds.append(pred_test)

        # OOF 저장
        oof_col[bld_idx] = bld_oof

        # Test 예측 평균 저장
        if bld_test_preds:
            test_col_pred[bld_test_idx] = np.mean(bld_test_preds, axis=0)

    # 결과 저장
    sun_oof_preds[col] = oof_col
    sun_pred_test[col] = test_col_pred

    # SMAPE 출력
    score = smape(y_all[col], oof_col)
    rmse = np.sqrt(mean_squared_error(y_all[col], oof_col))
    print(f"    > {col} SMAPE: {score:.6f}")
    print(f"    > {col}  RMSE: {rmse:.6f}")

# -----------------------------
# 예측 결과 반올림 및 적용
# -----------------------------
sun_oof_preds['일조(hr)'] = sun_oof_preds['일조(hr)'].round(1)
sun_pred_test['일조(hr)'] = sun_pred_test['일조(hr)'].round(1)
sun_oof_preds['일사(MJ/m2)'] = sun_oof_preds['일사(MJ/m2)'].round(2)
sun_pred_test['일사(MJ/m2)'] = sun_pred_test['일사(MJ/m2)'].round(2)

train[sun_target_cols] = sun_oof_preds
test[sun_target_cols] = sun_pred_test

# 해가 없는 시간대 마스크
sun_off_mask_train = train['cos_hour'] <= 0
sun_off_mask_test = test['cos_hour'] <= 0

# 음수 → 0 처리 함수
def clip_non_negative(df, cols):
    for c in cols:
        df[c] = df[c].clip(lower=0)
    return df

# 1. 일사: 해가 없는 시간대 0으로
train.loc[sun_off_mask_train, '일사(MJ/m2)'] = 0
test.loc[sun_off_mask_test, '일사(MJ/m2)'] = 0

# 2. 음수 보정 (clip)
train = clip_non_negative(train, sun_target_cols)
test = clip_non_negative(test, sun_target_cols)

train.to_csv('./Energy/(trier)train.csv', index=False)
test.to_csv('./Energy/(trier)test.csv', index=False)

# 변수 설정

train['target'] = np.log1p(train['전력소비량(kWh)'])
# X_all = train.drop(columns=['전력소비량(kWh)', 'target'])
X_all = train[['건물번호', '기온(°C)', '풍속(m/s)', '습도(%)', '일사(MJ/m2)',
             '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
             'PCS용량(kW)', '주말여부', '근무시간',
             'sin_hour', 'cos_hour', '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5',
             '요일_6', 'DI']]
test = test[['건물번호', '기온(°C)', '풍속(m/s)', '습도(%)', '일사(MJ/m2)',
             '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
             'PCS용량(kW)', '주말여부', '근무시간',
             'sin_hour', 'cos_hour', '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5',
             '요일_6', 'DI']]

y_all = train['target']

def get_models():
    return ['xgb', 'lgb', 'cat']

X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, shuffle=True)
val_df = X_val.copy()
val_df['target'] = y_val
val_df['pred'] = np.nan  # 예측값 저장용

# --------------------------
# Building별 학습 및 예측
# --------------------------
print("[5] Building별 학습 및 예측")

N_SPLITS = 5
pred_list = []
building_ids = sorted(train['건물번호'].unique())

for bno in building_ids:
    print(f"    > Building {bno}")

    # 개별 데이터 분리
    tr_b = X_tr[X_tr['건물번호'] == bno].copy()
    val_b = val_df[val_df['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(tr_b) < N_SPLITS:
        print(f"    >> ⚠️ Skipped due to insufficient samples: {len(tr_b)}")
        pred_list.append(np.zeros(len(test_b)))
        continue

    y_b = y_tr[tr_b.index]
    common_cols = list(set(tr_b.columns) & set(test_b.columns))
    common_cols.sort() 
    # 스케일링
    
    num_cols = [col for col in tr_b.columns if col not in ['건물번호']]
    ss = StandardScaler()
    x = ss.fit_transform(tr_b[num_cols])
    x_test_b = ss.transform(test_b[num_cols])

    # KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = {name: np.zeros(len(x)) for name in get_models()}
    test_preds_all = {name: np.zeros(len(x_test_b)) for name in get_models()}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(x)):
        print(f"       > 🔁 Fold {fold+1}/{N_SPLITS}")
        x_trn, x_val = x[tr_idx], x[val_idx]
        y_trn, y_val = y_b.iloc[tr_idx], y_b.iloc[val_idx]
        
        xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, early_stopping_rounds=50,verbosity=0)
        lgb_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbose=-1)
        cat_model = CatBoostRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbose=0)

        # 학습 (개별 모델)
        xgb_model.fit(x_trn, y_trn,
                      eval_set=[(x_val, y_val)],
                      )

        lgb_model.fit(x_trn, y_trn,
                      eval_set=[(x_val, y_val)],
                      early_stopping_rounds=50,
                      )

        cat_model.fit(x_trn, y_trn,
                      eval_set=(x_val, y_val),
                      early_stopping_rounds=50,
                      use_best_model=True)

        # OOF 예측값 저장
        oof_preds['xgb'][val_idx] = xgb_model.predict(x_val)
        oof_preds['lgb'][val_idx] = lgb_model.predict(x_val)
        oof_preds['cat'][val_idx] = cat_model.predict(x_val)

        # 테스트 예측값 저장 (fold 평균용)
        test_preds_all['xgb'] += xgb_model.predict(x_test_b) / N_SPLITS
        test_preds_all['lgb'] += lgb_model.predict(x_test_b) / N_SPLITS
        test_preds_all['cat'] += cat_model.predict(x_test_b) / N_SPLITS

    # 스태킹용 Feature 생성
    oof_stack = np.vstack([
        oof_preds['xgb'],
        oof_preds['lgb'],
        oof_preds['cat']
    ]).T

    test_stack = np.vstack([
        test_preds_all['xgb'],
        test_preds_all['lgb'],
        test_preds_all['cat']
    ]).T

    # 메타모델 학습 (RidgeCV)
    meta_model = RidgeCV()
    meta_model.fit(oof_stack, y_b)
    final_pred = meta_model.predict(test_stack)

    # 결과 저장
    pred_list.append(final_pred)
    val_df.loc[val_b.index, 'pred'] = meta_model.predict(oof_stack)
    
print("[6] 전체 Test 예측값 결합 및 로그 복원")

# pred_list는 건물번호 순서대로 test 데이터 예측값을 담고 있음
final_preds_log = np.concatenate(pred_list)
final_preds = np.expm1(final_preds_log)  # 로그 복원

# test 데이터와 길이가 맞는지 확인
assert len(final_preds) == len(test), "❌ 예측 결과와 테스트 데이터 길이가 맞지 않습니다."

# submission 파일 생성
submission = pd.DataFrame({
    'id': test.index,
    'target': final_preds
})

# 저장 디렉토리 및 파일명
output_dir = "./Energy/submission"
os.makedirs(output_dir, exist_ok=True)

from datetime import datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{output_dir}/submission_{now}_seed{SEED}.csv"
submission.to_csv(output_path, index=False)

print(f"✅ 제출 파일 저장 완료: {output_path}")

val_df = val_df.dropna(subset=['pred'])  # 예측이 없는 샘플 제거
val_df['target_inv'] = np.expm1(val_df['target'])
val_df['pred_inv'] = np.expm1(val_df['pred'])

# SMAPE, RMSE 계산
final_smape = smape(val_df['target_inv'].values, val_df['pred_inv'].values)
final_rmse = np.sqrt(mean_squared_error(val_df['target_inv'].values, val_df['pred_inv'].values))

print(f"📊 최종 SMAPE : {final_smape:.4f}")
print(f"📉 최종 RMSE  : {final_rmse:.4f}")