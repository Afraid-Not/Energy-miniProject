import numpy as np
import pandas as pd
import random
import json
import warnings
import os
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
seed_file = "./Energy/seed_count/08_02_seed.json"
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

def get_models():
    return {
        'xgb': XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=1.0,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=SEED,
            n_jobs=-4,
            verbosity=0
        ),
        'lgb': LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=7,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-4,
            verbose=-1
        ),
        # 'cat': CatBoostRegressor(
        #     iterations=500,
        #     learning_rate=0.01,
        #     depth=6,
        #     l2_leaf_reg=5,
        #     bagging_temperature=1.0,
        #     random_strength=1.0,
        #     subsample=0.8,
        #     loss_function='RMSE',
        #     random_state=SEED,
        #     verbose=0
        # )
    }

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
    # df['기온×습도'] = df['기온(°C)'] * df['습도(%)']
    # df['풍속×기온'] = df['풍속(m/s)'] * df['기온(°C)']
    df['습도(%)'] = df['습도(%)'] / 100
    df['강수여부'] = (df['강수량(mm)'] > 0).astype(int)
    df['근무시간'] = df['시간'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['시간'] / 24)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['건물유형'])

train['건물유형_LE'] = le.transform(train['건물유형'])
test['건물유형_LE'] = le.transform(test['건물유형'])  # train 기반으로 transform만!

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

etr_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)']
etest_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',]
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
    loss_function='RMSE',       # 회귀는 일반적으로 RMSE
    early_stopping_rounds=100,   # 조기 종료
    random_state=SEED,
    verbose=0                  # 100 step마다 로그 출력
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
print(f"    > 일사 RMSE : {e_rmse}")
print(f"    > ✅ 일사 결측치 {len(pred_e)}건 train에 반영 완료")


print("[3] 일조/일사 보조모델 학습 및 예측")


print("    > 🌞 [SUN] 건물별 보조모델 학습 시작")
sun_target_cols = ['일조(hr)', '일사(MJ/m2)']
sun_feature_cols = ['건물번호', '기온(°C)', '강수량(mm)','강수여부', '풍속(m/s)', '습도(%)', 'sin_hour', 'cos_hour']

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
                verbose=-1
            )

    # 2. 멀티아웃풋으로 감싸기 (일조, 일사 예측 동시에)
    stack_model = MultiOutputRegressor(stacking_model)

    stack_model.fit(X_train, y_train)

    y_pred_val = stack_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    sun_rmses.append(rmse)
    print(f"    - ✅ RMSE: {rmse:.6f}")

    pred_test = stack_model.predict(X_test)
    pred_test = np.maximum(pred_test, 0)
    pred_test[:,0] = np.round(pred_test[:,0], 1)
    pred_test[:,1] = np.round(pred_test[:,1], 2)
    sun_test_preds.append(pred_test)

# 최종 결합 및 저장
sun_final_pred = np.vstack(sun_test_preds)
test[['일조(hr)', '일사(MJ/m2)']] = sun_final_pred

# 🌙 야간 시간대 처리: 해가 없으면 일조/일사는 0
if '시간' not in test.columns and '일시' in test.columns:
    test['시간'] = pd.to_datetime(test['일시']).dt.hour

test.loc[(test['시간'] < 6) | (test['시간'] > 18), ['일조(hr)', '일사(MJ/m2)']] = 0

final_rmse = np.mean(sun_rmses)
print(f"\n    > 🌞 건물별 검증 평균 RMSE: {final_rmse:.6f}")

# 불필요한 컬럼 제거
drop_cols = ['일시', '시간', '요일', '건물유형', '건물유형_LE']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

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

s3_time = time.time()
# exit()

# --------------------------
# Train / Validation Split
# --------------------------
train['target'] = np.log1p(train['전력소비량(kWh)'])
# X_all = train.drop(columns=['전력소비량(kWh)', 'target'])
X_all = train[['건물번호', '기온(°C)', '풍속(m/s)', '습도(%)', 'sin_hour', 
                'cos_hour', '일사(MJ/m2)', '연면적(m2)', '냉방면적(m2)', 
                'has_solar', 'has_ess', '주말여부', '근무시간',
                ]]
test = test[['건물번호', '기온(°C)', '풍속(m/s)', '습도(%)', 'sin_hour', 
                'cos_hour', '일사(MJ/m2)', '연면적(m2)', '냉방면적(m2)', 
                'has_solar', 'has_ess', '주말여부', '근무시간',
                ]]


# region(피쳐 임포턴스 측정)
# X_all.columns = [
#     'building_id', 'temperature', 'is_rain', 'wind_speed', 'humidity',
#     'sin_hour', 'cos_hour', 'sunshine_hour', 'solar_radiation_MJ',
#     'total_floor_area', 'cooling_area',
#     'has_solar', 'has_ess', 'has_pcs', 'is_weekend', 'working_hour',
#     'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',
#     'weekday_4', 'weekday_5', 'weekday_6'
# ]

# test.columns = X_all.columns

# X = X_all
# y = train['target']

# # 모델 학습
# model = LGBMRegressor(random_state=42)
# model.fit(X, y)

# # 피처 중요도 추출
# importances = model.feature_importances_
# features = X.columns
# importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# # 중요도 순으로 정렬
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
# import matplotlib.pyplot as plt
# import seaborn as sns
# # 시각화
# plt.figure(figsize=(12, 8))
# sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
# plt.title('Feature Importance (LightGBM)')
# plt.tight_layout()
# plt.show()

# exit()
# 바탕화면 : 전력량 feature importance.png 참고
# endregion()

y_all = train['target']

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
    
    # print(x.shape)
    # print(x_test_b.shape)

    # KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = {name: np.zeros(len(x)) for name in get_models()}
    test_preds_all = {name: np.zeros(len(x_test_b)) for name in get_models()}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(x)):
        print(f"       > 🔁 Fold {fold+1}/{N_SPLITS}")
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
        val_scaled = ss.transform(val_b[num_cols])
        val_preds = {name: model.predict(val_scaled) for name, model in models.items()}
        meta_val_X = pd.DataFrame(val_preds)
        val_pred_log = meta_model.predict(meta_val_X)
        val_df.loc[val_b.index, 'pred'] = np.expm1(val_pred_log)
s4_time = time.time()

# --------------------------
# 결과 저장 및 평가
# --------------------------
print("[6] 결과 저장 및 평가")

submission_df = pd.read_csv(data_path + 'sample_submission.csv')
submission_df['answer'] = np.concatenate(pred_list)
os.makedirs(subm_path, exist_ok=True)

val_df = val_df.dropna(subset=['pred'])
final_smape = smape(np.expm1(val_df['target']), val_df['pred'])
save_path = subm_path + f'Energy_08_02_{SEED}.csv'
submission_df.to_csv(save_path, index=False)

print(f"\n✅ 저장 완료: {save_path}")
print(f">>    Train 일사 예측 RMSE : {e_rmse:.6f}")
print(f">>Test 일조/일사 예측 RMSE : {final_rmse:.6f}")
print(f">>   📊 전체 Hold-out SMAPE: {final_smape:.6f}")

print("[7] 결과 기록")
s5_time = time.time()


with open("./Energy/08_submission/08_02_log.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f">>    Train 일사 예측 RMSE : {e_rmse:.6f}\n")
    f.write(f">>Test 일조/일사 예측 RMSE : {final_rmse:.6f}\n")
    f.write(f"✅ 저장 완료: Energy_08_02_{SEED}.csv\n")
    f.write(f"최종 SMAPE 점수 : {final_smape}\n")
    f.write(f">>        전처리 소요 시간 : {np.round(s3_time - s_time, 2)} sec\n")
    f.write(f">>     전체 모델 소요 시간 : {np.round(s4_time - s3_time, 2)} sec\n")
    f.write(f">>            총 소요 시간 : {np.round(s5_time - s_time, 2)} sec\n")
    f.write("="*40 + "\n")

print("\n>>    전처리 소요 시간 :", np.round(s3_time - s_time, 2), 'sec')
print(">> 전체 모델 소요 시간 :", np.round(s4_time - s3_time, 2), 'sec')
print(">>        총 소요 시간 :", np.round(s5_time - s_time, 2), 'sec')
    
print("[8] 완료")