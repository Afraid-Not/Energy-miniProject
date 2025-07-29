# ========================
# 임포트 및 랜덤 시드 고정
print(f"[13_01_preprocessing] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping

seed_file = "./Energy/13_submission/(SEED_COUNT)13_01_preprocessing.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 1}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 65 #seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

# ========================
# 데이터 로드
print(f"[1] 데이터 로드")
# ========================

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def peak_weighted_mae(y_true, y_pred):
    weight = y_true / (y_true.max() + 1e-8)
    return np.mean(weight * np.abs(y_true - y_pred))


data_path = './Energy/'
save_path = './Energy/13_submission/'
os.makedirs(save_path, exist_ok=True)
train_save_path = './Energy/13_submission/train_csv/'
test_save_path = './Energy/13_submission/test_csv/'
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)


train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')

# ========================
# building_csv 전처리
print(f"[2] 전처리 시작")
# ========================

building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col :
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)

building_csv = building_csv.fillna(0)

train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

# ========================
# train, test 전처리
# ========================

def feature_engineering(df):
    df = df.copy()

    # ======================
    # 날짜·시간 기반 파생 피처
    # ======================
    df['일시'] = pd.to_datetime(df['일시'])
    df['시각'] = df['일시'].dt.hour                       # 시각(0~23)
    df['요일'] = df['일시'].dt.dayofweek                  # 요일(0=월 ~ 6=일)
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['근무시간'] = df['시각'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)  # 주기적 패턴
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)
    df['정오거리'] = (df['시각'] - 12).abs()  # 12시(정오) 기준 거리
    df['정오거리_INV'] = 1 / (df['정오거리'] + 1)
    df['peak_time'] = df['시각'].apply(lambda x: 1 if 10 <= x <= 16 else 0) # Peak time (11 AM to 4 PM)

    # 요일 One-hot
    dayofweek_ohe = pd.get_dummies(df['요일'], prefix='요일')
    df = pd.concat([df, dayofweek_ohe], axis=1)

    # ======================
    # 건물유형 One-hot
    # ======================
    if '건물유형' in df.columns:
        building_type_ohe = pd.get_dummies(df['건물유형'])
        df = pd.concat([df, building_type_ohe], axis=1)

    # ======================
    # 기상/에너지 관련 파생 피처
    # ======================

    # 불쾌지수(DI, Discomfort Index)
    if '기온(°C)' in df.columns and '습도(%)' in df.columns:
        df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3

    # 냉방 면적 대비 태양광 용량
    if '태양광용량(kW)' in df.columns and '냉방면적(m2)' in df.columns:
        df['태양광per냉방면적'] = df['태양광용량(kW)'] / (df['냉방면적(m2)'] + 1e-6)

    # ESS/PCS 설치 여부
    if 'ESS저장용량(kWh)' in df.columns:
        df['ESS설치여부'] = df['ESS저장용량(kWh)'].apply(lambda x: 1 if x > 0 else 0)
    if 'PCS용량(kW)' in df.columns:
        df['PCS설치여부'] = df['PCS용량(kW)'].apply(lambda x: 1 if x > 0 else 0)

    # ESS+PCS 총용량 대비 연면적 (설비 밀도)
    if 'ESS저장용량(kWh)' in df.columns and 'PCS용량(kW)' in df.columns and '연면적(m2)' in df.columns:
        df['설비밀도'] = (df['ESS저장용량(kWh)'] + df['PCS용량(kW)']) / (df['연면적(m2)'] + 1e-6)

    return df

def predict_train_solar(train_df, zero_bnos, kfold=5):
    print(f"\n[일사 보간] 앙상블 모델 예측 시작")

    def train_models(df_filtered, label='일사(MJ/m2)'):
        df_filtered['hour'] = pd.to_datetime(df_filtered['일시']).dt.hour
        train_data = df_filtered[
            (~df_filtered['건물번호'].isin(zero_bnos)) &
            (df_filtered[label] > 0)
        ].copy()
        X = train_data[['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']].values
        y = train_data[label].values

        # KFold 성능 평가
        print(f"\n      > {label} KFold={kfold} 성능 평가")
        kf = KFold(n_splits=kfold, shuffle=True, random_state=SEED)
        scores = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            models = [
                LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbosity=-1),
                XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbosity=0),
                CatBoostRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbose=0)
            ]
            preds = [model.fit(X_train, y_train).predict(X_valid) for model in models]
            pred_mean = np.mean(preds, axis=0)

            score = peak_weighted_mae(y_valid, pred_mean)
            scores.append(score)
            print(f"         > Fold {fold+1}: {score:.6f}")

        score_prepro = np.mean(scores)
        print(f"      > 평균 PWMAE: {score_prepro:.6f}")

        # 전체 학습 후 반환
        model_lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbosity=-1)
        model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbosity=0)
        model_cat = CatBoostRegressor(n_estimators=1000, learning_rate=0.05, random_state=SEED, verbose=0)

        model_lgb.fit(X, y)
        model_xgb.fit(X, y)
        model_cat.fit(X, y)

        return model_lgb, model_xgb, model_cat

    # 모델 학습
    ensemble_models = train_models(train_df.copy(), label='일사(MJ/m2)')

    # 보간 대상 전체
    df_filled = train_df.copy()
    df_filled['hour'] = pd.to_datetime(df_filled['일시']).dt.hour

    for bno in zero_bnos:
        print(f"      > [건물 {bno}] 일사(MJ/m2) 예측 중...")
        target = df_filled[(df_filled['건물번호'] == bno) & (df_filled['일사(MJ/m2)'] == 0)].copy()

        if target.empty:
            continue

        # 시간대 보정
        target['hour'] = pd.to_datetime(target['일시']).dt.hour
        target.loc[(target['hour'] < 5) | (target['hour'] > 21), '일사(MJ/m2)'] = 0.0

        condition = (target['hour'] >= 5) & (target['hour'] <= 21)
        target_idx = target[condition].index
        X_target = target.loc[target_idx, ['기온(°C)', '습도(%)', '풍속(m/s)', 'hour']].values

        # 앙상블 모델로 예측
        preds = np.mean([model.predict(X_target) for model in ensemble_models], axis=0)

        df_filled.loc[target_idx, '일사(MJ/m2)'] = np.clip(preds, 0, None)

    return df_filled.drop(columns='hour')


train_all = feature_engineering(train)
test_all = feature_engineering(test)

###########################################################################
# train 일사 보간
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

train_all_fixed = predict_train_solar(train_all, zero_bnos, kfold=5)

submission_df = pd.read_csv(data_path + 'sample_submission.csv')

# submission_df에 건물번호 컬럼 추가 (num_date_time에서 추출)
# num_date_time 컬럼은 '건물번호_일시' 형식으로 되어 있음.
submission_df['건물번호'] = submission_df['num_date_time'].apply(lambda x: int(x.split('_')[0]))

submission_df['answer'] = 0.0 # 초기값 0으로 설정

# 건물별 예측을 위해 건물 번호 리스트 가져오기
building_ids = sorted(train_all_fixed['건물번호'].unique())

# 전력 소비량 예측 모델의 전체 SMAPE 합계를 저장할 변수
total_power_smape = 0
total_evaluated_buildings = 0

peak_weight = 0.1

# 일조 / 일사 예측의 전체 MAE를 저장할 변수 초기화
overall_sunshine_mae = 0.0
overall_insolation_mae = 0.0
count_sunshine_evaluated = 0
count_insolation_evaluated = 0
# ========================
# 건물별 모델 학습 및 예측
# ========================

for building_id in building_ids:
    print(f"\n[BUILDING {building_id}] 건물별 예측 시작")

    # 건물별 데이터 분리
    train_building = train_all_fixed[train_all_fixed['건물번호'] == building_id].copy()
    test_building = test_all[test_all['건물번호'] == building_id].copy()

    # ========================
    # test['일조(hr)', '일사(MJ/m2)'] 예측 준비 (건물별)
    # ========================
    train_feature = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)','연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
                     'ESS저장용량(kWh)', 'PCS용량(kW)', '요일', '월', '일', '주말여부', '근무시간', '시각',
                     'SIN_시', 'COS_시', '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5',
                     '요일_6', 'IDC(전화국)', '건물기타', '공공', '백화점', '병원', '상용', '아파트', '연구소', '학교',
                     '호텔', '불쾌지수', '태양광per냉방면적', 'ESS설치여부', 'PCS설치여부', '설비밀도']

    # train_building에서 해당 feature만 추출
    X = train_building[train_feature].copy()
    Y1 = train_building['일조(hr)'].copy()
    Y2 = train_building['일사(MJ/m2)'].copy()
    test1 = test_building[train_feature].copy()

    log_col = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '태양광per냉방면적']
    mms_col = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '불쾌지수']

    # 로그 변환 및 스케일링
    X.loc[:, log_col] = np.log1p(X[log_col])
    test1.loc[:, log_col] = np.log1p(test1[log_col])
    mms = MinMaxScaler()
    X.loc[:, mms_col] = mms.fit_transform(X[mms_col])
    test1.loc[:, mms_col] = mms.transform(test1[mms_col])

    # ========================
    # test['일조(hr)'] 예측 (건물별)
    # ========================

    n_split = 5
    cv = KFold(n_splits=n_split, random_state=SEED, shuffle=True)
    print(f"      test['일조(hr)'] 예측 시작")

    X_sunshine = X.copy()
    y_sunshine = Y1.copy()

    valid_sunshine_indices = y_sunshine.dropna().index
    X_sunshine = X_sunshine.loc[valid_sunshine_indices]
    y_sunshine = y_sunshine.loc[valid_sunshine_indices]

    if X_sunshine.empty or y_sunshine.empty:
        print(f"      일조(hr) 학습 데이터 부족. 예측 건너뜁니다.")
        test_building['일조(hr)'] = 0.0
    else:
        sun_total_mae = 0
        sunshine_preds_building = np.zeros(test1.shape[0])

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_sunshine, y_sunshine)):
            print(f"         > [일조(hr)] Fold {fold+1}/{n_split}")
            X_train, X_val = X_sunshine.iloc[train_idx], X_sunshine.iloc[val_idx]
            y_train, y_val = y_sunshine.iloc[train_idx], y_sunshine.iloc[val_idx]

            # ==========================
            # 1차 모델 앙상블 예측
            # ==========================

            # XGBoost
            xgb = XGBRegressor(
                random_state=SEED, n_estimators=1200, learning_rate=0.03,
                max_depth=6, subsample=0.85, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, gamma=0,
                min_child_weight=5, early_stopping_rounds=100,
                n_jobs=-1, verbosity=0
            )
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred_xgb_val = xgb.predict(X_val)
            pred_xgb_test = xgb.predict(test1)

            # LightGBM
            lgb = LGBMRegressor(
                random_state=SEED, n_estimators=1200, learning_rate=0.03,
                max_depth=7, num_leaves=64, subsample=0.85, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, min_child_samples=20,
                n_jobs=-1, verbosity=-1
            )
            lgb.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
            pred_lgb_val = lgb.predict(X_val)
            pred_lgb_test = lgb.predict(test1)

            # CatBoost
            cb = CatBoostRegressor(
                random_seed=SEED, iterations=1200, learning_rate=0.03,
                depth=6, l2_leaf_reg=3.0, bagging_temperature=1.0,
                random_strength=1.0, loss_function='MAE',
                od_type='Iter', od_wait=100, verbose=False
            )
            cb.fit(X_train, y_train, eval_set=(X_val, y_val))
            pred_cb_val = cb.predict(X_val)
            pred_cb_test = cb.predict(test1)

            # 앙상블 평균
            pred_val_ensemble = (pred_xgb_val + pred_lgb_val + pred_cb_val) / 3
            pred_test_ensemble = (pred_xgb_test + pred_lgb_test + pred_cb_test) / 3

            # ==========================
            # 2차 보정: 잔차 모델
            # ==========================
            residual = y_val - pred_val_ensemble
            residual_model = LGBMRegressor(
                random_state=SEED, n_estimators=300, learning_rate=0.05,
                max_depth=5, num_leaves=32, reg_alpha=0.5, reg_lambda=1.0,
                verbosity=-1
            )
            residual_model.fit(X_val, residual)
            residual_val_pred = residual_model.predict(X_val)
            residual_test_pred = residual_model.predict(test1)

            # 최종 예측 (앙상블 + 잔차 보정)
            final_val = pred_val_ensemble + residual_val_pred
            final_test = pred_test_ensemble + residual_test_pred

            # 일조(hr) - 최종 Peak_Weightes_MAE 보고
            fold_mae = peak_weighted_mae(y_val, final_val)
            print(f"         > 최종 Peak_Weightes_MAE : {fold_mae:.6f}")

            sun_total_mae += fold_mae
            sunshine_preds_building += final_test / n_split

        avg_sunshine_mae_building = sun_total_mae / n_split
        test_building['일조(hr)'] = sunshine_preds_building
        print(f"      [일조(hr)] MAE (건물별 평균): {avg_sunshine_mae_building:.6f}")
        overall_sunshine_mae += avg_sunshine_mae_building
        count_sunshine_evaluated += 1
        
        test_all.loc[test_all['건물번호'] == building_id, '일조(hr)'] = sunshine_preds_building

    # ========================
    # test['일사(MJ/m2)'] 예측 (건물별)
    # ========================
    print(f"\n      test['일사(MJ/m2)'] 예측 시작")

    X_insolation = X.copy()
    y_insolation = Y2.copy()

    valid_insolation_indices = y_insolation.dropna().index
    X_insolation = X_insolation.loc[valid_insolation_indices]
    y_insolation = y_insolation.loc[valid_insolation_indices]

    if X_insolation.empty or y_insolation.empty:
        print(f"      일사(MJ/m2) 학습 데이터 부족. 예측 건너뜁니다.")
        test_building['일사(MJ/m2)'] = 0.0
    else:
        insolation_total_mae = 0
        insolation_preds_building = np.zeros(test1.shape[0])

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_insolation, y_insolation)):
            print(f"         > [일사(MJ/m2)] Fold {fold+1}/{n_split}")
            X_train, X_val = X_insolation.iloc[train_idx], X_insolation.iloc[val_idx]
            y_train, y_val = y_insolation.iloc[train_idx], y_insolation.iloc[val_idx]

            # === 앙상블 기반 1차 예측 ===
            xgb = XGBRegressor(
                random_state=SEED,
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                gamma=0,
                min_child_weight=5,
                early_stopping_rounds=100,
                n_jobs=-1,
                verbosity=0
            )
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred_xgb_val = xgb.predict(X_val)
            pred_xgb_test = xgb.predict(test1)

            lgb = LGBMRegressor(
                random_state=SEED,
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=7,
                num_leaves=64,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=2.0,
                min_child_samples=20,
                n_jobs=-1,
                verbosity=-1
            )
            lgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(100, verbose=False),
                    log_evaluation(0)
                ]
            )
            pred_lgb_val = lgb.predict(X_val)
            pred_lgb_test = lgb.predict(test1)

            cb = CatBoostRegressor(
                random_seed=SEED,
                iterations=1200,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                bagging_temperature=1.0,
                random_strength=1.0,
                loss_function='MAE',
                od_type='Iter',
                od_wait=100,
                verbose=False
            )
            cb.fit(X_train, y_train, eval_set=(X_val, y_val))
            pred_cb_val = cb.predict(X_val)
            pred_cb_test = cb.predict(test1)

            # 앙상블 평균
            pred_val_ensemble = (pred_xgb_val + pred_lgb_val + pred_cb_val) / 3
            pred_test_ensemble = (pred_xgb_test + pred_lgb_test + pred_cb_test) / 3

            # === 2차 보정: 잔차 모델 ===
            residual = y_val - pred_val_ensemble
            residual_model = LGBMRegressor(
                random_state=SEED, n_estimators=300, learning_rate=0.05,
                max_depth=5, num_leaves=32, reg_alpha=0.5, reg_lambda=1.0,
                verbosity=-1
            )
            residual_model.fit(X_val, residual)

            # 잔차 보정 예측
            residual_pred_val = residual_model.predict(X_val)
            residual_pred_test = residual_model.predict(test1)

            # 최종 예측: 앙상블 + 잔차 보정
            final_val = pred_val_ensemble + residual_pred_val
            final_test = pred_test_ensemble + residual_pred_test

            # 일사(MJ/m2) - 최종 Peak_Weightes_MAE 보고
            fold_mae = peak_weighted_mae(y_val, final_val)
            print(f"         > 최종 Peak_Weightes_MAE : {fold_mae:.6f}")

            insolation_total_mae += fold_mae
            insolation_preds_building += final_test / n_split

        avg_insolation_mae_building = insolation_total_mae / n_split
        test_building['일사(MJ/m2)'] = insolation_preds_building
        print(f"      [일사(MJ/m2)] MAE (건물별 평균): {avg_insolation_mae_building:.6f}")
        overall_insolation_mae += avg_insolation_mae_building
        count_insolation_evaluated += 1
        test_all.loc[test_all['건물번호'] == building_id, '일사(MJ/m2)'] = insolation_preds_building
# Calculate overall average MAEs
final_avg_sunshine_mae = overall_sunshine_mae / count_sunshine_evaluated if count_sunshine_evaluated > 0 else 0.0
final_avg_insolation_mae = overall_insolation_mae / count_insolation_evaluated if count_insolation_evaluated > 0 else 0.0

if '시각' not in test_all.columns:
    test_all['일시'] = pd.to_datetime(test_all['일시'])
    test_all['시각'] = test_all['일시'].dt.hour

# 일조(hr): 20시~06시는 0
test_all.loc[test_all['시각'].isin([0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23]), '일조(hr)'] = 0.0

# 일사(MJ/m2): 21시~05시는 0
test_all.loc[test_all['시각'].isin([0, 1, 2, 3, 4, 5, 21, 22, 23]), '일사(MJ/m2)'] = 0.0

# 음수 값 clip
test_all['일조(hr)'] = test_all['일조(hr)'].clip(lower=0)
test_all['일사(MJ/m2)'] = test_all['일사(MJ/m2)'].clip(lower=0)

test_filename = f'13_01_test_SEED{SEED}_no_peak.csv'
train_filename = f'13_01_train_SEED{SEED}_no_peak.csv'

# Save preprocessed dataframes
train_all_fixed.to_csv(train_save_path  + train_filename, index=False)
test_all.to_csv(test_save_path +test_filename, index=False)
print(f"[4] 전처리된 데이터 저장 완료: {test_filename}")

print(f"\n[5] 전체 평균 일조(hr) Peak_Weightes_MAE: {final_avg_sunshine_mae:.6f}")
print(f"[5] 전체 평균 일사(MJ/m2) Peak_Weightes_MAE: {final_avg_insolation_mae:.6f}")

with open(save_path + "(LOG)13_01_preprocessing.txt", "a") as f:
    f.write(f"<파일명 : 13_01_preprocessing_no_peak_train_predict.py>\n")
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"{train_filename}\n")
    f.write(f"{test_filename}\n")
    f.write(f"Overall Average 일조(hr) Peak_Weightes_MAE : {final_avg_sunshine_mae:.6f}\n")
    f.write(f"Overall Average 일사(MJ/m2) Peak_Weightes_MAE : {final_avg_insolation_mae:.6f}\n")
    f.write("="*40 + "\n")
    
print(f"[12_03_preprocessing] 종료")