# ========================
# 임포트 및 랜덤 시드 고정
print(f"[11_00] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

seed_file = "./Energy/11_submission/11_00.json"

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

# ========================
# 데이터 로드
print(f"[1] 데이터 로드")
# ========================

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

data_path = './Energy/'
save_path = './Energy/11_submission/'
os.makedirs(save_path, exist_ok=True)

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
    df['시각'] = df['일시'].dt.hour                    # 시각(0~23)
    df['요일'] = df['일시'].dt.dayofweek              # 요일(0=월 ~ 6=일)
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부
    df['근무시간'] = df['시각'].apply(lambda x: 1 if 9 <= x <= 18 else 0)  # 근무시간 여부
    df['SIN_시'] = np.sin(2 * np.pi * df['시각'] / 24)  # 주기적 패턴
    df['COS_시'] = np.cos(2 * np.pi * df['시각'] / 24)

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

train_all = feature_engineering(train)
test_all = feature_engineering(test)
# print(train_all.columns)
# print(test_all.columns)

# 최종 제출을 위한 DataFrame 초기화
submission_df = pd.read_csv(data_path + 'sample_submission.csv')

# submission_df에 건물번호 컬럼 추가 (num_date_time에서 추출)
# num_date_time 컬럼은 '건물번호_일시' 형식으로 되어 있음.
submission_df['건물번호'] = submission_df['num_date_time'].apply(lambda x: int(x.split('_')[0]))

submission_df['answer'] = 0.0 # 초기값 0으로 설정

# 건물별 예측을 위해 건물 번호 리스트 가져오기
building_ids = sorted(train_all['건물번호'].unique())

# 전력 소비량 예측 모델의 전체 SMAPE 합계를 저장할 변수
total_power_smape = 0 # 변수명 변경: total_power_rmse -> total_power_smape
total_evaluated_buildings = 0

# ========================
# 건물별 모델 학습 및 예측
# ========================

for building_id in building_ids:
    print(f"\n[BUILDING {building_id}] 건물별 예측 시작")

    # 건물별 데이터 분리
    train_building = train_all[train_all['건물번호'] == building_id].copy()
    test_building = test_all[test_all['건물번호'] == building_id].copy()

    # ========================
    # test['일조(hr)', '일사(MJ/m2)'] 예측 준비 (건물별)
    # ========================
    train_feature = ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)','연면적(m2)', '냉방면적(m2)', '태양광용량(kW)',
           'ESS저장용량(kWh)', 'PCS용량(kW)', '요일', '월', '일', '주말여부', '근무시간',
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

    print(f"  [BUILDING {building_id}] test['일조(hr)'] 예측 시작")

    X_sunshine = X.copy()
    y_sunshine = Y1.copy()

    valid_sunshine_indices = y_sunshine.dropna().index
    X_sunshine = X_sunshine.loc[valid_sunshine_indices]
    y_sunshine = y_sunshine.loc[valid_sunshine_indices]

    # 해당 건물에 일조 데이터가 아예 없는 경우 건너뛰기
    if X_sunshine.empty or y_sunshine.empty:
        print(f"  [BUILDING {building_id}] 일조(hr) 학습 데이터 부족. 예측 건너뜁니다.")
        test_building['일조(hr)'] = 0.0 # 기본값 설정
    else:
        sun_total_smape = 0 # 변수명 변경: sun_total_rmse -> sun_total_smape
        sunshine_preds_building = np.zeros(test1.shape[0])
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_sunshine, y_sunshine)):
            print(f"    > [BUILDING {building_id} - 일조(hr)] Fold {fold+1}/{n_split}")
            X_train, X_val = X_sunshine.iloc[train_idx], X_sunshine.iloc[val_idx]
            y_train, y_val = y_sunshine.iloc[train_idx], y_sunshine.iloc[val_idx]

            xgb_sunshine = XGBRegressor(random_state=SEED, n_estimators=1000,
                                         learning_rate=0.05, early_stopping_rounds=100, n_jobs=-1)
            xgb_sunshine.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            sunshine_preds_building += xgb_sunshine.predict(test1) / n_split
            sun_val = xgb_sunshine.predict(X_val)
            sunshine_smape = smape(y_val, sun_val) # SMAPE로 변경
            print(f"    > Fold {fold +1} SMAPE : {sunshine_smape:.6f}")
            sun_total_smape += sunshine_smape # 변수명 변경

        sun_frmse = sun_total_smape / n_split # 변수명 변경
        test_building['일조(hr)'] = sunshine_preds_building
        print(f"  [BUILDING {building_id}] 일조(hr) SMAPE: {sun_frmse:.6f}") # SMAPE로 변경

    # ========================
    # test['일사(MJ/m2)'] 예측 (건물별)
    # ========================

    print(f"  [BUILDING {building_id}] test['일사(MJ/m2)'] 예측 시작")

    X_insolation = X.copy()
    y_insolation = Y2.copy()

    valid_insolation_indices = y_insolation.dropna().index
    X_insolation = X_insolation.loc[valid_insolation_indices]
    y_insolation = y_insolation.loc[valid_insolation_indices]

    # 해당 건물에 일사 데이터가 아예 없는 경우 건너뛰기
    if X_insolation.empty or y_insolation.empty:
        print(f"  [BUILDING {building_id}] 일사(MJ/m2) 학습 데이터 부족. 예측 건너킵니다.")
        test_building['일사(MJ/m2)'] = 0.0 # 기본값 설정
    else:
        insolation_total_smape = 0 # 변수명 변경: insolation_total_rmse -> insolation_total_smape
        insolation_preds_building = np.zeros(test1.shape[0]) # test1의 shape를 기준으로 예측값 배열 초기화
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_insolation, y_insolation)):
            print(f"    > [BUILDING {building_id} - 일사(MJ/m2)] Fold {fold+1}/{n_split}")
            X_train, X_val = X_insolation.iloc[train_idx], X_insolation.iloc[val_idx]
            y_train, y_val = y_insolation.iloc[train_idx], y_insolation.iloc[val_idx]

            xgb_insolation = XGBRegressor(random_state=SEED, n_estimators=1000, learning_rate=0.05,
                                         early_stopping_rounds=100, n_jobs=-1)
            xgb_insolation.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            solar_val = xgb_insolation.predict(X_val)
            solar_smape = smape(y_val, solar_val) # SMAPE로 변경
            print(f"    > Fold {fold +1} SMAPE : {solar_smape:.6f}")
            insolation_total_smape += solar_smape # 변수명 변경
            insolation_preds_building += xgb_insolation.predict(test1) / n_split # test1에 대한 예측

        solar_frmse = insolation_total_smape / n_split # 변수명 변경
        test_building['일사(MJ/m2)'] = insolation_preds_building
        print(f"  [BUILDING {building_id}] 일사(MJ/m2) SMAPE: {solar_frmse:.6f}") # SMAPE로 변경


    # ========================
    # 전력 소비량 예측 모델 학습 및 앙상블 (건물별)
    # ========================
    print(f"  [BUILDING {building_id}] 전력 소비량 예측 모델")

    # 전력 소비량 예측을 위한 최종 피처 정의
    features = [
        '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        '시각', '요일', '월', '일', '주말여부', '근무시간', 'SIN_시', 'COS_시',
        '요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '요일_5', '요일_6',
        '불쾌지수', '태양광per냉방면적', 'ESS설치여부', 'PCS설치여부', '설비밀도'
    ]

    # 건물 유형에 따른 One-hot 인코딩된 컬럼 추가 (train과 test 모두 있는 컬럼만)
    # 건물 유형 OHE 컬럼은 train_all에서만 추출하고, 해당 건물의 데이터에는 없을 수 있으므로 전체 컬럼에서 필터링
    building_type_cols_all = [col for col in train_all.columns if '사업시설' in col or '기타' in col or '아파트' in col or '학교' in col or '공공' in col or '백화점' in col or '병원' in col or '호텔' in col or '연구소' in col or 'IDC' in col or '상용' in col] # '상용' 추가
    building_type_cols = [col for col in building_type_cols_all if col in train_building.columns and col in test_building.columns]
    features.extend(building_type_cols)
    features = list(set(features)) # 중복 제거

    # 존재하지 않는 피처 제거 (건물유형 OHE 중 해당 건물에 없는 유형은 제외)
    # X_train_final에서 실제 사용할 수 있는 피처만 남김
    X_train_final = train_building[features].copy()
    y_train_final = train_building['전력소비량(kWh)'].copy()

    # test_building에서 예측된 일조, 일사를 포함하여 최종 test_X 생성
    X_test_final = test_building[features].copy()

    # 로그 변환 컬럼 재적용 (건물별로 다시 적용)
    # 이미 위에 일조/일사 예측 전 X, test1에서 변환되었으므로 여기서는 확인만 필요.
    # 하지만 건물별로 다시 진행하므로, 일조/일사 예측 후 업데이트된 test_building에 적용
    log_col_final = ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '태양광per냉방면적']
    for col in log_col_final:
        if col in X_train_final.columns:
            X_train_final[col] = np.log1p(train_building[col]) # train_building 원본 데이터에서 다시 변환
        if col in X_test_final.columns:
            X_test_final[col] = np.log1p(test_building[col]) # test_building 원본 데이터에서 다시 변환


    # MinMaxScaler 재적용 (건물별로 다시 적용)
    mms_col_final = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '불쾌지수', '일조(hr)', '일사(MJ/m2)'] # 일조, 일사 포함
    mms_final = MinMaxScaler()
    for col in mms_col_final:
        if col in X_train_final.columns:
            X_train_final[col] = mms_final.fit_transform(X_train_final[[col]])
        if col in X_test_final.columns:
            X_test_final[col] = mms_final.transform(X_test_final[[col]])


    xgb_preds_building = np.zeros(X_test_final.shape[0])
    lgbm_preds_building = np.zeros(X_test_final.shape[0])

    # 교차 검증 중 SMAPE 합계를 위한 변수
    fold_power_smapes = [] # 변수명 변경: fold_power_rmses -> fold_power_smapes
    
    # KFold는 X_train_final의 행 개수에 따라 동적으로 조정
    # 데이터 수가 n_split보다 적은 경우 warning 발생 가능
    if len(X_train_final) < n_split:
        current_n_split = len(X_train_final)
        if current_n_split == 0:
            print(f"  [BUILDING {building_id}] 전력 소비량 학습 데이터 부족. 예측 건너닙니다.")
            # 해당 건물에 대한 예측값을 0으로 설정하거나, 특정 기본값으로 채울 수 있습니다.
            submission_df.loc[submission_df['건물번호'] == building_id, 'answer'] = 0.0
            continue # 다음 건물로 넘어감
        print(f"  [BUILDING {building_id}] KFold n_splits를 {current_n_split}로 조정합니다. (데이터 부족)")
        cv_final = KFold(n_splits=current_n_split, random_state=SEED, shuffle=True)
    else:
        cv_final = cv # 원래 n_split 사용

    for fold, (train_idx, val_idx) in enumerate(cv_final.split(X_train_final, y_train_final)):
        print(f"    > [BUILDING {building_id} - 전력 소비량 예측] Fold {fold+1}/{cv_final.n_splits} 시작")
        X_train, X_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
        y_train, y_val = y_train_final.iloc[train_idx], y_train_final.iloc[val_idx]

        # XGBoost 모델 학습
        xgb_model = XGBRegressor(random_state=SEED, n_estimators=1500, learning_rate=0.03,
                                 subsample=0.7, colsample_bytree=0.7, max_depth=8, n_jobs=-1,
                                 early_stopping_rounds=100)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_preds_building += xgb_model.predict(X_test_final) / cv_final.n_splits
        
        # XGBoost 검증 세트 예측 및 SMAPE 계산
        xgb_val_preds = xgb_model.predict(X_val)
        xgb_val_smape = smape(y_val, xgb_val_preds) # SMAPE로 변경

        # LightGBM 모델 학습
        lgbm_model = LGBMRegressor(random_state=SEED, n_estimators=1500, learning_rate=0.03,
                                   subsample=0.7, colsample_bytree=0.7, max_depth=8, n_jobs=-1,
                                   early_stopping_rounds=100, verbosity=-1)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgbm_preds_building += lgbm_model.predict(X_test_final) / cv_final.n_splits
        
        # LightGBM 검증 세트 예측 및 SMAPE 계산
        lgbm_val_preds = lgbm_model.predict(X_val)
        lgbm_val_smape = smape(y_val, lgbm_val_preds) # SMAPE로 변경

        # 앙상블 예측 (검증 세트에 대한) 및 SMAPE 계산
        ensemble_val_preds = (xgb_val_preds + lgbm_val_preds) / 2
        ensemble_val_smape = smape(y_val, ensemble_val_preds) # SMAPE로 변경
        
        fold_power_smapes.append(ensemble_val_smape) # 변수명 변경
        print(f"    > Fold {fold + 1} Ensemble SMAPE: {ensemble_val_smape:.6f}")


    # 최종 앙상블 예측 (가중 평균)
    final_preds_building = (xgb_preds_building + lgbm_preds_building) / 2
    
    # 음수 예측값 0으로 처리 (전력소비량은 음수가 될 수 없음)
    final_preds_building[final_preds_building < 0] = 0

    # 해당 건물에 대한 예측 결과를 submission DataFrame에 업데이트
    submission_df.loc[submission_df['건물번호'] == building_id, 'answer'] = final_preds_building

    # 건물별 평균 SMAPE 계산 및 누적
    if fold_power_smapes: # 데이터가 있어서 SMAPE가 계산된 경우에만
        avg_building_smape = np.mean(fold_power_smapes)
        total_power_smape += avg_building_smape # 변수명 변경
        total_evaluated_buildings += 1
        print(f"  [BUILDING {building_id}] 전력 소비량 SMAPE: {avg_building_smape:.6f}")
    else:
        print(f"  [BUILDING {building_id}] 전력 소비량 예측 완료 (SMAPE 평가 생략 - 데이터 부족).")

# ========================
# 최종 결과 저장
# ========================

# 전체 건물에 대한 평균 SMAPE 출력 (학습에 사용된 건물만 포함)
if total_evaluated_buildings > 0:
    overall_avg_power_smape = total_power_smape / total_evaluated_buildings # 변수명 변경
    print(f"\n[FINAL EVALUATION] 전체 건물 평균 전력 소비량 SMAPE: {overall_avg_power_smape:.6f}")
else:
    print(f"\n[FINAL EVALUATION] 전력 소비량 SMAPE를 평가할 건물이 없습니다.")
    
print(f"\n[6] 최종 결과 저장")
submission_df.drop(columns=['건물번호'], inplace=True) # submission_df에서 '건물번호' 컬럼 제거
import datetime
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{overall_avg_power_smape:.6f}".replace('.', '_')
filename = f"({SEED})_11_00_({today})_({score_str}).csv"
submission_df.to_csv(save_path + filename, index=False)

print(f"!! 저장 완료: {filename} !!")
# 로그 저장
with open(save_path + "11_00.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"Mean SMAPE : {overall_avg_power_smape:.6f}\n")
    f.write("="*40 + "\n")
    
print(f"[11_00] 종료")