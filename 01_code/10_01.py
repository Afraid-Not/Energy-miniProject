import pandas as pd
import numpy as np
import warnings
import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import os
import json
import random

warnings.filterwarnings('ignore')

seed_file = "./Energy/10_00_submission/10_01_seed.json"

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

# --- 1. 데이터 로드 및 초기 병합 ---
data_path = './Energy/'
save_path = './Energy/10_00_submission/' # 저장 경로 정의

train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')
submit = pd.read_csv(data_path + 'sample_submission.csv')

for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_csv[col] = building_csv[col].replace('-', 0).astype(float)

train = pd.merge(train_csv, building_csv, on=['건물번호'], how='left')
test = pd.merge(test_csv, building_csv, on=['건물번호'], how='left')
print("데이터 로드 및 초기 병합 완료.")
print(f"Train 데이터셋 크기: {train.shape}")
print(f"Test 데이터셋 크기: {test.shape}")

# --- '-' 값을 0으로 변환 및 float 타입으로 변경 ---
cols_to_convert_str_to_num = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '전력소비량(kWh)']

for col in cols_to_convert_str_to_num:
    if col in train.columns:
        train[col] = train[col].replace('-', '0').astype(float)
for col in cols_to_convert_str_to_num:
    if col in test.columns and col != '전력소비량(kWh)':
        test[col] = test[col].replace('-', '0').astype(float)
print("특정 컬럼 '-' 값을 0으로 변환 및 float 타입으로 변경 완료.")

# --- 2. 전처리 1: 파생 피처 생성 ---
def feature_engineering_1(df):
    df['parsed_datetime'] = pd.to_datetime(df['일시'])
    df['연'] = df['parsed_datetime'].dt.year
    df['월'] = df['parsed_datetime'].dt.month
    df['일'] = df['parsed_datetime'].dt.day
    df['시간'] = df['parsed_datetime'].dt.hour
    df['요일'] = df['parsed_datetime'].dt.dayofweek
    df['sin_hour'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['시간'] / 24)
    df['주말 여부'] = ((df['요일'] == 5) | (df['요일'] == 6)).astype(int)
    df['공휴일'] = 0
    df.loc[(df['월'] == 6) & (df['일'] == 6), '공휴일'] = 1
    df.loc[(df['월'] == 8) & (df['일'] == 15), '공휴일'] = 1
    df['불쾌지수'] = 0.81 * df['기온(°C)'] + 0.01 * df['습도(%)'] * (0.99 * df['기온(°C)'] - 14.3) + 46.3
    df['냉방도일'] = df['기온(°C)'].apply(lambda x: max(0, x - 26))
    df['체감온도'] = 13.12 + 0.6215 * df['기온(°C)'] - 11.37 * (df['풍속(m/s)']**0.16) + 0.3965 * df['기온(°C)'] * (df['풍속(m/s)']**0.16)
    df.loc[df['풍속(m/s)'] == 0, '체감온도'] = df['기온(°C)']
    df['근무시간여부'] = ((df['시간'] >= 9) & (df['시간'] < 18)).astype(int)
    df['강수량변화율'] = df.groupby('건물번호')['강수량(mm)'].diff().fillna(0)
    df['is_rain_x_working_hours'] = df['강수량(mm)'].apply(lambda x: 1 if x > 0 else 0) * df['근무시간여부']
    df = pd.get_dummies(df, columns=['건물유형', '요일'], drop_first=True)
    return df.drop(columns=['parsed_datetime'])
from sklearn.model_selection import TimeSeriesSplit
train = feature_engineering_1(train)
test = feature_engineering_1(test)
print("전처리 1 (파생 피처 생성) 완료.")

# --- 3. train '일사(MJ/m2)' 예측 보조 모델 ---
def train_solar_radiation_imputation(df):
    imputed_df = df.copy()
    imputed_df['is_day'] = ((imputed_df['시간'] >= 6) & (imputed_df['시간'] <= 21)).astype(int)
    imputed_df['일사(MJ/m2)_original'] = imputed_df['일사(MJ/m2)']
    
    # 낮 시간(06시~21시)에 일사가 0인 경우를 np.nan으로 변경하여 예측 대상으로 만듦
    imputed_df.loc[(imputed_df['일사(MJ/m2)'] == 0) & (imputed_df['is_day'] == 1), '일사(MJ/m2)'] = np.nan
    
    features_solar = ['기온(°C)', '강수량(mm)', '풍속(m/s)', 'sin_hour', 'cos_hour', '일조(hr)',
                      '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '전력소비량(kWh)']
    target_solar = '일사(MJ/m2)'
    
    rmse_1 =[]

    for building_type_col in [col for col in imputed_df.columns if '건물유형_' in col]:
        building_type_value = 1
        type_df_train = imputed_df[imputed_df[building_type_col] == building_type_value].copy()
        
        # 모델 훈련에 사용할 NaN 없는 데이터 추출
        type_df_train_cleaned = type_df_train.dropna(subset=[target_solar])

        X_train_solar = type_df_train_cleaned[features_solar]
        y_train_solar = type_df_train_cleaned[target_solar]
        X_predict_solar = type_df_train[type_df_train[target_solar].isna()][features_solar]

        # 훈련 데이터가 비어있지 않고 예측할 데이터가 있는 경우에만 모델 훈련
        if not X_train_solar.empty and not X_predict_solar.empty:
            X_temp_solar, X_eval_solar, y_temp_solar, y_eval_solar = train_test_split(
                X_train_solar, y_train_solar, test_size=0.2, random_state=SEED
            )
            tscv = TimeSeriesSplit(n_splits=3)
            oof_preds_xgb = np.zeros(X_temp_solar.shape[0])
            oof_preds_lgb = np.zeros(X_temp_solar.shape[0])
            test_preds_xgb = []
            test_preds_lgb = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_temp_solar, y_temp_solar)):
                X_train_fold, X_val_fold = X_temp_solar.iloc[train_idx], X_temp_solar.iloc[val_idx]
                y_train_fold, y_val_fold = y_temp_solar.iloc[train_idx], y_temp_solar.iloc[val_idx]

                # 타겟에 NaN/Inf 확인 (학습 전 최종 점검)
                if y_train_fold.isnull().any() or not np.isfinite(y_train_fold).all() or \
                   y_val_fold.isnull().any() or not np.isfinite(y_val_fold).all():
                    print(f"  [Train 일사] 건물유형 {building_type_col} 값 {building_type_value}, Fold {fold+1}: NaN/Inf in target. 이 폴드 건너뛰기.")
                    continue

                # XGBoost 모델
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, tree_method='hist', early_stopping_rounds=50)
                xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                oof_preds_xgb[val_idx] = xgb_model.predict(X_val_fold)
                test_preds_xgb.append(xgb_model.predict(X_predict_solar))

                # LightGBM 모델
                lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, early_stopping_round=50, verbose=-1)
                lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
                oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
                test_preds_lgb.append(lgb_model.predict(X_predict_solar))

                print(f"  [Train 일사] {building_type_col} - Fold {fold+1} XGBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_xgb[val_idx])):.4f}")
                print(f"  [Train 일사] {building_type_col} - Fold {fold+1} LightGBM RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_lgb[val_idx])):.4f}")

            # 모든 폴드가 스킵되어 예측값이 없는 경우 처리
            if not test_preds_xgb or not test_preds_lgb:
                print(f"  [Train 일사] {building_type_col} 값 {building_type_value}: 모든 폴드에서 유효한 예측값 생성 실패. 이 건물 유형의 일사량 예측 건너뛰기.")
                # 해당 건물 유형의 일사량은 NaN으로 남겨두고 최종 단계에서 0으로 채워질 것임.
                continue

            ensemble_oof_preds = (oof_preds_xgb + oof_preds_lgb) / 2
            rmse_oof = np.sqrt(mean_squared_error(y_temp_solar, ensemble_oof_preds))
            print(f"[Train 일사] {building_type_col} 값 {building_type_value} 앙상블 OOF RMSE: {rmse_oof:.4f}")

            final_xgb_eval_preds = xgb_model.predict(X_eval_solar)
            final_lgb_eval_preds = lgb_model.predict(X_eval_solar)
            ensemble_eval_preds = (final_xgb_eval_preds + final_lgb_eval_preds) / 2
            rmse_eval = np.sqrt(mean_squared_error(y_eval_solar, ensemble_eval_preds))
            print(f"[Train 일사] {building_type_col} 값 {building_type_value} 앙상블 평가 세트 RMSE: {rmse_eval:.4f}")
            rmse_1.append(rmse_eval)
            
            predicted_solar_radiation = (np.mean(test_preds_xgb, axis=0) + np.mean(test_preds_lgb, axis=0)) / 2
            
            # 예측된 일사량을 원래 위치에 채워넣기
            imputed_df.loc[imputed_df[building_type_col] == building_type_value, target_solar] = imputed_df.loc[
                imputed_df[building_type_col] == building_type_value, target_solar].fillna(pd.Series(predicted_solar_radiation, index=X_predict_solar.index))
            
    # 밤 시간(21시 이후 ~ 6시 이전)의 일사는 모두 0으로 설정
    imputed_df.loc[(imputed_df['시간'] > 21) | (imputed_df['시간'] < 6), '일사(MJ/m2)'] = 0
    # 일사량이 음수일 경우 0으로 설정
    imputed_df.loc[imputed_df['일사(MJ/m2)'] < 0, '일사(MJ/m2)'] = 0

    # 마지막으로, 모델 훈련이 불가능하여 채워지지 않은 모든 NaN 값을 0으로 채움
    initial_nan_count = imputed_df['일사(MJ/m2)'].isnull().sum()
    if initial_nan_count > 0:
        imputed_df['일사(MJ/m2)'].fillna(0, inplace=True)
        print(f"  [최종] 예측 불가능하여 남아있던 일사(MJ/m2) NaN 값 {initial_nan_count}개를 0으로 채웠습니다.")
    
    return imputed_df.drop(columns=['is_day', '일사(MJ/m2)_original']), np.mean(np.array(rmse_1)), features_solar

train, rmse_1, feature_1 = train_solar_radiation_imputation(train)
print("\nTrain '일사(MJ/m2)' 예측 보조 모델 훈련 및 결측치 처리 완료.")
print(f"Train '일사(MJ/m2)' NaN 개수: {train['일사(MJ/m2)'].isnull().sum()}")
print(f"Train '일사(MJ/m2)' RMSE : {rmse_1:.4f}")

# exit()
# --- 4. test '일조(hr)', '일사(MJ/m2)' 예측 보조 모델 ---
def test_sunshine_solar_imputation(train_df, test_df):
    imputed_test_df = test_df.copy()
    
    imputed_test_df['is_day'] = ((imputed_test_df['시간'] >= 6) & (imputed_test_df['시간'] <= 21)).astype(int)
    train_df['연면적(m2)'] = np.log1p(train_df['연면적(m2)'])
    test_df['연면적(m2)'] = np.log1p(test_df['연면적(m2)'])
    features_final_test_solar = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'sin_hour', 'cos_hour', '연면적(m2)']
    
    # 예측된 값을 저장할 컬럼을 미리 0.0으로 초기화
    imputed_test_df['일조(hr)'] = 0.0
    imputed_test_df['일사(MJ/m2)'] = 0.0
    rmse_2 = []
    rmse_3 = []
    print("\nTest '일조(hr)', '일사(MJ/m2)' 예측 보조 모델 훈련 시작...")

    for building_num in sorted(train_df['건물번호'].unique()):
        train_building = train_df[train_df['건물번호'] == building_num].copy()
        test_building = imputed_test_df[imputed_test_df['건물번호'] == building_num].copy()

        # 학습 데이터에서 타겟 변수에 NaN이 있는 행 제거 (이전에도 있었지만, 명시적 강조)
        train_building_for_model = train_building.copy()
        initial_rows = train_building_for_model.shape[0]
        train_building_for_model.dropna(subset=['일조(hr)', '일사(MJ/m2)'], inplace=True)
        dropped_rows = initial_rows - train_building_for_model.shape[0]
        if dropped_rows > 0:
            print(f"  [건물 {building_num}] '일조(hr)' 또는 '일사(MJ/m2)'의 NaN 값으로 인해 {dropped_rows}개 행 제거됨.")
        
        # 모델 학습에 사용할 데이터가 비어있는 경우 스킵
        if train_building_for_model.empty:
            print(f"  [건물 {building_num}] NaN 제거 후 유효한 학습 데이터가 없어 '일조(hr)' / '일사(MJ/m2)' 예측을 건너뜀.")
            continue

        X_train_building = train_building_for_model[features_final_test_solar]
        y_train_sunshine = train_building_for_model['일조(hr)']
        y_train_solar = train_building_for_model['일사(MJ/m2)']
        X_test_building_orig = test_building[features_final_test_solar] # test 데이터 피처 원본

        # X_train_building, X_test_building_orig 의 NaN/Inf 처리 (사전 처리)
        X_train_building = X_train_building.replace([np.inf, -np.inf], np.nan).fillna(X_train_building.median())
        X_test_building_orig = X_test_building_orig.replace([np.inf, -np.inf], np.nan).fillna(X_test_building_orig.median())


        X_temp_sun, X_eval_sun, y_temp_sun, y_eval_sun = train_test_split(
            X_train_building, y_train_sunshine, test_size=0.2, random_state=SEED
        )
        X_temp_solar_b, X_eval_solar_b, y_temp_solar_b, y_eval_solar_b = train_test_split(
            X_train_building, y_train_solar, test_size=0.2, random_state=SEED
        )

        tscv = TimeSeriesSplit(n_splits=3)

        for target_col, y_temp, y_eval in [('일조(hr)', y_temp_sun, y_eval_sun), ('일사(MJ/m2)', y_temp_solar_b, y_eval_solar_b)]:
            # 현재 타겟에 맞는 X_temp를 할당 (이전 오류 수정 반영)
            current_X_temp = X_temp_sun if target_col == '일조(hr)' else X_temp_solar_b

            # 타겟 변수에 NaN/Inf가 여전히 있는지 확인
            if y_temp.isnull().any() or not np.isfinite(y_temp).all() or \
               y_eval.isnull().any() or not np.isfinite(y_eval).all():
                print(f"  [건물 {building_num}] 경고: {target_col} 타겟에 NaN/Inf가 발견되어 이 타겟의 학습/검증을 건너뜁니다.")
                continue

            # oof_preds 초기화 시 current_X_temp.shape[0] 사용
            oof_preds_xgb = np.zeros(current_X_temp.shape[0])
            oof_preds_lgb = np.zeros(current_X_temp.shape[0])
            oof_preds_cat = np.zeros(current_X_temp.shape[0])
            test_preds_xgb = []
            test_preds_lgb = []
            test_preds_cat = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(current_X_temp, y_temp)):
                X_train_fold, X_val_fold = current_X_temp.iloc[train_idx], current_X_temp.iloc[val_idx]
                y_train_fold, y_val_fold = y_temp.iloc[train_idx], y_temp.iloc[val_idx]

                # 폴드 데이터에 NaN/Inf 최종 확인 및 건너뛰기
                if y_train_fold.isnull().any() or not np.isfinite(y_train_fold).all() or \
                   y_val_fold.isnull().any() or not np.isfinite(y_val_fold).all():
                    print(f"  [건물 {building_num}] Fold {fold+1} {target_col}: y_train_fold 또는 y_val_fold에 NaN/Inf가 있어 이 폴드를 건너뜁니다.")
                    continue
                
                # X_train_fold, X_val_fold에 NaN/Inf가 있을 경우 중앙값으로 채우기 (이미 X_train_building에서 처리되었지만, 만약의 경우 대비)
                if X_train_fold.isnull().any().any() or not np.isfinite(X_train_fold).all().all():
                     X_train_fold = X_train_fold.replace([np.inf, -np.inf], np.nan).fillna(X_train_fold.median())
                if X_val_fold.isnull().any().any() or not np.isfinite(X_val_fold).all().all():
                     X_val_fold = X_val_fold.replace([np.inf, -np.inf], np.nan).fillna(X_val_fold.median())
                
                # test 데이터 피처는 이미 X_test_building_orig에서 처리됨
                X_test_building_cleaned_for_pred = X_test_building_orig


                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, tree_method='hist', early_stopping_rounds=50)
                xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                oof_preds_xgb[val_idx] = xgb_model.predict(X_val_fold)
                test_preds_xgb.append(xgb_model.predict(X_test_building_cleaned_for_pred))

                lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, early_stopping_round=50, verbose=-1)
                lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
                oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
                test_preds_lgb.append(lgb_model.predict(X_test_building_cleaned_for_pred))

                cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, random_seed=SEED, verbose=0, early_stopping_rounds=50)
                cat_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
                oof_preds_cat[val_idx] = cat_model.predict(X_val_fold)
                test_preds_cat.append(cat_model.predict(X_test_building_cleaned_for_pred))

                print(f"  [Test {target_col}] 건물 {building_num}, Fold {fold+1} XGBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_xgb[val_idx])):.4f}")
                print(f"  [Test {target_col}] 건물 {building_num}, Fold {fold+1} LightGBM RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_lgb[val_idx])):.4f}")
                print(f"  [Test {target_col}] 건물 {building_num}, Fold {fold+1} CatBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_cat[val_idx])):.4f}")
            
            # 모든 폴드가 스킵되어 test_preds_xgb 등이 비어있는 경우 처리
            # 해당 target_col에 대한 모든 test_preds_list가 비어있으면 건너뛰기
            if not test_preds_xgb or not test_preds_lgb or not test_preds_cat:
                print(f"  [Test {target_col}] 건물 {building_num}: 모든 폴드에서 유효한 예측값 생성 실패. 해당 타겟 건너뛰기.")
                imputed_test_df.loc[imputed_test_df['건물번호'] == building_num, target_col] = np.nan # 예측 실패 시 NaN으로 남겨두기
                continue # 다음 target_col로 넘어감 (일조 -> 일사)

            meta_X_train = np.column_stack((oof_preds_xgb, oof_preds_lgb, oof_preds_cat))
            meta_X_test = np.column_stack((np.mean(test_preds_xgb, axis=0), np.mean(test_preds_lgb, axis=0), np.mean(test_preds_cat, axis=0)))
            
            ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=3, scoring='neg_root_mean_squared_error')
            ridge_model.fit(meta_X_train, y_temp)

            predicted_values = ridge_model.predict(meta_X_test)
            
            final_xgb_eval_preds = xgb_model.predict(X_eval_sun if target_col == '일조(hr)' else X_eval_solar_b)
            final_lgb_eval_preds = lgb_model.predict(X_eval_sun if target_col == '일조(hr)' else X_eval_solar_b)
            final_cat_eval_preds = cat_model.predict(X_eval_sun if target_col == '일조(hr)' else X_eval_solar_b)
            
            meta_X_eval = np.column_stack((final_xgb_eval_preds, final_lgb_eval_preds, final_cat_eval_preds))
            ridge_eval_preds = ridge_model.predict(meta_X_eval)
            rmse_eval = np.sqrt(mean_squared_error(y_eval, ridge_eval_preds))
            print(f"[Test {target_col}] 건물 {building_num} 스택킹 평가 세트 RMSE: {rmse_eval:.4f}")
            
            if target_col == '일조(hr)' :
                rmse_2.append(rmse_eval)
            else :
                rmse_3.append(rmse_eval)
            
            imputed_test_df.loc[imputed_test_df['건물번호'] == building_num, target_col] = predicted_values
            
            imputed_test_df.loc[((imputed_test_df['시간'] > 21) | (imputed_test_df['시간'] < 6)) & (imputed_test_df['건물번호'] == building_num), target_col] = 0
            imputed_test_df.loc[(imputed_test_df['건물번호'] == building_num) & (imputed_test_df[target_col] < 0), target_col] = 0

    return imputed_test_df.drop(columns=['is_day']), np.mean(np.array(rmse_2)), np.mean(np.array(rmse_3)), features_final_test_solar

test, rmse_2, rmse_3, feature_2  = test_sunshine_solar_imputation(train, test)
print("\nTest '일조(hr)', '일사(MJ/m2)' 예측 보조 모델 훈련 및 결측치 처리 완료.")
print(f"Test '일조(hr)' NaN 개수: {test['일조(hr)'].isnull().sum()}")
print(f"Test '일사(MJ/m2)' NaN 개수: {test['일사(MJ/m2)'].isnull().sum()}")
print(f"Test '일조(hr)' RMSE : {rmse_2:.4f}")
print(f"Test '일사(MJ/m2)' RMSE : {rmse_3:.4f}")

# --- 5. 전처리 2: 추가 파생 피처 생성 ---
def feature_engineering_2(df):
    df['태양광사용량'] = df['일사(MJ/m2)'] * df['태양광용량(kW)']
    df['sun_x_temp'] = df['일조(hr)'] * df['기온(°C)']
    df['sun_x_cooling_area'] = df['일조(hr)'] * df['냉방면적(m2)']
    df['sun_x_working_hours'] = df['일조(hr)'] * df['근무시간여부']
    df['sun_x_solar_capacity'] = df['일조(hr)'] * df['태양광용량(kW)']
    df['daily_total_sunshine'] = df.groupby(['건물번호', '연', '월', '일'])['일조(hr)'].transform('sum')
    df['solar_x_cooling_area'] = df['일사(MJ/m2)'] * df['냉방면적(m2)']
    df['rolling_hourly_solar_radiation'] = df.groupby('건물번호')['일사(MJ/m2)'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['rolling_hourly_sunshine'] = df.groupby('건물번호')['일조(hr)'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['solar_radiation_change'] = df.groupby('건물번호')['일사(MJ/m2)'].diff().fillna(0)
    df['solar_x_working_hours'] = df['일사(MJ/m2)'] * df['근무시간여부']
    df['solar_x_solar_capacity'] = df['일사(MJ/m2)'] * df['태양광용량(kW)']
    return df

train = feature_engineering_2(train)
test = feature_engineering_2(test)
print("\n전처리 2 (추가 파생 피처 생성) 완료.")

# train.to_csv(save_path + 'train_pre.csv', index=False)
# test.to_csv(save_path + 'test_pre.csv', index=False)

# exit()


# --- 6. 피처 Log 변환 (중복 제거) ---
print("\n--- 피처 Log 변환 시작 ---")
log_transform_candidates = [
    '태양광사용량', 'sun_x_cooling_area', 'sun_x_solar_capacity',
    'solar_x_cooling_area', 'solar_x_solar_capacity',
    '연면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
]

cols_to_apply_log = []
for col in log_transform_candidates:
    if col in train.columns and col in test.columns:
        # log1p는 0을 0으로 변환하고, 음수를 처리하지 못하므로 최소값이 0 이상인지 확인
        if train[col].min() >= 0 and test[col].min() >= 0:
            cols_to_apply_log.append(col)
        else:
            print(f"경고: '{col}' 컬럼에 음수 값이 포함되어 있어 log1p 변환을 건너뜁니다.")

for col in cols_to_apply_log:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])
    print(f"'{col}' 컬럼에 log1p 변환 적용 완료.")

print("\n--- Log 변환 요약 ---")
print("Log1p 변환이 적용된 컬럼:")
for col in cols_to_apply_log:
    print(f"- {col}")


# --- 7. 최종 모델 구축 및 예측 ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

print("\n최종 전력소비량 예측 모델 훈련 시작...")

final_features = [
    'sin_hour', 'cos_hour', '주말 여부', '공휴일', '태양광사용량', '불쾌지수',
    '냉방도일', '체감온도', '근무시간여부'
]
test_predictions = np.zeros(test.shape[0])
smape_1 = []
rmse_4 = []
for building_num in sorted(train['건물번호'].unique()):
    print(f"\n===== 건물번호: {building_num} 학습 시작 =====")
    
    train_building = train[train['건물번호'] == building_num].copy()
    test_building = test[test['건물번호'] == building_num].copy()

    # 최종 모델 학습을 위한 타겟 NaN 제거
    train_building.dropna(subset=['전력소비량(kWh)'], inplace=True)
    if train_building.empty:
        print(f"  [건물 {building_num}] '전력소비량(kWh)'의 NaN 값 제거 후 유효한 학습 데이터가 없어 건너뜀.")
        continue

    X_train_b = train_building[final_features]
    y_train_b = train_building['전력소비량(kWh)']
    X_test_b = test_building[final_features]
    
    y_train_b_log = np.log1p(y_train_b)

    # 피처에 NaN/Inf 확인 및 중앙값으로 대체
    X_train_b = X_train_b.replace([np.inf, -np.inf], np.nan).fillna(X_train_b.median())
    X_test_b = X_test_b.replace([np.inf, -np.inf], np.nan).fillna(X_test_b.median())

    X_temp_b, X_eval_b, y_temp_b, y_eval_b = train_test_split(
        X_train_b, y_train_b_log, test_size=0.2, random_state=SEED
    )

    tscv = TimeSeriesSplit(n_splits=3)

    oof_preds_xgb = np.zeros(X_temp_b.shape[0])
    oof_preds_lgb = np.zeros(X_temp_b.shape[0])

    test_preds_xgb = []
    test_preds_lgb = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_temp_b, y_temp_b)):
        X_train_fold, X_val_fold = X_temp_b.iloc[train_idx], X_temp_b.iloc[val_idx]
        y_train_fold, y_val_fold = y_temp_b.iloc[train_idx], y_temp_b.iloc[val_idx]

        # 폴드 데이터에 NaN/Inf 최종 확인
        if y_train_fold.isnull().any() or not np.isfinite(y_train_fold).all() or \
           y_val_fold.isnull().any() or not np.isfinite(y_val_fold).all() or \
           X_train_fold.isnull().any().any() or not np.isfinite(X_train_fold).all().all() or \
           X_val_fold.isnull().any().any() or not np.isfinite(X_val_fold).all().all():
            print(f"  [건물 {building_num}] Fold {fold+1}: 학습/검증 데이터에 NaN/Inf가 있어 이 폴드를 건너뜁니다.")
            continue

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, tree_method='hist', early_stopping_rounds=50)
        xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val_fold)
        test_preds_xgb.append(xgb_model.predict(X_test_b))

        lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, early_stopping_round=50, verbose=-1)
        lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
        test_preds_lgb.append(lgb_model.predict(X_test_b))

        y_val_fold_expm1 = np.expm1(y_val_fold)
        oof_preds_xgb[val_idx] = np.expm1(oof_preds_xgb[val_idx])
        oof_preds_lgb[val_idx] = np.expm1(oof_preds_lgb[val_idx])
        
        print(f"  [건물 {building_num}] Fold {fold+1} XGBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold_expm1, oof_preds_xgb[val_idx])):.4f}")
        print(f"  [건물 {building_num}] Fold {fold+1} LightGBM RMSE: {np.sqrt(mean_squared_error(y_val_fold_expm1, oof_preds_lgb[val_idx])):.4f}")

    # 모든 폴드가 스킵되어 test_preds_xgb 등이 비어있는 경우 처리
    if not test_preds_xgb or not test_preds_lgb:
        print(f"  [건물 {building_num}] 모든 폴드에서 유효한 예측값 생성 실패. 해당 건물 예측 건너뛰기.")
        # 이 경우, 해당 건물의 test_predictions 값은 0으로 남아있게 됩니다.
        continue

    meta_X_train = np.column_stack((oof_preds_xgb, oof_preds_lgb))
    meta_X_test = np.column_stack((np.mean(test_preds_xgb, axis=0), np.mean(test_preds_lgb, axis=0)))

    ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=3, scoring='neg_root_mean_squared_error')
    ridge_model.fit(meta_X_train, y_temp_b)

    final_preds_level2 = ridge_model.predict(meta_X_test)
    
    final_xgb_eval_preds = xgb_model.predict(X_eval_b)
    final_lgb_eval_preds = lgb_model.predict(X_eval_b)
    meta_X_eval = np.column_stack((final_xgb_eval_preds, final_lgb_eval_preds))
    ridge_eval_preds = ridge_model.predict(meta_X_eval)
    
    y_eval_b_exmp1 = np.expm1(y_eval_b)
    ridge_eval_preds_exmp1 = np.expm1(ridge_eval_preds)
    
    rmse_eval = np.sqrt(mean_squared_error(y_eval_b_exmp1, ridge_eval_preds_exmp1))
    smape_eval = smape(y_eval_b_exmp1, ridge_eval_preds_exmp1)
    print(f"  [건물 {building_num}] Level 2 (RidgeCV) 평가 세트 RMSE: {rmse_eval:.4f}")
    print(f"  [건물 {building_num}] Level 2 (RidgeCV) 평가 세트 SMAPE: {smape_eval:.4f}%")

    rmse_4.append(rmse_eval)
    smape_1.append(smape_eval)
    
    final_preds_level2_exmp1 = np.expm1(final_preds_level2)
    final_building_predictions = final_preds_level2_exmp1
    
    final_building_predictions[final_building_predictions < 0] = 0

    test_predictions[test_building.index - test.index.min()] = final_building_predictions

rmse_4 = np.mean(np.array(rmse_4))
smape_1 = np.mean(np.array(smape_1))

print("\n최종 전력소비량 예측 모델 훈련 및 예측 완료.")
print(f"최종 전력소비량 예측모델 RMSE  : {rmse_4:.4f}")
print(f"최종 전력소비량 예측모델 SMAPE : {smape_1:.4f}")

# --- 8. 최종 저장 ---
os.makedirs(save_path, exist_ok=True)
submit['answer'] = test_predictions
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{smape_1:.4f}".replace('.', '_')
os.makedirs(save_path, exist_ok=True)

filename = f"({SEED})10_01_{today}_SMAPE_{score_str}.csv"
submit.to_csv(save_path + filename, index=False)

print(f"\n최종 예측 결과가 {save_path}{filename} 에 저장되었습니다.")

print(f"\n--- 최종 모델 종합 성적 ---")
print(f"사용된 랜덤 시드: {SEED}")
print(f"Train '일사(MJ/m2)' RMSE : {rmse_1:.4f}")
print(f"Test '일조(hr)' RMSE : {rmse_2:.4f}")
print(f"Test '일사(MJ/m2)' RMSE : {rmse_3:.4f}")
print(f"최종 전력소비량 예측모델 RMSE  : {rmse_4:.4f}")
print(f"최종 전력소비량 예측모델 SMAPE : {smape_1:.4f}%")

with open("./Energy/10_00_submission/10_01_log.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"1단계('일사(MJ/m2)')      RMSE : {rmse_1:.6f}\n")
    f.write(f"2단계('일조(hr)')         RMSE : {rmse_2:.6f}\n")
    f.write(f"2단계('일사(MJ/m2)')      RMSE : {rmse_3:.6f}\n")
    f.write(f"3단계('전력소비량(kWh)')  RMSE : {rmse_4:.6f}\n")    
    f.write(f"3단계('전력소비량(kWh)') SMAPE : {smape_1:.6}%\n")
    f.write(f"1단계 컬럼 : {feature_1}\n") 
    f.write(f"2단계 컬럼 : {feature_2}\n") 
    f.write(f"3단계 컬럼 : {final_features}\n") 
    f.write("="*40 + "\n")
    
    