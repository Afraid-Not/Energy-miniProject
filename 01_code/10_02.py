import pandas as pd
import numpy as np
import warnings
import datetime
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit # KFold is still imported but TimeSeriesSplit is used
from sklearn.preprocessing import LabelEncoder, RobustScaler # RobustScaler imported
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import os

warnings.filterwarnings('ignore')

SEED = 1
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

    for building_type_col in [col for col in imputed_df.columns if '건물유형_' in col]:
        building_type_value = 1
        type_df_train = imputed_df[imputed_df[building_type_col] == building_type_value].copy()
        
        # 모델 훈련에 사용할 NaN 없는 데이터 추출
        type_df_train_cleaned = type_df_train.dropna(subset=[target_solar])

        X_train_solar_orig = type_df_train_cleaned[features_solar]
        y_train_solar = type_df_train_cleaned[target_solar]
        X_predict_solar_orig = type_df_train[type_df_train[target_solar].isna()][features_solar]

        # 훈련 데이터가 비어있지 않고 예측할 데이터가 있는 경우에만 모델 훈련
        if not X_train_solar_orig.empty and not X_predict_solar_orig.empty:
            # RobustScaler 적용
            scaler = RobustScaler()
            X_train_solar = scaler.fit_transform(X_train_solar_orig)
            X_predict_solar = scaler.transform(X_predict_solar_orig)
            
            # DataFrame으로 다시 변환 (컬럼명 유지를 위함)
            X_train_solar = pd.DataFrame(X_train_solar, columns=features_solar, index=X_train_solar_orig.index)
            X_predict_solar = pd.DataFrame(X_predict_solar, columns=features_solar, index=X_predict_solar_orig.index)

            # train_test_split은 tscv 외부에서 진행되므로, 전체 스케일링된 데이터 사용
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
                    print(f"   [Train 일사] 건물유형 {building_type_col} 값 {building_type_value}, Fold {fold+1}: NaN/Inf in target. 이 폴드 건너뛰기.")
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

                print(f"   [Train 일사] {building_type_col} - Fold {fold+1} XGBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_xgb[val_idx])):.4f}")
                print(f"   [Train 일사] {building_type_col} - Fold {fold+1} LightGBM RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_lgb[val_idx])):.4f}")

            # 모든 폴드가 스킵되어 예측값이 없는 경우 처리
            if not test_preds_xgb or not test_preds_lgb:
                print(f"   [Train 일사] {building_type_col} 값 {building_type_value}: 모든 폴드에서 유효한 예측값 생성 실패. 이 건물 유형의 일사량 예측 건너뛰기.")
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

            predicted_solar_radiation = (np.mean(test_preds_xgb, axis=0) + np.mean(test_preds_lgb, axis=0)) / 2
            
            # 예측된 일사량을 원래 위치에 채워넣기
            imputed_df.loc[imputed_df[building_type_col] == building_type_value, target_solar] = imputed_df.loc[
                imputed_df[building_type_col] == building_type_value, target_solar].fillna(pd.Series(predicted_solar_radiation, index=X_predict_solar_orig.index))
            
    # 밤 시간(21시 이후 ~ 6시 이전)의 일사는 모두 0으로 설정
    imputed_df.loc[(imputed_df['시간'] > 21) | (imputed_df['시간'] < 6), '일사(MJ/m2)'] = 0
    # 일사량이 음수일 경우 0으로 설정
    imputed_df.loc[imputed_df['일사(MJ/m2)'] < 0, '일사(MJ/m2)'] = 0

    # 마지막으로, 모델 훈련이 불가능하여 채워지지 않은 모든 NaN 값을 0으로 채움
    initial_nan_count = imputed_df['일사(MJ/m2)'].isnull().sum()
    if initial_nan_count > 0:
        imputed_df['일사(MJ/m2)'].fillna(0, inplace=True)
        print(f"   [최종] 예측 불가능하여 남아있던 일사(MJ/m2) NaN 값 {initial_nan_count}개를 0으로 채웠습니다.")
    
    return imputed_df.drop(columns=['is_day', '일사(MJ/m2)_original'])

train = train_solar_radiation_imputation(train)
print("\nTrain '일사(MJ/m2)' 예측 보조 모델 훈련 및 결측치 처리 완료.")
print(f"Train '일사(MJ/m2)' NaN 개수: {train['일사(MJ/m2)'].isnull().sum()}")

# --- 4. test '일조(hr)', '일사(MJ/m2)' 예측 보조 모델 ---
def test_sunshine_solar_imputation(train_df, test_df):
    imputed_test_df = test_df.copy()
    
    imputed_test_df['is_day'] = ((imputed_test_df['시간'] >= 6) & (imputed_test_df['시간'] <= 21)).astype(int)

    # features_final_test_solar 에서 더미 변수 제거하여 스케일링 대상만 남김
    features_to_scale_test_solar = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'sin_hour', 'cos_hour', '연면적(m2)']
    features_final_test_solar = features_to_scale_test_solar + [col for col in imputed_test_df.columns if '건물유형_' in col]
    
    # 예측된 값을 저장할 컬럼을 미리 0.0으로 초기화
    imputed_test_df['일조(hr)'] = 0.0
    imputed_test_df['일사(MJ/m2)'] = 0.0

    print("\nTest '일조(hr)', '일사(MJ/m2)' 예측 보조 모델 훈련 시작...")

    for building_num in sorted(train_df['건물번호'].unique()):
        print(f"===== Test 데이터 건물번호: {building_num} '일조(hr)', '일사(MJ/m2)' 예측 시작 =====")
        train_building = train_df[train_df['건물번호'] == building_num].copy()
        test_building = imputed_test_df[imputed_test_df['건물번호'] == building_num].copy()

        # 학습 데이터에서 타겟 변수에 NaN이 있는 행 제거 (이전에도 있었지만, 명시적 강조)
        train_building_for_model = train_building.copy()
        initial_rows = train_building_for_model.shape[0]
        train_building_for_model.dropna(subset=['일조(hr)', '일사(MJ/m2)'], inplace=True)
        dropped_rows = initial_rows - train_building_for_model.shape[0]
        if dropped_rows > 0:
            print(f"   [건물 {building_num}] '일조(hr)' 또는 '일사(MJ/m2)'의 NaN 값으로 인해 {dropped_rows}개 행 제거됨.")
        
        # 모델 학습에 사용할 데이터가 비어있는 경우 스킵
        if train_building_for_model.empty:
            print(f"   [건물 {building_num}] NaN 제거 후 유효한 학습 데이터가 없어 '일조(hr)' / '일사(MJ/m2)' 예측을 건너뜀.")
            continue

        X_train_building_orig = train_building_for_model[features_final_test_solar]
        y_train_sunshine = train_building_for_model['일조(hr)']
        y_train_solar = train_building_for_model['일사(MJ/m2)']
        X_test_building_orig = test_building[features_final_test_solar] # test 데이터 피처 원본

        # X_train_building_orig, X_test_building_orig 의 NaN/Inf 처리 (사전 처리)
        X_train_building_orig = X_train_building_orig.replace([np.inf, -np.inf], np.nan).fillna(X_train_building_orig.median())
        X_test_building_orig = X_test_building_orig.replace([np.inf, -np.inf], np.nan).fillna(X_test_building_orig.median())

        # RobustScaler 적용
        scaler = RobustScaler()
        X_train_building_scaled_num = scaler.fit_transform(X_train_building_orig[features_to_scale_test_solar])
        X_test_building_scaled_num = scaler.transform(X_test_building_orig[features_to_scale_test_solar])

        X_train_building = pd.DataFrame(X_train_building_scaled_num, columns=features_to_scale_test_solar, index=X_train_building_orig.index)
        X_test_building_cleaned_for_pred = pd.DataFrame(X_test_building_scaled_num, columns=features_to_scale_test_solar, index=X_test_building_orig.index)

        # 원-핫 인코딩된 컬럼 다시 추가
        for col in [c for c in X_train_building_orig.columns if '건물유형_' in c]:
            X_train_building[col] = X_train_building_orig[col]
            X_test_building_cleaned_for_pred[col] = X_test_building_orig[col]

        tscv = TimeSeriesSplit(n_splits=3)

        for target_col, y_train_target in [('일조(hr)', y_train_sunshine), ('일사(MJ/m2)', y_train_solar)]:
            oof_preds_xgb = np.zeros(X_train_building.shape[0])
            oof_preds_lgb = np.zeros(X_train_building.shape[0])
            oof_preds_cat = np.zeros(X_train_building.shape[0])
            test_preds_xgb = []
            test_preds_lgb = []
            test_preds_cat = []

            # 타겟 변수에 NaN/Inf가 여전히 있는지 확인
            if y_train_target.isnull().any() or not np.isfinite(y_train_target).all():
                print(f"   [건물 {building_num}] 경고: {target_col} 타겟에 NaN/Inf가 발견되어 이 타겟의 학습을 건너뜁니다.")
                imputed_test_df.loc[imputed_test_df['건물번호'] == building_num, target_col] = 0 # Default to 0 if no training possible
                continue

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_building, y_train_target)):
                X_train_fold, X_val_fold = X_train_building.iloc[train_idx], X_train_building.iloc[val_idx]
                y_train_fold, y_val_fold = y_train_target.iloc[train_idx], y_train_target.iloc[val_idx]

                # 폴드 데이터에 NaN/Inf 최종 확인 및 건너뛰기
                if y_train_fold.isnull().any() or not np.isfinite(y_train_fold).all() or \
                   y_val_fold.isnull().any() or not np.isfinite(y_val_fold).all() or \
                   X_train_fold.isnull().any().any() or not np.isfinite(X_train_fold).all().all() or \
                   X_val_fold.isnull().any().any() or not np.isfinite(X_val_fold).all().all():
                    print(f"   [건물 {building_num}] Fold {fold+1} {target_col}: y_train_fold 또는 y_val_fold에 NaN/Inf 또는 X_fold에 NaN/Inf가 있어 이 폴드를 건너뜁니다.")
                    continue
                
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

                print(f"   [Test {target_col}] 건물 {building_num}, Fold {fold+1} XGBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_xgb[val_idx])):.4f}")
                print(f"   [Test {target_col}] 건물 {building_num}, Fold {fold+1} LightGBM RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_lgb[val_idx])):.4f}")
                print(f"   [Test {target_col}] 건물 {building_num}, Fold {fold+1} CatBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_cat[val_idx])):.4f}")
            
            # 모든 폴드가 스킵되어 test_preds_xgb 등이 비어있는 경우 처리
            if not test_preds_xgb or not test_preds_lgb or not test_preds_cat:
                print(f"   [Test {target_col}] 건물 {building_num}: 모든 폴드에서 유효한 예측값 생성 실패. 해당 타겟 건너뛰기.")
                imputed_test_df.loc[imputed_test_df['건물번호'] == building_num, target_col] = np.nan # 예측 실패 시 NaN으로 남겨두기
                continue

            meta_X_train = np.column_stack((oof_preds_xgb, oof_preds_lgb, oof_preds_cat))
            meta_X_test = np.column_stack((np.mean(test_preds_xgb, axis=0), np.mean(test_preds_lgb, axis=0), np.mean(test_preds_cat, axis=0)))
            
            ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=3, scoring='neg_root_mean_squared_error')
            ridge_model.fit(meta_X_train, y_train_target) # Fit on the full X_train_building
            predicted_values = ridge_model.predict(meta_X_test)
            
            rmse_oof_stacker = np.sqrt(mean_squared_error(y_train_target, ridge_model.predict(meta_X_train)))
            print(f"[Test {target_col}] 건물 {building_num} 스택킹 OOF RMSE: {rmse_oof_stacker:.4f}")
            
            imputed_test_df.loc[imputed_test_df['건물번호'] == building_num, target_col] = predicted_values
            
            imputed_test_df.loc[((imputed_test_df['시간'] > 21) | (imputed_test_df['시간'] < 6)) & (imputed_test_df['건물번호'] == building_num), target_col] = 0
            imputed_test_df.loc[(imputed_test_df['건물번호'] == building_num) & (imputed_test_df[target_col] < 0), target_col] = 0

    # Final fillna for any remaining NaNs in imputed_test_df for '일조(hr)' and '일사(MJ/m2)'
    for col in ['일조(hr)', '일사(MJ/m2)']:
        initial_nan_count = imputed_test_df[col].isnull().sum()
        if initial_nan_count > 0:
            imputed_test_df[col].fillna(0, inplace=True)
            print(f"   [최종] 예측 불가능하여 남아있던 Test {col} NaN 값 {initial_nan_count}개를 0으로 채웠습니다.")

    return imputed_test_df.drop(columns=['is_day'])

train = train_solar_radiation_imputation(train)
print("\nTrain '일사(MJ/m2)' 예측 보조 모델 훈련 및 결측치 처리 완료.")
print(f"Train '일사(MJ/m2)' NaN 개수: {train['일사(MJ/m2)'].isnull().sum()}")

test = test_sunshine_solar_imputation(train, test)
print("\nTest '일조(hr)', '일사(MJ/m2)' 예측 보조 모델 훈련 및 결측치 처리 완료.")
print(f"Test '일조(hr)' NaN 개수: {test['일조(hr)'].isnull().sum()}")
print(f"Test '일사(MJ/m2)' NaN 개수: {test['일사(MJ/m2)'].isnull().sum()}")

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

# RobustScaler를 적용할 최종 수치형 피처 리스트 (더미 변수 및 이진 피처 제외)
numerical_final_features_to_scale = [
    'sin_hour', 'cos_hour', '태양광사용량', '불쾌지수',
    '냉방도일', '체감온도', 'sun_x_temp', 'sun_x_cooling_area',
    'sun_x_working_hours', 'sun_x_solar_capacity', 'daily_total_sunshine',
    'solar_x_cooling_area', 'rolling_hourly_solar_radiation', 'rolling_hourly_sunshine',
    'solar_radiation_change', 'solar_x_working_hours', 'solar_x_solar_capacity',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
    '연면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'
]


test_predictions = np.zeros(test.shape[0])

for building_num in sorted(train['건물번호'].unique()):
    print(f"\n===== 건물번호: {building_num} 학습 시작 =====")
    
    train_building = train[train['건물번호'] == building_num].copy()
    test_building = test[test['건물번호'] == building_num].copy()

    # 최종 모델 학습을 위한 타겟 NaN 제거
    train_building.dropna(subset=['전력소비량(kWh)'], inplace=True)
    if train_building.empty:
        print(f"   [건물 {building_num}] '전력소비량(kWh)'의 NaN 값 제거 후 유효한 학습 데이터가 없어 건너뜀.")
        continue

    # 전체 최종 피처 목록 (원-핫 인코딩 및 이진 피처 포함)
    current_final_features = numerical_final_features_to_scale + \
                             [col for col in train.columns if '건물유형_' in col and col in train_building.columns] + \
                             [col for col in train.columns if '요일_' in col and col in train_building.columns] + \
                             ['주말 여부', '공휴일', '근무시간여부'] # Binary features, ensure they are in the list


    X_train_b_orig = train_building[current_final_features]
    y_train_b = train_building['전력소비량(kWh)']
    X_test_b_orig = test_building[current_final_features]


    # 피처에 NaN/Inf 확인 및 중앙값으로 대체
    X_train_b_orig = X_train_b_orig.replace([np.inf, -np.inf], np.nan).fillna(X_train_b_orig.median())
    X_test_b_orig = X_test_b_orig.replace([np.inf, -np.inf], np.nan).fillna(X_test_b_orig.median())

    # RobustScaler 적용
    scaler = RobustScaler()
    X_train_b_scaled_num = scaler.fit_transform(X_train_b_orig[numerical_final_features_to_scale])
    X_test_b_scaled_num = scaler.transform(X_test_b_orig[numerical_final_features_to_scale])

    X_train_b = pd.DataFrame(X_train_b_scaled_num, columns=numerical_final_features_to_scale, index=X_train_b_orig.index)
    X_test_b = pd.DataFrame(X_test_b_scaled_num, columns=numerical_final_features_to_scale, index=X_test_b_orig.index)

    # 원-핫 인코딩된 컬럼 및 이진 컬럼 다시 추가
    for col in [c for c in current_final_features if c not in numerical_final_features_to_scale]:
        X_train_b[col] = X_train_b_orig[col]
        X_test_b[col] = X_test_b_orig[col]


    tscv = TimeSeriesSplit(n_splits=3)

    oof_preds_xgb = np.zeros(X_train_b.shape[0])
    oof_preds_lgb = np.zeros(X_train_b.shape[0])

    test_preds_xgb = []
    test_preds_lgb = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_b, y_train_b)):
        X_train_fold, X_val_fold = X_train_b.iloc[train_idx], X_train_b.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_b.iloc[train_idx], y_train_b.iloc[val_idx]

        # 폴드 데이터에 NaN/Inf 최종 확인
        if y_train_fold.isnull().any() or not np.isfinite(y_train_fold).all() or \
           y_val_fold.isnull().any() or not np.isfinite(y_val_fold).all() or \
           X_train_fold.isnull().any().any() or not np.isfinite(X_train_fold).all().all() or \
           X_val_fold.isnull().any().any() or not np.isfinite(X_val_fold).all().all():
            print(f"   [건물 {building_num}] Fold {fold+1}: 학습/검증 데이터에 NaN/Inf가 있어 이 폴드를 건너뜁니다.")
            continue

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, tree_method='hist', early_stopping_rounds=50)
        xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val_fold)
        test_preds_xgb.append(xgb_model.predict(X_test_b))

        lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, early_stopping_round=50, verbose=-1)
        lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
        test_preds_lgb.append(lgb_model.predict(X_test_b))

        print(f"   [건물 {building_num}] Fold {fold+1} XGBoost RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_xgb[val_idx])):.4f}")
        print(f"   [건물 {building_num}] Fold {fold+1} LightGBM RMSE: {np.sqrt(mean_squared_error(y_val_fold, oof_preds_lgb[val_idx])):.4f}")

    # 모든 폴드가 스킵되어 test_preds_xgb 등이 비어있는 경우 처리
    if not test_preds_xgb or not test_preds_lgb:
        print(f"   [건물 {building_num}] 모든 폴드에서 유효한 예측값 생성 실패. 해당 건물 예측 건너뛰기.")
        continue

    meta_X_train = np.column_stack((oof_preds_xgb, oof_preds_lgb))
    meta_X_test = np.column_stack((np.mean(test_preds_xgb, axis=0), np.mean(test_preds_lgb, axis=0)))

    ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=3, scoring='neg_root_mean_squared_error')
    ridge_model.fit(meta_X_train, y_train_b) 
    
    final_preds_level2 = ridge_model.predict(meta_X_test)
    
    rmse_oof_stacker = np.sqrt(mean_squared_error(y_train_b, ridge_model.predict(meta_X_train)))
    smape_oof_stacker = smape(y_train_b, ridge_model.predict(meta_X_train))
    print(f"   [건물 {building_num}] Level 2 (RidgeCV) OOF RMSE: {rmse_oof_stacker:.4f}")
    print(f"   [건물 {building_num}] Level 2 (RidgeCV) OOF SMAPE: {smape_oof_stacker:.4f}%")

    final_building_predictions = final_preds_level2
    
    final_building_predictions[final_building_predictions < 0] = 0

    test_predictions[test_building.index - test.index.min()] = final_building_predictions

print("\n최종 전력소비량 예측 모델 훈련 및 예측 완료.")

# --- 8. 최종 저장 ---
os.makedirs(save_path, exist_ok=True)
submit['answer'] = test_predictions
submit.to_csv(save_path + 'submission.csv', index=False)

print(f"\n최종 예측 결과가 {save_path}submission.csv 에 저장되었습니다.")

# --- 9. 최종 성적 보고 ---
overall_rmse_eval_list = []
overall_smape_eval_list = []

for building_num in sorted(train['건물번호'].unique()):
    train_building = train[train['건물번호'] == building_num].copy()
    
    train_building.dropna(subset=['전력소비량(kWh)'], inplace=True)
    if train_building.empty:
        continue

    # 전체 최종 피처 목록 (원-핫 인코딩 포함)
    current_final_features_eval = numerical_final_features_to_scale + \
                                  [col for col in train.columns if '건물유형_' in col and col in train_building.columns] + \
                                  [col for col in train.columns if '요일_' in col and col in train_building.columns] + \
                                  ['주말 여부', '공휴일', '근무시간여부']

    X_train_b_orig_eval = train_building[current_final_features_eval]
    y_train_b_eval = train_building['전력소비량(kWh)']

    X_train_b_orig_eval = X_train_b_orig_eval.replace([np.inf, -np.inf], np.nan).fillna(X_train_b_orig_eval.median())

    # RobustScaler 적용
    scaler_eval = RobustScaler()
    X_train_b_scaled_num_eval = scaler_eval.fit_transform(X_train_b_orig_eval[numerical_final_features_to_scale])
    X_train_b_eval = pd.DataFrame(X_train_b_scaled_num_eval, columns=numerical_final_features_to_scale, index=X_train_b_orig_eval.index)

    # 원-핫 인코딩된 컬럼 및 이진 컬럼 다시 추가
    for col in [c for c in current_final_features_eval if c not in numerical_final_features_to_scale]:
        X_train_b_eval[col] = X_train_b_orig_eval[col]
    
    tscv_eval = TimeSeriesSplit(n_splits=3)
    
    all_oof_xgb = np.zeros(X_train_b_eval.shape[0])
    all_oof_lgb = np.zeros(X_train_b_eval.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(tscv_eval.split(X_train_b_eval, y_train_b_eval)):
        X_train_fold, X_val_fold = X_train_b_eval.iloc[train_idx], X_train_b_eval.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_b_eval.iloc[train_idx], y_train_b_eval.iloc[val_idx]

        if y_train_fold.isnull().any() or not np.isfinite(y_train_fold).all() or \
           y_val_fold.isnull().any() or not np.isfinite(y_val_fold).all() or \
           X_train_fold.isnull().any().any() or not np.isfinite(X_train_fold).all().all() or \
           X_val_fold.isnull().any().any() or not np.isfinite(X_val_fold).all().all():
            print(f"   [건물 {building_num}] 평가 Fold {fold+1}: 학습/검증 데이터에 NaN/Inf가 있어 이 폴드를 건너뜁니다.")
            continue

        xgb_model_eval = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1)
        xgb_model_eval.fit(X_train_fold, y_train_fold)
        
        lgb_model_eval = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1)
        lgb_model_eval.fit(X_train_fold, y_train_fold)

        all_oof_xgb[val_idx] = xgb_model_eval.predict(X_val_fold)
        all_oof_lgb[val_idx] = lgb_model_eval.predict(X_val_fold)

    meta_X_overall_oof = np.column_stack((all_oof_xgb, all_oof_lgb))
    
    # Filter out entries where OOF predictions might not have been generated (e.g., due to skipped folds)
    # Using a more robust check for non-zero predictions, as 0 can be a valid prediction
    valid_indices = np.where((all_oof_xgb != 0) | (all_oof_lgb != 0))[0] 
    # Or, if 0 is a possible valid prediction, check if they were assigned a value in the oof array.
    # For now, stick with checking if they were set to something non-zero, implies they were processed in a fold.

    if len(valid_indices) == 0:
        print(f"   [건물 {building_num}] 평가: 유효한 OOF 예측이 없어 전체 성능 평가를 건너뜁니다.")
        continue

    meta_X_overall_oof_filtered = meta_X_overall_oof[valid_indices]
    y_train_b_valid = y_train_b_eval.iloc[valid_indices]

    if meta_X_overall_oof_filtered.shape[0] == 0:
        print(f"   [건물 {building_num}] 평가: 유효한 메타 학습 데이터가 없어 전체 성능 평가를 건너뜁니다.")
        continue

    ridge_model_eval = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=3, scoring='neg_root_mean_squared_error')
    ridge_model_eval.fit(meta_X_overall_oof_filtered, y_train_b_valid)
    
    overall_ridge_preds = ridge_model_eval.predict(meta_X_overall_oof_filtered)
    
    overall_ridge_preds[overall_ridge_preds < 0] = 0
    
    rmse = np.sqrt(mean_squared_error(y_train_b_valid, overall_ridge_preds))
    smape_val = smape(y_train_b_valid, overall_ridge_preds)

    overall_rmse_eval_list.append(rmse)
    overall_smape_eval_list.append(smape_val)

print(f"\n--- 최종 모델 종합 성적 ---")
print(f"사용된 랜덤 시드: {SEED}")
print(f"평균 Level 2 (RidgeCV) OOF RMSE: {np.mean(overall_rmse_eval_list):.4f}")
print(f"평균 Level 2 (RidgeCV) OOF SMAPE: {np.mean(overall_smape_eval_list):.4f}%")