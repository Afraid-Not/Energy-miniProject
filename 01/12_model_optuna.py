import pandas as pd
import numpy as np
import datetime
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("[11_model_optuna] 시작")

seed_file = "./Energy/01/(SEED_COUNT)11_model.json"

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

xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.1,
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': 0
            }

lgbm_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': -1,
                'force_col_wise': True
            }

cat_params = {
                'loss_function': 'MAE',
                'iterations': 500,
                'learning_rate': 0.08,
                'depth': 5,
                'l2_leaf_reg': 2,
                'subsample': 0.9,
                'colsample_bylevel': 0.9,
                'random_strength': 0.5,
                'bagging_temperature': 0.5,
                'border_count': 32,
                'random_seed': SEED,
                'thread_count': -1,
                'verbose': False,
                'allow_writing_files': False
            }

print(f"[1] 데이터 로드 및 전처리 (생략 - 기존 코드에서 이미 수행)")
data_path = './Energy/'
csv_path = './Energy/01/'
save_path = './Energy/01/submission/'
os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
train = pd.read_csv(csv_path + 'predict_train.csv')
test = pd.read_csv(csv_path + 'predict_test.csv')
samplesub = pd.read_csv(data_path +'sample_submission.csv')

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

def one_hot_building_id(df):
    temp = pd.get_dummies(df['건물번호'], prefix='건물')
    df = pd.concat([df, temp], axis=1)
    return df

train = one_hot_building_id(train)
test = one_hot_building_id(test)

exclude_cols = ['건물번호', '일시', '전력소비량(kWh)', '건물유형', '날짜']

# train과 test에 공통으로 존재하는 피처만 선택
train_features = [col for col in train.columns if col not in exclude_cols]
test_features = [col for col in test.columns if col not in exclude_cols]

# 교집합으로 공통 피처만 사용
features = list(set(train_features) & set(test_features))

print(f"총 피처 수: {len(features)}")
print(f"제외된 train 전용 피처: {set(train_features) - set(test_features)}")
print(f"제외된 test 전용 피처: {set(test_features) - set(train_features)}")

# 또는 더 안전하게 명시적으로 제외할 피처들을 지정:
excluded_lag_features = [
    '일조_lag_24h', '일조_lag_48h', 
    '일사_lag_24h', '일사_lag_48h'
]

# 기존 features에서 test에 없는 피처들 제거
features = [col for col in train.columns 
           if col not in exclude_cols 
           and col not in excluded_lag_features
           and col in test.columns]

target = '전력소비량(kWh)'

print(f"최종 사용할 피처 수: {len(features)}")
print("최종 피처 리스트 확인:")
for i, feat in enumerate(features[:10]):  # 처음 10개만 출력
    print(f"  {i+1}: {feat}")
if len(features) > 10:
    print(f"  ... 외 {len(features)-10}개")

# 피처가 train과 test에 모두 존재하는지 최종 확인
missing_in_test = [f for f in features if f not in test.columns]
missing_in_train = [f for f in features if f not in train.columns]

if missing_in_test:
    print(f"❌ Test에 없는 피처: {missing_in_test}")
    features = [f for f in features if f not in missing_in_test]
    
if missing_in_train:
    print(f"❌ Train에 없는 피처: {missing_in_train}")
    features = [f for f in features if f not in missing_in_train]

print(f"✅ 최종 검증된 피처 수: {len(features)}")


N_SPLIT = 5
KFOLD = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

########################### Optuna 가중평균 최적화 클래스 ############################

class OptimizedEnsembleWeighting:
    def __init__(self, seed=42):
        self.seed = seed
        self.best_weights = None
        self.study = None
        
    def objective(self, trial, predictions_dict, y_true):
        """Optuna objective function for ensemble weight optimization"""
        # 각 예측 방법의 가중치를 0~1 사이에서 샘플링
        model_names = list(predictions_dict.keys())
        weights = []
        
        for i, name in enumerate(model_names):
            if i == len(model_names) - 1:
                # 마지막 가중치는 1에서 나머지 가중치들의 합을 뺀 값
                weight = 1.0 - sum(weights)
                if weight < 0:
                    weight = 0.1
            else:
                weight = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
            weights.append(weight)
        
        # 가중치 정규화
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]
        
        # 가중 앙상블 예측
        ensemble_pred = np.zeros(len(y_true))
        for i, (name, weight) in enumerate(zip(model_names, weights)):
            ensemble_pred += weight * predictions_dict[name]
        
        # SMAPE 계산 (최소화하려는 목표)
        smape_score = smape(np.expm1(y_true), np.expm1(ensemble_pred))
        return smape_score
    
    def optimize_weights(self, predictions_dict, y_true, n_trials=100):
        """가중치 최적화"""
        print(f"    > Optuna 앙상블 가중치 최적화 시작 (trials: {n_trials})")
        
        # Optuna study 생성
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=f'ensemble_weighting_{self.seed}'
        )
        
        # 최적화 실행
        self.study.optimize(
            lambda trial: self.objective(trial, predictions_dict, y_true),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # 최적 가중치 추출
        best_params = self.study.best_params
        model_names = list(predictions_dict.keys())
        weights = []
        
        for i, name in enumerate(model_names):
            if i == len(model_names) - 1:
                weight = 1.0 - sum(weights)
                if weight < 0:
                    weight = 0.1
            else:
                weight = best_params.get(f'weight_{name}', 1.0 / len(model_names))
            weights.append(weight)
        
        # 가중치 정규화
        total_weight = sum(weights)
        self.best_weights = [w / total_weight for w in weights]
        
        print(f"    > 최적화 완료 - Best SMAPE: {self.study.best_value:.6f}")
        for name, weight in zip(model_names, self.best_weights):
            print(f"    > {name}: {weight:.4f}")
        
        return self.best_weights
    
    def predict(self, predictions_dict):
        """최적화된 가중치로 앙상블 예측"""
        if self.best_weights is None:
            raise ValueError("가중치가 최적화되지 않았습니다. optimize_weights를 먼저 실행하세요.")
        
        model_names = list(predictions_dict.keys())
        ensemble_pred = np.zeros(len(list(predictions_dict.values())[0]))
        
        for name, weight in zip(model_names, self.best_weights):
            ensemble_pred += weight * predictions_dict[name]
        
        return ensemble_pred

########################### 기존 함수들 ############################

def peak_model(train_df, test_df, building_type, current_features, current_target, seed_val):
    peak_hours_apartment_hotel = [18, 19, 20, 21, 22]
    peak_hours_other = [12, 13, 14, 15, 16]

    # peak_idx_bool은 이제 train_df/test_df의 현재 인덱스를 따릅니다.
    train_peak_idx_bool = train_df['시각'].isin(peak_hours_apartment_hotel) if building_type in ['아파트', '호텔'] else train_df['시각'].isin(peak_hours_other)
    test_peak_idx_bool = test_df['시각'].isin(peak_hours_apartment_hotel) if building_type in ['아파트', '호텔'] else test_df['시각'].isin(peak_hours_other)

    # 이 시점에서 X_train_peak, y_train_peak, X_test_peak는
    # 현재 train_df/test_df 내에서 필터링된 데이터프레임의 copy가 됩니다.
    # .copy()를 명시적으로 추가하여 SettingWithCopyWarning을 피합니다.
    X_train_peak_subset = train_df[train_peak_idx_bool][current_features].copy()
    y_train_peak_subset = np.log1p(train_df[train_peak_idx_bool][current_target]).copy()
    X_test_peak_subset = test_df[test_peak_idx_bool][current_features].copy()

    if len(X_train_peak_subset) == 0 or len(X_test_peak_subset) == 0:
        return None, None # 피크 시간대에 데이터가 없는 경우

    # KFold는 항상 0부터 시작하는 인덱스를 기준으로 작동하므로,
    # StandardScaler에 전달하기 전에 DataFrame을 NumPy 배열로 변환하는 것이 일반적입니다.
    # 인덱스를 유지할 필요가 없습니다.
    scaler_peak = StandardScaler()
    X_train_peak_scaled = scaler_peak.fit_transform(X_train_peak_subset)
    X_test_peak_scaled = scaler_peak.transform(X_test_peak_subset)

    peak_model_lgbm = LGBMRegressor(**lgbm_params)

    # 이 oof_preds와 test_preds는 필터링된 서브셋의 길이만큼 생성됩니다.
    oof_preds_for_peak_subset = np.zeros(len(X_train_peak_subset))
    test_preds_for_peak_subset = np.zeros(len(X_test_peak_subset))

    # KFold.split은 이제 X_train_peak_scaled (NumPy 배열)의 내부 인덱스를 사용합니다.
    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_peak_scaled, y_train_peak_subset)):
        X_train_fold, X_val_fold = X_train_peak_scaled[train_idx], X_train_peak_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_peak_subset.iloc[train_idx], y_train_peak_subset.iloc[val_idx]

        peak_model_lgbm.fit(X_train_fold, y_train_fold)
        oof_preds_for_peak_subset[val_idx] = peak_model_lgbm.predict(X_val_fold)
        test_preds_for_peak_subset += peak_model_lgbm.predict(X_test_peak_scaled) / N_SPLIT
    
    # 이제 반환되는 값은 해당 서브셋 (train_df[train_peak_idx_bool])에 대한 예측값입니다.
    # NaN으로 채워진 전체 배열이 아니라, 피크 시간대에 대한 예측값만 반환합니다.
    return oof_preds_for_peak_subset, test_preds_for_peak_subset

def tune_and_predict_with_optuna(train_df_subset, test_df_subset, 
                                 current_features, current_target, seed_val, n_trials=15):
    """
    Optuna를 사용하여 LGBMRegressor의 하이퍼파라미터를 튜닝하고,
    최적의 모델로 OOF 및 테스트 예측을 수행합니다.
    """
    X_train_full = train_df_subset[current_features].reset_index(drop=True)
    y_train_full = np.log1p(train_df_subset[current_target]).reset_index(drop=True)
    X_test_full = test_df_subset[current_features].reset_index(drop=True)

    # 데이터가 부족하면 튜닝을 건너뜁니다.
    if len(X_train_full) == 0 or len(X_test_full) == 0:
        return None, None, float('inf') # 예측값, 테스트 예측값, 높은 SMAPE 반환

    def objective(trial):
        # LGBMRegressor 하이퍼파라미터 탐색 공간 정의
        param = {
            'objective': 'mae', # MAE를 최소화하도록 설정
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000), # 더 넓은 범위
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'random_state': seed_val,
            'n_jobs': -1,
            'verbose': -1,
        }

        kf = KFold(n_splits=2, shuffle=True, random_state=seed_val)
        oof_trial_preds = np.zeros(len(X_train_full))
        
        # OOF 예측을 위한 스케일러를 매 폴드마다 새로 초기화하지 않고,
        # 전체 데이터에 대해 한 번 fit_transform 한 후 인덱싱하여 사용합니다.
        # 이렇게 하면 스케일링 일관성이 유지됩니다.
        scaler = StandardScaler()
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full_scaled, y_train_full)):
            X_train_fold, X_val_fold = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
            y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            model = LGBMRegressor(**param)
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      eval_metric='mae',
                      callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)]) # Early stopping rounds 증가

            oof_trial_preds[val_idx] = model.predict(X_val_fold)
            
            # Pruning 콜백. MAE가 일정 기준 이상으로 나쁘면 Trial 종료
            trial.report(mean_absolute_error(y_val_fold, oof_trial_preds[val_idx]), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # OOF MAE 반환 (Optuna는 이 값을 최소화하려고 시도)
        return mean_absolute_error(y_train_full, oof_trial_preds)

    # Optuna Study 생성 및 최적화
    # sampler=optuna.samplers.TPESampler(seed=seed_val)로 시드 고정
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials)

    print(f"    > Optuna Best MAE: {study.best_value:.6f}")
    print(f"    > Optuna Best Params: {study.best_params}")

    # 최적의 하이퍼파라미터로 최종 모델 학습 및 예측
    best_params = study.best_params
    best_lgbm_model = LGBMRegressor(objective='mae', random_state=seed_val, **best_params, n_jobs=-1, verbose=-1)

    oof_final_preds = np.zeros(len(X_train_full))
    test_final_preds = np.zeros(len(X_test_full))

    # KFold를 사용하여 최적 모델로 최종 OOF 및 테스트 예측 수행
    # 스케일러는 다시 전체 X_train_full에 fit_transform 후 사용
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full_scaled, y_train_full)): # 전역 KFOLD 사용
        X_train_fold, X_val_fold = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        best_lgbm_model.fit(X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            eval_metric='mae',
                            callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)])

        oof_final_preds[val_idx] = best_lgbm_model.predict(X_val_fold)
        test_final_preds += best_lgbm_model.predict(X_test_full_scaled) / N_SPLIT

    # SMAPE 계산 (여기서는 현재 서브셋에 대한 SMAPE를 반환)
    final_smape = smape(np.expm1(y_train_full), np.expm1(oof_final_preds))
    print(f"    > Optuna 최종 모델 SMAPE: {final_smape:.6f}")

    return oof_final_preds, test_final_preds, final_smape

def train_and_predict_subset(train_subset, test_subset, name="Subset"):
    X_train_full = train_subset[features].reset_index(drop=True)
    y_train_full = np.log1p(train_subset[target].reset_index(drop=True))
    X_test = test_subset[features].reset_index(drop=True)

    oof_preds_lvl1 = np.zeros((len(X_train_full), 3)) # xgb, lgb, cat
    test_preds_lvl1 = np.zeros((len(X_test), 3))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = XGBRegressor(**xgb_params)
        lgb_model = LGBMRegressor(**lgbm_params)
        cat_model = CatBoostRegressor(**cat_params)
        
        xgb_model.fit(X_train_scaled, y_train)
        lgb_model.fit(X_train_scaled, y_train)
        cat_model.fit(X_train_scaled, y_train)

        oof_preds_lvl1[val_idx, 0] = xgb_model.predict(X_val_scaled)
        oof_preds_lvl1[val_idx, 1] = lgb_model.predict(X_val_scaled)
        oof_preds_lvl1[val_idx, 2] = cat_model.predict(X_val_scaled)

        test_preds_lvl1[:, 0] += xgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_lvl1[:, 1] += lgb_model.predict(X_test_scaled) / N_SPLIT
        test_preds_lvl1[:, 2] += cat_model.predict(X_test_scaled) / N_SPLIT
    
    meta_model = RidgeCV()
    meta_model.fit(oof_preds_lvl1, y_train_full)

    oof_pred_final = meta_model.predict(oof_preds_lvl1)
    test_pred_final = meta_model.predict(test_preds_lvl1)

    # --- Peak Model Integration ---
    building_type_for_peak = train_subset['건물유형'].iloc[0] if not train_subset.empty else None

    # 최종 예측 배열을 초기화합니다. 이들은 train_subset과 test_subset의 길이를 따릅니다.
    oof_pred_final_combined = np.array(oof_pred_final) # 기본 모델의 예측으로 초기화
    test_pred_final_combined = np.array(test_pred_final) # 기본 모델의 예측으로 초기화

    if building_type_for_peak:
        peak_oof_pred_subset, peak_test_pred_subset = peak_model(train_subset, test_subset, building_type_for_peak, features, target, SEED)

        if peak_oof_pred_subset is not None and peak_test_pred_subset is not None:
            # 피크 시간대에 해당하는 인덱스를 다시 얻습니다.
            train_peak_idx_bool = train_subset['시각'].isin([18, 19, 20, 21, 22]) if building_type_for_peak in ['아파트', '호텔'] else train_subset['시각'].isin([12, 13, 14, 15, 16])
            test_peak_idx_bool = test_subset['시각'].isin([18, 19, 20, 21, 22]) if building_type_for_peak in ['아파트', '호텔'] else test_subset['시각'].isin([12, 13, 14, 15, 16])

            # oof_pred_final_combined와 test_pred_final_combined에 피크 모델 예측을 덮어씁니다.
            # train_subset[train_peak_idx_bool].index는 해당 서브셋 내의 인덱스가 아니라,
            # X_train_full (reset_index(drop=True)된)의 내부 인덱스와 직접적으로 매핑될 수 없습니다.
            # 대신, train_peak_idx_bool의 boolean 배열을 직접 사용합니다.
            oof_pred_final_combined[train_peak_idx_bool.values] = peak_oof_pred_subset
            test_pred_final_combined[test_peak_idx_bool.values] = peak_test_pred_subset
    
    current_smape = smape(np.expm1(y_train_full), np.expm1(oof_pred_final_combined))
    print(f"    > {name} SMAPE: {current_smape:.6f}")

    return oof_pred_final_combined, test_pred_final_combined, current_smape


########################### 메인 실행 코드 ############################

# --- 1. 유형별 사전학습 및 예측 ---
print("\n--- [Step 1] 건물유형별 사전학습 및 예측 시작 ---")

type_wise_oof_preds = np.zeros(len(train))
type_wise_test_preds = np.zeros(len(test))
type_wise_smapes = []
type_wise_results = {} # Store results for conditional re-training

building_types = train['건물유형'].unique()

for btype in building_types:
    print(f"   > 건물유형: {btype} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_bt = train[train['건물유형'] == btype].copy()
    test_bt = test[test['건물유형'] == btype].copy()

    if len(train_bt) == 0 or len(test_bt) == 0:
        continue # Skip if no data for this type in train or test

    oof_pred, test_pred, current_smape = train_and_predict_subset(train_bt, test_bt, name=f"Type {btype}")
    
    type_wise_smapes.append(current_smape)
    type_wise_oof_preds[train_bt.index] = oof_pred
    type_wise_test_preds[test_bt.index] = test_pred
    type_wise_results[btype] = {"smape": current_smape, "train_indices": train_bt.index, "test_indices": test_bt.index}

print("--- [Step 1] 건물유형별 사전학습 및 예측 완료 ---")
t_w_s = np.mean(type_wise_smapes)
print(f"--- 평균 SMAPE (유형별): {t_w_s:.6f} ---")

# Conditional Re-training for Type-wise models
print("\n--- [Step 1.5] 건물유형별 조건부 재학습 시작 ---")
re_trained_type_wise_count = 0
for btype, result in type_wise_results.items():
    if result["smape"] > t_w_s: # If SMAPE is worse than average
        print(f"    > 재학습 필요: 건물유형 {btype} (SMAPE: {result['smape']:.6f} > 평균: {t_w_s:.6f})")
        train_bt = train.loc[result["train_indices"]].copy()
        test_bt = test.loc[result["test_indices"]].copy()

        # Optuna 튜닝 및 예측 함수 호출
        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna(
            train_bt, test_bt, features, target, SEED, n_trials=15 # n_trials 설정
        )
        
        if oof_pred_re is not None: # 데이터가 부족하여 건너뛴 경우가 아닐 때만 업데이트
            type_wise_oof_preds[train_bt.index] = oof_pred_re
            type_wise_test_preds[test_bt.index] = test_pred_re
            re_trained_type_wise_count += 1
            print(f"    > Type {btype} Re-train (Optuna) SMAPE: {current_re_smape:.6f}")
        else:
            print(f"    > Type {btype} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물유형별 조건부 재학습 완료 ({re_trained_type_wise_count}개 유형 재학습) ---")

y_true_all_train = np.log1p(train[target])
final_type_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(type_wise_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (유형별): {final_type_wise_smape_after_retrain:.6f} ---")


# --- 2. 건물별 사전학습 및 예측 ---
print("\n--- [Step 2] 건물별 사전학습 및 예측 시작 ---")

building_wise_oof_preds = np.zeros(len(train))
building_wise_test_preds = np.zeros(len(test))
building_wise_smapes = []
building_wise_results = {} # Store results for conditional re-training

building_ids = train['건물번호'].unique()

for bno in building_ids:
    print(f"   > 건물번호: {bno} 모델 학습 시작 (KFold 교차 검증 적용)")
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(train_b) == 0 or len(test_b) == 0:
        continue # Skip if no data for this building in train or test

    oof_pred, test_pred, current_smape = train_and_predict_subset(train_b, test_b, name=f"Building {bno}")
    
    building_wise_smapes.append(current_smape)
    building_wise_oof_preds[train_b.index] = oof_pred
    building_wise_test_preds[test_b.index] = test_pred
    building_wise_results[bno] = {"smape": current_smape, "train_indices": train_b.index, "test_indices": test_b.index}


print("--- [Step 2] 건물별 사전학습 및 예측 완료 ---")
b_w_s = np.mean(building_wise_smapes)
print(f"--- 평균 SMAPE (건물별): {b_w_s:.6f} ---")

print("\n--- [Step 2.5] 건물별 조건부 재학습 시작 ---")
re_trained_building_wise_count = 0
for bno, result in building_wise_results.items():
    if result["smape"] > b_w_s: # If SMAPE is worse than average
        print(f"    > 재학습 필요: 건물번호 {bno} (SMAPE: {result['smape']:.6f} > 평균: {b_w_s:.6f})")
        train_b = train.loc[result["train_indices"]].copy()
        test_b = test.loc[result["test_indices"]].copy()

        # tune_and_predict_with_optuna 함수 호출 시 필요한 모든 인자를 전달
        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna(
            train_b, test_b, features, target, SEED, n_trials=15 # n_trials 설정 (예: 15)
        )
        
        # Optuna 튜닝이 데이터를 반환한 경우에만 업데이트
        if oof_pred_re is not None:
            building_wise_oof_preds[train_b.index] = oof_pred_re
            building_wise_test_preds[test_b.index] = test_pred_re
            re_trained_building_wise_count += 1
            print(f"    > Building {bno} Re-train (Optuna) SMAPE: {current_re_smape:.6f}")
        else:
            print(f"    > Building {bno} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물별 조건부 재학습 완료 ({re_trained_building_wise_count}개 건물 재학습) ---")

y_true_all_train = np.log1p(train[target]) # 원본 train 데이터의 타겟 값
final_building_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(building_wise_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (건물별): {final_building_wise_smape_after_retrain:.6f} ---")

building_groups = [
    [28],                                  
    [72],                                   
    [19,58,75,91],                         
    [77],                                  
    [24],                                  
    [61,74,81],                             
    [32,42,65,79,99],                      
    [11,12,13,41,68,83,88],                
    [20,26,44,45,70,100],                  
    [1,2,3,4,5,6,7,8,27,33,34,35,37,47,67,86,96],
    [71],                                  
    [54,84],                               
    [17,18,29,30,31,40,43,48,49,51,52,53,60,63,64,76,78],
    [66],                                   
    [85],                                   
    [55,82],                                
    [15,16,39,59,73,92],                    
    [80,87],                                
    [89,90],                                
    [98],                                   
    [50],                                   
    [21,22,23],                             
    [46,93,94,95],                          
    [14,69],                                
    [57],                                   
    [97],                                   
    [36,38,56],                             
    [25,62],                                
    [9,10],                                  
]
print("\n--- [Step 3] 건물 그룹별 예측 시작 ---")

group_oof_preds = np.zeros(len(train))
group_test_preds = np.zeros(len(test))
group_smapes = []
group_results = {} # Store results for conditional re-training

for group_idx, group in enumerate(building_groups):
    print(f"   > 그룹 {group_idx+1}: 건물들 {group} 모델 학습 시작")

    train_g = train[train['건물번호'].isin(group)].copy()
    test_g = test[test['건물번호'].isin(group)].copy()

    if len(train_g) < 10 or len(test_g) < 1:
        print(f"   > 그룹 {group_idx+1} (건물: {group}) 데이터 부족으로 건너뜀.")
        continue   # 너무 작은 그룹 제외

    oof_pred, test_pred, current_smape = train_and_predict_subset(train_g, test_g, name=f"Group {group_idx+1}")
    
    group_smapes.append(current_smape)
    group_oof_preds[train_g.index] = oof_pred
    group_test_preds[test_g.index] = test_pred
    group_results[group_idx] = {"smape": current_smape, "train_indices": train_g.index, "test_indices": test_g.index, "buildings": group}


print("--- [Step 3] 건물 그룹별 예측 완료 ---")
g_w_s = np.mean(group_smapes)
print(f"--- 평균 SMAPE (그룹별): {g_w_s:.6f} ---")

# Conditional Re-training for Group-wise models
print("\n--- [Step 3.5] 건물 그룹별 조건부 재학습 시작 ---")
re_trained_group_wise_count = 0
for group_idx, result in group_results.items():
    if result["smape"] > g_w_s: # If SMAPE is worse than average
        print(f"    > 재학습 필요: 그룹 {group_idx+1} (건물: {result['buildings']}) (SMAPE: {result['smape']:.6f} > 평균: {g_w_s:.6f})")
        train_g = train.loc[result["train_indices"]].copy()
        test_g = test.loc[result["test_indices"]].copy()

        # tune_and_predict_with_optuna 함수 호출 시 필요한 모든 인자를 전달
        oof_pred_re, test_pred_re, current_re_smape = tune_and_predict_with_optuna(
            train_g, test_g, features, target, SEED, n_trials=15 # n_trials 설정 (예: 15)
        )
        
        # Optuna 튜닝이 데이터를 반환한 경우에만 업데이트
        if oof_pred_re is not None:
            group_oof_preds[train_g.index] = oof_pred_re
            group_test_preds[test_g.index] = test_pred_re
            re_trained_group_wise_count += 1
            print(f"    > Group {group_idx+1} Re-train (Optuna) SMAPE: {current_re_smape:.6f}")
        else:
            print(f"    > Group {group_idx+1} (Optuna) 데이터 부족으로 재학습 건너김.")

print(f"--- 건물 그룹별 조건부 재학습 완료 ({re_trained_group_wise_count}개 그룹 재학습) ---")

# --- 추가된 부분: 재학습 후 전체 그룹별 SMAPE 다시 계산 ---
# 재학습된 결과를 반영한 최종 group_oof_preds로 SMAPE를 다시 계산합니다.
# 이 값은 train DataFrame의 원래 타겟 값을 사용해야 합니다.
y_true_all_train = np.log1p(train[target]) # 원본 train 데이터의 타겟 값
final_group_wise_smape_after_retrain = smape(np.expm1(y_true_all_train), np.expm1(group_oof_preds))
print(f"--- 재학습 후 최종 평균 SMAPE (그룹별): {final_group_wise_smape_after_retrain:.6f} ---")


# --- 4. Optuna 가중평균 최적화 및 최종 예측 ---
print("\n--- [Step 4] Optuna 가중평균 최적화 및 최종 예측 시작 ---")

# 각 방법별 예측 결과를 딕셔너리로 구성
predictions_dict = {
    'type_wise': type_wise_oof_preds,
    'building_wise': building_wise_oof_preds,
    'group_wise': group_oof_preds
}

test_predictions_dict = {
    'type_wise': type_wise_test_preds,
    'building_wise': building_wise_test_preds,
    'group_wise': group_test_preds
}

# OptimizedEnsembleWeighting 인스턴스 생성
ensemble_optimizer = OptimizedEnsembleWeighting(seed=SEED)

# 가중치 최적화 (50 trials)
print("Optuna를 사용한 앙상블 가중치 최적화 시작...")
best_weights = ensemble_optimizer.optimize_weights(
    predictions_dict, 
    y_true_all_train, 
    n_trials=50
)

print(f"\n최적화된 앙상블 가중치:")
for method, weight in zip(predictions_dict.keys(), best_weights):
    print(f"  {method}: {weight:.4f}")

# 최적화된 가중치로 OOF 예측
optimized_oof_preds = ensemble_optimizer.predict(predictions_dict)

# 최적화된 가중치로 테스트 예측
optimized_test_preds = ensemble_optimizer.predict(test_predictions_dict)

# 잔차 계산 및 잔차 모델 학습
print("\n잔차 모델 학습 시작...")
residuals = y_true_all_train - optimized_oof_preds

# 잔차 모델 (LGBMRegressor)
residual_model = LGBMRegressor(objective='mae', random_state=SEED, n_estimators=300, 
                               learning_rate=0.05, num_leaves=31, max_depth=-1, n_jobs=-1, verbose=-1)

# 잔차 모델 학습을 위한 피처 스케일링
scaler_res = StandardScaler()
X_train_scaled_res = scaler_res.fit_transform(train[features])
X_test_scaled_res = scaler_res.transform(test[features])

# 전체 훈련 데이터로 잔차 모델 학습
residual_model.fit(X_train_scaled_res, residuals)

# 테스트 데이터에 대한 잔차 예측
predicted_residuals = residual_model.predict(X_test_scaled_res)

# 최종 예측: 최적화된 앙상블 예측 + 예측된 잔차
final_preds_residual = optimized_test_preds + predicted_residuals

# 예측값이 음수가 되지 않도록 하고 원래 스케일로 변환
final_predictions_exp = np.expm1(final_preds_residual)
final_predictions_exp[final_predictions_exp < 0] = 0 # 음수 값 0으로 처리

# 최종 OOF SMAPE 계산 (모델 성능 평가용)
final_oof_combined_preds = optimized_oof_preds + residual_model.predict(X_train_scaled_res)
final_oof_smape = smape(np.expm1(y_true_all_train), np.expm1(final_oof_combined_preds))

print(f"--- [Step 4] Optuna 가중평균 최적화 및 최종 예측 완료 ---")
print(f"최종 SMAPE (Optuna 최적화 + 잔차 모델링): {final_oof_smape:.6f}")
print(f"Optuna 최적화만 적용한 SMAPE: {ensemble_optimizer.study.best_value:.6f}")

# 성능 비교를 위한 기존 평균 앙상블 SMAPE 계산
simple_ensemble_oof = (type_wise_oof_preds + building_wise_oof_preds + group_oof_preds) / 3
simple_ensemble_smape = smape(np.expm1(y_true_all_train), np.expm1(simple_ensemble_oof))
print(f"기존 단순 평균 앙상블 SMAPE: {simple_ensemble_smape:.6f}")
print(f"Optuna 최적화 개선효과: {simple_ensemble_smape - ensemble_optimizer.study.best_value:.6f}")

print("[3] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions_exp
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_oof_smape:.4f}".replace('.', '_')

filename = f"01_{today}_SMAPE_{score_str}_optuna.csv"
samplesub.to_csv(save_path + filename, index=False)

with open("./Energy/01/11_model_optuna.py", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"건물 유형별 학습 SMAPE : {final_type_wise_smape_after_retrain}\n")
    f.write(f"건물 번호별 학습 SMAPE : {final_building_wise_smape_after_retrain}\n")
    f.write(f"건물 그룹별 학습 SMAPE : {final_group_wise_smape_after_retrain}\n") 
    f.write(f"단순 평균 앙상블 SMAPE : {simple_ensemble_smape}\n")
    f.write(f"Optuna 최적화 앙상블 SMAPE : {ensemble_optimizer.study.best_value}\n")
    f.write(f"최종 잔차학습 SMAPE : {final_oof_smape}\n")
    f.write(f"Optuna 최적 가중치: {dict(zip(['type_wise', 'building_wise', 'group_wise'], best_weights))}\n")
    f.write("="*40 + "\n")

# Optuna 최적화 결과 상세 저장
optuna_results = {
    'seed': SEED,
    'best_weights': {
        'type_wise': best_weights[0],
        'building_wise': best_weights[1], 
        'group_wise': best_weights[2]
    },
    'best_smape': ensemble_optimizer.study.best_value,
    'simple_ensemble_smape': simple_ensemble_smape,
    'final_smape_with_residual': final_oof_smape,
    'improvement': simple_ensemble_smape - ensemble_optimizer.study.best_value,
    'individual_smapes': {
        'type_wise': final_type_wise_smape_after_retrain,
        'building_wise': final_building_wise_smape_after_retrain,
        'group_wise': final_group_wise_smape_after_retrain
    }
}

with open(csv_path + f"optuna_ensemble_results_seed_{SEED}.json", "w") as f:
    json.dump(optuna_results, f, indent=2, ensure_ascii=False)

print(f"[4] 종료")
print(f"Optuna 앙상블 최적화 결과가 저장되었습니다: optuna_ensemble_results_seed_{SEED}.json")