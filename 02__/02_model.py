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
from sklearn.linear_model import RidgeCV, LinearRegression
from lightgbm import early_stopping, log_evaluation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Optuna 로그 완전 억제
optuna.logging.set_verbosity(optuna.logging.ERROR)

print("[02 - model with Residual Learning] 시작")

seed_file = "./Energy/02/(SEED_COUNT)02_model.json"

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

print(f"[1] 데이터 로드 및 전처리 (생략 - 기존 코드에서 이미 수행)")
data_path = './Energy/'
csv_path = './Energy/02/'
save_path = f'./Energy/02/{SEED}_submission/'
os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
train = pd.read_csv(csv_path + '06_train_47_2.csv')
test_call = '06_test_47_2.csv'

test = pd.read_csv(csv_path + test_call)
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
features = [col for col in train.columns if col not in exclude_cols]
target = '전력소비량(kWh)'

N_SPLIT = 5
KFOLD = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

def optimize_xgb_hyperparams(X_train, y_train, seed_val, n_trials=50):
    """XGBoost 하이퍼파라미터 최적화"""
    
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': seed_val,
            'n_jobs': -1,
        }

        # 단순히 train/validation split으로 평가
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed_val
        )

        model = XGBRegressor(**param)
        model.fit(X_train_split, y_train_split)
        pred = model.predict(X_val_split)
        score = smape(y_val_split, pred)
            
        return score

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value

def optimize_lgbm_hyperparams(X_train, y_train, seed_val, n_trials=50):
    """LightGBM 하이퍼파라미터 최적화"""
    
    def objective(trial):
        param = {
            'objective': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
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

        # 단순히 train/validation split으로 평가
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed_val
        )

        model = LGBMRegressor(**param)
        model.fit(X_train_split, y_train_split)
        pred = model.predict(X_val_split)
        score = mean_absolute_error(y_val_split, pred)
            
        return score

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value

def optimize_residual_hyperparams(X_train, y_residual, seed_val, n_trials=30):
    """잔차 학습용 모델 하이퍼파라미터 최적화"""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['xgb', 'lgbm', 'ridge'])
        
        if model_type == 'xgb':
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': seed_val,
                'n_jobs': -1,
            }
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_residual, test_size=0.2, random_state=seed_val
            )
            model = XGBRegressor(**param)
            model.fit(X_train_split, y_train_split)
            pred = model.predict(X_val_split)
            score = mean_squared_error(y_val_split, pred)
            
        elif model_type == 'lgbm':
            param = {
                'objective': 'regression',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 60),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
                'random_state': seed_val,
                'n_jobs': -1,
                'verbose': -1,
            }
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_residual, test_size=0.2, random_state=seed_val
            )
            model = LGBMRegressor(**param)
            model.fit(X_train_split, y_train_split)
            pred = model.predict(X_val_split)
            score = mean_squared_error(y_val_split, pred)
            
        else:  # ridge
            alpha = trial.suggest_float('alpha', 0.1, 100.0, log=True)
            param = {'alpha': alpha}
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_residual, test_size=0.2, random_state=seed_val
            )
            model = RidgeCV(alphas=[alpha], cv=3)
            model.fit(X_train_split, y_train_split)
            pred = model.predict(X_val_split)
            score = mean_squared_error(y_val_split, pred)
        
        return score

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value

def optimize_ensemble_weights(base_oof, residual_oof, y_true, seed_val, n_trials=100):
    """앙상블 가중치 최적화 (기본 모델 + 잔차 모델)"""
    
    def objective(trial):
        base_weight = trial.suggest_float('base_weight', 0.5, 1.0)
        residual_weight = trial.suggest_float('residual_weight', 0.0, 0.5)
        
        ensemble_pred = base_weight * base_oof + residual_weight * residual_oof
        score = smape(np.expm1(y_true), np.expm1(ensemble_pred))
        
        return score
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value

def select_features_with_importance(X_train, y_train, X_test, feature_names, building_name):
    """Feature importance 기반 피처 선택"""
    print(f"    > {building_name} - Feature Importance 분석 및 피처 선택 중...")
    
    # XGBoost와 LightGBM으로 feature importance 측정
    xgb_base = XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    lgbm_base = LGBMRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbose=-1)
    
    # 기본 모델 학습
    xgb_base.fit(X_train, y_train)
    lgbm_base.fit(X_train, y_train)
    
    # Feature importance 평균 계산
    xgb_importance = xgb_base.feature_importances_
    lgbm_importance = lgbm_base.feature_importances_
    avg_importance = (xgb_importance + lgbm_importance) / 2
    
    # Importance 기준으로 피처 순위 매기기
    importance_indices = np.argsort(avg_importance)[::-1]  # 내림차순 정렬
    
    # 상위 피처들 선택 (최소 10개, 최대 전체 피처의 80% 또는 50개 중 작은 값)
    min_features = min(10, len(feature_names))
    max_features = min(70, int(len(feature_names) * 0.8))
    
    # Importance가 0보다 큰 피처들 중에서 선택
    valid_features = importance_indices[avg_importance[importance_indices] > 0]
    
    # 선택할 피처 수 결정
    if len(valid_features) < min_features:
        selected_indices = importance_indices[:min_features]  # 최소 개수 보장
    elif len(valid_features) > max_features:
        selected_indices = valid_features[:max_features]  # 최대 개수 제한
    else:
        selected_indices = valid_features  # 모든 valid 피처 사용
    
    # 선택된 피처로 데이터 변환
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # 선택된 피처 이름들
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"    > {building_name} - 원본 피처 수: {len(feature_names)}, 선택된 피처 수: {len(selected_features)}")
    print(f"    > {building_name} - Top 5 피처: {selected_features[:5]}")
    
    # 선택된 피처들의 importance 출력
    selected_importance = avg_importance[selected_indices]
    print(f"    > {building_name} - Top 5 Importance: {selected_importance[:5].round(4)}")
    
    return X_train_selected, X_test_selected, selected_features

def train_residual_model(X_train, y_residual, X_test, best_params, building_name):
    """잔차 모델 학습"""
    model_type = best_params['model_type']
    
    residual_oof_preds = np.zeros(len(X_train))
    residual_test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train, y_residual)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_residual.iloc[train_idx], y_residual.iloc[val_idx]
        
        if model_type == 'xgb':
            model_params = {k: v for k, v in best_params.items() if k != 'model_type'}
            model = XGBRegressor(**model_params)
        elif model_type == 'lgbm':
            model_params = {k: v for k, v in best_params.items() if k != 'model_type'}
            model = LGBMRegressor(**model_params)
        else:  # ridge
            model = RidgeCV(alphas=[best_params['alpha']], cv=3)
        
        model.fit(X_train_fold, y_train_fold)
        residual_oof_preds[val_idx] = model.predict(X_val_fold)
        residual_test_preds += model.predict(X_test) / N_SPLIT
    
    return residual_oof_preds, residual_test_preds

def train_and_predict_building(train_subset, test_subset, building_name):
    """건물별 XGB+LGBM 앙상블 + 잔차 학습 모델"""
    
    X_train = train_subset[features].reset_index(drop=True)
    y_train = np.log1p(train_subset[target].reset_index(drop=True))
    X_test = test_subset[features].reset_index(drop=True)

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, float('inf')

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature importance 기반 피처 선택
    X_train_selected, X_test_selected, selected_features = select_features_with_importance(
        X_train_scaled, y_train, X_test_scaled, features, building_name
    )
    
    X_train_scaled_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_scaled_df = pd.DataFrame(X_test_selected, columns=selected_features)

    print(f"    > {building_name} - XGB 하이퍼파라미터 최적화 중...")
    # XGB 최적화
    with tqdm(total=30, desc=f"XGB Optimization", leave=False) as pbar:
        best_xgb_params, best_xgb_score = optimize_xgb_hyperparams(X_train_scaled_df, y_train, SEED, n_trials=30)
        pbar.update(30)
    
    print(f"    > {building_name} - XGB Best SMAPE: {best_xgb_score:.6f}")

    print(f"    > {building_name} - LGBM 하이퍼파라미터 최적화 중...")
    # LGBM 최적화
    with tqdm(total=30, desc=f"LGBM Optimization", leave=False) as pbar:
        best_lgbm_params, best_lgbm_score = optimize_lgbm_hyperparams(X_train_scaled_df, y_train, SEED, n_trials=30)
        pbar.update(30)
    
    print(f"    > {building_name} - LGBM Best SMAPE: {best_lgbm_score:.6f}")

    # === STEP 1: 기본 앙상블 모델 학습 ===
    xgb_oof_preds = np.zeros(len(X_train_selected))
    lgbm_oof_preds = np.zeros(len(X_train_selected))
    xgb_test_preds = np.zeros(len(X_test_selected))
    lgbm_test_preds = np.zeros(len(X_test_selected))

    for fold, (train_idx, val_idx) in enumerate(KFOLD.split(X_train_selected, y_train)):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # XGB
        xgb_model = XGBRegressor(**best_xgb_params)
        xgb_model.fit(X_train_fold, y_train_fold)
        xgb_oof_preds[val_idx] = xgb_model.predict(X_val_fold)
        xgb_test_preds += xgb_model.predict(X_test_selected) / N_SPLIT

        # LGBM
        lgbm_model = LGBMRegressor(**best_lgbm_params)
        lgbm_model.fit(X_train_fold, y_train_fold)
        lgbm_oof_preds[val_idx] = lgbm_model.predict(X_val_fold)
        lgbm_test_preds += lgbm_model.predict(X_test_selected) / N_SPLIT

    # 기본 앙상블 예측 (가중 평균)
    base_oof_preds = 0.5 * xgb_oof_preds + 0.5 * lgbm_oof_preds
    base_test_preds = 0.5 * xgb_test_preds + 0.5 * lgbm_test_preds
    
    base_smape = smape(np.expm1(y_train), np.expm1(base_oof_preds))
    print(f"    > {building_name} - Base Ensemble SMAPE: {base_smape:.6f}")

    # === STEP 2: 잔차 계산 및 잔차 모델 학습 ===
    print(f"    > {building_name} - 잔차 모델 최적화 중...")
    
    # 잔차 계산
    residuals = y_train - base_oof_preds
    
    # 잔차 모델 최적화
    with tqdm(total=30, desc=f"Residual Model Opt", leave=False) as pbar:
        best_residual_params, best_residual_score = optimize_residual_hyperparams(
            X_train_scaled_df, residuals, SEED, n_trials=30
        )
        pbar.update(30)
    
    print(f"    > {building_name} - Best Residual Model: {best_residual_params['model_type']}")
    print(f"    > {building_name} - Residual MSE: {best_residual_score:.6f}")
    
    # 잔차 모델 학습
    residual_oof_preds, residual_test_preds = train_residual_model(
        X_train_selected, residuals, X_test_selected, best_residual_params, building_name
    )

    # === STEP 3: 최종 앙상블 가중치 최적화 ===
    print(f"    > {building_name} - 최종 앙상블 가중치 최적화 중...")
    with tqdm(total=50, desc=f"Final Ensemble Opt", leave=False) as pbar:
        best_weights, best_ensemble_score = optimize_ensemble_weights(
            base_oof_preds, base_oof_preds + residual_oof_preds, y_train, SEED, n_trials=50
        )
        pbar.update(50)
    
    print(f"    > {building_name} - Best Final SMAPE: {best_ensemble_score:.6f}")
    print(f"    > {building_name} - Best Weights: Base={best_weights['base_weight']:.4f}, Residual={best_weights['residual_weight']:.4f}")

    # === STEP 4: 최종 예측 ===
    final_oof_preds = (best_weights['base_weight'] * base_oof_preds + 
                       best_weights['residual_weight'] * (base_oof_preds + residual_oof_preds))
    
    final_test_preds = (best_weights['base_weight'] * base_test_preds + 
                        best_weights['residual_weight'] * (base_test_preds + residual_test_preds))

    return final_oof_preds, final_test_preds, best_ensemble_score

# --- 건물별 학습 및 예측 ---
print("\n--- [Step 1] 건물별 잔차 학습 포함 앙상블 모델 학습 시작 ---")

building_wise_oof_preds = np.zeros(len(train))
building_wise_test_preds = np.zeros(len(test))
building_wise_smapes = []
building_wise_results = {}

building_ids = train['건물번호'].unique()

for bno in building_ids:
    print(f"\n   > 건물번호: {bno} 모델 학습 시작")
    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    if len(train_b) == 0 or len(test_b) == 0:
        print(f"   > 건물번호: {bno} 데이터 부족으로 건너뛰기")
        continue

    oof_pred, test_pred, current_smape = train_and_predict_building(train_b, test_b, f"Building {bno}")
    
    if oof_pred is not None:
        building_wise_smapes.append(current_smape)
        building_wise_oof_preds[train_b.index] = oof_pred
        building_wise_test_preds[test_b.index] = test_pred
        building_wise_results[bno] = {
            "smape": current_smape, 
            "train_indices": train_b.index, 
            "test_indices": test_b.index
        }
        print(f"   > 건물번호: {bno} 최종 SMAPE: {current_smape:.6f}")

print("\n--- [Step 1] 건물별 잔차 학습 포함 앙상블 모델 학습 완료 ---")

# 전체 성능 계산
y_true_all_train = np.log1p(train[target])
final_building_wise_smape = smape(np.expm1(y_true_all_train), np.expm1(building_wise_oof_preds))
print(f"--- 전체 평균 SMAPE (잔차 학습 포함): {final_building_wise_smape:.6f} ---")

# 최종 예측값 준비
final_predictions = np.expm1(building_wise_test_preds)
final_predictions[final_predictions < 0] = 0

print("\n[2] 예측 결과 저장 시작")
samplesub['answer'] = final_predictions
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{final_building_wise_smape:.4f}".replace('.', '_')

filename = f"02_{today}_RESIDUAL_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

# 로그 저장
with open(f"./Energy/02/{SEED}_submission/(LOG)model_residual.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"test file name : {test_call}\n")
    f.write(f"건물별 잔차학습 앙상블 SMAPE : {final_building_wise_smape:.6f}\n")
    f.write("건물별 세부 결과:\n")
    for bno, result in building_wise_results.items():
        f.write(f"  - 건물 {bno}: SMAPE {result['smape']:.6f}\n")
    f.write("="*40 + "\n")

print(f"[3] 종료 - 최종 SMAPE (잔차 학습 포함): {final_building_wise_smape:.6f}")