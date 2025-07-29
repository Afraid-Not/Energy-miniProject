# ========================
# 임포트 및 랜덤 시드 고정
print(f"[03_hyperparameter_tuning] 시작")
# ========================
import pandas as pd
import numpy as np
import optuna
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping

seed_file = "./Energy/_best_code/(SEED_COUNT)hyperparameter.json"

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

# 이 값들은 전역 변수로 정의되어 있다고 가정합니다.
# 예시:
def log_smape(y_true, y_pred):
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.maximum(0, y_pred))
    numerator = np.abs(y_true_log - y_pred_log)
    denominator = (np.abs(y_true_log) + np.abs(y_pred_log)) / 2
    denominator[denominator == 0] = 1
    smape_values = numerator / denominator
    return np.mean(smape_values) * 100


print(f"[1] 데이터 로드 및 전처리 (생략 - 기존 코드에서 이미 수행)")
data_path = './Energy/'
save_path = './Energy/_best_code/'
traintest_save_path = './Energy/_best_code/best_train_test/'
os.makedirs(save_path, exist_ok=True)

# Preprocessed data 로드
train_call = 'best_train_SEED44_up.csv'
test_call = 'best_test_SEED44_up.csv'

train = pd.read_csv(traintest_save_path + train_call)
test = pd.read_csv(traintest_save_path + test_call)
samplesub = pd.read_csv(data_path +'sample_submission.csv')

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

exclude_cols = ['건물번호', '일시', '전력소비량(kWh)', '건물유형', '날짜']
features = [col for col in train.columns if col not in exclude_cols]
target = '전력소비량(kWh)'

N_SPLIT = 5
KFOLD = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

def tune_model_with_optuna(model_type, X_train_full, y_train_full, seed_val, n_trials=15):
    
    def objective(trial):
        if model_type == 'lgbm':
            param = {
                'objective': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
                'random_state': seed_val, 'n_jobs': -1, 'verbosity': -1,
            }
            model = LGBMRegressor(**param)
        elif model_type == 'xgb':
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
                'random_state': seed_val, 'n_jobs': -1, 'verbosity': 0,
                'early_stopping_rounds' : 50
            }
            model = XGBRegressor(**param)
        elif model_type == 'cat':
            param = {
                'loss_function': 'MAE',
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 5, 15),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': seed_val, 'verbose': -1
            }
            model = CatBoostRegressor(**param)
        else:
            raise ValueError("Invalid model_type")

        # KFold for tuning
        kf_tune = KFold(n_splits=2, shuffle=True, random_state=seed_val)
        oof_preds = np.zeros(len(X_train_full))
        
        for fold, (train_idx, val_idx) in enumerate(kf_tune.split(X_train_full, y_train_full)):
            X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            if model_type == 'xgb' :
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif model_type == 'lgbm' :
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(0)] if model_type == 'lgbm' else None)
            else :
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

            oof_preds[val_idx] = model.predict(X_val)
            trial.report(mean_absolute_error(y_val, oof_preds[val_idx]), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return mean_absolute_error(y_train_full, oof_preds)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_val))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n[{model_type.upper()}] Optuna Best MAE: {study.best_value:.6f}")
    print(f"[{model_type.upper()}] Optuna Best Params: {study.best_params}")
    return study.best_params, study.best_value

# 가상의 데이터 (실제 데이터프레임으로 대체)
# X_train, y_train, X_test는 사전에 준비되어 있어야 합니다.
X_train_full = train[features].reset_index(drop=True)
y_train_full = np.log1p(train[target]).reset_index(drop=True)
X_test_full = test[features].reset_index(drop=True)

# 모델별로 최적 파라미터 찾기
best_params_dict = {}
best_params_dict['lgbm'], lgbm_smape = tune_model_with_optuna('lgbm', X_train_full, y_train_full, SEED, n_trials=1)
best_params_dict['xgb'], xgb_smape = tune_model_with_optuna('xgb', X_train_full, y_train_full, SEED, n_trials=1)
best_params_dict['cat'], cat_smape = tune_model_with_optuna('cat', X_train_full, y_train_full, SEED, n_trials=1)

best_smape = (lgbm_smape + xgb_smape + cat_smape) / 3

# 최적 파라미터를 JSON 파일로 저장
with open('./Energy/_best_code/best_params.json', 'w') as f:
    json.dump(best_params_dict, f, indent=4)

with open(save_path + "(LOG)03_tuning.txt", "a") as f:
    f.write(f"<파일명 : 03_hyperparameter_tuning.py>\n")
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"BEST SMAPE : {best_smape:.6f}\n")
    f.write("="*40 + "\n")
    
print(f"[01_preprocessing] 종료")