import os, json, shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna

# Seed 관리
seed_path = './Energy/seed_count/seed.json'
os.makedirs(os.path.dirname(seed_path), exist_ok=True)
if not os.path.exists(seed_path):
    seed = 73
else:
    with open(seed_path, 'r') as f:
        seed_data = json.load(f)
    seed = seed_data['seed'] + 1
with open(seed_path, 'w') as f:
    json.dump({'seed': seed}, f)

# 데이터 로딩
train = pd.read_csv('./Energy/train.csv')
test = pd.read_csv('./Energy/test.csv')
building = pd.read_csv('./Energy/building_info.csv')
building.replace('-', np.nan, inplace=True)
building = building.fillna(0)
building = building.rename(columns={
    '건물번호': 'building_id', '건물유형': 'building_type',
    '연면적(m2)': 'total_area', '냉방면적(m2)': 'cool_area',
    '태양광용량(kW)': 'solar_capacity', 'ESS저장용량(kWh)': 'ess_capacity',
    'PCS용량(kW)': 'pcs_capacity'
})
for col in ['total_area', 'cool_area', 'solar_capacity', 'ess_capacity', 'pcs_capacity']:
    building[col] = pd.to_numeric(building[col], errors='coerce')

# 시간 파생
def create_time_features(df):
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    return df

train = create_time_features(train)
test = create_time_features(test)
train = train.merge(building, left_on='건물번호', right_on='building_id', how='left')
test = test.merge(building, left_on='건물번호', right_on='building_id', how='left')

# 평가 함수
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff) * 100

# Optuna 튜닝 함수
def tune_model(model_name, X, y):
    def objective(trial):
        if model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': seed
            }
            model = XGBRegressor(**params)
        elif model_name == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': seed
            }
            model = LGBMRegressor(**params)
        else:  # cat
            params = {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'random_seed': seed,
                'verbose': 0
            }
            model = CatBoostRegressor(**params)

        model.fit(X, y)
        pred = model.predict(X)
        return smape(y, pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    return study.best_params

# 예측 수행
predictions = []
all_losses = []
SMAPE_FILTER = 30
ROLLING_N = 6

for bid in train['건물번호'].unique():
    train_b = train[train['건물번호'] == bid].copy()
    test_b = test[test['건물번호'] == bid].copy()

    train_b['target'] = train_b['전력소비량(kWh)']
    train_b = train_b.sort_values('일시')
    train_b['lag1'] = train_b['target'].shift(1)
    train_b['rolling6'] = train_b['target'].shift(1).rolling(ROLLING_N).mean()
    train_b = train_b.fillna(0)

    test_b['lag1'] = train_b['target'].iloc[-1] if not train_b.empty else 0
    test_b['rolling6'] = train_b['target'].iloc[-ROLLING_N:].mean() if not train_b.empty else 0
    test_b = test_b.fillna(0)

    feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
                    'month', 'day', 'hour', 'weekday', 'is_weekend',
                    'total_area', 'cool_area', 'solar_capacity',
                    'ess_capacity', 'pcs_capacity', 'lag1', 'rolling6']
    feature_cols = [c for c in feature_cols if c in train_b.columns and c in test_b.columns]
    X_train, y_train, X_test = train_b[feature_cols], train_b['target'], test_b[feature_cols]

    if len(X_train) < 10:
        print(f"[{bid}] 학습불가 → 평균값 대체")
        preds = [y_train.mean()] * len(X_test)
    else:
        # 각 모델 파라미터 튜닝 후 학습
        best_xgb = tune_model('xgb', X_train, y_train)
        best_lgb = tune_model('lgb', X_train, y_train)
        best_cat = tune_model('cat', X_train, y_train)

        model_xgb = XGBRegressor(**best_xgb)
        model_lgb = LGBMRegressor(**best_lgb)
        model_cat = CatBoostRegressor(**best_cat, verbose=0)

        model_xgb.fit(X_train, y_train)
        model_lgb.fit(X_train, y_train)
        model_cat.fit(X_train, y_train)

        pred_xgb = model_xgb.predict(X_test)
        pred_lgb = model_lgb.predict(X_test)
        pred_cat = model_cat.predict(X_test)

        preds = (pred_xgb + pred_lgb + pred_cat) / 3
        try:
            local_smape = smape(y_train[-len(preds):], preds)
        except:
            local_smape = 999.0

        if local_smape > SMAPE_FILTER:
            print(f"[{bid}] SMAPE={local_smape:.2f} → 대체 적용")
            preds = [y_train.mean()] * len(X_test)
        all_losses.append((bid, local_smape))

    predictions.extend(preds)

# 제출 파일 저장
sub = pd.read_csv('./Energy/sample_submission.csv')
sub['answer'] = predictions

avg_smape = np.mean([loss for _, loss in all_losses])
epochs = 1
date_str = datetime.now().strftime('%Y%m%d')
save_name = f'energy_{seed}_{date_str}_{epochs}_{avg_smape:.4f}.csv'
save_path = Path('./Energy/submission')
save_path.mkdir(parents=True, exist_ok=True)
sub.to_csv(save_path / save_name, index=False)

# 출력
print(f"# 저장완료")
print(f"#      Seed = {seed}")
print(f"#     Smape = {avg_smape:.4f}")
print(f"# File_name = {save_name}")

# 삭제 여부
# SCORE_THRESHOLD = 7.48062
# if avg_smape > SCORE_THRESHOLD:
#     shutil.rmtree("./Energy/submission")
#     print(f"🚫 Score {avg_smape:.5f} > 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
# else:
#     print(f"🎉 Score {avg_smape:.5f} < 기준 {SCORE_THRESHOLD} → 디렉토리 유지")
