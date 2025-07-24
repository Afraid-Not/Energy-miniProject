import os
import pandas as pd
import numpy as np
import random
import datetime, json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

seed_file = "./Energy/02/log/(Gift_model)SEED_COUNT.json"

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

# 경로 설정
data_path = './Energy/'
csv_path = './Energy/02/'
trainer = './Energy/02/trainer/'
save_path = f'./Energy/02/{SEED}_(GIFT)submissions/'
os.makedirs(save_path, exist_ok=True)

train_call = '06_train_63_optuna.csv'
test_call = '06_test_63_optuna.csv'

train = pd.read_csv(trainer + train_call)
test = pd.read_csv(trainer + test_call)
samplesub = pd.read_csv(data_path + 'sample_submission.csv')

print(list(set(train.columns)))
# exit()
print("[1] 데이터 로딩 및 초기 전처리 완료")

exclude_cols = ['건물번호', '일시', '전력소비량(kWh)', '건물유형', 'date']
features = [col for col in train.columns if col not in exclude_cols]
target = '전력소비량(kWh)'

print("[2] 전처리 완료")

# Optuna 하이퍼파라미터 최적화 함수
def optimize_hyperparameters(x_sample, y_sample, n_trials=50):
    def objective(trial):
        # XGBoost 파라미터
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 800),  # 범위 축소
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.05, 0.15),  # 범위 축소
            'max_depth': trial.suggest_int('xgb_max_depth', 4, 6),  # 범위 축소
            'subsample': trial.suggest_float('xgb_subsample', 0.8, 0.95),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.8, 0.95),
        }

        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 500, 800),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.05, 0.15),
            'max_depth': trial.suggest_int('lgb_max_depth', 4, 6),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 30, 60),  # 범위 축소
            'subsample': trial.suggest_float('lgb_subsample', 0.8, 0.95),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.8, 0.95),
        }

        cat_params = {
            'n_estimators': trial.suggest_int('cat_n_estimators', 500, 800),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.05, 0.15),
            'max_depth': trial.suggest_int('cat_max_depth', 4, 6),
        }
        
        # K-Fold 교차 검증
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(x_sample):
            x_train_cv, x_val_cv = x_sample.iloc[train_idx], x_sample.iloc[val_idx]
            y_train_cv, y_val_cv = y_sample.iloc[train_idx], y_sample.iloc[val_idx]
            
            # 스케일링
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_cv)
            x_val_scaled = scaler.transform(x_val_cv)
            
            # 모델 학습
            xgb_model = XGBRegressor(random_state=SEED, **xgb_params, objective='reg:squarederror')
            lgb_model = LGBMRegressor(random_state=SEED, **lgb_params, objective='mae', verbose=-1)
            cat_model = CatBoostRegressor(random_seed=SEED, **cat_params, verbose=0, loss_function='MAE')
            
            xgb_model.fit(x_train_scaled, y_train_cv, eval_set=[(x_val_scaled, y_val_cv)], verbose=False)
            lgb_model.fit(x_train_scaled, y_train_cv, eval_set=[(x_val_scaled, y_val_cv)], 
                         callbacks=[lgb.early_stopping(50, verbose=False)])
            cat_model.fit(x_train_scaled, y_train_cv, eval_set=(x_val_scaled, y_val_cv), early_stopping_rounds=50)
            
            # 앙상블 예측
            pred_xgb = xgb_model.predict(x_val_scaled)
            pred_lgb = lgb_model.predict(x_val_scaled)
            pred_cat = cat_model.predict(x_val_scaled)
            
            # 간단한 평균 앙상블
            ensemble_pred = (pred_xgb + pred_lgb + pred_cat) / 3
            
            # SMAPE 계산
            smape = np.mean(200 * np.abs(np.expm1(ensemble_pred) - np.expm1(y_val_cv)) /
                           (np.abs(np.expm1(ensemble_pred)) + np.abs(np.expm1(y_val_cv)) + 1e-6))
            cv_scores.append(smape)
        
        return np.mean(cv_scores)
    
    # Optuna 스터디 생성 (로그 억제)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    
    # 진행상태바와 함께 최적화 실행
    with tqdm(total=n_trials, desc="Optuna Optimization", leave=False) as pbar:
        for i in range(n_trials):
            study.optimize(objective, n_trials=1)
            pbar.update(1)
            pbar.set_postfix({'Best SMAPE': f"{study.best_value:.4f}"})
    
    return study.best_params

# 모델 학습 및 예측 함수
def train_and_predict_model(x, y, x_test_final, best_params):
    """모델 학습 및 예측을 수행하는 함수"""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    fold_predictions = []
    fold_smapes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 스케일링
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_fold)
        x_val_scaled = scaler.transform(x_val_fold)
        x_test_scaled = scaler.transform(x_test_final)
        
        # 최적 파라미터로 모델 학습
        xgb_model = XGBRegressor(
            n_estimators=best_params['xgb_n_estimators'],
            learning_rate=best_params['xgb_learning_rate'],
            max_depth=best_params['xgb_max_depth'],
            subsample=best_params['xgb_subsample'],
            colsample_bytree=best_params['xgb_colsample_bytree'],
            random_state=SEED, objective='reg:squarederror'
        )
        
        lgb_model = LGBMRegressor(
            n_estimators=best_params['lgb_n_estimators'],
            learning_rate=best_params['lgb_learning_rate'],
            max_depth=best_params['lgb_max_depth'],
            num_leaves=best_params['lgb_num_leaves'],
            subsample=best_params['lgb_subsample'],
            colsample_bytree=best_params['lgb_colsample_bytree'],
            random_state=SEED, objective='mae', verbose=-1
        )
        
        cat_model = CatBoostRegressor(
            n_estimators=best_params['cat_n_estimators'],
            learning_rate=best_params['cat_learning_rate'],
            max_depth=best_params['cat_max_depth'],
            random_seed=SEED, verbose=0, loss_function='MAE'
        )
        
        # 모델 학습
        xgb_model.fit(x_train_scaled, y_train_fold, eval_set=[(x_val_scaled, y_val_fold)], verbose=False)
        lgb_model.fit(x_train_scaled, y_train_fold, eval_set=[(x_val_scaled, y_val_fold)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        cat_model.fit(x_train_scaled, y_train_fold, eval_set=(x_val_scaled, y_val_fold), early_stopping_rounds=50)
        
        # Level 1 predictions
        oof_train_lvl1 = np.vstack([
            xgb_model.predict(x_train_scaled),
            lgb_model.predict(x_train_scaled),
            cat_model.predict(x_train_scaled)
        ]).T
        oof_val_lvl1 = np.vstack([
            xgb_model.predict(x_val_scaled),
            lgb_model.predict(x_val_scaled),
            cat_model.predict(x_val_scaled)
        ]).T
        oof_test_lvl1 = np.vstack([
            xgb_model.predict(x_test_scaled),
            lgb_model.predict(x_test_scaled),
            cat_model.predict(x_test_scaled)
        ]).T
        
        # Level 2 Meta model
        meta_model = RidgeCV()
        meta_model.fit(oof_train_lvl1, y_train_fold)
        val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
        test_pred_lvl2 = meta_model.predict(oof_test_lvl1)
        
        # Level 3 Final model
        final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
        final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val_fold)
        
        # 검증 점수 계산
        val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
        fold_smape = np.mean(200 * np.abs(np.expm1(val_final) - np.expm1(y_val_fold)) /
                            (np.abs(np.expm1(val_final)) + np.abs(np.expm1(y_val_fold)) + 1e-6))
        fold_smapes.append(fold_smape)
        
        # 테스트 예측
        fold_pred = final_model.predict(test_pred_lvl2.reshape(-1, 1))
        fold_predictions.append(fold_pred)
    
    # K-Fold 평균 예측
    final_prediction = np.mean(fold_predictions, axis=0)
    avg_fold_smape = np.mean(fold_smapes)
    
    return final_prediction, avg_fold_smape

# 최종 예측 결과 저장용
final_preds = []
val_smapes = []
building_smapes = {}  # 건물별 SMAPE 기록
building_predictions = {}  # 건물별 예측값 저장

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()

print("[3] 건물별 학습 시작")

for bno in building_ids:
    print(f"\n    > 🏢 건물번호 {bno} 모델링 중...")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy()

    x = train_b[features]
    y = np.log1p(train_b[target])
    x_test_final = test_b[features]
    
    # 30% 샘플링으로 하이퍼파라미터 최적화
    sample_size = int(len(x) * 0.3)
    sample_indices = np.random.choice(len(x), sample_size, replace=False)
    x_sample = x.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]
    
    print(f"    📊 Optuna 최적화 중... (샘플 크기: {sample_size})")
    best_params = optimize_hyperparameters(x_sample, y_sample, n_trials=2)
    
    # 최종 모델 학습 및 예측
    print("    🔄 K-Fold 교차검증으로 최종 모델 학습 중...")
    final_prediction, avg_fold_smape = train_and_predict_model(x, y, x_test_final, best_params)
    
    # 건물별 결과 저장
    building_smapes[bno] = avg_fold_smape
    building_predictions[bno] = final_prediction
    val_smapes.append(avg_fold_smape)
    
    print(f"    ✅ 건물 {bno} 완료 - 평균 SMAPE: {avg_fold_smape:.4f}")

print("\n[4] 건물별 학습 완료")

# 전체 평균 SMAPE 계산
overall_avg_smape = np.mean(val_smapes)
print(f"📊 전체 평균 SMAPE: {overall_avg_smape:.4f}")

# 재학습 단계
print("\n[5] 재학습 단계 시작")
retrain_count = 0
improved_count = 0

for bno in building_ids:
    current_smape = building_smapes[bno]
    
    # 평균 SMAPE보다 높은 경우 재학습
    if current_smape > overall_avg_smape:
        print(f"\n    🔄 건물 {bno} 재학습 시작 (현재 SMAPE: {current_smape:.4f} > 평균: {overall_avg_smape:.4f})")
        retrain_count += 1
        
        train_b = train[train['건물번호'] == bno].copy()
        test_b = test[test['건물번호'] == bno].copy()

        x = train_b[features]
        y = np.log1p(train_b[target])
        x_test_final = test_b[features]
        
        # 50% 샘플링으로 하이퍼파라미터 재최적화
        sample_size = int(len(x) * 0.5)
        sample_indices = np.random.choice(len(x), sample_size, replace=False)
        x_sample = x.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        print(f"    📊 재학습 Optuna 최적화 중... (샘플 크기: {sample_size})")
        new_best_params = optimize_hyperparameters(x_sample, y_sample, n_trials=2)
        
        # 재학습 모델 학습 및 예측
        print("    🔄 재학습 K-Fold 교차검증 중...")
        new_prediction, new_smape = train_and_predict_model(x, y, x_test_final, new_best_params)
        
        # 성능 개선 여부 확인
        if new_smape < current_smape:
            print(f"    ✅ 성능 개선! {current_smape:.4f} -> {new_smape:.4f} (개선: {current_smape - new_smape:.4f})")
            building_smapes[bno] = new_smape
            building_predictions[bno] = new_prediction
            
            # val_smapes에서도 업데이트
            building_idx = list(building_ids).index(bno)
            val_smapes[building_idx] = new_smape
            improved_count += 1
        else:
            print(f"    ❌ 성능 하락. 원래 모델 유지 ({current_smape:.4f} vs {new_smape:.4f})")

# 최종 예측값 구성
for bno in building_ids:
    final_preds.extend(building_predictions[bno])

print(f"\n[6] 재학습 완료")
print(f"    - 재학습 대상: {retrain_count}개 건물")
print(f"    - 성능 개선: {improved_count}개 건물")
print(f"    - 개선률: {improved_count/retrain_count*100 if retrain_count > 0 else 0:.1f}%")

print("\n[7] 저장 시작")

# 결과 저장
samplesub['answer'] = np.expm1(final_preds)
today = datetime.datetime.now().strftime('%Y%m%d')
final_avg_smape = np.mean(val_smapes)
score_str = f"{final_avg_smape:.4f}".replace('.', '_')

filename = f"{SEED}_energy_{today}_{score_str}_retrain.csv"
samplesub.to_csv(save_path + filename, index=False)

print(f"[8] 📁 저장 완료 ")
print(f"✅ 최종 SMAPE 점수 : {final_avg_smape:.6f}")
print(f"📈 성능 개선 : {overall_avg_smape:.6f} -> {final_avg_smape:.6f}")

# 건물별 SMAPE 상세 로그 저장
smape_log = []
for bno in building_ids:
    smape_log.append(f"건물 {bno}: {building_smapes[bno]:.4f}")

with open(save_path+ "(Gift_model)LOG.txt", "a") as f:
    f.write(f"<{SEED} 회차>\n")
    f.write(f"<02_1gift_kfold_optuna.py>\n")
    f.write(f"{filename}\n")
    f.write(f"{train_call}\n")
    f.write(f"{test_call}\n")
    f.write(f"초기 평균 SMAPE : {overall_avg_smape:.6f}\n")
    f.write(f"최종 SMAPE 점수 : {final_avg_smape:.6f}\n")
    f.write(f"재학습 대상: {retrain_count}개, 개선: {improved_count}개\n")
    f.write("건물별 SMAPE:\n")
    f.write("\n".join(smape_log) + "\n")
    f.write("="*40 + "\n")