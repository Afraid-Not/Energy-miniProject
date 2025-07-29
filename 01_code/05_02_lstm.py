import os
import pandas as pd
import numpy as np
import random
import datetime, json
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

# PyTorch 관련 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

seed_file = "./Energy/05_submission/05_02.json"

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
torch.manual_seed(SEED)

# 경로 설정
data_path = './Energy/'

# 데이터 로드
buildinginfo = pd.read_csv(data_path + 'building_info.csv')
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
samplesub = pd.read_csv(data_path + 'sample_submission.csv')

print("[1] 데이터 로딩 및 초기 전처리 완료")

# 결측치 처리
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    buildinginfo[col] = buildinginfo[col].replace('-', 0).astype(float)

# Feature Engineering
def feature_engineering(df, is_test=False):
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['is_weekend'] = df['일시'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df['is_working_hours'] = df['일시'].dt.hour.apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # LSTM 예측 후에는 이 부분에서 '일조'와 '일사'를 0으로 채우지 않습니다.
    # 단, train 데이터에는 0으로 채워 학습 데이터로 활용합니다.
    if not is_test:
        for col in ['일조(hr)', '일사(MJ/m2)']:
            if col in df.columns:
                df[col] = df[col].fillna(0) # train 데이터에 대해서는 0으로 채움
    
    temp = df['기온(°C)']
    humidity = df['습도(%)']
    df['DI'] = 9/5 * temp - 0.55 * (1 - humidity/100) * (9/5 * temp - 26) + 32
    return df

# 전처리 (LSTM 예측 전에 먼저 수행)
train = feature_engineering(train, is_test=False) # train은 기존처럼 0으로 채움
test_processed_for_lstm = feature_engineering(test.copy(), is_test=True) # test는 LSTM으로 채울 것이므로 0으로 안 채움

train = train.merge(buildinginfo, on='건물번호', how='left')
test_processed_for_lstm = test_processed_for_lstm.merge(buildinginfo, on='건물번호', how='left')

train['건물유형'] = train['건물유형'].astype('category').cat.codes
test_processed_for_lstm['건물유형'] = test_processed_for_lstm['건물유형'].astype('category').cat.codes


# --- LSTM 모델 for '일조(hr)' and '일사(MJ/m2)' 예측 ---
print("[1.5] '일조' 및 '일사' 예측을 위한 LSTM 모델 학습 시작")

# PyTorch GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    > PyTorch 모델이 사용할 디바이스: {device}")


# PyTorch LSTM 모델 정의
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # hidden/cell state를 현재 디바이스에 맞게 생성
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # 마지막 타임스텝의 아웃풋 사용
        return out

# 시퀀스 데이터 생성 함수
def create_sequences(data_x, data_y, sequence_length):
    xs, ys = [], []
    # 건물별로 시퀀스 생성 (건물 경계를 넘지 않도록)
    for bno in data_x['건물번호'].unique():
        b_data_x = data_x[data_x['건물번호'] == bno].drop(columns=['건물번호'])
        b_data_y = data_y[data_y['건물번호'] == bno].drop(columns=['건물번호'])
        
        for i in range(len(b_data_x) - sequence_length):
            xs.append(b_data_x.iloc[i:(i + sequence_length)].values)
            ys.append(b_data_y.iloc[i + sequence_length].values)
    return np.array(xs), np.array(ys)

# LSTM 모델 학습을 위한 피처 및 타겟 설정
lstm_features = ['기온(°C)', '습도(%)', 'hour', 'dayofweek', 'month', 'day', 'sin_hour', 'cos_hour']
lstm_targets = ['일조(hr)', '일사(MJ/m2)']

# LSTM 학습 데이터 준비 (일조, 일사 결측치가 없는 train 데이터만 사용)
train_for_lstm = train.dropna(subset=lstm_targets).copy()
# 시퀀스 생성을 위해 건물번호와 일시 기준으로 정렬
train_for_lstm = train_for_lstm.sort_values(by=['건물번호', '일시']).reset_index(drop=True)

# 스케일러 정의
scaler_lstm_features = StandardScaler()
scaler_lstm_targets = StandardScaler()

# LSTM 학습 피처와 타겟을 스케일링
X_lstm_train_scaled = scaler_lstm_features.fit_transform(train_for_lstm[lstm_features])
y_lstm_train_scaled = scaler_lstm_targets.fit_transform(train_for_lstm[lstm_targets])

# 스케일링된 데이터를 DataFrame으로 변환하여 건물번호 포함 (시퀀스 생성 함수에서 사용)
X_lstm_train_scaled_df = pd.DataFrame(X_lstm_train_scaled, columns=lstm_features, index=train_for_lstm.index)
X_lstm_train_scaled_df['건물번호'] = train_for_lstm['건물번호']

y_lstm_train_scaled_df = pd.DataFrame(y_lstm_train_scaled, columns=lstm_targets, index=train_for_lstm.index)
y_lstm_train_scaled_df['건물번호'] = train_for_lstm['건물번호']


SEQUENCE_LENGTH = 24 # 과거 24시간 데이터를 사용하여 예측
lstm_train_X, lstm_train_y = create_sequences(X_lstm_train_scaled_df, y_lstm_train_scaled_df, SEQUENCE_LENGTH)

# PyTorch Tensor로 변환 및 GPU로 이동
X_tensor = torch.FloatTensor(lstm_train_X).to(device)
y_tensor = torch.FloatTensor(lstm_train_y).to(device)

# Dataset 및 DataLoader 생성
dataset = TensorDataset(X_tensor, y_tensor)
# TimeSeriesSplit for LSTM training (using the last split for simplicity)
tscv_lstm = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv_lstm.split(X_tensor): # TimeSeriesSplit은 x만 받음
    train_data = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_data = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

train_loader = DataLoader(train_data, batch_size=64, shuffle=False) # 시계열이므로 shuffle=False
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# --- Early Stopping 클래스 정의 ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 검증 손실이 개선되지 않아도 기다릴 에폭 수.
                            이 횟수만큼 개선이 없으면 학습을 중단합니다.
            verbose (bool): True이면 각 검증 손실 개선 시 메시지를 출력합니다.
            delta (float): 개선으로 간주하기 위한 최소 변화량.
            path (str): 모델 체크포인트를 저장할 경로.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # <-- 여기서 np.Inf를 np.inf로 변경
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'        EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """검증 손실이 감소하면 모델을 저장합니다."""
        if self.verbose:
            print(f'        Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 모델, 손실 함수, 옵티마이저 정의
input_size = len(lstm_features)
hidden_size = 50
num_layers = 2
output_size = len(lstm_targets) # 일조, 일사 두 가지 예측

model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device) # 모델을 GPU로 이동
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Early Stopping 인스턴스 생성
early_stopping = EarlyStopping(patience=10, verbose=True, path=f'./Energy/05_submission/({SEED})lstm_model.pt')


# LSTM 모델 학습
num_epochs = 100000 # 에폭 수 조절 가능 (early stopping으로 인해 실제 학습은 더 일찍 중단될 수 있음)
print(f"    > LSTM 모델 학습 시작 (epochs: {num_epochs}, timesteps: {SEQUENCE_LENGTH})")
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # 데이터를 GPU로 이동
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x_val, batch_y_val in val_loader:
            batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device) # 데이터를 GPU로 이동
            outputs_val = model(batch_x_val)
            val_loss += criterion(outputs_val, batch_y_val).item()
    val_loss /= len(val_loader)
    
    print(f"        >> Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}") # 매 에폭마다 출력

    # Early Stopping 체크
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("        Early stopping")
        break

# 최적의 모델 가중치 로드
model.load_state_dict(torch.load(f'./Energy/05_submission/({SEED})lstm_model.pt'))
print("    > LSTM 모델. 최적 모델 로드.")

# test 데이터에 대한 일조, 일사 예측
print("    > test 데이터에 대한 '일조', '일사' 예측 시작...")

# test 데이터도 건물별로 시퀀스 생성
test_lstm_X_scaled = scaler_lstm_features.transform(test_processed_for_lstm[lstm_features])
test_lstm_X_scaled_df = pd.DataFrame(test_lstm_X_scaled, columns=lstm_features, index=test_processed_for_lstm.index)
test_lstm_X_scaled_df['건물번호'] = test_processed_for_lstm['건물번호']


# 건물별로 순회하며 시퀀스 생성 및 예측
test_preds_lstm = np.zeros((len(test_processed_for_lstm), len(lstm_targets)))

for bno in test_lstm_X_scaled_df['건물번호'].unique():
    b_data_x = test_lstm_X_scaled_df[test_lstm_X_scaled_df['건물번호'] == bno].drop(columns=['건물번호'])
    
    # 예측을 위한 시퀀스 생성 (마지막 시퀀스만 필요)
    if len(b_data_x) < SEQUENCE_LENGTH:
        # 시퀀스 길이가 부족하면 예측 불가, 0으로 채우거나 다른 전략 필요
        print(f"        건물번호 {bno}: 데이터 길이가 시퀀스 길이({SEQUENCE_LENGTH})보다 짧습니다. 예측 생략.")
        continue # 이 건물은 예측 생략하고 넘어감
    
    # test 데이터는 순차적으로 예측해야 함
    b_preds = []
    with torch.no_grad():
        for i in range(len(b_data_x) - SEQUENCE_LENGTH + 1):
            input_seq = torch.FloatTensor(b_data_x.iloc[i:(i + SEQUENCE_LENGTH)].values).unsqueeze(0).to(device) # 데이터를 GPU로 이동
            output = model(input_seq)
            b_preds.append(output.squeeze(0).cpu().numpy()) # 예측 결과를 CPU로 다시 가져옴
    
    # b_preds는 SEQUENCE_LENGTH 이후부터의 예측이므로,
    # 해당 건물의 첫 SEQUENCE_LENGTH-1개의 데이터에 대해서는 예측값이 없음.
    
    # 0으로 채워진 초기 배열 생성
    current_building_indices = test_processed_for_lstm[test_processed_for_lstm['건물번호'] == bno].index
    temp_preds_for_building = np.zeros((len(current_building_indices), len(lstm_targets)))
    
    # 예측된 값들을 올바른 위치에 할당
    if len(b_preds) > 0:
        temp_preds_for_building[SEQUENCE_LENGTH-1:] = np.array(b_preds)
    
    # 스케일링 역변환
    temp_preds_for_building = scaler_lstm_targets.inverse_transform(temp_preds_for_building)

    # test_processed_for_lstm에 예측 결과 반영
    test_processed_for_lstm.loc[current_building_indices, lstm_targets] = temp_preds_for_building


print("    > '일조', '일사' 예측 완료 및 test 데이터에 반영.")

# 이제 원래 test 변수에 LSTM 예측이 반영된 데이터를 할당
test = test_processed_for_lstm

test.to_csv('./Energy/05_submission/05_02_test.csv', index=False)
exit()
features = [
    '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
    'is_weekend', 'is_working_hours', 'sin_hour', 'cos_hour', 'DI',
    'hour' #, 'dayofweek', 'month', 'day', 
]

target = '전력소비량(kWh)'
print("[2] 전처리 완료 (LSTM 예측 포함)")

# 최종 예측 결과 저장용
final_preds = []
val_smapes = []

# 건물별로 모델 학습 및 예측
building_ids = train['건물번호'].unique()

print("[3] 건물별 학습 시작")

# TimeSeriesSplit 설정 (주요 앙상블 모델 학습용)
tscv = TimeSeriesSplit(n_splits=5) # TimeSeriesSplit 추가
kfol = KFold(n_splits=5, random_state=SEED, shuffle=True) # KFold도 여전히 사용 가능

for bno in building_ids:
    print(f"      > 건물번호 {bno} 모델링 중...")

    train_b = train[train['건물번호'] == bno].copy()
    test_b = test[test['건물번호'] == bno].copy() # 이제 test_b는 LSTM 예측값이 반영된 데이터

    x = train_b[features]
    y = np.log1p(train_b[target])
    x_test_final = test_b[features]

    # TimeSeriesSplit을 사용하여 훈련/검증 데이터 분할
    # 실제 시계열 예측과 유사하게, 마지막 폴드의 데이터를 훈련 및 검증에 사용합니다.
    # 이는 일반적인 K-Fold보다 시계열 데이터에 더 적합합니다.
    train_index, val_index = None, None
    for tr_idx, va_idx in tscv.split(x): # TimeSeriesSplit 사용
        train_index, val_index = tr_idx, va_idx # 마지막 폴드를 사용

    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_final_scaled = scaler.transform(x_test_final)

    # Base models (GPU 가속 파라미터 추가)
    xgb_model = XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                             random_state=SEED, early_stopping_rounds=50, objective='reg:squarederror',
                             ) 
    xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)

    lgb_model = LGBMRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                              random_state=SEED, objective='mae', verbose=-1,
                              ) 
    lgb_model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    cat_model = CatBoostRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                  random_seed=SEED, verbose=0, loss_function='MAE',
                                  ) 
    cat_model.fit(x_train_scaled, y_train, eval_set=(x_val_scaled, y_val), early_stopping_rounds=50)

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
        xgb_model.predict(x_test_final_scaled),
        lgb_model.predict(x_test_final_scaled),
        cat_model.predict(x_test_final_scaled)
    ]).T

    # Level 2 Meta model
    meta_model = RidgeCV()
    meta_model.fit(oof_train_lvl1, y_train)
    val_pred_lvl2 = meta_model.predict(oof_val_lvl1)
    test_pred_lvl2 = meta_model.predict(oof_test_lvl1)

    # Level 3 Final model
    final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)
    final_model.fit(val_pred_lvl2.reshape(-1, 1), y_val)

    val_final = final_model.predict(val_pred_lvl2.reshape(-1, 1))
    val_smape = np.mean(200 * np.abs(np.expm1(val_final) - np.expm1(y_val)) /
                         (np.abs(np.expm1(val_final)) + np.abs(np.expm1(y_val)) + 1e-6))
    val_smapes.append(val_smape)

    pred = final_model.predict(test_pred_lvl2.reshape(-1, 1))
    final_preds.extend(pred)
    
print("[4] 건물별 학습 완료")
print("[5] 저장 시작")

# 결과 저장
samplesub['answer'] = final_preds
today = datetime.datetime.now().strftime('%Y%m%d')
avg_smape = np.mean(val_smapes)
# avg_smape_exp는 로그 역변환된 값이므로, 실제 SMAPE와 혼동하지 않도록 주의
score_str = f"{avg_smape:.4f}".replace('.', '_')
filepath = './Energy/05_submission/'

os.makedirs(filepath, exist_ok=True)

filename = f"({SEED})0502_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(filepath + filename, index=False)

print(f"[6] 📁 저장 완료 ")
print(f"✅ 최종 SMAPE 점수 : {avg_smape:.6f}")
    
with open("./Energy/05_submission/05_02_log.txt", "a") as f:
    f.write(f"<SEED : {SEED}>\n")
    f.write(f"저장  : {filename}\n")
    f.write(f"SMAPE : {avg_smape}\n")
    f.write("="*40 + "\n")