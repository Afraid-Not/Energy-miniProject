print(f"#0. 시작")
import pandas as pd
import numpy as np
import time
import os
import json
import random
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')

seed_file = "./Energy/05_submission/05_02_model.json"

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
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

data_path = './Energy/'
save_path = './Energy/10_00_submission/'
train = pd.read_csv(save_path + f'{SEED}_new_train.csv', index_col=0)
test = pd.read_csv(save_path + f'{SEED}_new_train.csv', index_col=0)
submit = pd.read_csv(data_path + 'sample_submission.csv')

# test의 순서를 보장하는 인덱스 기준 고유키 붙이기
test = test.reset_index()
test['num_date_time'] = submit['num_date_time']

from torch.utils.data import Dataset
class EnergyLSTMDataset(Dataset):
    def __init__(self, df, building_id, features, target, seq_len=24):
        self.df = df[df['건물번호'] == building_id].copy()
        self.features = features
        self.target = target
        self.seq_len = seq_len

        self.X = self.df[features].values
        self.y = self.df[target].values

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.seq_len]
        y_target = self.y[idx+self.seq_len]  # 다음 시점 예측
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]
        last_out = lstm_out[:, -1, :]  # 마지막 시점
        out = self.fc(last_out)
        return out.squeeze(1)
    
from torch.utils.data import DataLoader

def train_lstm_model(df, building_id, features, target, device='cpu', epochs=20):
    dataset = EnergyLSTMDataset(df, building_id, features, target)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = LSTMRegressor(input_size=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    
    return model


def predict_lstm(model, df, building_id, features, seq_len=24, device='cpu'):
    model.eval()
    data = df[df['건물번호'] == building_id][features].values
    inputs = torch.tensor(data[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inputs).item()
    return pred

features = ['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형',
       '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연',
       '월', '일', '요일', 'hour', 'sin_hour', 'cos_hour', 'is_weekend',
       'is_working_hours', 'is_holiday', 'feels_like', 'discomfort_index',
       'cooling_degree_day', 'rainfall_change_rate', 'is_rain_x_working_hours',
       '일조(hr)', '일사(MJ/m2)', '태양광사용량', 'sun_x_temp', 'sun_x_cooling_area',
       'sun_x_working_hours', 'sun_x_solar_capacity', 'solar_x_cooling_area',
       'solar_x_working_hours', 'solar_x_solar_capacity', 'date',
       'daily_total_sunshine', 'rolling_hourly_sunshine',
       'rolling_hourly_solar_radiation', 'solar_radiation_change']

target = ["전력소비량(kWh)"]

for i in range(1, 101) :
    building_id = i+1
    model = train_lstm_model(train, building_id, features, target, DEVICE, epochs=10)
    prediction = predict_lstm(model, test, building_id, features)
    print(f"[건물 {building_id}] 예측값: {prediction:.4f}")