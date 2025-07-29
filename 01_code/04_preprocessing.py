import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
train = pd.read_csv("./Energy/train.csv")
test = pd.read_csv("./Energy/test.csv")
building = pd.read_csv("./Energy/building_info.csv")

# ---------------------------
# 2. building 전처리 및 파생
# ---------------------------
building.replace('-', np.nan, inplace=True)
for col in ['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building[col] = pd.to_numeric(building[col], errors='coerce')
building.fillna(0, inplace=True)

building['면적당_냉방비율'] = building['냉방면적(m2)'] / building['연면적(m2)']
building['태양광_설비_유무'] = (building['태양광용량(kW)'] > 0).astype(int)
building['ESS_설비_유무'] = (building['ESS저장용량(kWh)'] > 0).astype(int)
building['PCS_설비_유무'] = (building['PCS용량(kW)'] > 0).astype(int)
building['설비_총용량'] = building['태양광용량(kW)'] + building['ESS저장용량(kWh)'] + building['PCS용량(kW)']

building = pd.get_dummies(building, columns=['건물유형'])

# ---------------------------
# 3. train 전처리 및 피처 생성
# ---------------------------
train_merged = pd.merge(train, building, on='건물번호', how='left')

train_merged['일시'] = pd.to_datetime(train_merged['일시'], format='%Y%m%d %H')
train_merged['월'] = train_merged['일시'].dt.month
train_merged['일'] = train_merged['일시'].dt.day
train_merged['시간'] = train_merged['일시'].dt.hour
train_merged['요일'] = train_merged['일시'].dt.weekday
train_merged['주말여부'] = (train_merged['요일'] >= 5).astype(int)

train_merged['기온×습도'] = train_merged['기온(°C)'] * train_merged['습도(%)']
train_merged['일조×일사'] = train_merged['일조(hr)'] * train_merged['일사(MJ/m2)']
train_merged['풍속×일사'] = train_merged['풍속(m/s)'] * train_merged['일사(MJ/m2)']
train_merged['풍속×기온'] = train_merged['풍속(m/s)'] * train_merged['기온(°C)']
train_merged['강수여부'] = (train_merged['강수량(mm)'] > 0).astype(int)

train_merged['연면적당_전력소비량'] = train_merged['전력소비량(kWh)'] / train_merged['연면적(m2)']
train_merged['냉방면적당_전력소비량'] = train_merged['전력소비량(kWh)'] / train_merged['냉방면적(m2)']

# 건물유형별 평균 전력소비량
building_types = [col for col in train_merged.columns if col.startswith('건물유형_')]
for col in building_types:
    type_mean = train_merged[train_merged[col] == 1]['전력소비량(kWh)'].mean()
    train_merged[f'{col}_평균전력'] = type_mean

# ---------------------------
# 4. test 전처리 및 피처 생성
# ---------------------------
test_merged = pd.merge(test, building, on='건물번호', how='left')

test_merged['일시'] = pd.to_datetime(test_merged['일시'], format='%Y%m%d %H')
test_merged['월'] = test_merged['일시'].dt.month
test_merged['일'] = test_merged['일시'].dt.day
test_merged['시간'] = test_merged['일시'].dt.hour
test_merged['요일'] = test_merged['일시'].dt.weekday
test_merged['주말여부'] = (test_merged['요일'] >= 5).astype(int)

# 일조/일사 선형 회귀 예측
predictors = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)']
for target in ['일조(hr)', '일사(MJ/m2)']:
    df = train[predictors + [target]].dropna()
    model = LinearRegression()
    model.fit(df[predictors], df[target])
    test_merged[target] = model.predict(test_merged[predictors])

# 날씨 파생 피처
test_merged['기온×습도'] = test_merged['기온(°C)'] * test_merged['습도(%)']
test_merged['일조×일사'] = test_merged['일조(hr)'] * test_merged['일사(MJ/m2)']
test_merged['풍속×일사'] = test_merged['풍속(m/s)'] * test_merged['일사(MJ/m2)']
test_merged['풍속×기온'] = test_merged['풍속(m/s)'] * test_merged['기온(°C)']
test_merged['강수여부'] = (test_merged['강수량(mm)'] > 0).astype(int)

# 소비량 없음 → NaN 처리
test_merged['연면적당_전력소비량'] = np.nan
test_merged['냉방면적당_전력소비량'] = np.nan

# 건물유형별 평균 전력소비량 적용
for col in building_types:
    if col in test_merged.columns:
        type_mean = train_merged[train_merged[col] == 1]['전력소비량(kWh)'].mean()
        test_merged[f'{col}_평균전력'] = type_mean
    else:
        test_merged[f'{col}_평균전력'] = 0

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 입력 피처/타겟 분리
target_col = '전력소비량(kWh)'
drop_cols = ['num_date_time', '일시', target_col]

X = train_merged.drop(columns=drop_cols)
y = train_merged[target_col]

# 2. 모델 학습
model = lgb.LGBMRegressor(random_state=42)
model.fit(X, y)

# 3. 피처 중요도 추출
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 1. FontProperties 객체 직접 생성 (이름 자동 인식 없이 경로로만 처리)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)

# 2. 시각화에 직접 fontproperties 전달
plt.figure(figsize=(10, 12))
sns.barplot(data=feature_importance.head(30), y='Feature', x='Importance')
plt.title('Top 30 Feature Importance (LGBM)', fontproperties=fontprop)
plt.xlabel('Importance', fontproperties=fontprop)
plt.ylabel('Feature', fontproperties=fontprop)

# y축 눈금 글꼴도 개별 지정
for label in plt.gca().get_yticklabels():
    label.set_fontproperties(fontprop)

plt.tight_layout()
plt.show()