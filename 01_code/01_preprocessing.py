import numpy as np
import pandas as pd
import os

data_path = './Energy/'
save_path = './Energy/data/'
os.makedirs(save_path, exist_ok=True)

building = pd.read_csv(data_path + 'building_info.csv')

# print(building.columns)
# Index(['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)',
#        'PCS용량(kW)'],

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
building['건물유형'] = le.fit_transform(building['건물유형'])

train = pd.read_csv(data_path + 'train.csv', index_col=0)
# print(train.columns)
# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
#        '일사(MJ/m2)', '전력소비량(kWh)'],
#       dtype='object')
test = pd.read_csv(data_path + 'test.csv', index_col=0)

building = building.replace('-', np.nan)
building = building.fillna(0)
# print(building)

sun = train[['건물번호', '일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)']]

# print(sun.shape)        # (204000, 4)
# print(building.shape)   # (100, 7)

merge = pd.merge(building, sun, on=['건물번호'], how='left')
# print(merge)
# print(merge.shape)  #(204000, 10)

merge['일사(MJ/m2)'] = pd.to_numeric(merge['일사(MJ/m2)'], errors='coerce')
merge['태양광용량(kW)'] = pd.to_numeric(merge['태양광용량(kW)'], errors='coerce')
merge['예상_태양광_발전량(kWh)'] = merge['일사(MJ/m2)'] * 0.2778 * merge['태양광용량(kW)']

# print(merge['예상_발전량(kWh)'].isna().sum()) # 0
merge['순수_전력소비량(kWh)'] = merge['전력소비량(kWh)'] - merge['예상_태양광_발전량(kWh)']
print('min:', np.min(merge['순수_전력소비량(kWh)']), '\nmax:', np.max(merge['순수_전력소비량(kWh)']))
# min: -1290.0388145999998 
# max: 26994.96979008

# print(train.columns)
# print(train.shape)
# print(merge.columns)
# print(merge.shape)

merge_subset = merge[['건물번호', '일시', '건물유형', '예상_태양광_발전량(kWh)', '순수_전력소비량(kWh)']]

train_new = pd.merge(train, merge_subset, on=['건물번호', '일시'], how='left')

train_new['일시'] = pd.to_datetime(train_new['일시'], format='%Y%m%d %H')

# 요일 컬럼 추가 (0: 월요일 ~ 6: 일요일)
train_new['요일'] = train_new['일시'].dt.weekday
# 시간 컬럼 추가
train_new['시간'] = train_new['일시'].dt.hour

# 주말 여부 (1: 주말, 0: 평일)
train_new['주말여부'] = train_new['요일'].apply(lambda x: 1 if x >= 5 else 0)


# 전력소비량 제거
train_new = train_new.drop(columns=['전력소비량(kWh)', '일시'])
train_new.to_csv(save_path + 'train_new.csv', index=False)
# exit()
# 확인
# print(train_new.head())
# print(np.min(train_new['순수_전력소비량(kWh)']), '\n', np.max(train_new['순수_전력소비량(kWh)']))
# -1290.0388145999998 
#  26994.96979008

# 1. test에 건물 정보 병합
test_merged = pd.merge(test, building, on='건물번호', how='left')

# 2. 수치형 변환 및 결측치 처리
# test_merged['일사(MJ/m2)'] = pd.to_numeric(test_merged['일사(MJ/m2)'], errors='coerce')
# test_merged['태양광용량(kW)'] = pd.to_numeric(test_merged['태양광용량(kW)'], errors='coerce')

# 3. 태양광 발전량 및 순수 전력소비량 계산
# test_merged['예상_태양광_발전량(kWh)'] = test_merged['일사(MJ/m2)'] * 0.2778 * test_merged['태양광용량(kW)']
# test_merged['순수_전력소비량(kWh)'] = test_merged['전력소비량(kWh)'] - test_merged['예상_태양광_발전량(kWh)']

# 4. 일시 datetime 변환 → 요일, 주말 컬럼 생성
test_merged['일시'] = pd.to_datetime(test_merged['일시'], format='%Y%m%d %H')
test_merged['요일'] = test_merged['일시'].dt.weekday
test_merged['시간'] = test_merged['일시'].dt.hour
test_merged['주말여부'] = test_merged['요일'].apply(lambda x: 1 if x >= 5 else 0)

test_merged = test_merged.drop(['연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '일시'], axis=1)

test_new = test_merged.copy()
test_new.to_csv(save_path + 'test_new.csv', index=False)
print(train_new.columns)
print(test_new.columns)

# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
#        '건물유형', '예상_태양광_발전량(kWh)', '순수_전력소비량(kWh)', '요일', '주말여부'],
#       dtype='object')
# Index(['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형', '요일', '주말여부'], dtype='object')






