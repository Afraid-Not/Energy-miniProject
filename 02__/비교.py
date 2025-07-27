################### 기간으로 보기 ####################
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

ids = random.sample(range(1, 101), 3)
# 입력값
building_ids = [1, 20, 50] # 건물 번호
start_date = "2024-08-25"
end_date = "2024-09-01"

# 파일 경로
# test_path = "./Energy/01/06_test_43_.csv"
# test_path = "./Energy/01/06_test_50_.csv"
# test_path = "./Energy/01/06_test_51_.csv"
# test_path = "./Energy/01/06_test_52_.csv"
test_path = "./Energy/02/06_test_47_2.csv"
# test_path = "./Energy/6.3664130255/13_01_test_SEED5.csv"


train_path = "./Energy/dont_touch/dont_touch_test_filled.csv"

# 데이터 불러오기
test = pd.read_csv(test_path)
train = pd.read_csv(train_path, encoding='utf-8-sig')

# '일시' 컬럼 datetime 변환
test['일시'] = pd.to_datetime(test['일시'])
train['일시'] = pd.to_datetime(train['일시'])

# 기간 범위 datetime 변환
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# 서브플롯 설정: 1x2 배열, 그림 크기 조정
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
axes = axes.flatten() # 2차원 배열을 1차원으로 변환하여 쉽게 접근

for i, building_id in enumerate(building_ids):
    # 현재 건물 ID에 해당하는 서브플롯 선택 (axes 배열 범위를 벗어나지 않도록 확인)
    if i < len(axes):
        ax = axes[i]

        # 건물 및 기간 필터링 (집계하지 않고 모든 시간 데이터 사용)
        train_filtered = train[
            (train['건물번호'] == building_id) &
            (train['일시'] >= start_dt) &
            (train['일시'] <= end_dt)
        ].sort_values("일시")

        test_filtered = test[
            (test['건물번호'] == building_id) &
            (test['일시'] >= start_dt) &
            (test['일시'] <= end_dt)
        ].sort_values("일시")

        # 플롯 그리기
        ax.plot(train_filtered['일시'], train_filtered['일조(hr)'], label='Train Sunshine(hr)', linestyle='--', marker='o')
        ax.plot(test_filtered['일시'], test_filtered['일조(hr)'], label='Test Sunshine(hr)', linestyle='-', marker='o')
        ax.plot(train_filtered['일시'], train_filtered['일사(MJ/m2)'], label='Train Insolation(MJ/m2)', linestyle='--', marker='x')
        ax.plot(test_filtered['일시'], test_filtered['일사(MJ/m2)'], label='Test Insolation(MJ/m2)', linestyle='-', marker='x')

        # x축 포맷 설정 (날짜와 시간이 잘 보이도록)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 서브플롯 제목, 라벨, 그리드 설정
        ax.set_title(f"43__{building_id}")
        ax.set_xlabel("Date and Time")
        ax.set_ylabel("Value")
        ax.grid(True)

# 사용되지 않는 서브플롯 제거 (building_ids의 개수가 axes의 개수보다 적을 경우)
for j in range(len(building_ids), len(axes)):
    fig.delaxes(axes[j])

# 공통 범례 설정
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.96))

# 전체 레이아웃 조정 및 제목 설정
plt.tight_layout(rect=[0, 0, 1, 0.93]) # 범례와 겹치지 않도록 여백 조정
plt.suptitle(f"test {start_date} to {end_date}", fontsize=16, y=0.98)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae =mean_absolute_error(train['일조(hr)'], test['일조(hr)'])
mae2 = mean_absolute_error(train['일사(MJ/m2)'], test['일사(MJ/m2)'])
r2 =r2_score(train['일조(hr)'], test['일조(hr)'])
r22 = r2_score(train['일사(MJ/m2)'], test['일사(MJ/m2)'])
rmse = np.sqrt(mean_squared_error(train['일조(hr)'], test['일조(hr)']))
rmse2 = np.sqrt(mean_squared_error(train['일사(MJ/m2)'], test['일사(MJ/m2)']))

total = (mae + (1-r2) + rmse) / 3
total2 = (mae2 + (1-r22) + rmse2) / 3

print(f"{test_path}")
print(f"일조 mae  : {mae:.6f}| 일사 mae  : {mae2:.6f}")
print(f"일조 R2   : {1-r2:.6f}| 일사 R2   : {1-r22:.6f}")
print(f"일조 RMSE : {rmse:.6f}| 일사 RMSE : {rmse2:.6f}")
print(f"평균 점수 : {total:.6f}| 평균점수 : {total2:.6f}")

# ./Energy/01/06_test_43_.csv
# 일조 mae  : 0.103821| 일사 mae  : 0.199223
# 일조 R2   : 0.194544| 일사 R2   : 0.215395
# 일조 RMSE : 0.182995| 일사 RMSE : 0.465659
# 평균 점수 : 0.160453| 평균점수 : 0.293426

# ./Energy/01/06_test_50_.csv
# 일조 mae  : 0.105672| 일사 mae  : 0.202011
# 일조 R2   : 0.198902| 일사 R2   : 0.217044
# 일조 RMSE : 0.185033| 일사 RMSE : 0.467437
# 평균 점수 : 0.163202| 평균점수 : 0.295497

# ./Energy/01/06_test_51_.csv
# 일조 mae  : 0.102838| 일사 mae  : 0.198285
# 일조 R2   : 0.191659| 일사 R2   : 0.219477
# 일조 RMSE : 0.181633| 일사 RMSE : 0.470050
# 평균 점수 : 0.158710| 평균점수 : 0.295937

# ./Energy/01/06_test_52_.csv
# 일조 mae  : 0.103561| 일사 mae  : 0.196434
# 일조 R2   : 0.194088| 일사 R2   : 0.213545
# 일조 RMSE : 0.182780| 일사 RMSE : 0.463654
# 평균 점수 : 0.160143| 평균점수 : 0.291211

# ./Energy/02/06_test_43_.csv
# 일조 mae  : 0.105350| 일사 mae  : 0.197479
# 일조 R2   : 0.198176| 일사 R2   : 0.216478
# 일조 RMSE : 0.184695| 일사 RMSE : 0.466828
# 평균 점수 : 0.162741| 평균점수 : 0.293595

# ./Energy/02/06_test_48_.csv
# 일조 mae  : 0.104521| 일사 mae  : 0.198728
# 일조 R2   : 0.195863| 일사 R2   : 0.218381
# 일조 RMSE : 0.183614| 일사 RMSE : 0.468875
# 평균 점수 : 0.161333| 평균점수 : 0.295328

# ./Energy/02/06_test_52_.csv
# 일조 mae  : 0.104501| 일사 mae  : 0.198738
# 일조 R2   : 0.194731| 일사 R2   : 0.218964
# 일조 RMSE : 0.183083| 일사 RMSE : 0.469501
# 평균 점수 : 0.160771| 평균점수 : 0.295734

# ./Energy/02/06_test_47_.csv
# 일조 mae  : 0.103583| 일사 mae  : 0.198787
# 일조 R2   : 0.193030| 일사 R2   : 0.218860
# 일조 RMSE : 0.182281| 일사 RMSE : 0.469389
# 평균 점수 : 0.159632| 평균점수 : 0.295679

# ./Energy/02/06_test_47_2.csv
# 일조 mae  : 0.103971| 일사 mae  : 0.198621
# 일조 R2   : 0.195069| 일사 R2   : 0.219158
# 일조 RMSE : 0.183241| 일사 RMSE : 0.469708
# 평균 점수 : 0.160761| 평균점수 : 0.295829