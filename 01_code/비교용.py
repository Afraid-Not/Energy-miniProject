""" 
##################### 날짜 하루 보기 #################

import pandas as pd
import matplotlib.pyplot as plt


# 입력값
building_ids = [1, 72] # 건물 번호 1부터 4까지
train_date = "2024-08-26"
test_date = "2024-08-26"

# 파일 경로
# test_path = "./Energy/13_submission/test_csv/13_01_test_SEED56.csv"
test_path = "./Energy/dont_touch/dont_touch_test_filled.csv"
train_path = "./Energy/dont_touch/dont_touch_test_filled.csv"

test = pd.read_csv(test_path)
train = pd.read_csv(train_path, encoding='utf-8-sig')

# train['일조(hr)'] = train['일조(hr)'].fillna(0)
# train['일사(MJ/m2)'] = train['일사(MJ/m2)'].fillna(0)

# 컬럼 이름 정리

# train[['일조(hr)', '일사(MJ/m2)']] = train[['일조(hr)', '일사(MJ/m2)']].fillna(0)
# print(test.columns)
# print(train.columns)
# print(test[['일조(hr)', '일사(MJ/m2)']].head())
# print(train[['일조(hr)', '일사(MJ/m2)']].head())
# exit()

# 일시 datetime 변환
test['일시'] = pd.to_datetime(test['일시'])
train['일시'] = pd.to_datetime(train['일시'])

# 날짜 파싱
train_day = pd.to_datetime(train_date).date()
test_day = pd.to_datetime(test_date).date()

# 서브플롯 설정: 4x2 배열, 그림 크기 조정
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
axes = axes.flatten() # 2차원 배열을 1차원으로 변환하여 쉽게 접근

for i, building_id in enumerate(building_ids):
    # 현재 건물 ID에 해당하는 서브플롯 선택 (axes 배열 범위를 벗어나지 않도록 확인)
    if i < len(axes):
        ax = axes[i]

        # 건물 및 날짜 필터링
        train_filtered = train[(train['건물번호'] == building_id) & (train['일시'].dt.date == train_day)]
        test_filtered = test[(test['건물번호'] == building_id) & (test['일시'].dt.date == test_day)]

        # 데이터 정렬
        train_filtered = train_filtered.sort_values("일시")
        test_filtered = test_filtered.sort_values("일시")

        # 시각(시간) 추출
        train_hours = train_filtered['일시'].dt.hour
        test_hours = test_filtered['일시'].dt.hour

        # 플롯 그리기
        ax.plot(train_hours, train_filtered['일조(hr)'], label='Train Sunshine(hr)', linestyle='--', marker='o')
        ax.plot(test_hours, test_filtered['일조(hr)'], label='Test Sunshine(hr)', linestyle='-', marker='o')
        ax.plot(train_hours, train_filtered['일사(MJ/m2)'], label='Train Insolation(MJ/m2)', linestyle='--', marker='x')
        ax.plot(test_hours, test_filtered['일사(MJ/m2)'], label='Test Insolation(MJ/m2)', linestyle='-', marker='x')

        # 서브플롯 제목, 라벨, 그리드 설정
        ax.set_title(f"{building_id}")
        ax.set_xlabel("time")
        ax.set_ylabel("answer")
        ax.grid(True)

# 사용되지 않는 서브플롯 제거 (building_ids의 개수가 axes의 개수보다 적을 경우)
for j in range(len(building_ids), len(axes)):
    fig.delaxes(axes[j])

# 공통 범례 설정
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.96))

# 전체 레이아웃 조정 및 제목 설정
plt.tight_layout(rect=[0, 0, 1, 0.93]) # 범례와 겹치지 않도록 여백 조정
plt.suptitle("(2024-08-17 vs 2024-08-25)", fontsize=16, y=0.98)
plt.show()
 """
################### 기간으로 보기 ####################
import pandas as pd
import matplotlib.pyplot as plt

# 입력값
building_ids = [1, 20] # 건물 번호
start_date = "2024-08-25"
end_date = "2024-08-31"

# 파일 경로
test_path = "./Energy/_best_code/best_train_test/best_test_SEED44_up.csv"
# test_path = "./Energy/13_submission/6.7025145136/13_01_test_SEED65_up.csv"
# test_path = "./Energy/dont_touch/dont_touch_test_filled.csv"
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
fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
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
        ax.set_title(f"Building {building_id}")
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