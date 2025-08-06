import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 설정 ======
building_id = 43
start_date = "2024-08-25"
end_date = "2024-09-01"

# 파일 경로
train_path = "./Energy/dont_touch/dont_touch_test_filled.csv"
test_paths = [
    "./Energy/02/06_test_47_2.csv",
    "./Energy/02/06_test_47_.csv"
]
test_labels = ["Test 모델 A", "Test 모델 B"]

# ====== 데이터 불러오기 ======
train = pd.read_csv(train_path, encoding='utf-8-sig')
train['일시'] = pd.to_datetime(train['일시'])
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# Train 필터링
train_filtered = train[
    (train['건물번호'] == building_id) &
    (train['일시'] >= start_dt) &
    (train['일시'] <= end_dt)
].sort_values("일시")

# Plot setup
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot Train 데이터
axes[0].plot(train_filtered['일시'], train_filtered['일조(hr)'], label='Train 일조(hr)', linestyle='--', color='black', marker='o')
axes[1].plot(train_filtered['일시'], train_filtered['일사(MJ/m2)'], label='Train 일사(MJ/m2)', linestyle='--', color='black', marker='x')

# 각 Test 파일 비교
for test_path, label in zip(test_paths, test_labels):
    test = pd.read_csv(test_path)
    test['일시'] = pd.to_datetime(test['일시'])

    test_filtered = test[
        (test['건물번호'] == building_id) &
        (test['일시'] >= start_dt) &
        (test['일시'] <= end_dt)
    ].sort_values("일시")

    axes[0].plot(test_filtered['일시'], test_filtered['일조(hr)'], label=f'{label} - 일조(hr)', marker='o')
    axes[1].plot(test_filtered['일시'], test_filtered['일사(MJ/m2)'], label=f'{label} - 일사(MJ/m2)', marker='x')

# 공통 설정
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("일시")
    ax.set_ylabel("값")
    ax.grid(True)
    ax.legend()

axes[0].set_title(f"건물번호 {building_id} - 일조량(hr) 비교")
axes[1].set_title(f"건물번호 {building_id} - 일사량(MJ/m2) 비교")
plt.suptitle(f"기간: {start_date} ~ {end_date}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()