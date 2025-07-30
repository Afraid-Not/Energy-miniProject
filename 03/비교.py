################### 기간으로 보기 ####################
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

# test = pd.read_csv('./Energy/02/trainer/06_test_68_norol.csv')
# train = pd.read_csv('./Energy/02/trainer/06_train_68_norol.csv')

# testcol = set(test.columns)
# traincol = set(train.columns)

# print(list(traincol - testcol))

# exit()
log_path = "./Energy/03/log/"
import os
os.makedirs(log_path, exist_ok=True)

################### 기간으로 보기 ####################
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

# 일사량이 0인 건물들 (SMAPE 계산에서 제외)
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

ids = random.sample(range(1, 101), 3)
# 입력값
building_ids = ids # 건물 번호
start_date = "2024-08-25"
end_date = "2024-09-01"

# 파일 경로
# test_path = "./Energy/01/06_test_56_knn_full.csv"
# test_path = "./Energy/02/submission/test_interpolated_top30_features_SEED43.csv"
# test_path = "./Energy/02/submission/test_interpolated_optuna_ensemble_SEED43.csv"
test_path = "./Energy/03/trainer/new_test_ver5.csv"
# test_path = "./Energy/02/trainer/06_test_82_ver2.csv"

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

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return 100 * np.mean(ratio)

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
        title_suffix = " (제외 건물)" if building_id in zero_bnos else ""
        ax.set_title(f"43__{building_id}{title_suffix}")
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

# ===== 전체 데이터에서 제외 건물 필터링 =====
print(f"\n=== SMAPE 계산 (제외 건물: {zero_bnos}) ===")

# 제외 건물이 아닌 데이터만 선택
train_filtered_all = train[~train['건물번호'].isin(zero_bnos)]
test_filtered_all = test[~test['건물번호'].isin(zero_bnos)]

print(f"전체 데이터: Train {len(train)}, Test {len(test)}")
print(f"필터링 후: Train {len(train_filtered_all)}, Test {len(test_filtered_all)}")

# 일조 시간 성능 계산 (모든 건물)
mae_sunshine_all = r2_score(train['일조(hr)'], test['일조(hr)'])
smape_sunshine_all = smape(train['일조(hr)'], test['일조(hr)'])

# 일조 시간 성능 계산 (제외 건물 빼고)
mae_sunshine_filtered = r2_score(train_filtered_all['일조(hr)'], test_filtered_all['일조(hr)'])
smape_sunshine_filtered = smape(train_filtered_all['일조(hr)'], test_filtered_all['일조(hr)'])

# 일사량 성능 계산 (모든 건물)
mae_solar_all = r2_score(train['일사(MJ/m2)'], test['일사(MJ/m2)'])
smape_solar_all = smape(train['일사(MJ/m2)'], test['일사(MJ/m2)'])

# 일사량 성능 계산 (제외 건물 빼고)
mae_solar_filtered = r2_score(train_filtered_all['일사(MJ/m2)'], test_filtered_all['일사(MJ/m2)'])
smape_solar_filtered = smape(train_filtered_all['일사(MJ/m2)'], test_filtered_all['일사(MJ/m2)'])

print(f"\n{test_path}")
print(f"=== 전체 건물 포함 ===")
print(f"일조 R2     : {mae_sunshine_all:.6f} | 일사 R2     : {mae_solar_all:.6f}")
print(f"일조 SMAPE  : {smape_sunshine_all:.6f} | 일사 SMAPE  : {smape_solar_all:.6f}")

print(f"\n=== 제외 건물 빼고 계산 ===")
print(f"일조 R2     : {mae_sunshine_filtered:.6f} | 일사 R2     : {mae_solar_filtered:.6f}")
print(f"일조 SMAPE  : {smape_sunshine_filtered:.6f} | 일사 SMAPE  : {smape_solar_filtered:.6f}")

# 개선도 계산
sunshine_improvement = smape_sunshine_all - smape_sunshine_filtered
solar_improvement = smape_solar_all - smape_solar_filtered

print(f"\n=== SMAPE 개선도 ===")
print(f"일조 개선: {sunshine_improvement:.6f} (전체 {smape_sunshine_all:.4f} → 필터링 {smape_sunshine_filtered:.4f})")
print(f"일사 개선: {solar_improvement:.6f} (전체 {smape_solar_all:.4f} → 필터링 {smape_solar_filtered:.4f})")

# 로그 파일에 저장
with open(log_path + "comparison.txt", "a") as f:
    f.write(f"<'{test_path}'>\n")
    f.write(f"=== 전체 건물 포함 ===\n")
    f.write(f"일조 R2     : {mae_sunshine_all:.6f} | 일사 R2     : {mae_solar_all:.6f}\n")
    f.write(f"일조 SMAPE  : {smape_sunshine_all:.6f} | 일사 SMAPE  : {smape_solar_all:.6f}\n")
    
    f.write(f"=== 제외 건물({zero_bnos}) 빼고 계산 ===\n")
    f.write(f"일조 R2     : {mae_sunshine_filtered:.6f} | 일사 R2     : {mae_solar_filtered:.6f}\n")
    f.write(f"일조 SMAPE  : {smape_sunshine_filtered:.6f} | 일사 SMAPE  : {smape_solar_filtered:.6f}\n")
    
    f.write(f"=== SMAPE 개선도 ===\n")
    f.write(f"일조 개선: {sunshine_improvement:.6f}\n")
    f.write(f"일사 개선: {solar_improvement:.6f}\n")
    f.write("="*50 + "\n")

print(f"\n로그 파일 저장 완료: {log_path}comparison.txt")









