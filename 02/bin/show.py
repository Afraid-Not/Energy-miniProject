import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("./Energy/train.csv")
df['일시'] = pd.to_datetime(df['일시'])

# 건물번호별, 일별 최대 전력소비량 계산
daily_max = df.groupby(['건물번호', df['일시'].dt.date])['전력소비량(kWh)'].max().reset_index()
daily_max.columns = ['건물번호', '일자', '최대 전력소비량(kWh)']

# 비교할 건물번호 리스트
# target_buildings = [1, 2, 3]  # 원하는 건물번호 입력

# 시각화

for i in range(1, 101) :
    plt.figure(figsize=(14, 7))
    building_data = daily_max[daily_max['건물번호'] == i]
    plt.plot(building_data['일자'], building_data['최대 전력소비량(kWh)'], marker='o', label=f'건물 {i}')

    plt.title("건물별 일별 최대 전력소비량 (선그래프)")
    plt.xlabel("일자")
    plt.ylabel("최대 전력소비량(kWh)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
