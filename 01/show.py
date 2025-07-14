
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./Energy/12_submission/preprocessed_train.csv')
answer_df = pd.read_csv('./Energy/12_submission/6.6559843954/12_02_20250730_SMAPE_3_9408.csv')
df['일시'] = pd.to_datetime(df['일시'])
answer_df['num_date_time'] = answer_df['num_date_time'].str.extract(r'_(\d{8} \d{2})')[0]
answer_df['num_date_time'] = pd.to_datetime(answer_df['num_date_time'], format='%Y%m%d %H')

# 전주 시간 계산
answer_df['prev_week_time'] = answer_df['num_date_time'] - pd.Timedelta(weeks=1)

# 전주 시간과 현재 예측 시간 둘 다 포함된 df 추출
df_slice = df[df['일시'].isin(answer_df['prev_week_time'])].copy()
df_slice.rename(columns={'전력소비량(kWh)': 'prev_week_actual'}, inplace=True)

# 현재 예측 시각 + 예측값만 추출
answer_slice = answer_df[['num_date_time', 'answer', 'prev_week_time']].copy()

# 병합
merged = pd.merge(answer_slice, df_slice[['일시', 'prev_week_actual']], left_on='prev_week_time', right_on='일시', how='inner')

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(merged['num_date_time'], merged['prev_week_actual'], label='Actual (1 week before)', marker='o', alpha=0.7)
plt.plot(merged['num_date_time'], merged['answer'], label='Predicted', marker='x', alpha=0.7)
plt.title('예측값 vs 전주 실제값 (요일+시각 기준)')
plt.xlabel('예측 시각')
plt.ylabel('전력소비량(kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()