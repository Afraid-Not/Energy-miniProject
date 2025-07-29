import pandas as pd
import matplotlib.pyplot as plt

# 파일 로드
pred = pd.read_csv("./Energy/13_submission/test_csv/13_01_test_SEED56.csv")
dont = pd.read_csv('./Energy/dont_touch/donttouch_testera.csv', encoding='euc-kr')
building_info = pd.read_csv('./Energy/dont_touch/dont_touch_building_info.csv', encoding='euc-kr')

# datetime 변환
pred['일시'] = pd.to_datetime(pred['일시'], errors='coerce')
dont['일시'] = pd.to_datetime(dont['일시'], errors='coerce')

# 건물 ↔ 지점 매핑
pred = pd.merge(pred, building_info[['건물번호', '지점명']], on='건물번호', how='left')

# 기상청 측정값 연결
dont_small = dont[['지점명', '일시', '일사(MJ/m2)']].rename(columns={'일사(MJ/m2)': 'True'})
merged = pd.merge(pred, dont_small, on=['지점명', '일시'], how='left')

# 특정 건물 선택
b1 = merged[merged['건물번호'] == 2].sort_values('일시')

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

# 예측 및 측정값
ax.plot(b1['일시'], b1['일사(MJ/m2)'], label='Predicted', linestyle='--', color='deepskyblue')
ax.plot(b1['일시'], b1['True'], label='Measured', color='black')

# 피크 시간대 강조 (10시~16시)
peak_mask = b1['일시'].dt.hour.between(10, 16)
ax.fill_between(b1['일시'], 0, b1[['일사(MJ/m2)', 'True']].max(axis=1),
                where=peak_mask, facecolor='gray', alpha=0.2, label='Peak Hours')

# 스타일 설정
ax.set_title('Building insolation prediction vs actual')
ax.set_ylabel('Insolation (MJ/m2)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
