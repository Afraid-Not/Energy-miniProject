import pandas as pd

# 데이터 불러오기
df = pd.read_csv("./Energy/train.csv")

# 건물번호별 이상치 비율 계산
outlier_summary = []

for bno, group in df.groupby('건물번호'):
    Q1 = group['전력소비량(kWh)'].quantile(0.25)
    Q3 = group['전력소비량(kWh)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_count = ((group['전력소비량(kWh)'] < lower_bound) | 
                     (group['전력소비량(kWh)'] > upper_bound)).sum()
    total_count = len(group)
    outlier_ratio = outlier_count / total_count * 100
    
    outlier_summary.append({
        '건물번호': bno,
        '데이터수': total_count,
        '이상치수': outlier_count,
        '이상치비율(%)': round(outlier_ratio, 2),
        '하한값': round(lower_bound, 2),
        '상한값': round(upper_bound, 2)
    })

# DataFrame 생성 및 정렬
outlier_df = pd.DataFrame(outlier_summary).sort_values('이상치비율(%)', ascending=False)

# CSV 저장
outlier_df.to_csv("건물번호별_전력소비량_이상치분석.csv", index=False, encoding="utf-8-sig")

print("CSV 저장 완료: 건물번호별_전력소비량_이상치분석.csv")
