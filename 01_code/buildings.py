import pandas as pd

df_train = pd.read_csv('./Energy/train.csv')
df_test = pd.read_csv('./Energy/test.csv')

# 일시별 기상값 그룹 → 건물별 기상값 분포가 같은지 보기
check_train = df_train.groupby(['일시'])[['기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)']].nunique()
check_test = df_test.groupby(['일시'])[['기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)']].nunique()

print(check_train)
print(check_test)