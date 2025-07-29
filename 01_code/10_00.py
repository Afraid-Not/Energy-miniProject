
import numpy as np
import pandas as pd

data_path = './Energy/'
save_path = './Energy/10_00_submission/'
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')
submit = pd.read_csv(data_path + 'sample_submission.csv')

train = pd.merge(train_csv, building_csv, on=['건물번호'], how='left')
test = pd.merge(test_csv, building_csv, on=['건물번호'], how='left')

# Index(['건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '건물유형',
#        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'],

# ['일조(hr)','일사(MJ/m2)', '전력소비량(kWh)']

'''(full_pipeline 요약)
<전처리 1>
1. 파생피쳐 1 
    - '연', '월', '일', '요일', 'sin_hour', 'cos_hour', '주말 여부', '공휴일', '불쾌지수', '냉방도일', '체감온도', '근무시간여부'(09:00~18:00),
        '강수량변화율', 'is_rain_x_working_hours' 
    - 공휴일 목록 : 6월 6일 현충일, 8월 15일 광복절
    - onehot = ['건물유형', '요일']
<train['일사(MJ/m2)'] 예측 보조 모델>
2. train의 '일사(MJ/m2)' 의 결측치를 예측한다.
    - train['일사(MJ/m2)']에 낮시간에 일사가 0인 건물 번호를 확인한다.
    - 그 건물번호의 건물 유형을 가지고 같은 건물 유형들을 통해 결측치를 예측한다.
    - 모델은 xgboost, lighgbm 앙상블, early_stop을 각각 사용한다.
    - feature = ['기온(°C)', '강수량(mm)', '풍속(m/s)', 'sin_hour', 'cos_hour', '일조(hr)', '태양광용량(kW)', 
        'ESS저장용량(kWh)', 'PCS용량(kW)', '전력소비량(kWh)']
    - target = ['일사(MJ/m2)']
    - 예측된 '일사(MJ/m2)'는 낮시간(06:00~21:00)을 기준으로 밤시간은 모두 0으로 만든다.
    - 평가는 RMSE
<test[['일조(hr)','일사(MJ/m2)']] 예측 보조 모델>
3. test의 '일조(hr)','일사(MJ/m2)'를 예측한다.
    - test에 없는 '일조(hr)','일사(MJ/m2)'를 각각 예측하여 붙여 넣는다.
    - 건물번호로 분류하여 100번의 모델을 돌린다.
    - xgboost, lighgbm, catboost를 스택킹해서 사용한다. 
    - early_stop을 각각 사용한다.
    - feature = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 'sin_hour', 'cos_hour', '건물유형', '일조(hr)', '연면적(m2)']
    - target = ['일조(hr)', '일사(MJ/m2)']
    - 예측된 '일조(hr)', '일사(MJ/m2)'는 낮시간(06:00~21:00)을 기준으로 밤시간은 모두 0으로 만든다.
    - 평가는 RMSE
<전처리 2>
4. 파생피쳐 2
    - 예측한 '일조(hr)', '일사(MJ/m2)'를 바탕으로 또 다른 파생피쳐들을 만든다.
    - '태양광사용량',  'sun_x_temp', 'sun_x_cooling_area', 'sun_x_working_hours', 'sun_x_solar_capacity', 'daily_total_sunshine',
        'solar_x_cooling_area', 'rolling_hourly_solar_radiation', 'rolling_hourly_sunshine', 'solar_radiation_change',
        'solar_x_working_hours', 'solar_x_solar_capacity' 등
        
<test['전력소비량(kWh)'] 예측 최종 모델>
5. 최종 모델을 만든다.
    - feature = ['sin_hour', 'cos_hour', '요일', '주말 여부', '공휴일', '태양광사용량', '불쾌지수', '냉방도일', '체감온도', '근무시간여부'(09:00~18:00), 
        'sun_x_temp', 'sun_x_cooling_area', 'sun_x_working_hours', 'sun_x_solar_capacity', 'daily_total_sunshine',
        'solar_x_cooling_area', 'rolling_hourly_solar_radiation', 'rolling_hourly_sunshine', 'solar_radiation_change',
        'solar_x_working_hours', 'solar_x_solar_capacity', '주말 여부']
    - target = ['전력소비량(kWh)']
    - 훈련은 '건물번호' 별로 나눠서 진행해서 마지막에 붙인다.
    - 모델은 Level1 ~ Level3으로 만든다.
        level1 : xgboost, lightgbm (early_stop)
        level2 : RidgeCV    (early_stop)
        level3 : 강력한거
6. 평가는 smape
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

<최종 저장>
7. 최종 저장 경로


<주의점>
1. 맨위에서 SEED=1 로 고정하고 모든 모델을 만들때 랜덤값은 SEED로 사용한다.
2. 각 모델 별로 하드 파라미터 튜닝을 적용한다.
3. 모든 모델은 훈련 이전에 이전에 x_temp, x_test, y_temp, y_test로 평가 전용 세트를 만든다. (train_test_split)
    그리고 훈련은 x_temp, y_temp를 x_train, x_val, y_train, y_val로 나눠서 진행한다.
    훈련을 할 떄는 kfold를 n_split=3 으로 훈련한다.
4. 성능 검사는 x_test, y_test로 한다. 모델의 kfold마다 x_val, y_val로 평가 점수를 보여준다.
5. 맨 마지막에 3개의 모델의 최종성적과 랜덤시드를 보여준다.

<경로>
data_path = './Energy/'
save_path = './Energy/10_00_submission/'
train_csv = pd.read_csv(data_path + 'train.csv', index_col=0)
test_csv = pd.read_csv(data_path + 'test.csv', index_col=0)
building_csv = pd.read_csv(data_path + 'building_info.csv')
submit = pd.read_csv(data_path + 'sample_submission.csv')

train = pd.merge(train_csv, building_csv, on=['건물번호'], how='left')
test = pd.merge(test_csv, building_csv, on=['건물번호'], how='left')

'''