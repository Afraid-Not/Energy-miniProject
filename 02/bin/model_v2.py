print(f"[model] 시작")
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from lightgbm import log_evaluation, early_stopping
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

seed_file = "./Energy/02/log/(model)SEED_COUNTS.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 1 # seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)

# train = pd.read_csv(trainer + '06_train_120_ver3.csv', index_col=0)

data_path = './Energy/'
csv_path = './Energy/02/'
log_path = './Energy/02/log/'
trainer = './Energy/02/trainer/'

os.makedirs(log_path, exist_ok=True)
os.makedirs(trainer, exist_ok=True)


# print(train.shape)
# print(train['전력소비량(kWh)'].head())
# print(test.shape)
# print(test['전력소비량(kWh)'].head())
# print(list(set(train.columns)-set(test.columns)))
# 사용 예시
import pandas as pd
import numpy as np
import re

# =========================
# 1) 규칙 파일 로드 & 정규화
# =========================
def parse_dt(x):
    if pd.isna(x) or (isinstance(x, str) and x.strip().lower() in {'', 'x', 'nan'}):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x)
    s = str(x).strip()
    # 허용 예: 20240803 12, 2024080312, 2024-08-03 12, 2024/08/03 12
    s = re.sub(r'[^0-9]', '', s)   # 숫자만 남김
    # YYYYMMDDHH or YYYYMMDD
    if len(s) == 10:  # YYYYMMDDHH
        return pd.to_datetime(s, format='%Y%m%d%H', errors='coerce')
    elif len(s) == 8: # YYYYMMDD (시각 없으면 00시로)
        return pd.to_datetime(s, format='%Y%m%d', errors='coerce')
    else:
        return pd.to_datetime(x, errors='coerce')

# 규칙 컬럼 패턴들
RANGE_START_PAT = re.compile(r'\(start\)', re.IGNORECASE)
RANGE_END_PAT   = re.compile(r'\(end\)', re.IGNORECASE)
POINT_PAT       = re.compile(r'이상치\(\s*시간\s*\)|이상치\(시간\)', re.IGNORECASE)
AFTER_UP_PAT    = re.compile(r'이후사용량급증', re.IGNORECASE)
AFTER_DOWN_PAT  = re.compile(r'이후사용량감소', re.IGNORECASE)
SPLIT_NEED_PAT  = re.compile(r'기간분리필요', re.IGNORECASE)
SWITCH_PAT      = re.compile(r'구간\s*전환', re.IGNORECASE)  # 구간 전환1/2 등

def load_outlier_rules(path='./Energy/outlier2.csv'):
    rules = pd.read_csv(path, encoding='cp949')
    # 건물번호 정수화
    # print(rules.columns )
    if '건물번호' not in rules.columns:
        raise ValueError("outlier2.csv에 '건물번호' 컬럼이 필요합니다.")
    rules['건물번호'] = rules['건물번호'].astype(int)

    # datetime으로 파싱해야 할 열들 선택
    dt_cols = []
    for c in rules.columns:
        name = str(c)
        if (RANGE_START_PAT.search(name) or RANGE_END_PAT.search(name) or
            POINT_PAT.search(name) or AFTER_UP_PAT.search(name) or AFTER_DOWN_PAT.search(name) or
            SWITCH_PAT.search(name)):
            dt_cols.append(c)

    for c in dt_cols:
        rules[c] = rules[c].apply(parse_dt)

    # '기간분리필요'는 불리언/표식으로 정규화
    for c in rules.columns:
        if SPLIT_NEED_PAT.search(str(c)):
            rules[c] = rules[c].apply(lambda x: False if pd.isna(x) else (str(x).strip().lower() not in {'', 'x', '0', 'false', 'nan'}))

    return rules

# =======================================
# 2) 메인 처리: 기간/포인트 제거 or 보간
# =======================================
def clean_with_rules(df, rules,
                     time_col='일시', bno_col='건물번호', y_col='전력소비량(kWh)',
                     mode='remove',
                     after_window_hours=0,
                     shift_map=None,
                     split_gap_hours=6):
    """
    mode: 'remove' or 'interpolate'
    after_window_hours: 이후사용량급증/감소 시점 직후 몇 시간 제거/보정할지(0이면 제거 안함, 세그먼트만 구분)
    shift_map: { 건물번호: {'hours': +6 or -6}, ... }  # 기간분리필요 시 구간전환 시점 이후를 시간 시프트
    split_gap_hours: 시프트할 경우, 구간전환 시점 ~ 시점+gap 구간은 겹침 방지를 위해 제거
    """
    df = df.copy()

    # 일시를 datetime으로
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = df[time_col].apply(parse_dt)
    df = df.sort_values([bno_col, time_col]).reset_index(drop=True)

    # 세그먼트 표기 컬럼
    df['segment'] = 0  # 기본 0

    # 이상치 마스크(제거/보정용)
    drop_mask = pd.Series(False, index=df.index)

    # 보정 시 결측으로 만들 배열
    to_nan = pd.Series(False, index=df.index)

    # 규칙 열 분류
    range_pairs = []   # [(start_col, end_col)]
    point_cols  = []   # ['이상치(num)(시간)' 등]
    after_cols_up = []     # '이후사용량급증'
    after_cols_down = []   # '이후사용량감소'
    switch_cols = []       # '구간 전환1,2,...'
    split_need_cols = []   # 기간분리필요

    for c in rules.columns:
        s = str(c)
        if RANGE_START_PAT.search(s):
            # start/end 짝 매칭
            base = RANGE_START_PAT.sub('', s).strip()
            # 같은 base로 end를 찾기
            candidates = [col for col in rules.columns if RANGE_END_PAT.search(str(col)) and RANGE_END_PAT.sub('', str(col)).strip() == base]
            if candidates:
                range_pairs.append((c, candidates[0]))
        elif POINT_PAT.search(s):
            point_cols.append(c)
        elif AFTER_UP_PAT.search(s):
            after_cols_up.append(c)
        elif AFTER_DOWN_PAT.search(s):
            after_cols_down.append(c)
        elif SWITCH_PAT.search(s):
            switch_cols.append(c)
        elif SPLIT_NEED_PAT.search(s):
            split_need_cols.append(c)

    # 건물별 처리
    for bno, grp_rule in rules.groupby('건물번호'):
        idx_b = (df[bno_col] == bno)

        # 2-1) 기간 이상치 처리
        for sc, ec in range_pairs:
            for _, r in grp_rule[[sc, ec]].dropna(how='all').iterrows():
                st, en = r.get(sc, pd.NaT), r.get(ec, pd.NaT)
                if pd.isna(st) or pd.isna(en):  # 둘 다 있어야 범위
                    continue
                m = idx_b & (df[time_col] >= st) & (df[time_col] <= en)
                if mode == 'remove':
                    drop_mask |= m
                else:
                    to_nan |= m

        # 2-2) 단일 시각 이상치(한 시간)
        for pc in point_cols:
            times = grp_rule[pc].dropna().unique()
            if len(times) == 0:
                continue
            m = idx_b & (df[time_col].isin(times))
            if mode == 'remove':
                drop_mask |= m
            else:
                to_nan |= m

        # 2-3) 이후사용량 급증/감소: 세그먼트 분리 + (선택) 직후 윈도우 제거/보정
        after_times = []
        for ac in after_cols_up + after_cols_down:
            ts = grp_rule[ac].dropna().unique()
            after_times.extend(list(ts))
        # 중복 제거 & 정렬
        after_times = sorted(set([t for t in after_times if not pd.isna(t)]))
        for i, ts in enumerate(after_times, start=1):
            # ts 이후를 다음 세그먼트로 표기
            m_after = idx_b & (df[time_col] >= ts)
            df.loc[m_after, 'segment'] = df.loc[m_after, 'segment'].mask(df.loc[m_after, 'segment'] < i, i)
            if after_window_hours and after_window_hours > 0:
                # ts 직후 after_window_hours 시간 제거/보정
                m_win = idx_b & (df[time_col] > ts) & (df[time_col] <= ts + pd.Timedelta(hours=after_window_hours))
                if mode == 'remove':
                    drop_mask |= m_win
                else:
                    to_nan |= m_win

        # 2-4) 기간분리필요 + 시간 시프트(건물별 지정)
        #    - 구간 전환 시점(들) 읽기
        switches = []
        for sc in switch_cols:
            ts = grp_rule[sc].dropna().unique()
            switches.extend(list(ts))
        switches = sorted([t for t in switches if not pd.isna(t)])

        need_split = False
        for spc in split_need_cols:
            # 값이 True/표시가 있으면 기간분리 필요
            if grp_rule[spc].astype(bool).any():
                need_split = True
                break

        if need_split and switches:
            # 시프트 설정
            shift_hours = 0
            if shift_map and bno in shift_map:
                shift_hours = int(shift_map[bno].get('hours', 0))

            pivot = switches[0]  # 보통 '구간 전환1'을 기준
            # 세그먼트 분리(기본)
            df.loc[idx_b & (df[time_col] >= pivot), 'segment'] = df.loc[idx_b & (df[time_col] >= pivot), 'segment'].max() + 1

            if shift_hours != 0:
                # pivot 이후 데이터 시간 시프트
                m_after = idx_b & (df[time_col] >= pivot)
                df.loc[m_after, time_col] = df.loc[m_after, time_col] + pd.Timedelta(hours=shift_hours)

                # 겹치는 구간 제거(중복/겹침 완화)
                if split_gap_hours and split_gap_hours > 0:
                    gap_start = pivot
                    gap_end = pivot + pd.Timedelta(hours=abs(shift_hours))  # 보통 6시간
                    m_gap = idx_b & (df[time_col] >= gap_start) & (df[time_col] < gap_end + pd.Timedelta(hours=split_gap_hours-abs(shift_hours)))
                    # 안전하게 gap_start ~ gap_start+split_gap_hours 제거
                    m_gap = idx_b & (df[time_col] >= gap_start) & (df[time_col] < gap_start + pd.Timedelta(hours=split_gap_hours))
                    if mode == 'remove':
                        drop_mask |= m_gap
                    else:
                        to_nan |= m_gap

    # 실제 적용
    if mode == 'remove':
        cleaned = df.loc[~drop_mask].copy()
    else:
        cleaned = df.copy()
        cleaned.loc[to_nan, y_col] = np.nan
        # 건물별 선형보간(시간축 기준)
        cleaned = (cleaned
                   .set_index(time_col)
                   .groupby(bno_col, group_keys=False)
                   .apply(lambda g: g.sort_index().assign(**{
                       y_col: g[y_col].interpolate(method='time').bfill().ffill()
                   }))
                   .reset_index())

    # 중복 일시(시프트 후) 처리: 건물번호+일시가 겹치면 평균
    cleaned = (cleaned
               .groupby([bno_col, time_col], as_index=False)
               .agg({**{c:'first' for c in cleaned.columns if c not in [y_col, bno_col, time_col, 'segment']},
                     **{y_col: 'mean'}, 'segment':'max'}))

    return cleaned


# =========================
# 사용 예시
# =========================
# 1) 규칙 로드
rules = load_outlier_rules('./Energy/outlier2.csv')  # 경로는 너 환경에 맞게
test_call= '06_test_120_ver3.csv'
test = pd.read_csv(trainer + test_call)

train_call = '06_train_120_ver3.csv'
# 2) 원본 데이터 로드 예시
df = pd.read_csv(trainer + train_call)  # 예시
df['일시'] = pd.to_datetime(df['일시'])  # 가능하면 미리 파싱

shift_map = {
    80: {'hours': 0},
    87: {'hours': +6},   
}

# 4) 실행 (삭제 모드)
cleaned_remove = clean_with_rules(df, rules, mode='remove',
                                  after_window_hours=0,
                                  shift_map=shift_map,
                                  split_gap_hours=6)

# 5) 실행 (보간 모드)
# cleaned_interp = clean_with_rules(df, rules, mode='interpolate',
#                                   after_window_hours=0,
#                                   shift_map=shift_map,
#                                   split_gap_hours=6)
test['segment'] = 0
train = cleaned_remove.copy()

# 1) 휴무/피크 설정 로드
holiday_path = './Energy/holiday.csv'  # 경로 맞게 수정 가능
holiday_df = pd.read_csv(holiday_path, encoding='cp949')

# 요일 매핑
weekday_map = {'월요일':0, '화요일':1, '수요일':2, '목요일':3, '금요일':4, '토요일':5, '일요일':6}
weekday_cols = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']

# 건물별 피크 요일(주중) 추출: 월~금 중 'peak' 표시된 요일
peak_map = {}
for _, row in holiday_df.iterrows():
    bno = row['건물번호']
    peak_idx = None
    for day in ['월요일','화요일','수요일','목요일','금요일']:
        if str(row.get(day, '')).strip() == 'peak':
            peak_idx = weekday_map[day]
            break
    peak_map[bno] = peak_idx

# 건물별 정기휴무 요일(여러 개 가능)
regular_closed_map = {}
for _, row in holiday_df.iterrows():
    bno = row['건물번호']
    closed_set = set()
    for day in weekday_cols:
        if str(row.get(day, '')).strip() == '휴무':
            closed_set.add(weekday_map[day])
    regular_closed_map[bno] = closed_set

# 건물별 임시휴무일/휴무예정일(YYYYMMDD -> date)
def _to_date(x):
    if pd.isna(x): 
        return None
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return None
    # 허용 포맷: YYYYMMDD / YYYY-MM-DD / 기타 to_datetime 파싱
    try:
        if len(s) == 8 and s.isdigit():
            return pd.to_datetime(s, format='%Y%m%d').date()
        return pd.to_datetime(s).date()
    except Exception:
        return None

temp_cols = ['임시휴무1','임시휴무2','임시휴무3','임시휴무4','임시휴무5']

temp_closed_map = {}
planned_closed_map = {}
for _, row in holiday_df.iterrows():
    bno = row['건물번호']
    # 임시휴무 여러 개를 set으로
    tset = set()
    for c in temp_cols:
        d = _to_date(row.get(c))
        if d is not None:
            tset.add(d)
    temp_closed_map[bno] = tset
    # 휴무예정일(단일)
    planned_closed_map[bno] = _to_date(row.get('휴무예정일'))

# 2) 공통 처리 함수
def add_peak_and_holidays(df):
    df = df.copy()
    # 일시 -> datetime, 날짜/요일 파생
    if not np.issubdtype(df['일시'].dtype, np.datetime64):
        df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
    df['date'] = df['일시'].dt.date
    df['weekday'] = df['일시'].dt.weekday  # 0=월 ~ 6=일

    # peak_dayofweek: 건물별 peak 요일 == 행의 weekday
    df['peak_dayofweek'] = df.apply(
        lambda r: int(peak_map.get(r['건물번호']) is not None and r['weekday'] == peak_map.get(r['건물번호'])),
        axis=1
    )

    # 전역 공휴일(6/6, 8/15) - 연도 무관 처리
    df['is_global_holiday'] = ((df['일시'].dt.month == 6) & (df['일시'].dt.day == 6)) | \
                              ((df['일시'].dt.month == 8) & (df['일시'].dt.day == 15))

    # 정기휴무(건물별 요일)
    df['is_regular_closed'] = df.apply(
        lambda r: int(r['weekday'] in regular_closed_map.get(r['건물번호'], set())),
        axis=1
    )

    # 임시휴무(건물별 날짜 세트)
    df['is_temp_closed'] = df.apply(
        lambda r: int(r['date'] in temp_closed_map.get(r['건물번호'], set())),
        axis=1
    )

    # 휴무예정일(건물별 단일 날짜)
    df['is_planned_closed'] = df.apply(
        lambda r: int(planned_closed_map.get(r['건물번호']) is not None and r['date'] == planned_closed_map.get(r['건물번호'])),
        axis=1
    )

    # 최종 holidays: 전역공휴일 OR 정기휴무 OR 임시휴무 OR 휴무예정일
    df['holidays'] = (
        df['is_global_holiday'] |
        (df['is_regular_closed'] == 1) |
        (df['is_temp_closed'] == 1) |
        (df['is_planned_closed'] == 1)
    ).astype(int)

    # 불필요 중간 컬럼 정리(원하면 남겨도 됨)
    df.drop(columns=['date'], inplace=True)
    return df

# 3) train/test에 적용
train = add_peak_and_holidays(train)
test  = add_peak_and_holidays(test)

""" 
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_feature_importance(X, y, methods=['random_forest', 'correlation', 'mutual_info'], 
                               save_path='feature_importance.csv'):

    
    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({'feature': feature_names})
    
    print(f"피처 개수: {len(feature_names)}")
    print(f"데이터 크기: {X.shape}")
    
    # 1. Random Forest Feature Importance
    if 'random_forest' in methods:
        print("Random Forest 피처 중요도 계산 중...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance_df['rf_importance'] = rf.feature_importances_
        importance_df['rf_rank'] = importance_df['rf_importance'].rank(ascending=False)
    
    # 2. Correlation with target
    if 'correlation' in methods:
        print("상관관계 기반 중요도 계산 중...")
        correlations = []
        for col in feature_names:
            corr = abs(np.corrcoef(X[col], y)[0, 1])
            correlations.append(corr if not np.isnan(corr) else 0)
        importance_df['correlation'] = correlations
        importance_df['corr_rank'] = importance_df['correlation'].rank(ascending=False)
    
    # 3. Mutual Information
    if 'mutual_info' in methods:
        print("Mutual Information 중요도 계산 중...")
        # 샘플링으로 계산 속도 향상 (데이터가 클 경우)
        if len(X) > 50000:
            sample_idx = np.random.choice(len(X), 50000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
            
        mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
        importance_df['mutual_info'] = mi_scores
        importance_df['mi_rank'] = importance_df['mutual_info'].rank(ascending=False)
    
    # 4. F-statistic (univariate statistical test)
    if 'f_statistic' in methods:
        print("F-statistic 중요도 계산 중...")
        f_scores, _ = f_regression(X, y)
        importance_df['f_statistic'] = f_scores
        importance_df['f_rank'] = importance_df['f_statistic'].rank(ascending=False)
    
    # 종합 점수 계산 (rank 기반)
    rank_cols = [col for col in importance_df.columns if col.endswith('_rank')]
    if rank_cols:
        # 평균 순위 계산 (낮은 순위가 더 중요함)
        importance_df['avg_rank'] = importance_df[rank_cols].mean(axis=1)
        importance_df['combined_rank'] = importance_df['avg_rank'].rank()
    
    # 중요도 순으로 정렬
    if 'combined_rank' in importance_df.columns:
        importance_df = importance_df.sort_values('combined_rank')
    elif 'rf_importance' in importance_df.columns:
        importance_df = importance_df.sort_values('rf_importance', ascending=False)
    
    # CSV 저장
    importance_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"피처 중요도가 '{save_path}'에 저장되었습니다.")
    
    # 상위 20개 피처 출력
    print("\n=== 상위 20개 중요 피처 ===")
    display_cols = ['feature']
    if 'rf_importance' in importance_df.columns:
        display_cols.append('rf_importance')
    if 'correlation' in importance_df.columns:
        display_cols.append('correlation')
    if 'combined_rank' in importance_df.columns:
        display_cols.append('combined_rank')
    
    print(importance_df[display_cols].head(20).to_string(index=False))
    
    return importance_df

# 실행 코드
if __name__ == "__main__":
    # 피처 중요도 계산 및 저장
    importance_result = calculate_feature_importance(
        X, y, 
        methods=['random_forest', 'correlation', 'mutual_info'],
        save_path='./Energy/02/log/feature_importance_analysis.csv'
    )
    
    # 추가 분석: 중요도별 피처 그룹 생성
    print("\n=== 피처 중요도 그룹 분석 ===")
    
    # Random Forest 기준으로 그룹 나누기
    if 'rf_importance' in importance_result.columns:
        rf_importance = importance_result['rf_importance']
        
        # 상위 10%, 20%, 50% 피처들
        top_10_pct = int(len(importance_result) * 0.1)
        top_20_pct = int(len(importance_result) * 0.2)
        top_50_pct = int(len(importance_result) * 0.5)
        
        high_importance = importance_result.head(top_10_pct)['feature'].tolist()
        medium_importance = importance_result.iloc[top_10_pct:top_20_pct]['feature'].tolist()
        low_importance = importance_result.iloc[top_20_pct:top_50_pct]['feature'].tolist()
        
        print(f"고중요도 피처 (상위 10%): {len(high_importance)}개")
        print(f"중중요도 피처 (10-20%): {len(medium_importance)}개") 
        print(f"저중요도 피처 (20-50%): {len(low_importance)}개")
        
        # 그룹별 피처 저장
        pd.DataFrame({'high_importance_features': pd.Series(high_importance)}).to_csv(
            'high_importance_features.csv', index=False, encoding='utf-8-sig'
        )
        print("고중요도 피처가 'high_importance_features.csv'에 저장되었습니다.")
 """

le = LabelEncoder()
train['건물유형'] = le.fit_transform(train['건물유형'])
test['건물유형'] = le.transform(test['건물유형'])

target_col = ['전력소비량(kWh)']
drop_col = ['일시']
X = train.drop(target_col + drop_col, axis=1)
y = train['전력소비량(kWh)']
test = test.drop(target_col + drop_col, axis=1).copy()


















import datetime
save_path = f'./Energy/02/{SEED}_submission/'
os.makedirs(save_path , exist_ok=True )
samplesub = pd.read_csv(data_path + 'sample_submission.csv')
samplesub['answer'] = ensemble_predictions
today = datetime.datetime.now().strftime('%Y%m%d')
score_str = f"{ensemble_smape:.4f}".replace('.', '_')

filename = f"13_02_{today}_SMAPE_{score_str}.csv"
samplesub.to_csv(save_path + filename, index=False)

with open(save_path + "(LOG)model.txt", "a") as f:
    f.write(f"<SEED :{SEED}>\n")
    f.write(f"{filename}\n")
    f.write(f"test file name : {test_call}\n")
    f.write(f"train file name : {train_call}\n")
    f.write(f"전체 데이터 모델 SMAPE: {whole_avg_smape:.4f}\n")
    f.write(f"건물유형별 모델 SMAPE: {building_type_avg_smape:.4f}\n")
    f.write(f"건물번호별 모델 SMAPE: {building_num_avg_smape:.4f}\n") 
    f.write(f"앙상블 모델 SMAPE: {ensemble_smape:.4f}\n")
    f.write("="*40 + "\n")

print(f"[4] 종료 ")