# Energy Mini Project ⚡

> **전력소비량 예측 모델** - 고급 머신러닝 파이프라인

<p align="center">
  <img src="https://img.shields.io/badge/Score-5.15%25_SMAPE-success?style=flat-square" alt="Score"/>
  <img src="https://img.shields.io/badge/Rank-8th_Public_|_6th_Private-orange?style=flat-square" alt="Rank"/>
  <img src="https://img.shields.io/badge/Period-2025.07.14_~_2025.08.25-lightgrey?style=flat-square" alt="Period"/>
</p>

## 📊 발표 자료

프로젝트 상세 분석 및 결과는 **[📄 발표 자료 PDF](./presentation.pdf)** 를 참고해주세요.

---

## 📋 Project Overview

100개 건물의 시간대별 전력소비량 예측을 위한 통합적인 분석 솔루션

### 🎯 목표
- **평가지표**: SMAPE (Symmetric Mean Absolute Percentage Error) 최소화
- **데이터**: 건물 정보, 기상 데이터 (온도, 습도, 강수량, 일사, 일조)
- **예측 대상**: 전력소비량 (kWh)

### 🏆 최종 성적
- **최고 점수**: SMAPE **5.15397**
- **Public 순위**: 8위
- **Private 순위**: 6위
- **프로젝트 기간**: 2025.07.14 ~ 2025.08.25

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **ML Libraries**: LightGBM, XGBoost, CatBoost
- **Optimization**: Optuna (Hyperparameter Tuning)
- **Data Processing**: Pandas, NumPy
- **Feature Engineering**: Time Series, Statistical Features

---

## 📁 Project Structure

| 폴더 | 설명 |
|------|------|
| `01/` | 초기 버전 - 전처리 및 기본 모델 |
| `01_0812/` | 8월 12일 버전 - 개선된 전처리 |
| `01_code/` | 버전별 코드 (01~10) |
| `01_best_code/` | 01 시리즈 최적 코드 |
| `02/` | 두 번째 접근 - 앙상블 모델 |
| `02_best_code/` | 02 시리즈 최적 코드 |
| `03/` | 세 번째 접근 - Feature Importance |
| `_final/` | 최종 제출 코드 |
| `best/` | 최고 성능 모델 |
| `best_model/` | **최종 선택 모델 (SMAPE: 5.15)** |

---

## 🔍 핵심 전략

### 1️⃣ 데이터 전처리

#### 시간 기반 피처
- **날짜/시간 분해**: 시간, 요일, 월, 주간/주말 구분
- **SIN/COS 인코딩**: 주기성 변수를 연속적 특성으로 변환
- **계절성 피처**: 여름철(6-9월) 구분, daylight 기반 특성

#### 기상 관련 피처
- **일사량/일조량 계산**: 위도, 경도 기반 태양 관련 특성
- **냉난방도일(CDH/HDH)**: 온도 기반 냉난방 수요 지표
- **불쾌지수**: (0.81T + 0.01H(0.99T-14.3) + 46.3)
- **건물 그룹화**: 유사 건물을 30개 그룹으로 군집화

### 2️⃣ 데이터 증강 및 이상치 처리

#### 시계열 데이터 증강
- **시간 간격 확장**: 15분 단위 → 증강
- **방법 1 (Linear Interpolation)**: 선형 보간
- **방법 2 (upsample_20min_linear)**: 구간 업샘플링

#### 이상치 탐지 및 처리
- **탐지 방법**: IQR(사분위수), 단일 시점 이상치(singles), 구간 이상치(intervals)
- **처리 방식**: 
  - 일사량(solar) 결측: 0으로 대체
  - 일조량(sunshine) 결측: groupby 건물별 선형 보간 또는 ffill/bfill
  - Z-Score 기반 시각적 분석으로 이상치 제거(12% 향상)

### 3️⃣ 통계 기반 피처 생성

5단계 계층적 통계 피처:
1. **가장 세분화된 통계**: 건물번호, 시간, 요일 기준
2. **휴일 기반 통계**: 건물번호, 시간, 휴일여부 기준
3. **시간별 통계**: 건물번호, 시간 기준
4. **건물 전체 통계**: 건물번호 기준
5. **글로벌 통계**: 전체 데이터 기준

### 4️⃣ 모델 앙상블 전략

#### 3가지 모델링 접근
- **세그먼트별 모델**: 건물을 기준으로 세그먼트화 (ESS 보유, PVC 보유, 일반 건물)
- **유형별 모델**: 건물 유형별 분류 (공장, 학교, 데이터센터, 아파트 등)
- **건물별 모델**: 개별 건물 특화 학습 (100개 건물별 독립 모델)

#### 앙상블 구조
XGBoost + LightGBM (TimeSeriesSplit)
↓
OOF 예측 1 + OOF 예측 2
↓
최종 앙상블 (가중 평균)

#### 핵심 모델링 기법
- **TimeSeriesSplit**: 시계열 특성을 고려한 K-Fold
- **log1p 변환**: 타겟 분포 개선
- **Optuna 최적화**: 하이퍼파라미터 자동 탐색
- **OOF(Out-of-Fold) 검증**: 전체 파이프라인 성능 검증

### 5️⃣ 하이퍼파라미터 최적화 (Optuna)

#### Optuna 베이지안 최적화
1. 목적함수 정의: SMAPE 평가지표
2. 탐색 공간 설정: XGBoost/LightGBM 파라미터 범위
3. TPE(Tree-structured Parzen Estimator) 샘플링
4. MedianPruner: 유망하지 않은 트라이얼 조기 종료
5. N_TRIALS_MODEL(25회): 반복 실행 후 최적 파라미터 선택 및 저장

#### MD5 해시 기반 캐시 시스템
- 피처셋 해시: 파라미터 조합의 해시값으로 캐시 키 생성
- 결과 캐싱: JSON으로 최적 파라미터 저장 및 재사용
- 재현성 보장: 모든 단계에서 동일한 시드 유지
- 실험 추적: 파사피라미터 구성별 성능 비교 가능

---

## 📈 Performance Progress

| Version | SMAPE | 주요 개선사항 |
|---------|-------|---------------|
| 초기 버전 | 13.8% | 세그먼트별 모델 OOF |
| v1 | 14.9% | 유형별 모델 OOF |
| v2 | 7.7% | 건물별 모델 OOF |
| v3 | 10.97% | 최종 앙상블 LOCAL |
| **Final (Public)** | **5.15%** | **8th 최종 제출** |
| **Final (Private)** | **5.52%** | **6th 최종 순위** |

### 핵심 성과
- **세그먼트/유형/건물별** 모델링 각 단계 검증
- **특정 건물 유형별** 성능 차이 발견 (산업시설 > 교육시설)
- **휴일/영업일** 구분 모델링으로 패턴 예측 향상

---

## 🚀 How to Run

### 최종 모델 실행
```bash
#1. 전처리 및 피크 시간대 모델
python best_model/13_01_preprocessing_peak_model.py
#2. 최종 모델 학습 및 예측
python best_model/13_02_model.py
```
### 버전별 실행
```bash
Version
#1 - 기본 접근
python 01/01_preprocessing.py
python 01/01_model.py
Version
#2 - 앙상블
python 02/02_model_ver10_ensemble_IDTypeTotal_solar.pyVersion 3 - Feature Importance
python 03/preprocessing_ver5.py
```

### 💡 Key Insights
## **데이터 분석**

- **일사량/일조량 보간**: KNN 기반 보간으로 기상 데이터 품질 향상
- **시간대별 패턴**: Peak/Off-peak 시간 구분이 효과적
- **건물 군집화**: 유사 특성 건물 그룹화로 일반화 성능 개선

##**모델링**

- **앙상블 효과**: XGBoost + LightGBM 조합이 단일 모델 대비 안정적
- **TimeSeriesSplit**: 시계열 특성 고려한 검증으로 과적합 방지
- **Optuna 최적화**: 자동 하이퍼파라미터 탐색으로 성능 향상

##**성능 향상 요인**

- **이상치 제거 효과**: Z-Score 기반 처리로 SMAPE 약 12% 향상
- **통계 기반 피처**: 계층적 통계 특성이 모델 성능에 큰 기여
- **MD5 캐싱**: 재현성 보장 및 실험 속도 ~80% 단축

### 🎓 Lessons Learned
## **재현성 보장**

- **다중 시드 고정**: PYTHONHASHSEED, random, numpy 시드 일괄 설정
- **버전 태깅**: 각 실험에 고유 버전 코드 부여 (version = "LETS_GO")
- **알고리즘 모든 랜덤 파라미터**: seed 파라미터 적용
- **분리된 경로**: 실험별 독립 저장 경로로 덮어쓰기 방지

## **확장성 및 향후 개선 방향**

- **모델 확장성**: 세그먼트/유형/건물별 특화 모델 추가 가능
- **피처 클라이언트**: 새로운 피처 생성 함수 즉시 통합 가능
- **멀티프로세싱**: 병렬화 위한 프레임워크 준비됨

## **추가 개선 가능 영역**

- **메타 피처 자동 생성 시스템**
- **클라우드 기반 분산 학습 지원**
- **실시간 예측 API 구현**

### 🙏 후기 및 소감
## **소감**

**"이번 대회의 교훈은 피처 엔지니어링의 중요성을 단적으로 확인했다는 점에 있습니다. 일조시간과 일조량을 예측하는 모델을 시간이 한계로 더욱 정교할 및 고도화하지 못했다는 아쉬움이 깊이 남습니다."**

## **모델 구성 전략 및 효과**

- **"보조 타깃 생성 → 메인 타깃 예측"의 3단계로 고정**
- **세로화 + 시계열 전용 피처 + 기본문 양상분이 가장 알맞다고 점검**
- **optuna sampling으로 모델 성능 시간을 획기적으로 줄였음**

Period: 2025.07.14 ~ 2025.08.25
Final Score: SMAPE 5.15% (Public 8th / Private 6th)
Team: [TEAM] 한국식_러다이트_운동원
