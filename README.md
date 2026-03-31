# 2025 전력사용량 예측 AI 경진대회

<p align="center">
  <img src="6등_2025_전력사용량_예측_AI_경진대회.png" alt="Competition Result - 6th Place" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/SMAPE-5.15%25-brightgreen?style=for-the-badge" alt="SMAPE Score"/>
  <img src="https://img.shields.io/badge/Public-8th-blue?style=for-the-badge" alt="Public Rank"/>
  <img src="https://img.shields.io/badge/Private-6th-blueviolet?style=for-the-badge" alt="Private Rank"/>
  <img src="https://img.shields.io/badge/Period-2025.07.14_~_2025.08.25-grey?style=for-the-badge" alt="Period"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LightGBM-Gradient_Boosting-9ACD32?style=flat-square" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient_Boosting-EC4E20?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/CatBoost-Gradient_Boosting-FFCC00?style=flat-square" alt="CatBoost"/>
  <img src="https://img.shields.io/badge/Optuna-Bayesian_Optimization-2196F3?style=flat-square" alt="Optuna"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
</p>

<p align="center">
  <b>Team: 한국식_러다이트_운동원</b><br/>
  100개 건물의 시간대별 전력소비량(kWh)을 예측하는 AI 솔루션
</p>

<p align="center">
  <a href="./보고서_2025_전력사용량_예측_AI_경진대회.pdf">보고서 PDF</a> | <a href="./presentation.pdf">발표 자료 PDF</a>
</p>

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [앙상블 파이프라인 아키텍처](#앙상블-파이프라인-아키텍처)
- [핵심 전략](#핵심-전략)
  - [1. 피처 엔지니어링](#1-피처-엔지니어링)
  - [2. 데이터 증강 및 이상치 처리](#2-데이터-증강-및-이상치-처리)
  - [3. 계층적 통계 피처](#3-계층적-통계-피처)
  - [4. 앙상블 전략](#4-앙상블-전략)
  - [5. Optuna + MD5 캐싱](#5-optuna--md5-캐싱)
  - [6. 모델링 접근법](#6-모델링-접근법)
- [성능 추이](#성능-추이)
- [프로젝트 구조](#프로젝트-구조)
- [실행 방법](#실행-방법)
- [재현성](#재현성)
- [개발자](#개발자)

---

## 프로젝트 개요

| 항목            | 내용                                                    |
| --------------- | ------------------------------------------------------- |
| **대회**        | 2025 전력사용량 예측 AI 경진대회                        |
| **목표**        | 100개 건물의 시간대별 전력소비량(kWh) 예측              |
| **평가지표**    | SMAPE (Symmetric Mean Absolute Percentage Error)        |
| **최종 점수**   | SMAPE 5.15% (Public 8위 / Private 6위)                  |
| **기간**        | 2025.07.14 ~ 2025.08.25                                 |
| **팀**          | 한국식*러다이트*운동원                                  |
| **입력 데이터** | 건물 정보, 기상 데이터 (온도, 습도, 강수량, 일사, 일조) |

---

## 기술 스택

| 분류                      | 기술                                          |
| ------------------------- | --------------------------------------------- |
| **언어**                  | Python 3.x                                    |
| **부스팅 모델**           | LightGBM, XGBoost, CatBoost                   |
| **하이퍼파라미터 최적화** | Optuna (TPE Sampler + MedianPruner)           |
| **데이터 처리**           | Pandas, NumPy                                 |
| **머신러닝 유틸**         | scikit-learn (TimeSeriesSplit, preprocessing) |

---

## 앙상블 파이프라인 아키텍처

```
+-------------------------------+
|        Raw Data Input         |
|  (건물 정보 + 기상 데이터)    |
+---------------+---------------+
                |
                v
+-------------------------------+
|       Preprocessing           |
|  - Peak / Off-peak 구분       |
|  - 이상치 제거 (IQR, Z-Score) |
|  - 시간 간격 확장 (15min+)    |
+---------------+---------------+
                |
                v
+-------------------------------+
|     Feature Engineering       |
|  - 시간 피처 (SIN/COS)       |
|  - 기상 피처 (CDH/HDH)       |
|  - 5단계 계층적 통계 피처     |
|  - 건물 군집화 (100 -> 30)    |
+---------------+---------------+
                |
        +-------+-------+
        |               |
        v               v
+---------------+ +---------------+
|   XGBoost     | |   LightGBM    |
| TimeSeriesSplit| | TimeSeriesSplit|
|   K-Fold      | |   K-Fold      |
+-------+-------+ +-------+-------+
        |               |
        v               v
+---------------+ +---------------+
| OOF Predict 1 | | OOF Predict 2 |
|  (log1p inv)  | |  (log1p inv)  |
+-------+-------+ +-------+-------+
        |               |
        +-------+-------+
                |
                v
+-------------------------------+
|    Weighted Average Ensemble  |
+-------------------------------+
                |
                v
+-------------------------------+
|     Final Prediction          |
|     SMAPE: 5.15%              |
+-------------------------------+
```

**Optuna 최적화 루프** (각 모델에 독립 적용):

```
Optuna TPE Sampler
       |
       v
Trial N -> 파라미터 조합 -> MD5 해시 생성
       |
       +--[캐시 HIT]--> 저장된 결과 반환 (~80% 속도 향상)
       |
       +--[캐시 MISS]--> 학습 + 평가 -> JSON 캐시 저장
       |
       v
MedianPruner -> 유망하지 않은 Trial 조기 종료
       |
       v
Best Parameters 선택
```

---

## 핵심 전략

### 1. 피처 엔지니어링

**시간 기반 피처**

| 피처                 | 설명                      |
| -------------------- | ------------------------- |
| hour, weekday, month | 기본 시간 분해            |
| weekend              | 주간/주말 구분            |
| SIN/COS encoding     | 주기성 변수의 연속적 변환 |
| seasonality          | 여름철(6~9월) 구분        |
| daylight             | 일광 기반 특성            |

**기상 관련 피처**

| 피처                       | 설명                               |
| -------------------------- | ---------------------------------- |
| 일사량/일조량              | 위도/경도 기반 태양 관련 특성 계산 |
| CDH (Cooling Degree Hours) | 냉방 수요 지표                     |
| HDH (Heating Degree Hours) | 난방 수요 지표                     |
| 불쾌지수                   | 온도/습도 기반 체감 지표           |
| 건물 그룹                  | 100개 건물을 30개 그룹으로 군집화  |

### 2. 데이터 증강 및 이상치 처리

**데이터 증강**

- 시간 간격 확장: 15분 단위에서 증강된 간격으로 변환
- Linear Interpolation: 선형 보간을 통한 데이터 밀도 증가
- upsample_20min_linear: 구간 업샘플링

**이상치 처리**

- IQR 기반 사분위수 탐지 (단일 시점 + 구간 이상치)
- Z-Score 기반 시각적 분석 후 제거 -- SMAPE 약 12% 향상
- 일사량 결측: 0으로 대체
- 일조량 결측: 건물별 선형 보간 또는 ffill/bfill

### 3. 계층적 통계 피처

5단계 계층 구조로 세분화 수준에 따라 통계량을 생성합니다:

| 레벨    | 기준                       | 세분화      |
| ------- | -------------------------- | ----------- |
| Level 1 | 건물번호 + 시간 + 요일     | 가장 세분화 |
| Level 2 | 건물번호 + 시간 + 휴일여부 | 휴일 기반   |
| Level 3 | 건물번호 + 시간            | 시간별      |
| Level 4 | 건물번호                   | 건물 전체   |
| Level 5 | 전체 데이터                | 글로벌      |

### 4. 앙상블 전략

- **XGBoost + LightGBM** 이중 모델 조합
- **TimeSeriesSplit K-Fold**: 시계열 특성을 고려한 교차 검증
- **OOF (Out-of-Fold) 예측**: 전체 파이프라인에 대한 일반화 성능 검증
- **log1p 타겟 변환**: 분포 정규화를 통한 예측 안정성 확보
- **가중 평균 앙상블**: 최종 예측값 도출

### 5. Optuna + MD5 캐싱

| 구성 요소     | 역할                                                |
| ------------- | --------------------------------------------------- |
| TPE Sampler   | Tree-structured Parzen Estimator 기반 베이지안 탐색 |
| MedianPruner  | 유망하지 않은 Trial 조기 종료                       |
| MD5 해시 캐시 | 파라미터 조합의 해시값으로 캐시 키 생성 + JSON 저장 |
| 속도 향상     | 캐시 재사용으로 실험 속도 약 80% 단축               |

### 6. 모델링 접근법

세 가지 독립적인 모델링 전략을 실험하였습니다:

| 접근법            | 설명                              | 분류 기준 |
| ----------------- | --------------------------------- | --------- |
| **세그먼트 기반** | ESS 보유, PVC 보유, 일반 건물     | 설비 특성 |
| **유형 기반**     | 공장, 학교, 데이터센터, 아파트 등 | 건물 용도 |
| **건물별 개별**   | 100개 건물별 독립 모델            | 건물 ID   |

---

## 성능 추이

| Version             | SMAPE     | 비고                |
| ------------------- | --------- | ------------------- |
| Initial             | 13.8%     | 세그먼트별 모델 OOF |
| v1                  | 14.9%     | 유형별 모델 OOF     |
| v2                  | 7.7%      | 건물별 모델 OOF     |
| v3                  | 10.97%    | 최종 앙상블 (LOCAL) |
| **Final (Public)**  | **5.15%** | **Public 8위**      |
| **Final (Private)** | **5.52%** | **Private 6위**     |

> 초기 13.8%에서 최종 5.15%까지 약 62% 개선을 달성하였습니다.

**주요 성능 향상 요인**

- 건물별 독립 모델링으로의 전환 (13.8% -> 7.7%)
- Z-Score 기반 이상치 제거 (약 12% SMAPE 감소)
- 5단계 계층적 통계 피처 도입
- Optuna 베이지안 최적화 + MD5 캐싱

---

## 프로젝트 구조

```
Energy-miniProject/
|
|-- best_model/                    # [FINAL] 최종 제출 모델 (SMAPE 5.15)
|   |-- 13_01_preprocessing_peak_model.py   # Peak/Off-peak 구분 전처리
|   |-- 13_02_model.py                      # XGBoost+LightGBM 앙상블
|
|-- 01/                            # 초기 버전: 기본 전처리 + 모델
|-- 02/                            # 두 번째 접근: 앙상블 도입
|-- 03/                            # 세 번째 접근: Feature Importance 분석
|
|-- 01_code/                       # 14개 버전 실험 (50+ 파일)
|-- 01_best_code/                  # 01 시리즈 최적 코드
|-- 02_best_code/                  # 02 시리즈 최적 코드
|
|-- 5.15397/                       # 점수별 실험 디렉토리 (14개)
|-- 5.18295/                       #   각 SMAPE 점수에 해당하는
|-- 5.21184/                       #   모델 및 파라미터 스냅샷
|-- ...                            #
|
|-- informations/                  # 참고 자료
|-- _final/                        # 최종 제출 코드
|-- best/                          # 최고 성능 모델 후보
|
|-- 보고서_2025_전력사용량_예측_AI_경진대회.pdf  # 분석 보고서
|-- presentation.pdf                             # 발표 자료
|-- 6등_2025_전력사용량_예측_AI_경진대회.png      # 최종 순위 캡처
|
|-- 총 200+ Python 파일
```

---

## 실행 방법

### 최종 모델 실행

```bash
# Step 1: 전처리 (Peak/Off-peak 구분 포함)
python best_model/13_01_preprocessing_peak_model.py

# Step 2: 모델 학습 및 예측
python best_model/13_02_model.py
```

### 이전 버전 실행

```bash
# Version 1 - 기본 접근
python 01/01_preprocessing.py
python 01/01_model.py

# Version 2 - 앙상블 모델
python 02/02_model_ver10_ensemble_IDTypeTotal_solar.py

# Version 3 - Feature Importance 분석
python 03/preprocessing_ver5.py
```

---

## 재현성

본 프로젝트는 완전한 재현성을 보장하기 위해 다음 사항을 적용하였습니다:

| 항목              | 적용 방법                                                          |
| ----------------- | ------------------------------------------------------------------ |
| **시드 고정**     | `PYTHONHASHSEED`, `random.seed()`, `numpy.random.seed()` 일괄 설정 |
| **버전 태깅**     | 각 실험에 고유 버전 코드 부여 (예: `version = "LETS_GO"`)          |
| **모델 시드**     | XGBoost/LightGBM/CatBoost의 모든 `seed` 파라미터 명시적 설정       |
| **독립 경로**     | 실험별 독립 저장 경로로 결과 덮어쓰기 방지                         |
| **파라미터 캐시** | MD5 해시 기반 JSON 캐시로 동일 파라미터 재현 보장                  |

---

## 개발자

**김재현 (Jaehyeon Kim)**

[![GitHub](https://img.shields.io/badge/GitHub-Afraid--Not-181717?style=flat-square&logo=github)](https://github.com/Afraid-Not)

---

---

# 2025 Power Consumption Prediction AI Competition

<p align="center">
  <img src="6등_2025_전력사용량_예측_AI_경진대회.png" alt="Competition Result - 6th Place" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/SMAPE-5.15%25-brightgreen?style=for-the-badge" alt="SMAPE Score"/>
  <img src="https://img.shields.io/badge/Public-8th-blue?style=for-the-badge" alt="Public Rank"/>
  <img src="https://img.shields.io/badge/Private-6th-blueviolet?style=for-the-badge" alt="Private Rank"/>
  <img src="https://img.shields.io/badge/Period-2025.07.14_~_2025.08.25-grey?style=for-the-badge" alt="Period"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LightGBM-Gradient_Boosting-9ACD32?style=flat-square" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient_Boosting-EC4E20?style=flat-square" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/CatBoost-Gradient_Boosting-FFCC00?style=flat-square" alt="CatBoost"/>
  <img src="https://img.shields.io/badge/Optuna-Bayesian_Optimization-2196F3?style=flat-square" alt="Optuna"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
</p>

<p align="center">
  <b>Team: 한국식_러다이트_운동원</b><br/>
  AI solution for predicting hourly power consumption (kWh) across 100 buildings
</p>

<p align="center">
  <a href="./보고서_2025_전력사용량_예측_AI_경진대회.pdf">Report PDF</a> | <a href="./presentation.pdf">Presentation PDF</a>
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Ensemble Pipeline Architecture](#ensemble-pipeline-architecture)
- [Key Strategies](#key-strategies)
  - [1. Feature Engineering](#1-feature-engineering)
  - [2. Data Augmentation and Outlier Handling](#2-data-augmentation-and-outlier-handling)
  - [3. Hierarchical Statistical Features](#3-hierarchical-statistical-features)
  - [4. Ensemble Strategy](#4-ensemble-strategy)
  - [5. Optuna + MD5 Caching](#5-optuna--md5-caching)
  - [6. Modeling Approaches](#6-modeling-approaches)
- [Performance Progression](#performance-progression)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Reproducibility](#reproducibility)
- [Developer](#developer)

---

## Project Overview

| Item            | Details                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| **Competition** | 2025 Power Consumption Prediction AI Competition                                                           |
| **Objective**   | Predict hourly power consumption (kWh) for 100 buildings                                                   |
| **Metric**      | SMAPE (Symmetric Mean Absolute Percentage Error)                                                           |
| **Final Score** | SMAPE 5.15% (Public 8th / Private 6th)                                                                     |
| **Period**      | 2025.07.14 ~ 2025.08.25                                                                                    |
| **Team**        | 한국식*러다이트*운동원                                                                                     |
| **Input Data**  | Building metadata, weather data (temperature, humidity, precipitation, solar radiation, sunshine duration) |

---

## Tech Stack

| Category                        | Technology                                    |
| ------------------------------- | --------------------------------------------- |
| **Language**                    | Python 3.x                                    |
| **Boosting Models**             | LightGBM, XGBoost, CatBoost                   |
| **Hyperparameter Optimization** | Optuna (TPE Sampler + MedianPruner)           |
| **Data Processing**             | Pandas, NumPy                                 |
| **ML Utilities**                | scikit-learn (TimeSeriesSplit, preprocessing) |

---

## Ensemble Pipeline Architecture

```
+-------------------------------+
|        Raw Data Input         |
|  (Building info + Weather)    |
+---------------+---------------+
                |
                v
+-------------------------------+
|       Preprocessing           |
|  - Peak / Off-peak split      |
|  - Outlier removal (IQR, Z)  |
|  - Time interval expansion    |
+---------------+---------------+
                |
                v
+-------------------------------+
|     Feature Engineering       |
|  - Time features (SIN/COS)   |
|  - Weather (CDH/HDH)         |
|  - 5-level hierarchical stats |
|  - Building clustering        |
|    (100 -> 30 groups)         |
+---------------+---------------+
                |
        +-------+-------+
        |               |
        v               v
+---------------+ +---------------+
|   XGBoost     | |   LightGBM    |
| TimeSeriesSplit| | TimeSeriesSplit|
|   K-Fold      | |   K-Fold      |
+-------+-------+ +-------+-------+
        |               |
        v               v
+---------------+ +---------------+
| OOF Predict 1 | | OOF Predict 2 |
|  (log1p inv)  | |  (log1p inv)  |
+-------+-------+ +-------+-------+
        |               |
        +-------+-------+
                |
                v
+-------------------------------+
|    Weighted Average Ensemble  |
+-------------------------------+
                |
                v
+-------------------------------+
|     Final Prediction          |
|     SMAPE: 5.15%              |
+-------------------------------+
```

**Optuna Optimization Loop** (applied independently to each model):

```
Optuna TPE Sampler
       |
       v
Trial N -> Parameter combination -> MD5 hash generation
       |
       +--[Cache HIT]--> Return stored result (~80% speedup)
       |
       +--[Cache MISS]--> Train + Evaluate -> Save to JSON cache
       |
       v
MedianPruner -> Early termination of unpromising trials
       |
       v
Best Parameters selected
```

---

## Key Strategies

### 1. Feature Engineering

**Time-based Features**

| Feature              | Description                                     |
| -------------------- | ----------------------------------------------- |
| hour, weekday, month | Basic temporal decomposition                    |
| weekend              | Weekday/weekend distinction                     |
| SIN/COS encoding     | Continuous transformation of cyclical variables |
| seasonality          | Summer period (June~September) flag             |
| daylight             | Daylight-based characteristics                  |

**Weather-related Features**

| Feature                    | Description                                  |
| -------------------------- | -------------------------------------------- |
| Solar radiation/sunshine   | Computed from latitude/longitude             |
| CDH (Cooling Degree Hours) | Cooling demand indicator                     |
| HDH (Heating Degree Hours) | Heating demand indicator                     |
| Discomfort index           | Temperature/humidity-based perceived comfort |
| Building group             | 100 buildings clustered into 30 groups       |

### 2. Data Augmentation and Outlier Handling

**Data Augmentation**

- Time interval expansion: 15-minute intervals augmented to higher density
- Linear interpolation for data densification
- upsample_20min_linear for interval-based upsampling

**Outlier Handling**

- IQR-based detection (single-point + interval outliers)
- Z-Score visual analysis and removal -- approximately 12% SMAPE improvement
- Solar radiation missing values: filled with 0
- Sunshine missing values: per-building linear interpolation or ffill/bfill

### 3. Hierarchical Statistical Features

Statistics generated at 5 hierarchical levels, from most granular to most general:

| Level   | Grouping Key              | Granularity   |
| ------- | ------------------------- | ------------- |
| Level 1 | Building + Hour + Weekday | Most granular |
| Level 2 | Building + Hour + Holiday | Holiday-aware |
| Level 3 | Building + Hour           | Hourly        |
| Level 4 | Building                  | Building-wide |
| Level 5 | All data                  | Global        |

### 4. Ensemble Strategy

- **XGBoost + LightGBM** dual-model combination
- **TimeSeriesSplit K-Fold**: cross-validation respecting temporal order
- **OOF (Out-of-Fold) prediction**: generalization validation across the full pipeline
- **log1p target transformation**: distribution normalization for prediction stability
- **Weighted average ensemble**: final prediction aggregation

### 5. Optuna + MD5 Caching

| Component         | Role                                                             |
| ----------------- | ---------------------------------------------------------------- |
| TPE Sampler       | Bayesian search via Tree-structured Parzen Estimator             |
| MedianPruner      | Early termination of unpromising trials                          |
| MD5 hash cache    | Hash-based cache key from parameter combinations, stored as JSON |
| Speed improvement | Approximately 80% reduction in experiment time via cache reuse   |

### 6. Modeling Approaches

Three independent modeling strategies were explored:

| Approach              | Description                                   | Grouping Criterion        |
| --------------------- | --------------------------------------------- | ------------------------- |
| **Segment-based**     | ESS-equipped, PVC-equipped, general buildings | Equipment characteristics |
| **Type-based**        | Factory, school, datacenter, apartment, etc.  | Building purpose          |
| **Building-specific** | 100 independent models per building           | Building ID               |

---

## Performance Progression

| Version             | SMAPE     | Notes                  |
| ------------------- | --------- | ---------------------- |
| Initial             | 13.8%     | Segment-based OOF      |
| v1                  | 14.9%     | Type-based OOF         |
| v2                  | 7.7%      | Building-specific OOF  |
| v3                  | 10.97%    | Final ensemble (LOCAL) |
| **Final (Public)**  | **5.15%** | **Public 8th place**   |
| **Final (Private)** | **5.52%** | **Private 6th place**  |

> Achieved approximately 62% improvement from initial 13.8% to final 5.15%.

**Key Performance Drivers**

- Transition to building-specific modeling (13.8% -> 7.7%)
- Z-Score-based outlier removal (approximately 12% SMAPE reduction)
- Introduction of 5-level hierarchical statistical features
- Optuna Bayesian optimization with MD5 caching

---

## Project Structure

```
Energy-miniProject/
|
|-- best_model/                    # [FINAL] Best submission model (SMAPE 5.15)
|   |-- 13_01_preprocessing_peak_model.py   # Peak/Off-peak preprocessing
|   |-- 13_02_model.py                      # XGBoost+LightGBM ensemble
|
|-- 01/                            # Version 1: Basic preprocessing + model
|-- 02/                            # Version 2: Ensemble introduction
|-- 03/                            # Version 3: Feature Importance analysis
|
|-- 01_code/                       # 14 versioned experiments (50+ files)
|-- 01_best_code/                  # Best code from 01 series
|-- 02_best_code/                  # Best code from 02 series
|
|-- 5.15397/                       # Score-named experiment directories (14 total)
|-- 5.18295/                       #   Each contains model and parameter
|-- 5.21184/                       #   snapshots at that SMAPE score
|-- ...                            #
|
|-- informations/                  # Reference materials
|-- _final/                        # Final submission code
|-- best/                          # Best model candidates
|
|-- Total: 200+ Python files across all versions
```

---

## How to Run

### Final Model

```bash
# Step 1: Preprocessing (includes Peak/Off-peak distinction)
python best_model/13_01_preprocessing_peak_model.py

# Step 2: Model training and prediction
python best_model/13_02_model.py
```

### Previous Versions

```bash
# Version 1 - Basic approach
python 01/01_preprocessing.py
python 01/01_model.py

# Version 2 - Ensemble model
python 02/02_model_ver10_ensemble_IDTypeTotal_solar.py

# Version 3 - Feature Importance analysis
python 03/preprocessing_ver5.py
```

---

## Reproducibility

This project ensures full reproducibility through the following measures:

| Item                  | Implementation                                                                |
| --------------------- | ----------------------------------------------------------------------------- |
| **Seed fixing**       | `PYTHONHASHSEED`, `random.seed()`, `numpy.random.seed()` all set consistently |
| **Version tagging**   | Each experiment assigned a unique version code (e.g., `version = "LETS_GO"`)  |
| **Model seeds**       | All `seed` parameters explicitly set for XGBoost/LightGBM/CatBoost            |
| **Independent paths** | Separate storage paths per experiment to prevent result overwriting           |
| **Parameter cache**   | MD5 hash-based JSON cache ensures identical parameter reproduction            |

---

## Developer

**Jaehyeon Kim**

[![GitHub](https://img.shields.io/badge/GitHub-Afraid--Not-181717?style=flat-square&logo=github)](https://github.com/Afraid-Not)
