import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def calculate_feature_importance(X, y, method='all', sample_ratio=0.3, n_estimators=100, random_state=42):
    """
    피쳐 임포턴스를 여러 방법으로 계산
    
    Parameters:
    -----------
    X : DataFrame
        피쳐 데이터
    y : Series
        타겟 변수
    method : str, default='all'
        'tree', 'permutation', 'correlation', 'all'
    sample_ratio : float, default=0.3
        샘플링 비율 (1.0이면 전체 데이터 사용)
    n_estimators : int, default=100
        랜덤포레스트 트리 개수
    random_state : int, default=42
        재현성을 위한 시드
    
    Returns:
    --------
    dict : 각 방법별 피쳐 임포턴스 결과
    """
    
    print(f"피쳐 임포턴스 계산 시작...")
    print(f"데이터 크기: {X.shape}")
    
    # 샘플링
    if sample_ratio < 1.0:
        n_samples = int(len(X) * sample_ratio)
        sample_idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        y_sample = y.iloc[sample_idx].reset_index(drop=True)
        print(f"샘플링: {len(X):,} → {len(X_sample):,} 행 ({sample_ratio*100:.0f}%)")
    else:
        X_sample = X.copy()
        y_sample = y.copy()
    
    results = {}
    
    # 1. Tree-based Feature Importance
    if method in ['tree', 'all']:
        print("Tree-based importance 계산 중...")
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X_sample, y_sample)
        
        tree_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': rf.feature_importances_,
            'rank': range(1, len(X_sample.columns) + 1)
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # 순위 다시 매기기
        tree_importance['rank'] = range(1, len(tree_importance) + 1)
        results['tree_based'] = tree_importance
        print(f"  완료! Top 5: {tree_importance.head(5)['feature'].tolist()}")
    
    # 2. Permutation Importance
    if method in ['permutation', 'all']:
        print("Permutation importance 계산 중...")
        if 'rf' not in locals():
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            rf.fit(X_sample, y_sample)
        
        perm_imp = permutation_importance(rf, X_sample, y_sample, n_repeats=5, random_state=random_state, n_jobs=-1)
        
        permutation_importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': perm_imp.importances_mean,
            'std': perm_imp.importances_std,
            'rank': range(1, len(X_sample.columns) + 1)
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # 순위 다시 매기기
        permutation_importance_df['rank'] = range(1, len(permutation_importance_df) + 1)
        results['permutation'] = permutation_importance_df
        print(f"  완료! Top 5: {permutation_importance_df.head(5)['feature'].tolist()}")
    
    # 3. Correlation-based Importance
    if method in ['correlation', 'all']:
        print("Correlation 계산 중...")
        correlations = []
        for col in X_sample.columns:
            corr = abs(np.corrcoef(X_sample[col], y_sample)[0, 1])
            correlations.append(corr if not np.isnan(corr) else 0)
        
        correlation_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'correlation': correlations,
            'rank': range(1, len(X_sample.columns) + 1)
        }).sort_values('correlation', ascending=False).reset_index(drop=True)
        
        # 순위 다시 매기기
        correlation_importance['rank'] = range(1, len(correlation_importance) + 1)
        results['correlation'] = correlation_importance
        print(f"  완료! Top 5: {correlation_importance.head(5)['feature'].tolist()}")
    
    print("피쳐 임포턴스 계산 완료!")
    return results

def plot_feature_importance(importance_results, top_n=15, figsize=(15, 10)):
    """피쳐 임포턴스 시각화"""
    
    methods = list(importance_results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        data = importance_results[method].head(top_n)
        
        if method == 'tree_based':
            y_values = data['importance']
            title = 'Tree-based Importance'
        elif method == 'permutation':
            y_values = data['importance']
            title = 'Permutation Importance'
        elif method == 'correlation':
            y_values = data['correlation']
            title = 'Correlation with Target'
        
        # 가로 막대 그래프
        y_pos = np.arange(len(data))
        axes[i].barh(y_pos, y_values, alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(data['feature'])
        axes[i].set_title(title)
        axes[i].invert_yaxis()  # 순서 뒤집기
        
        # 값 표시
        for j, v in enumerate(y_values):
            axes[i].text(v, j, f'{v:.3f}', va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def get_consensus_features(importance_results, top_n=10):
    """여러 방법의 합의를 통한 중요 피쳐 선별"""
    
    consensus_scores = {}
    
    # 각 방법별 점수 정규화 후 합산
    for method, data in importance_results.items():
        # 순위를 점수로 변환 (1등 = 최고점)
        max_rank = len(data)
        for _, row in data.iterrows():
            feature = row['feature']
            rank_score = (max_rank - row['rank'] + 1) / max_rank  # 1등=1.0, 꼴등=작은값
            
            if feature not in consensus_scores:
                consensus_scores[feature] = 0
            consensus_scores[feature] += rank_score
    
    # 합의 점수로 정렬
    consensus_df = pd.DataFrame([
        {'feature': feature, 'consensus_score': score, 'rank': i+1}
        for i, (feature, score) in enumerate(sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True))
    ])
    
    print(f"\n=== 합의 기반 Top {top_n} 피쳐 ===")
    top_consensus = consensus_df.head(top_n)
    for _, row in top_consensus.iterrows():
        print(f"{row['rank']:2d}. {row['feature']} (점수: {row['consensus_score']:.3f})")
    
    return consensus_df

def save_importance_results(importance_results, consensus_df, filename="feature_importance_results.json"):
    """결과를 JSON으로 저장"""
    import json
    from datetime import datetime
    
    # JSON 직렬화 가능한 형태로 변환
    json_data = {
        'analysis_info': {
            'created_at': datetime.now().isoformat(),
            'methods_used': list(importance_results.keys())
        }
    }
    
    # 각 방법별 결과 추가
    for method, df in importance_results.items():
        json_data[method] = df.head(20).to_dict('records')  # 상위 20개만 저장
    
    # 합의 결과 추가
    json_data['consensus'] = consensus_df.head(20).to_dict('records')
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 '{filename}' 파일로 저장되었습니다.")

# 사용 예시
def analyze_importance(X, y, target_name="target", save_path=None):
    """전체 분석 실행"""
    print(f"=== {target_name} 피쳐 임포턴스 분석 ===")
    
    # 1. 피쳐 임포턴스 계산
    results = calculate_feature_importance(X, y, method='all', sample_ratio=0.3)
    
    # 2. 결과 출력
    for method, data in results.items():
        print(f"\n{method.upper()} Top 10:")
        print(data.head(10)[['rank', 'feature', data.columns[1]]].to_string(index=False))
    
    # 3. 시각화
    plot_feature_importance(results, top_n=15)
    
    # 4. 합의 기반 피쳐 선별
    consensus = get_consensus_features(results, top_n=15)
    
    # 5. 결과 저장
    if save_path:
        save_importance_results(results, consensus, save_path)
    
    return results, consensus

# 실행 예시:
# 


log_path = './Energy/02/log/'
trainer = './Energy/02/trainer/'
SEED = 118
train_all = pd.read_csv(trainer + f'06_train_{SEED}_ver3.csv')
test_all = pd.read_csv(trainer + f'06_test_{SEED}_ver3.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_all['건물유형'] = le.fit_transform(train_all['건물유형'])
test_all['건물유형'] = le.transform(test_all['건물유형'])

# 일조시간 예측용: 일조 관련 피쳐 모두 제거
drop1_col = ['일조(hr)', '일사(MJ/m2)', '일시', 'date']
drop1_col.extend([col for col in train_all.columns if 'sunshine_' in col])
drop1_col.extend([col for col in train_all.columns if 'solar_' in col])
drop1_col.extend([col for col in train_all.columns if 'sun_' in col])  # sun_bin_* 제거
drop1_col.extend([col for col in train_all.columns if '일조' in col])  # 일조 관련 모든 피쳐 제거
drop1_col = list(set(drop1_col))
X1_train = train_all.drop(drop1_col, axis=1)
y_sunshine = train_all['일조(hr)']

# 일사량 예측용: 일사 관련 피쳐 제거 (일조 관련은 유지)
drop2_col = ['일사(MJ/m2)', '일시', 'date']
drop2_col.extend([col for col in train_all.columns if 'solar_' in col])
drop2_col.extend([col for col in train_all.columns if '일사' in col])  # 일사 관련 피쳐 제거 추가
drop2_col = list(set(drop2_col))
X2_train = train_all.drop(drop2_col, axis=1)
y_solar = train_all['일사(MJ/m2)']


# # 일조시간 분석
sunshine_imp, sunshine_consensus = analyze_importance(
    X1_train, y_sunshine,'일조(hr)', 
    save_path="./Energy/02/log/sunshine_importance.json"
)
#
# # 일사량 분석  
solar_imp, solar_consensus = analyze_importance(
    X2_train, y_solar, '일사(MJ/m2)',
    save_path="./Energy/02/log/solar_importance.json"
)