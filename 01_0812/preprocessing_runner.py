import subprocess
import re
import time
import os
from datetime import datetime

# -------------------------------
# 설정
# -------------------------------
# 실행 횟수 설정
NUM_RUNS = 10

# 실행 간격 (초)
DELAY_BETWEEN_RUNS = 10

# 실행할 스크립트 경로
SCRIPT_PATH = "./Energy/01/01_preprocessing_must_be_perfect.py"

# -------------------------------
# 실행 함수
# -------------------------------
def run_script_and_extract_metrics(run_number):
    print(f"\n{'='*80}")
    print(f"🚀 실행 #{run_number}/{NUM_RUNS} 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # 스크립트 파일이 존재하는지 확인
    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ 오류: 스크립트 파일을 찾을 수 없습니다: {SCRIPT_PATH}")
        return None

    start_time = time.time()
    
    process = subprocess.Popen(
        ["python", SCRIPT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 추출할 지표들
    current_seed = None
    train_insolation_rmse = None
    test_sunshine_rmse = None
    test_insolation_rmse = None

    for line in process.stdout:
        print(line, end='')  # 로그 실시간 출력

        # SEED 값 추출
        seed_match = re.search(r"\[Current Run SEED\]:\s*(\d+)", line)
        if seed_match:
            current_seed = int(seed_match.group(1))

        # Train 일사량 예측 RMSE 추출
        train_insolation_match = re.search(r"Train 일사량 예측 RMSE:\s*([\d.]+)", line)
        if train_insolation_match:
            train_insolation_rmse = float(train_insolation_match.group(1))

        # Test 일조시간 예측 Validation RMSE 추출
        test_sunshine_match = re.search(r"Test 일조시간 예측 Validation RMSE:\s*([\d.]+)", line)
        if test_sunshine_match:
            test_sunshine_rmse = float(test_sunshine_match.group(1))

        # Test 일사량 예측 Validation RMSE 추출
        test_insolation_match = re.search(r"Test 일사량 예측 Validation RMSE:\s*([\d.]+)", line)
        if test_insolation_match:
            test_insolation_rmse = float(test_insolation_match.group(1))

    return_code = process.wait()
    end_time = time.time()
    execution_time = end_time - start_time

    # 결과 요약
    result = {
        'run_number': run_number,
        'seed': current_seed,
        'execution_time': execution_time,
        'return_code': return_code,
        'success': return_code == 0,
        'train_insolation_rmse': train_insolation_rmse,
        'test_sunshine_rmse': test_sunshine_rmse,
        'test_insolation_rmse': test_insolation_rmse,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print(f"\n{'='*80}")
    if return_code == 0:
        print(f"✅ 실행 #{run_number} 완료!")
        print(f"⏱️  소요 시간: {execution_time:.2f}초")
        print(f"🔢 SEED: {current_seed}")
        if train_insolation_rmse:
            print(f"📊 Train 일사량 RMSE: {train_insolation_rmse:.6f}")
        if test_sunshine_rmse:
            print(f"📊 Test 일조시간 RMSE: {test_sunshine_rmse:.6f}")
        if test_insolation_rmse:
            print(f"📊 Test 일사량 RMSE: {test_insolation_rmse:.6f}")
    else:
        print(f"❌ 실행 #{run_number} 실패! (종료 코드: {return_code})")
        print(f"⏱️  소요 시간: {execution_time:.2f}초")
    
    print(f"{'='*80}")
    
    return result

def print_final_summary(results):
    """최종 결과 요약 출력"""
    print(f"\n{'='*80}")
    print("🎯 최종 실행 요약")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"📈 총 실행 횟수: {len(results)}")
    print(f"✅ 성공: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
    print(f"❌ 실패: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
    
    if successful_results:
        # 실행 시간 통계
        execution_times = [r['execution_time'] for r in successful_results]
        print(f"\n⏱️  실행 시간 통계:")
        print(f"   평균: {sum(execution_times)/len(execution_times):.2f}초")
        print(f"   최단: {min(execution_times):.2f}초")
        print(f"   최장: {max(execution_times):.2f}초")
        
        # RMSE 통계
        metrics = [
            ('train_insolation_rmse', 'Train 일사량 RMSE'),
            ('test_sunshine_rmse', 'Test 일조시간 RMSE'),
            ('test_insolation_rmse', 'Test 일사량 RMSE')
        ]
        
        for metric_key, metric_name in metrics:
            values = [r[metric_key] for r in successful_results if r[metric_key] is not None]
            if values:
                print(f"\n📊 {metric_name} 통계:")
                print(f"   평균: {sum(values)/len(values):.6f}")
                print(f"   최소: {min(values):.6f}")
                print(f"   최대: {max(values):.6f}")
                
                # 최고 성능 실행 찾기
                best_idx = values.index(min(values))
                best_result = [r for r in successful_results if r[metric_key] == min(values)][0]
                print(f"   최고 성능: 실행 #{best_result['run_number']} (SEED: {best_result['seed']})")
        
        print(f"\n🏆 모든 실행 결과:")
        print(f"{'실행':<4} {'SEED':<8} {'시간(초)':<10} {'Train일사':<12} {'Test일조':<12} {'Test일사':<12}")
        print("-" * 70)
        
        for r in successful_results:
            train_val = f"{r['train_insolation_rmse']:.6f}" if r['train_insolation_rmse'] else "N/A"
            sunshine_val = f"{r['test_sunshine_rmse']:.6f}" if r['test_sunshine_rmse'] else "N/A"
            insolation_val = f"{r['test_insolation_rmse']:.6f}" if r['test_insolation_rmse'] else "N/A"
            
            print(f"#{r['run_number']:<3} {r['seed']:<8} {r['execution_time']:<10.2f} {train_val:<12} {sunshine_val:<12} {insolation_val:<12}")
    
    if failed_results:
        print(f"\n❌ 실패한 실행들:")
        for r in failed_results:
            print(f"   실행 #{r['run_number']}: 종료 코드 {r['return_code']}")

# -------------------------------
# Main Loop
# -------------------------------
if __name__ == "__main__":
    print(f"🚀 전처리 스크립트를 {NUM_RUNS}번 실행합니다")
    print(f"📄 스크립트: {SCRIPT_PATH}")
    print(f"⏰ 실행 간격: {DELAY_BETWEEN_RUNS}초")
    print(f"🕐 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    try:
        for i in range(1, NUM_RUNS + 1):
            # 스크립트 실행
            result = run_script_and_extract_metrics(i)
            
            if result:
                results.append(result)
            
            # 진행률 표시
            progress = i / NUM_RUNS * 100
            print(f"\n📊 진행률: {progress:.1f}% ({i}/{NUM_RUNS})")
            
            # 마지막 실행이 아니면 대기
            if i < NUM_RUNS:
                print(f"⏳ {DELAY_BETWEEN_RUNS}초 대기 중...")
                time.sleep(DELAY_BETWEEN_RUNS)
                
    except KeyboardInterrupt:
        print(f"\n⚠️  사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
    
    # 최종 요약 출력
    if results:
        print_final_summary(results)
        
        # 결과를 간단한 텍스트 파일로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"./Energy/runner/preprocessing_results_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(f"전처리 실행 결과 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for r in results:
                if r['success']:
                    f.write(f"실행 #{r['run_number']} (SEED: {r['seed']}) - 성공\n")
                    f.write(f"  실행 시간: {r['execution_time']:.2f}초\n")
                    if r['train_insolation_rmse']:
                        f.write(f"  Train 일사량 RMSE: {r['train_insolation_rmse']:.6f}\n")
                    if r['test_sunshine_rmse']:
                        f.write(f"  Test 일조시간 RMSE: {r['test_sunshine_rmse']:.6f}\n")
                    if r['test_insolation_rmse']:
                        f.write(f"  Test 일사량 RMSE: {r['test_insolation_rmse']:.6f}\n")
                else:
                    f.write(f"실행 #{r['run_number']} - 실패 (코드: {r['return_code']})\n")
                f.write("\n")
        
        print(f"\n💾 결과가 저장되었습니다: preprocessing_results_{timestamp}.txt")
    
    print(f"\n🎉 모든 실행이 완료되었습니다!")
    print(f"🕐 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")