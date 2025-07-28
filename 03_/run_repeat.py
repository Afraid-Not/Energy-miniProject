# run_repeat.py
import subprocess
import time

N_RUNS = 100   # 반복 횟수 (원하는 만큼 바꾸세요)

for i in range(N_RUNS):
    print(f"\n=== Run {i+1}/{N_RUNS} ===\n")
    
    # 로그를 그대로 터미널에 출력
    subprocess.run(["python", "./Energy/03/codes/END.py"])
    
    # 필요시 실행 사이 대기
    time.sleep(5)