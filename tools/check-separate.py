import os

# A 경로와 B 경로 설정
target_dir = "/path/to/A"
target_temp_dir = "/path/to/B"

# A 경로와 B 경로 내의 파일 목록 가져오기
a_files = os.listdir(target_dir)
b_files = os.listdir(target_temp_dir)

# B 경로에 있는 파일명을 세트로 저장
b_file_set = set(b_files)

# A 경로에 있는 파일 중 B 경로에 없는 파일명 추출
missing_files = [file for file in a_files if file not in b_file_set]

# 결과 출력
print("B 경로에 없는 파일 목록:")
for file in missing_files:
    print(file)
