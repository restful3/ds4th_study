from pathlib import Path
files = [
    Path('LLMs-from-scratch/ch05/gpt_train.py'),
    Path('LLMs-from-scratch/ch05/gpt_download.py'),
    Path('LLMs-from-scratch/ch05/gpt_generate.py'),
    Path('LLMs-from-scratch/ch05/previous_chapters.py'),
]

comment_map = {
    '# This file collects all the relevant code that we covered thus far': '# 지금까지 다룬 주요 코드를 모아 둔 파일입니다',
    '# throughout Chapters 2-4.': '# 2~4장 전반에 걸쳐 소개한 코드입니다.',
    '# This file can be run as a standalone script.': '# 이 파일은 단독 스크립트로 실행할 수 있습니다.',
    '# Chapter 2': '# 2장',
    '# Chapter 3': '# 3장',
    '# Chapter 4': '# 4장',
    '# Chapter 5': '# 5장',
    '# Import from local files': '# 로컬 파일에서 import',
    '# Initialize the tokenizer': '# 토크나이저 초기화',
    '# Create dataset': '# 데이터셋 생성',
    '# Create dataloader': '# 데이터 로더 생성',
    '# Tokenize the entire text': '# 전체 텍스트를 토큰화',
    '# Use a sliding window to chunk the book into overlapping sequences of max_length': '# 슬라이딩 윈도우로 책을 max_length 길이의 겹치는 시퀀스로 분할',
    '# We implicitly split the matrix by adding a `num_heads` dimension': '# `num_heads` 차원을 추가해 행렬을 암묵적으로 분할',
    '# Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)': '# 마지막 차원을 펼쳐 (b, num_tokens, d_out)을 (b, num_tokens, num_heads, head_dim)으로 변환',
    '# Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)': '# 전치: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)',
    '# Compute scaled dot-product attention (aka self-attention) with a causal mask': '# 인과 마스크를 적용한 스케일드 닷프로덕트 어텐션(셀프 어텐션) 계산',
    '# Original mask truncated to the number of tokens and converted to boolean': '# 원래 마스크를 토큰 수에 맞게 자르고 불리언으로 변환',
    '# Use the mask to fill attention scores': '# 마스크를 이용해 어텐션 점수를 채움',
    '# Shape: (b, num_tokens, num_heads, head_dim)': '# 형태: (b, num_tokens, num_heads, head_dim)',
    '# Combine heads, where self.d_out = self.num_heads * self.head_dim': '# self.d_out = self.num_heads * self.head_dim을 활용해 헤드를 결합',
    '# Shortcut connection for attention block': '# 어텐션 블록용 잔차 연결',
    '# Shortcut connection for feed-forward block': '# 피드포워드 블록용 잔차 연결',
    '# Crop current context if it exceeds the supported context size': '# 지원되는 컨텍스트 길이를 넘으면 현재 컨텍스트를 잘라냄',
    '# E.g., if LLM supports only 5 tokens, and the context size is 10': '# 예를 들어 컨텍스트 길이가 10이고 LLM이 5개 토큰만 지원한다면',
    '# then only the last 5 tokens are used as context': '# 마지막 5개 토큰만 컨텍스트로 사용',
    '# Get the predictions': '# 예측값 계산',
    '# Focus only on the last time step': '# 마지막 시점만 사용',
    '# (batch, n_token, vocab_size) becomes (batch, vocab_size)': '# (batch, n_token, vocab_size)이 (batch, vocab_size)로 변환',
    '# Get the idx of the vocab entry with the highest logits value': '# 로짓이 가장 큰 어휘 인덱스를 선택',
    '# Append sampled index to the running sequence': '# 샘플링한 인덱스를 시퀀스에 이어붙임',
    '# idx is (B, T) array of indices in the current context': '# idx는 현재 컨텍스트 인덱스를 담은 (B, T) 배열',
    '# Validate model size': '# 모델 크기 검증',
    '# Define paths': '# 경로 정의',
    '# Download files': '# 파일 다운로드',
    '# Load settings and params': '# 설정과 파라미터 로드',
    '# Get the total file size from headers, defaulting to 0 if not present': '# 헤더에서 파일 크기를 확인하고 없으면 0으로 설정',
    '# Check if file exists and has the same size': '# 같은 크기의 파일이 이미 존재하는지 확인',
    '# Initialize the progress bar with total file size': '# 총 파일 크기로 진행률 바 초기화',
    '# If we reach here, both attempts have failed': '# 여기까지 왔다면 기본 및 백업 다운로드가 모두 실패한 것입니다',
    '# Alternative way using `requests`': '# `requests`를 사용하는 대체 방법',
    '# Send a GET request to download the file in streaming mode': '# 스트리밍 모드로 파일을 받는 GET 요청 전송',
    '# Define the block size for reading the file': '# 파일을 읽어올 블록 크기 정의',
    '# Open the destination file in binary write mode': '# 대상 파일을 바이너리 쓰기 모드로 열기',
    '# Iterate over the file data in chunks': '# 파일 데이터를 청크 단위로 반복 읽기',
    '# Update progress bar': '# 진행률 갱신',
    '# Write the chunk to the file': '# 청크를 파일에 기록',
    '# Initialize parameters dictionary with empty blocks for each layer': '# 각 레이어에 해당하는 빈 딕셔너리 블록을 초기화',
    '# Iterate over each variable in the checkpoint': '# 체크포인트의 변수들을 순회',
    '# Load the variable and remove singleton dimensions': '# 변수를 로드하고 불필요한 차원 제거',
    '# Process the variable name to extract relevant parts': '# 변수 이름에서 필요한 부분 추출',
    "# Skip the 'model/' prefix": "# 'model/' 접두어 건너뜀",
    '# Identify the target dictionary for the variable': '# 변수를 저장할 대상 딕셔너리 식별',
    '# Recursively access or create nested dictionaries': '# 중첩된 딕셔너리를 재귀적으로 탐색하거나 생성',
    '# Assign the variable array to the last key': '# 마지막 키에 변수 배열 할당',
    '# Initialize lists to track losses and tokens seen': '# 손실과 처리한 토큰 수를 기록할 리스트 초기화',
    '# Main training loop': '# 메인 학습 루프',
    '# Optional evaluation step': '# 선택적 평가 단계',
    '# Print a sample text after each epoch': '# 각 에폭 이후 샘플 텍스트 출력',
    '# Plot training and validation loss against epochs': '# 에폭별 학습/검증 손실 그래프 그리기',
    '# Create a second x-axis for tokens seen': '# 처리한 토큰 수를 위한 두 번째 x축 생성',
    '# Create a second x-axis that shares the same y-axis': '# 동일한 y축을 공유하는 두 번째 x축 생성',
    '# Invisible plot for aligning ticks': '# 눈금 정렬용 보이지 않는 플롯',
    '# Calculate accuracy after each epoch': '# 각 에폭 이후 정확도 계산',
    '# Reduce the number of batches to match the total number of batches in the data loader': '# 데이터 로더의 총 배치 수에 맞추도록 배치 수를 줄임',
    '# if num_batches exceeds the number of batches in the data loader': '# 지정한 num_batches가 데이터 로더 배치 수보다 크면'
}

inline_map = {
    'Reduce the projection dim to match desired output dim': '원하는 출력 차원에 맞도록 투영 차원을 줄임',
    'Linear layer to combine head outputs': '헤드 출력을 결합하는 선형 계층',
    'Shape [batch_size, num_tokens, emb_size]': '형태 [batch_size, num_tokens, emb_dim]',
    'Shape: (b, num_tokens, d_out)': '형태: (b, num_tokens, d_out)',
    '(batch, 1)': '(batch, 1)',
    '(batch, n_tokens+1)': '(batch, n_tokens+1)',
    'Dot product for each head': '각 헤드별 닷프로덕트 계산',
    'optional projection': '필요 시 추가 투영',
    'Add the original input back': '원래 입력을 더해 잔차 연결을 형성',
    'Compact print format': '보기 좋은 형식으로 출력',
    'add batch dimension': '배치 차원 추가',
    'remove batch dimension': '배치 차원 제거',
    'Logits of last output token': '마지막 출력 토큰의 로짓',
    'Create a second x-axis that shares the same y-axis': '동일한 y축을 공유하는 두 번째 x축 생성'
}

for file in files:
    text = file.read_text()
    for old, new in comment_map.items():
        text = text.replace(old, new)
    for old, new in inline_map.items():
        text = text.replace(old, new)
    text = text.replace('return True  # Indicate success without re-downloading', 'return True  # 이미 최신 파일이므로 재다운로드하지 않음')
    text = text.replace('block_size = 1024  # 1 Kilobyte', 'block_size = 1024  # 1킬로바이트')
    text = text.replace('progress_bar_description = url.split("/")[-1]  # Extract filename from URL', 'progress_bar_description = url.split("/")[-1]  # URL에서 파일명 추출')
    file.write_text(text)
