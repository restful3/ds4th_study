from pathlib import Path

files = [
    Path('LLMs-from-scratch/ch04/gpt.py'),
    Path('LLMs-from-scratch/ch04/previous_chapters.py'),
]

comment_map = {
    '# This file collects all the relevant code that we covered thus far': '# 지금까지 다룬 주요 코드를 모아 둔 파일입니다',
    '# throughout Chapters 2-4.': '# 2~4장 전반에 걸쳐 소개한 코드입니다.',
    '# This file can be run as a standalone script.': '# 이 파일은 단독 스크립트로 실행할 수 있습니다.',
    '# Chapter 2': '# 2장',
    '# Chapter 3': '# 3장',
    '# Chapter 4': '# 4장',
    '# Create dataloader': '# 데이터 로더 생성',
    '# Create dataset': '# 데이터셋 생성',
    '# Initialize the tokenizer': '# 토크나이저 초기화',
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
    'Context length': '컨텍스트 길이',
    'Dropout rate': '드롭아웃 비율',
    'Embedding dimension': '임베딩 차원',
    'Number of attention heads': '어텐션 헤드 수',
    'Number of layers': '층 수',
    'Query-Key-Value bias': '쿼리-키-값 편향',
    'disable dropout': '드롭아웃 비활성화'
}

for file in files:
    text = file.read_text()
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in comment_map:
            new_lines.append(line.replace(stripped, comment_map[stripped]))
            continue
        new_line = line
        for en, ko in inline_map.items():
            if en in new_line:
                new_line = new_line.replace(en, ko)
        new_lines.append(new_line)
    file.write_text('\n'.join(new_lines) + '\n')
