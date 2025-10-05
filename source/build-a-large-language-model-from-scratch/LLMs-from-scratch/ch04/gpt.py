# 지금까지 다룬 주요 코드를 모아 둔 파일입니다
# 2~4장 전반에 걸쳐 소개한 코드입니다.
# 이 파일은 단독 스크립트로 실행할 수 있습니다.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# 2장
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 전체 텍스트를 토큰화
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 슬라이딩 윈도우로 책을 max_length 길이의 겹치는 시퀀스로 분할
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 토크나이저 초기화
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋 생성
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터 로더 생성
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 3장
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 원하는 출력 차원에 맞도록 투영 차원을 줄임

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 헤드 출력을 결합하는 선형 계층
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # 형태: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # `num_heads` 차원을 추가해 행렬을 암묵적으로 분할
        # 마지막 차원을 펼쳐 (b, num_tokens, d_out)을 (b, num_tokens, num_heads, head_dim)으로 변환
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 전치: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 인과 마스크를 적용한 스케일드 닷프로덕트 어텐션(셀프 어텐션) 계산
        attn_scores = queries @ keys.transpose(2, 3)  # 각 헤드별 닷프로덕트 계산

        # 원래 마스크를 토큰 수에 맞게 자르고 불리언으로 변환
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 마스크를 이용해 어텐션 점수를 채움
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 형태: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # self.d_out = self.num_heads * self.head_dim을 활용해 헤드를 결합
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 필요 시 추가 투영

        return context_vec


#####################################
# 4장
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 어텐션 블록용 잔차 연결
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 형태 [batch_size, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원래 입력을 더해 잔차 연결을 형성

        # 피드포워드 블록용 잔차 연결
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원래 입력을 더해 잔차 연결을 형성

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 형태 [batch_size, num_tokens, emb_dim]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx는 현재 컨텍스트 인덱스를 담은 (B, T) 배열
    for _ in range(max_new_tokens):

        # 지원되는 컨텍스트 길이를 넘으면 현재 컨텍스트를 잘라냄
        # 예를 들어 컨텍스트 길이가 10이고 LLM이 5개 토큰만 지원한다면
        # 마지막 5개 토큰만 컨텍스트로 사용
        idx_cond = idx[:, -context_size:]

        # 예측값 계산
        with torch.no_grad():
            logits = model(idx_cond)

        # 마지막 시점만 사용
        # (batch, n_token, vocab_size)이 (batch, vocab_size)로 변환
        logits = logits[:, -1, :]

        # 로짓이 가장 큰 어휘 인덱스를 선택
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 샘플링한 인덱스를 시퀀스에 이어붙임
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 어휘 수
        "context_length": 1024,  # 컨텍스트 길이
        "emb_dim": 768,          # 임베딩 차원
        "n_heads": 12,           # 어텐션 헤드 수
        "n_layers": 12,          # 층 수
        "drop_rate": 0.1,        # 드롭아웃 비율
        "qkv_bias": False        # 쿼리-키-값 편향
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 드롭아웃 비활성화

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
