# Copyright (c) Sebastian Raschka under Apache License 2.0 (LICENSE.txt 참조).
# "처음부터 대규모 언어 모델 구축하기" 소스
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 코드: https://github.com/rasbt/LLMs-from-scratch
#
# 연습문제를 실행하기 위한 코드; 자세한 내용은 exercise-solutions.ipynb를 참조하세요

from functools import partial
from importlib.metadata import version
import json
import math
import os
import re
import time
import urllib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 이 폴더의 로컬 파일에서 가져오기
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 텍스트 사전 토큰화
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetWithMasking(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 새로운 기능: 명령어 길이를 위한 별도 리스트
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            # 새로운 기능: 명령어 길이 수집
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        # 새로운 기능: 명령어 길이와 텍스트를 각각 반환
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetPhi(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 텍스트 사전 토큰화
        self.encoded_texts = []
        for entry in data:

            ###################################################################
            # 새로운 기능: `format_input_phi`를 사용하고 응답 텍스트 템플릿 조정
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            ###################################################################
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # 표준 가중치 초기화와 유사
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Linear 레이어를 LinearWithLoRA로 교체
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 자식 모듈에 동일한 함수를 재귀적으로 적용
            replace_linear_with_lora(module, rank, alpha)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 배치에서 가장 긴 시퀀스 찾기
    batch_max_length = max(len(item)+1 for item in batch)

    # 입력과 타겟을 패딩하고 준비
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # <|endoftext|> 토큰 추가
        new_item += [pad_token_id]
        # 시퀀스를 max_length로 패딩
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 입력에 대해 마지막 토큰 자르기
        targets = torch.tensor(padded[1:])  # 타겟에 대해 오른쪽으로 +1 이동

        # 새로운 기능: 타겟에서 첫 번째 패딩 토큰을 제외한 모든 패딩 토큰을 ignore_index로 교체
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 새로운 기능: 선택적으로 최대 시퀀스 길이로 자르기
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 입력 및 타겟 리스트를 텐서로 변환하고 대상 장치로 전송
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 배치에서 가장 긴 시퀀스 찾기
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)   # 새로운 기능: 이제 배치는 튜플입니다

    # 입력과 타겟을 패딩하고 준비
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:  # 새로운 기능: 이제 배치는 튜플입니다
        new_item = item.copy()
        # <|endoftext|> 토큰 추가
        new_item += [pad_token_id]
        # 시퀀스를 max_length로 패딩
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 입력에 대해 마지막 토큰 자르기
        targets = torch.tensor(padded[1:])  # 타겟에 대해 오른쪽으로 +1 이동

        # 타겟에서 첫 번째 패딩 토큰을 제외한 모든 패딩 토큰을 ignore_index로 교체
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 새로운 기능: 타겟의 모든 입력 및 명령어 토큰 마스킹
        targets[:instruction_length-1] = -100

        # 선택적으로 최대 시퀀스 길이로 자르기
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 입력 및 타겟 리스트를 텐서로 변환하고 대상 장치로 전송
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def format_input_phi(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 에포크에 대한 훈련 및 검증 손실 플롯
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # x축에 정수 레이블만 표시

    # 확인된 토큰에 대한 두 번째 x축 생성
    ax2 = ax1.twiny()  # 동일한 y축을 공유하는 두 번째 x축 생성
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 틱 정렬을 위한 보이지 않는 플롯
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 공간을 만들기 위해 레이아웃 조정
    print(f"플롯이 {plot_name}으로 저장되었습니다")
    plt.savefig(plot_name)
    # plt.show()


def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    #######################################
    # 패키지 버전 출력
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 플로팅 라이브러리
        "tiktoken",    # 토크나이저
        "torch",       # 딥러닝 라이브러리
        "tqdm",        # 진행률 표시줄
        "tensorflow",  # OpenAI의 사전 훈련된 가중치를 위함
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    #######################################
    # 데이터셋 다운로드 및 준비
    #######################################
    file_path = "instruction-data.json"

    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85%는 훈련용
    test_portion = int(len(data) * 0.1)    # 10%는 테스트용

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    if alpaca52k:
        allowed_max_length = 512
    else:
        allowed_max_length = 1024

    if mask_instructions and phi3_prompt:
        raise ValueError("Simultaneous support for instruction masking and the Phi-3 prompt template has not been implemented, yet.")

    if mask_instructions:
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking
    elif phi3_prompt:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi
    else:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset

    num_workers = 0

    if alpaca52k:
        batch_size = 4
    else:
        batch_size = 8

    torch.manual_seed(123)

    train_dataset = CustomDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = CustomDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # 사전 훈련된 모델 로드
    #######################################
    BASE_CONFIG = {
        "vocab_size": 50257,     # 어휘 크기
        "context_length": 1024,  # 컨텍스트 길이
        "drop_rate": 0.0,        # 드롭아웃 비율
        "qkv_bias": True         # 쿼리-키-값 편향
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after: {total_params:,}")
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)

    #######################################
    # 모델 미세 조정
    #######################################
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()

    num_epochs = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

    plot_name = "loss-plot.pdf"
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")

    #######################################
    # 결과 저장
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # 예쁘게 출력하기 위한 "indent"
    print(f"Responses saved as {test_data_path}")

    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="GPT 모델 명령어 미세 조정"
    )
    options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}
    parser.add_argument(
        "--exercise_solution",
        type=str,
        default="baseline",
        help=(
            f"Which experiment to run. Options: {options}."
        )
    )
    args = parser.parse_args()

    if args.exercise_solution == "baseline":
        main()
    elif args.exercise_solution == "mask_instructions":
        main(mask_instructions=True)
    elif args.exercise_solution == "alpaca_52k":
        main(alpaca52k=True)
    elif args.exercise_solution == "phi3_prompt":
        main(phi3_prompt=True)
    elif args.exercise_solution == "lora":
        main(lora=True)
    else:
        raise ValueError(f"{args.exercise_solution} is not a valid --args.exercise_solution option. Options: {options}")
