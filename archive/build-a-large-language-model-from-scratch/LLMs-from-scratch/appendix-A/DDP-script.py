# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW: 새로 추가한 import:
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# NEW: (GPU당 1개 프로세스) 분산 프로세스 그룹을 초기화하는 함수입니다
# 프로세스 간 통신을 가능하게 합니다
def ddp_setup(rank, world_size):
    """
    인자:
        rank: 고유한 프로세스 ID
        world_size: 그룹에 속한 전체 프로세스 수
    """
    # rank:0 프로세스를 실행하는 머신의 순위를 지정합니다
    # 여기서는 모든 GPU가 동일한 머신에 있다고 가정합니다
    os.environ["MASTER_ADDR"] = "localhost"
    # 머신에서 사용 가능한 임의의 포트를 지정합니다
    os.environ["MASTER_PORT"] = "12345"

    # 프로세스 그룹을 초기화합니다
    if platform.system() == "Windows":
        # Windows용 PyTorch는 libuv를 지원하지 않으므로 비활성화합니다
        os.environ["USE_LIBUV"] = "0"
        # Windows 사용자는 백엔드로 "nccl" 대신 "gloo"를 사용해야 할 수도 있습니다
        # gloo: Facebook의 집단 통신 라이브러리
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA 집단 통신 라이브러리
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 첫 번째 은닉층
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 두 번째 은닉층
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 출력층
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # 최대 8개의 GPU에서 실행하려면 아래 주석을 해제해 데이터셋 크기를 늘리세요:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: 아래 DistributedSampler 때문에 False로 설정합니다
        pin_memory=True,
        drop_last=True,
        # NEW: GPU 간 샘플이 겹치지 않도록 배치를 분할합니다:
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# NEW: 래퍼 함수
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: 프로세스 그룹을 초기화합니다

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # NEW: DDP로 모델을 감쌉니다
    # 이제 기본 모델은 model.module로 접근할 수 있습니다

    for epoch in range(num_epochs):
        # NEW: 각 에폭마다 셔플 순서가 달라지도록 샘플러를 설정합니다
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # NEW: rank와 동일한 디바이스를 사용
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # 손실 함수

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 기록
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW (책에는 없음):
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  # NEW: 분산 모드를 깔끔하게 종료


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # 데이터셋이 작으므로 GPU가 2개를 초과하면 제대로 동작하지 않을 수 있습니다
    # GPU가 2개보다 많다면 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`로 실행하세요
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)

    # NEW: 새 프로세스를 생성합니다
    # spawn은 rank 값을 자동으로 전달합니다
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size는 GPU당 하나의 프로세스를 생성합니다
