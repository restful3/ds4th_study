# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import urllib.request

# import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 모델 크기 검증
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 경로 정의
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 파일 다운로드
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 설정과 파라미터 로드
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            # 헤더에서 파일 크기를 확인하고 없으면 0으로 설정
            file_size = int(response.headers.get("Content-Length", 0))

            # 같은 크기의 파일이 이미 존재하는지 확인
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # 이미 최신 파일이므로 재다운로드하지 않음

            block_size = 1024  # 1킬로바이트

            # 총 파일 크기로 진행률 바 초기화
            progress_bar_description = os.path.basename(download_url)  # URL에서 파일명 추출  # URL에서 파일명 추출
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # 여기까지 왔다면 기본 및 백업 다운로드가 모두 실패한 것입니다
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# `requests`를 사용하는 대체 방법
"""
def download_file(url, destination):
    # 스트리밍 모드로 파일을 받는 GET 요청 전송
    response = requests.get(url, stream=True)

    # 헤더에서 파일 크기를 확인하고 없으면 0으로 설정
    file_size = int(response.headers.get("content-length", 0))

    # 같은 크기의 파일이 이미 존재하는지 확인
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # 파일을 읽어올 블록 크기 정의
    block_size = 1024  # 1킬로바이트

    # 총 파일 크기로 진행률 바 초기화
    progress_bar_description = url.split("/")[-1]  # URL에서 파일명 추출
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 대상 파일을 바이너리 쓰기 모드로 열기
        with open(destination, "wb") as file:
            # 파일 데이터를 청크 단위로 반복 읽기
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 진행률 갱신
                file.write(chunk)  # 청크를 파일에 기록
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 각 레이어에 해당하는 빈 딕셔너리 블록을 초기화
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 체크포인트의 변수들을 순회
    for name, _ in tf.train.list_variables(ckpt_path):
        # 변수를 로드하고 불필요한 차원 제거
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 변수 이름에서 필요한 부분 추출
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # 변수를 저장할 대상 딕셔너리 식별
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 중첩된 딕셔너리를 재귀적으로 탐색하거나 생성
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 마지막 키에 변수 배열 할당
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
