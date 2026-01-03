# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# "처음부터 대규모 언어 모델 구축하기" 소스
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 코드: https://github.com/rasbt/LLMs-from-scratch
#
# 7장 코드를 기반으로 한 최소한의 명령어 미세 조정 파일

import json
import psutil
from tqdm import tqdm
import urllib.request


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 데이터 페이로드를 딕셔너리로 생성
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # 결정론적 응답을 위해 아래 설정이 필요합니다
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # 딕셔너리를 JSON 형식의 문자열로 변환하고 바이트로 인코딩
    payload = json.dumps(data).encode("utf-8")

    # 요청 객체를 생성하고, 메서드를 POST로 설정하고 필요한 헤더를 추가
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 요청을 보내고 응답을 캡처
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 응답을 읽고 디코딩
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def main(file_path):
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    with open(file_path, "r") as file:
        test_data = json.load(file)

    model = "llama3"
    scores = generate_model_scores(test_data, "model_response", model)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue

    return scores


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model responses with ollama"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "'output'과 'model_response' 키를 포함하는"
            " 테스트 데이터셋의 `.json` 파일 경로"
        )
    )
    args = parser.parse_args()

    main(file_path=args.file_path)
