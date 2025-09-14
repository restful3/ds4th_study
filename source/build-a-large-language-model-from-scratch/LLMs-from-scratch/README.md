## LLMs-from-scratch (한국어 요약)

이 저장소는 Sebastian Raschka의 "Build a Large Language Model (From Scratch)" 공식 코드를 학습용으로 간소화하고, 폴더 깊이를 줄여 단원별로 바로 접근 가능하도록 재구성한 버전입니다.

- 공식 저장소: https://github.com/rasbt/LLMs-from-scratch
- 도서 정보: http://mng.bz/orYv


### 시스템 요구사항
- Python 3.x (권장: 최신 소버전)
- 필수 라이브러리: `requirements.txt` 참고 (PyTorch, JupyterLab 등)
- GPU가 있으면 PyTorch가 자동 사용


### 빠른 시작 (가상환경 포함)
아래 명령은 Linux/macOS 기준입니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 주피터 실행
jupyter lab
```

비활성화:

```bash
deactivate
```


### 폴더 구조 (단순화된 학습용)
```
LLMs-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── ch02/
├── ch03/
├── ch04/
├── ch05/
├── ch06/
├── ch07/
├── appendix-A/
├── appendix-D/
└── appendix-E/
```


### 단원별 핵심 파일
- ch02 (텍스트 데이터)
  - [ch02/ch02.ipynb](ch02/ch02.ipynb)
  - [ch02/dataloader.ipynb](ch02/dataloader.ipynb)
  - [ch02/exercise-solutions.ipynb](ch02/exercise-solutions.ipynb)

- ch03 (어텐션 메커니즘)
  - [ch03/ch03.ipynb](ch03/ch03.ipynb)
  - [ch03/multihead-attention.ipynb](ch03/multihead-attention.ipynb)
  - [ch03/exercise-solutions.ipynb](ch03/exercise-solutions.ipynb)

- ch04 (GPT 모델 구현)
  - [ch04/ch04.ipynb](ch04/ch04.ipynb)
  - [ch04/gpt.py](ch04/gpt.py)
  - [ch04/exercise-solutions.ipynb](ch04/exercise-solutions.ipynb)

- ch05 (사전 학습)
  - [ch05/ch05.ipynb](ch05/ch05.ipynb)
  - [ch05/gpt_train.py](ch05/gpt_train.py)
  - [ch05/gpt_generate.py](ch05/gpt_generate.py)
  - [ch05/exercise-solutions.ipynb](ch05/exercise-solutions.ipynb)

- ch06 (분류 파인튜닝)
  - [ch06/ch06.ipynb](ch06/ch06.ipynb)
  - [ch06/gpt_class_finetune.py](ch06/gpt_class_finetune.py)
  - [ch06/load-finetuned-model.ipynb](ch06/load-finetuned-model.ipynb)
  - [ch06/exercise-solutions.ipynb](ch06/exercise-solutions.ipynb)

- ch07 (지시사항 파인튜닝)
  - [ch07/ch07.ipynb](ch07/ch07.ipynb)
  - [ch07/gpt_instruction_finetuning.py](ch07/gpt_instruction_finetuning.py)
  - [ch07/ollama_evaluate.py](ch07/ollama_evaluate.py)
  - [ch07/load-finetuned-model.ipynb](ch07/load-finetuned-model.ipynb)
  - [ch07/exercise-solutions.ipynb](ch07/exercise-solutions.ipynb)

- appendix-A (PyTorch 기초)
  - [appendix-A/code-part1.ipynb](appendix-A/code-part1.ipynb)
  - [appendix-A/code-part2.ipynb](appendix-A/code-part2.ipynb)
  - [appendix-A/DDP-script.py](appendix-A/DDP-script.py)
  - [appendix-A/DDP-script-torchrun.py](appendix-A/DDP-script-torchrun.py)
  - [appendix-A/exercise-solutions.ipynb](appendix-A/exercise-solutions.ipynb)

- appendix-D (훈련 루프 개선)
  - [appendix-D/appendix-D.ipynb](appendix-D/appendix-D.ipynb)

- appendix-E (LoRA 파인튜닝)
  - [appendix-E/appendix-E.ipynb](appendix-E/appendix-E.ipynb)
  - [appendix-E/gpt_download.py](appendix-E/gpt_download.py)


### 자주 쓰는 명령
```bash
# 가상환경 생성/활성화/비활성화
python3 -m venv .venv
source .venv/bin/activate
deactivate

# 필수 패키지 설치/업데이트
pip install --upgrade pip
pip install -r requirements.txt

# 주피터 노트북/랩
jupyter lab
```


### 참고
- 본 저장소는 원 저자의 공식 저장소 구조를 학습 편의상 단순화한 것입니다.
- 보너스 자료/부가 스크립트는 제외되어 있을 수 있습니다.
- 원본 전체 자료와 최신 변경 사항: https://github.com/rasbt/LLMs-from-scratch


### 인용 (Citation)
> Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.

```bibtex
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```


