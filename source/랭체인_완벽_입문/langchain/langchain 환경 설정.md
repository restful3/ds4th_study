1. 원하는 설치폴더에서 깃허브로 깃클론 후 langchain폴더로 이동
```Powershall
git clone https://github.com/wikibook/langchain
```

```Powershall
cd langchain
```

2. 아나콘다 가상환경 생성 / 활성화
```Powershall
conda create -n lang310 python=3.10
```

3. 환경설정
```Powershall
conda activate lang310
conda install ipykernel
pip install -r requirements.txt
pip install mkl-fft
pip install python-dotenv
pip install openai
```

4. 주피터의 커널로 가상환경을 저장합니다
```Powershall
python -m ipykernel install --user --name=lang310
```
