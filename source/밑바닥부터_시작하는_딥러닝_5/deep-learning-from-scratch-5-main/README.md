# 『밑바닥부터 시작하는 딥러닝 ❺』<br>: 정규 분포부터 생성 모델까지!

<a href="https://product.kyobobook.co.kr/detail/S000214507043"><img src="https://github.com/WegraLee/deep-learning-from-scratch-5/blob/main/cover.png?raw=true" width="180" align=right></a>

**10단계로 알아보는 이미지 생성 모델의 원리!**

이 책은 정규 분포와 최대 가능도 추정과 같은 기본 개념에서 출발하여 가우스 혼합 모델, 변이형 오토인코더(VAE), 계층형 VAE 그리고 확산 모델에 이르기까지 다양한 생성 모델을 설명한다. 수식과 알고리즘을 꼼꼼하게 다루며 수학 이론과 파이썬 프로그래밍을 바탕으로 한 실제 구현 방법을 알려준다. 생성 모델을 이론뿐만 아니라 실습과 함께 명확하게 학습할 수 있다. 특히 확산 모델에 이르는 10단계의 과정을 하나의 스토리로 엮어 중요한 기술들을 서로 잇고 개선할 수 있도록 구성했다. 이 책과 함께 생성 모델을 밑바닥부터 시작해보자.

[미리보기](https://www.yes24.com/Product/Viewer/Preview/134648807) | [알려진 오류(정오표)](https://docs.google.com/document/d/1SU7b_emm3Lqha4BfVLTr4Ae6eTg32BkKFWMEXl6N_vA) | [본문 그림과 수식 이미지 모음](https://drive.google.com/file/d/1bMxCjB_SJzc7oJ913QT6Yn9sn3fjsymn/view?usp=drive_link)

## 파일 구성

|폴더명 |설명                             |
|:--        |:--                              |
|`step01`   |1장에서 사용할 코드  |
|`step02`   |2장에서 사용할 코드  |
|...        |...                              |
|`step10`   |10장에서 사용할 코드 |
|`notebooks`   |1〜10장까지의 코드（주피터 노트북 형식）|


## 주피터 노트북

이 책의 코드는 주피터 노트북에서도 확인할 수 있습니다. 다음 표의 버튼을 클릭하면 각각의 클라우드 서비스에서 노트북을 실행할 수 있습니다.

| 단계 | Colab | Kaggle | Studio Lab |
| :--- | :--- | :--- | :--- |
| 1. 정규 분포 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/01_normal.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/01_normal.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/01_normal.ipynb) |
| 2. 최대 가능도 추정 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/02_mle.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/02_mle.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/02_mle.ipynb) |
| 3. 다변량 정규 분포 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/03_multi_normal.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/03_multi_normal.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/03_multi_normal.ipynb) |
| 4. 가우스 혼합 모델 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/04_gmm.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/04_gmm.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/04_gmm.ipynb) |
| 5. EM 알고리즘 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/05_em.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/05_em.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/05_em.ipynb) |
| 6. 신경망 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/06_pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/06_pytorch.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/06_pytorch.ipynb) |
| 7. 변이형 오토인코더 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/07_vae.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/07_vae.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/07_vae.ipynb) |
| 8. 확산 모델 이론 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/08_hvae.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/08_hvae.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/08_hvae.ipynb) |
| 9. 확산 모델 구현 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/09_diffusion.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/09_diffusion.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/09_diffusion.ipynb) |
| 10. 확산 모델 응용 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/10_diffusion2.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/10_diffusion2.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/10_diffusion2.ipynb) |


## 파이썬과 외부 라이브러리

소스 코드를 실행하려면 다음과 같은 라이브러리가 필요합니다.

* NumPy
* Matplotlib
* PyTorch（버전 2.x）
* torchvision
* tqdm

※ 파이썬 버전은 3.x를 사용합니다.


## 실행 방법

각 장의 폴더로 이동하여 파이썬 명령어를 실행하면 됩니다.

```
$ cd step01
$ python norm_dist.py

$ cd ../step02
$ python generate.py
```

---

## 팬픽 - 바닷속 딥러닝 어드벤처 (5부작)

<img src="https://github.com/WegraLee/deep-learning-from-scratch-5/blob/main/posters/%E1%84%87%E1%85%A1%E1%84%83%E1%85%A1%E1%86%BA%E1%84%89%E1%85%A9%E1%86%A8%20%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%87%E1%85%A6%E1%86%AB%E1%84%8E%E1%85%A5.png?raw=true">

"<밑바닥부터 시작하는 딥러닝>의 주인공 생선들은 딥러닝 기술로 바닷속 생태계를 어떻게 혁신하고 있을까요? 어공지능의 첨단을 이끌어가는 밑시딥 생선들과 신나는 모험을 떠나보세요."

바닷속 세계를 배경으로, 해양 생물들이 자신의 특성과 필요에 맞는 딥러닝 기술을 개발하여 문제를 해결해 나가는 모험을 그린 연작 소설입니다. 시리즈를 읽으신 분은 더 많은 재미를 느끼실 수 있도록 딥러닝 요소들을 곳곳에 삽입하였습니다.

각 편의 주인공과 주제는 다음과 같습니다.

1. **시야를 찾아서**: 쏨뱅이(쏨)가 **이미지 처리 기술**을 개발하여 주변 환경을 선명하게 파악
1. **상어공주**: 괭이상어 공주(꽹)가 **자연어 처리** 기술로 돌고래 왕자와의 사랑을 쟁취
1. **DeZero의 창조자**: 나뭇잎해룡(잎룡)이 **딥러닝 프레임워크**를 만들어 기술 보급과 협업 촉진
1. **제발, 가즈아!**: 가자미(가즈아)가 **심층 강화 학습**으로 먹이가 풍부한 새로운 바다 개척
1. **피쉬카소와 천재의 초상**: 유령실고기(피쉬카소)가 **이미지 생성 모델**로 바닷속 예술계 혁신

<a href="https://www.hanbit.co.kr/channel/series/series_detail_list.html?hcs_idx=34" target="_blank" rel="noopener noreferrer">소설 보러 가기</a>

---

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다. 상업적/비상업적 용도로 자유롭게 사용하실 수 있습니다.
