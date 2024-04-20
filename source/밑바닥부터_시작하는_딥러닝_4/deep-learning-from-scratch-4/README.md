# 『밑바닥부터 시작하는 딥러닝 ❹』

<a href="http://www.yes24.com/Product/Goods/72173703"><img src="https://github.com/WegraLee/deep-learning-from-scratch-4/blob/master/cover.jpeg" width="150" align=right></a>

판매처: <a href="https://product.kyobobook.co.kr/detail/S000212020531">교보문고</a>, <a href="http://www.yes24.com/Product/Goods/124640233">예스24</a>, <a href="https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=332558447">알라딘</a>, <a href="http://hanbit.co.kr/store/books/look.php?p_code=B6031814045">한빛미디어</a> 등

[미리보기](https://preview2.hanbit.co.kr/books/yyxd/#p=1)

---

## 시리즈 소개
『밑바닥부터 시작하는 딥러닝』 시리즈는 현재 4편까지 출간되었고, 2024년 중으로 5편도 출간될 예정입니다. 5편까지의 핵심 주제와 관계는 대략 다음 그림처럼 정리할 수 있습니다.

<img src="https://github.com/WegraLee/deep-learning-from-scratch-4/blob/master/series overview.png" width="600">

**1편**과 **2편**에서는 각각 합성곱 신경망(CNN)과 순환 신경망(RNN)을 아무런 프레임워크 없이 ‘밑바닥부터’ 직접 구현합니다. 딥러닝에서 가장 기본적인 개념들을 확실하게 이해하기에 좋습니다.

**3편**에서는 파이토치와 유사한 딥러닝 프레임워크를 만들고, 그 위에서 CNN과 RNN 모델들이 동작함을 확인합니다. 이 과정에서 딥러닝의 원리를 또 다른 관점에서 들여다볼 수 있습니다.

이번 **4편**에서는 강화 학습에 딥러닝을 결합한 심층 강화 학습을 설명하며, 당연하게도 3편에서 제작한 프레임워크를 활용합니다.

**5편**에서는 요즘 가장 핫한 생성 모델을 다룰 계획입니다.

‘시리즈’입니다만, 이전 편을 읽지 않아도 상관없도록 꾸려졌습니다. 예를 들어 3편에서 만드는 프레임워크는 작동 원리뿐 아니라 API 형태까지 파이토치와 거의 같습니다. 그래서 3편을 읽지 않았어도 4편을 읽는 데 전혀 무리가 없습니다. 실제로 4편의 예제 소스용 깃허브 저장소에서는 파이토치 버전 코드도 제공합니다.

* [❶편의 깃허브 저장소](https://github.com/WegraLee/deep-learning-from-scratch)
* [❷편의 깃허브 저장소](https://github.com/WegraLee/deep-learning-from-scratch-2)
* [❸편의 깃허브 저장소](https://github.com/WegraLee/deep-learning-from-scratch-3)

## 파일 구성

|폴더 이름 |설명                         |
|:--        |:--                          |
|ch01       |1장에서 사용하는 소스 코드 |
|...        |...                          |
|ch09       |9장에서 사용하는 소스 코드    |
|common     |공통으로 사용하는 소스 코드   |
|notebooks  |주피터 노트북 형태의 소스 코드 |
|pytorch    |파이토치용으로 포팅된 소스 코드  |

## 주피터 노트북
이 책의 코드는 주피터 노트북으로도 제공됩니다. 다음 표의 링크를 클릭하면 구글과 캐글 같은 클라우드 서비스에서 노트북을 실행할 수 있습니다.

| 장 | Colab | 캐글 | Studio Lab |
| :--- | :--- | :--- | :--- |
| 1장 밴디트 문제| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/01_bandit.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/01_bandit.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/01_bandit.ipynb) |
| 4장 동적 프로그래밍 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/04_dynamic_programming.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/04_dynamic_programming.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/04_dynamic_programming.ipynb) |
| 5장 몬테카를로법 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/05_montecarlo.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/05_montecarlo.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/05_montecarlo.ipynb) |
| 6장 TD법 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) |
| 7장 신경망과 Q 러닝 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/07_neural_networks.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/07_neural_networks.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/07_neural_networks.ipynb) |
| 8장 DQN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/08_dqn.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/08_dqn.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/08_dqn.ipynb) |
| 9장 정책 경사법  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/09_policy_gradient.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/09_policy_gradient.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/09_policy_gradient.ipynb) |


## 요구사항
소스 코드를 실행하려면 아래의 소프트웨어가 설치되어 있어야 합니다.

* 파이썬 3.x
* NumPy
* Matplotlib
* OpenAI Gym
* DeZero (혹은 파이토치)
 
이 책은 딥러닝 프레임워크로 DeZero를 사용합니다. DeZero는 시리즈 3편에서 만든 프레임워크입니다('pip install dezero' 명령으로 설치할 수 있습니다).

파이토치를 사용한 구현은 [pytorch 디렉터리](https://github.com/WegraLee/deep-learning-from-scratch-4/tree/master/pytorch)에서 제공합니다.

## 실행 방법

예제 코드들은 장별로 나눠 저장되어 있습니다. 실행하려면 다음과 같이 파이썬 명령을 실행하세요.

```
$ python ch01/avg.py
$ python ch08/dqn.py

$ cd ch09
$ python actor_critic.py
```

보다시피 각 디렉터리로 이동 후 실행해도 되고, 상위 디렉터리에서 ch0x 디렉터리를 지정해 실행해도 됩니다.

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.
상업적 목적으로도 자유롭게 이용하실 수 있습니다.


## 책의 오류

이 책의 오탈자 등 오류 정보는 아래 페이지에서 확인하실 수 있습니다.

[정오표](https://docs.google.com/document/d/1fsPVXyPF0gpmN57VV6k0uxMfWXUbiQCwno8vCTYpMc8/edit)
