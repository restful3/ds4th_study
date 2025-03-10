## 1. 서론 (Introduction) 상세 요약: rStar-Math의 탄생 배경과 핵심 아이디어

**1.1. 배경: LLM의 수학 문제 해결 능력과 한계**

최근 연구들은 거대 언어 모델(LLM)이 수학 문제 해결에 놀라운 능력을 보임을 입증했습니다. 하지만, 기존 LLM은 **System 1 사고** 방식처럼 단일 추론으로 완전한 해답을 생성하는 경향이 있어, 빠르지만 오류가 발생하기 쉽습니다.

**1.2. System 2 사고 방식의 필요성 대두**

이에 대한 대안으로 **System 2 사고** 방식, 즉 인간처럼 느리고 깊이 있는 사고 과정을 모방하는 패러다임이 제시되었습니다. 이 패러다임에서는 LLM을 **정책 모델**로 활용하여 여러 단계의 수학적 추론 단계를 생성하고, 또 다른 LLM을 **보상 모델**로 활용하여 각 단계를 평가합니다. 더욱 정확하다고 판단되는 단계와 솔루션을 선택하고, 이 과정을 반복하여 최종 답을 도출합니다.

**1.3. rStar-Math: 작은 SLM 기반의 깊이 있는 사고 프레임워크**

본 논문에서는 **rStar-Math**라는 새로운 접근 방식을 제시합니다. rStar-Math는 **소규모 언어 모델(SLM)**이 **자기 진화적인 깊이 있는 사고(self-evolved deep thinking)**를 통해, 거대 모델인 OpenAI 01과 경쟁하거나 능가하는 수학적 추론 능력을 달성할 수 있음을 보여줍니다.

**1.4. rStar-Math의 핵심 메커니즘: MCTS와 자기 진화**

rStar-Math는 **몬테카를로 트리 탐색(MCTS)**를 활용하여 "깊이 있는 사고"를 구현합니다. 수학 정책 SLM은 MCTS를 통해 시험 시간 탐색을 수행하고, SLM 기반의 프로세스 보상 모델(PRM)의 안내를 받습니다. 핵심은 두 개의 SLM을 훈련하는 데 따르는 어려움을 해결하기 위해 **세 가지 혁신적인 기술**을 도입하는 것입니다.

**1.5. rStar-Math의 세 가지 혁신 기술**

1. **코드 증강 CoT 데이터 합성 (Code-augmented CoT data synthesis):** 정책 SLM 훈련을 위해 단계별 검증된 추론 궤적(reasoning trajectories)을 생성하는 새로운 방법입니다. 광범위한 MCTS 롤아웃을 통해 단계별로 검증된 추론 궤적을 생성하고, 이를 정책 SLM 훈련에 사용합니다.
    
2. **프로세스 선호 모델(PPM) 훈련 (Process Preference Model training):** 단순한 단계별 점수 주석(score annotation)을 피하고, 더 효과적인 프로세스 선호 모델(PPM)을 생성하는 새로운 보상 모델 훈련 방법입니다. 단계별 보상 점수 할당의 본질적인 노이즈와 부정확성을 극복합니다.
    
3. **자기 진화 레시피 (Self-evolution recipe):** 정책 SLM과 PPM을 처음부터 구축하고, 반복적으로 진화시켜 추론 능력을 향상시키는 자기 진화 레시피입니다. 4단계의 자기 진화 과정을 통해 SLM의 수학적 추론 능력을 최고 수준으로 끌어올립니다.
![[Pasted image 20250125011234.png]]

**1.6. rStar-Math의 성능 입증**

rStar-Math는 747,000개의 수학 문제에 대한 수백만 개의 합성 솔루션을 통해 4단계의 자기 진화를 거쳐 SLM의 수학적 추론 능력을 최고 수준으로 향상시켰습니다. MATH 벤치마크에서 Qwen2.5-Math-7B 모델의 성능을 58.8%에서 90.0%로, Phi3-mini-3.8B 모델의 성능을 41.4%에서 86.4%로 향상시키며, OpenAI 01-preview 모델을 능가하는 성능을 보여주었습니다. 미국 수학 올림피아드(AIME)에서도 뛰어난 문제 해결 능력을 입증했습니다.
![[Pasted image 20250125011151.png]]
**1.7. 코드 및 데이터 공개**

rStar-Math의 코드와 데이터는 GitHub ([https://github.com/microsoft/rStar](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fmicrosoft%2FrStar))를 통해 공개될 예정입니다.

**요약**: 서론에서는 기존 LLM의 한계를 지적하고, System 2 사고 방식의 필요성을 강조하며, rStar-Math라는 새로운 접근 방식을 소개합니다. rStar-Math의 핵심 아이디어와 세 가지 혁신 기술, 그리고 뛰어난 성능을 간략하게 제시하며, 논문의 전체적인 방향을 제시합니다.



## 2. 관련 연구 (Related Works) 상세 요약: rStar-Math의 차별점과 기존 연구와의 연관성

2장에서는 rStar-Math 연구와 관련된 기존 연구들을 살펴보고, rStar-Math의 차별점과 기여를 명확히 합니다. 크게 **수학 데이터 합성**, **테스트 시점 계산 확장**, **보상 모델** 관련 연구들을 다룹니다.

**2.1. 수학 데이터 합성 (Math Data Synthesis)**

- **GPT-4 증류(distillation) 기반 CoT 데이터**: LLM 수학 추론 능력 발전은 고품질 CoT(Chain-of-Thought) 데이터 구축에 크게 의존해 왔습니다. GPT-4와 같은 거대 모델을 사용하여 CoT 데이터를 합성하는 방식이 주류를 이루었지만, 이는 교사 모델(teacher LLM)의 능력에 의해 성능이 제한되는 단점이 있습니다. NuminaMath, MetaMath 등의 연구가 대표적입니다.
    
- **CoT 데이터 증강의 한계**: GPT-4 증류 방식은 효과적이지만, 교사 모델이 풀 수 없는 어려운 문제는 학습 데이터셋에서 제외됩니다. 또한, 풀 수 있는 문제조차 오류가 있는 중간 단계를 포함할 수 있으며, 이를 감지하기 어렵습니다. 데이터 품질 향상을 위한 Rejection Sampling 등의 방법이 있지만, 중간 단계의 정확성을 보장하지는 못합니다. CoT 데이터 규모를 늘리는 것만으로는 성능 향상에 한계에 다다르고 있습니다 (e.g., OpenMathInstruct-2).
    
- **rStar-Math의 차별점**: rStar-Math는 GPT-4와 같은 거대 모델에 의존하는 대신, **더 작은 SLM과 MCTS를 활용하여 자체적으로 고품질의 학습 데이터를 생성**합니다. 이를 통해 교사 모델의 능력에 종속되지 않고, 더욱 어려운 문제에 대한 학습 데이터를 확보하며, 데이터 합성 과정에서 중간 단계의 오류를 효과적으로 제거합니다.
    

**2.2. 테스트 시점 계산 확장 (Scaling Test-time Compute)**

- **새로운 스케일링 법칙**: 테스트 시점 계산량 확장은 LLM 성능을 향상시키는 새로운 방법론으로 주목받고 있습니다. 다양한 샘플을 생성하고, 보상 모델을 사용하여 최적의 솔루션을 선택하는 방식으로 성능을 개선합니다. Random Sampling, Tree-Search (MCTS 포함) 등 다양한 탐색 방법들이 연구되고 있습니다.
    
- **오픈 소스 방법의 한계**: 오픈 소스 LLM 기반 테스트 시점 계산 확장 연구는 정책 모델 또는 보상 모델의 한계로 인해 제한적인 성능 향상을 보여왔습니다.
    
- **rStar-Math의 차별점**: rStar-Math는 정책 모델과 보상 모델을 **반복적으로 진화**시켜 이러한 한계를 극복합니다. 이를 통해 OpenAI 01 모델과 경쟁 가능한 System 2 수준의 수학적 추론 성능을 달성합니다.
    

**2.3. 보상 모델 (Reward Models)**

- **System 2 추론의 핵심**: 효과적인 System 2 추론을 위해 **보상 모델**은 매우 중요하지만, 학습 데이터 확보가 어렵다는 문제점이 있습니다. LLM-as-a-Judge, Outcome Reward Model (ORM), Process Reward Model (PRM) 등 다양한 보상 모델 연구가 진행되고 있습니다.
    
- **PRM의 잠재력과 한계**: PRM은 단계별 보상 신호를 제공하여 복잡한 추론에 효과적이지만, 단계별 주석(annotation) 수집에 많은 비용이 소요됩니다. PRM800k와 같은 대규모 데이터셋 구축 노력도 있었지만, 여전히 데이터 희소성 문제가 존재합니다. Monte Carlo Sampling 또는 MCTS를 이용한 자동 주석 방식 연구도 있지만, 정확한 보상 점수 생성의 어려움으로 성능 향상에 제한이 있습니다.
    
- **rStar-Math의 차별점**: rStar-Math는 정확한 단계별 보상 점수 대신, **프로세스 선호 모델(PPM)**이라는 새로운 보상 모델을 도입합니다. PPM은 Q-값을 기반으로 단계별 선호도 쌍(preference pairs)을 학습하여, 정확한 단계별 점수 없이도 효과적인 보상 신호를 제공합니다. 이를 통해 기존 PRM의 한계를 극복하고, 더 안정적이고 효과적인 보상 모델 학습이 가능합니다.
    

**요약**: 2장에서는 rStar-Math 연구의 배경이 되는 기존 연구들을 세 가지 측면에서 심층적으로 분석합니다. 수학 데이터 합성, 테스트 시점 계산 확장, 보상 모델 관련 연구들의 한계를 지적하고, rStar-Math가 제시하는 코드 증강 CoT 데이터 합성, 자기 진화, PPM 학습 방식이 이러한 한계를 어떻게 극복하고 차별점을 가지는지 명확하게 설명합니다. rStar-Math의 혁신성과 기여를 더욱 돋보이게 하는 중요한 챕터입니다.



## 3. 방법론 (Methodology) 상세 요약: rStar-Math의 핵심 구성 요소 및 작동 방식

3장에서는 rStar-Math의 핵심 방법론을 상세하게 설명합니다. rStar-Math는 효과적인 System 2 추론을 위해 MCTS, 코드 증강 CoT 데이터 합성, PPM 학습, 자기 진화 레시피 등 다양한 요소들을 통합적으로 활용합니다.

**3.1. 디자인 선택 (Design Choices)**

- **효과적인 System 2 추론을 위한 MCTS (MCTS for Effective System 2 Reasoning)**
    
    - **MCTS 선택 이유**: rStar-Math는 System 2 깊이 사고를 위해 몬테카를로 트리 탐색(MCTS)을 핵심 구성 요소로 채택했습니다. MCTS를 선택한 이유는 크게 두 가지입니다.
        
        1. **복잡한 문제 분해**: MCTS는 복잡한 수학 문제를 더 간단한 단일 단계 생성 작업으로 분해하여 정책 SLM의 부담을 줄여줍니다. Best-of-N, Self-Consistency 와 같은 다른 System 2 방법들은 전체 솔루션을 한 번의 추론으로 생성해야 합니다.
            
        2. **단계별 학습 데이터 자동 생성**: MCTS의 단계별 생성 방식은 정책 모델과 보상 모델 모두를 위한 단계별 학습 데이터를 자연스럽게 생성합니다. 표준 MCTS 롤아웃은 최종 정답에 대한 기여도를 기반으로 각 단계에 Q-값을 자동으로 할당하므로, 프로세스 보상 모델 훈련에 필요한 인간 주석 단계별 레이블링의 필요성을 없애줍니다.
            
    - **MCTS의 과제**: 이상적으로 GPT-4와 같은 advanced LLM을 MCTS에 통합하여 학습 데이터를 생성할 수 있지만, 두 가지 주요 과제가 있습니다.
        
        1. **어려운 문제 해결 능력 부족**: 강력한 모델조차 올림피아드 수준의 어려운 문제를 일관되게 해결하는 데 어려움을 겪습니다. 따라서 생성되는 학습 데이터는 주로 쉬운 문제로 구성되어 다양성과 품질이 제한될 수 있습니다.
            
        2. **Q-값 주석의 비용**: 단계별 Q-값 주석을 위해서는 광범위한 MCTS 롤아웃이 필요합니다. 불충분한 트리 탐색은 최적 이하 단계 과대평가와 같은 잘못된 Q-값 할당으로 이어질 수 있습니다. 각 롤아웃은 여러 단계별 생성을 포함하고 모델은 계산 비용이 많이 들기 때문에 롤아웃 횟수를 늘리면 추론 비용이 크게 증가합니다.
            
    - **rStar-Math의 접근 방식**: 두 개의 7B SLM (정책 SLM, PRM)을 사용하여 고품질 학습 데이터를 생성합니다. 작은 모델 크기 덕분에 접근 가능한 하드웨어(e.g., 4x40GB A100 GPU)에서 광범위한 MCTS 롤아웃이 가능합니다. SLM의 약한 능력으로 인해 자체 데이터 생성은 더 큰 어려움이 있습니다. SLM은 정답 생성을 자주 실패하고, 최종 답변이 정확하더라도 중간 단계에 오류가 있거나 품질이 낮은 경우가 많습니다. 또한, SLM은 GPT-4와 같은 advanced 모델에 비해 어려운 문제를 훨씬 적게 해결합니다.
        
- **개요 (Overview)**
    
    - **코드 증강 CoT 합성**: 오류 및 낮은 품질의 중간 단계를 완화하기 위해 코드 증강 CoT 합성 방법을 도입하여 단계별 검증된 추론 궤적을 Q-값으로 주석 처리하여 생성합니다.
        
    - **자기 진화 레시피**: 어려운 문제에 대한 SLM 성능을 더욱 향상시키기 위해 4단계 자기 진화 레시피를 도입합니다. 각 라운드에서 정책 SLM과 보상 모델 모두 더 강력한 버전으로 업데이트되어 점진적으로 더 어려운 문제를 해결하고 더 높은 품질의 학습 데이터를 생성합니다.
        
    - **프로세스 선호 모델 (PPM)**: 정확한 단계별 보상 주석의 필요성을 없애고 더 효과적인 프로세스 선호 모델(PPM)을 생성하는 새로운 프로세스 보상 모델 훈련 접근 방식을 제시합니다.
        

**3.2. 단계별 검증된 추론 궤적 (Step-by-Step Verified Reasoning Trajectory)**

- **MCTS 기반 탐색**: 문제 x와 정책 모델 M이 주어지면 표준 MCTS를 실행하여 단계별 솔루션 탐색을 위한 탐색 트리를 점진적으로 구성합니다. 루트 노드는 질문 x를 나타내고, 자식 노드는 M에 의해 생성된 중간 단계 s에 해당합니다. 터미널 노드 sa에서 끝나는 루트-리프 경로는 궤적 t = x ⊕ s1 ⊕ s2 ⊕ ... ⊕ sd를 형성하고, 각 단계 si에는 Q-값 Q(si)가 할당됩니다. 탐색 트리 T에서 솔루션 궤적 T = {t1, t2, ..., tn}(n ≥ 1)를 추출합니다. 목표는 T에서 고품질 궤적을 선택하여 학습 데이터셋을 구성하는 것입니다.
    
- **코드 증강 CoT 생성 (Code-augmented CoT Generation)**
    
    - **기존 MCTS 방식의 한계**: 기존 MCTS 접근 방식은 주로 자연어(NL) CoT를 생성합니다. 그러나 LLM은 환각(hallucination)으로 인해 부정확하거나 관련 없는 단계를 생성하고도 우연히 정답에 도달하는 경우가 있습니다. 이러한 결함 있는 단계를 감지하고 제거하는 것은 어렵습니다.
        
    - **코드 실행 증강 CoT**: 이 문제를 해결하기 위해 새로운 코드 실행 증강 CoT를 제안합니다. 정책 모델은 Python 주석으로 포함된 NL CoT와 함께 해당 Python 코드를 생성합니다. 성공적으로 실행된 Python 코드를 가진 생성만 유효한 후보로 유지합니다.
    ![[Pasted image 20250125011515.png]]

- **광범위한 롤아웃을 통한 Q-값 주석 (Extensive Rollouts for Q-value Annotation)**
    
    - **정확한 Q-값의 중요성**: 정확한 Q-값 Q(s) 주석은 MCTS 노드 선택을 올바른 문제 해결 경로로 안내하고 궤적 내에서 고품질 단계를 식별하는 데 매우 중요합니다. Q-값 신뢰성을 향상시키기 위해, 게임 결과에 따라 각 수의 보상을 사후 평가하는 Go 플레이어로부터 영감을 얻었습니다. 초기 추정치는 부정확할 수 있지만, 반복적인 게임 플레이를 통해 시간이 지남에 따라 이러한 평가가 개선됩니다. 마찬가지로 각 롤아웃에서 최종 정답 달성에 대한 기여도를 기반으로 각 단계의 Q-값을 업데이트합니다.
        
    - **두 가지 자체 주석 방법**: 단계별 Q-값을 얻기 위해 두 가지 자체 주석 방법을 도입합니다.
        
        1. **터미널 가이드 주석 (Terminal-guided annotation)**: PPM을 사용할 수 없거나 정확도가 충분하지 않은 처음 두 라운드 동안 터미널 가이드 주석을 사용합니다. 각 중간 노드의 점수를 최종 정답에 대한 기여도를 기준으로 합니다. 단계가 정답으로 자주 이어지면 Q-값이 증가하고, 그렇지 않으면 감소합니다. 터미널 노드는 정답의 경우 q(sa) = 1점, 오답의 경우 q(sa) = -1점을 받습니다.
            
        2. **PRM 증강 주석 (PRM-augmented annotation)**: 세 번째 라운드부터는 PPM을 사용하여 각 단계의 점수를 매겨 생성 효율성을 높입니다. PPM 증강 MCTS는 정책 모델이 더 높은 품질의 단계를 생성하고 솔루션을 올바른 경로로 안내하는 데 도움이 됩니다. 단계 si에 대해 PPM은 부분 궤적을 기반으로 초기 q(si)º 값을 예측합니다. 이 q 값은 Eq. 2의 MCTS 백프로파게이션을 통해 터미널 노드의 q(sa) 값을 기반으로 업데이트됩니다. 터미널 노드 sd의 경우 학습 데이터 생성 중 점수
            





## 3.3 프로세스 선호 모델 (Process Preference Model) 상세 요약
> PPM의 작동 방식 및 장점

- **PRM의 필요성과 한계**: 프로세스 보상 모델(PRM)은 세분화된 단계별 보상 신호를 제공하여 어려운 수학 문제 해결에 매우 유용합니다. 하지만 고품질 단계별 학습 데이터 확보는 여전히 어려운 과제입니다. 기존 방법들은 인간 주석 또는 MCTS 생성 점수를 사용하여 각 단계의 점수를 할당하고, MSE 손실 또는 pointwise 손실과 같은 방법을 사용하여 예측 점수와 레이블 점수 간의 차이를 최소화합니다. 결과적으로 주석 처리된 단계별 보상 점수의 정확성이 프로세스 보상 모델의 효과를 직접적으로 결정합니다.
    
- **단계별 점수 부여의 어려움**: 정확한 단계별 점수 부여는 여전히 해결되지 않은 문제입니다. 올바른 단계 집합 중에서 최고, 차선, 평균을 순위를 매기고 정확한 점수를 할당하기는 어렵습니다. 마찬가지로, 잘못된 단계 중에서도 최악의 단계와 중간 정도의 잘못된 단계를 구별하는 것은 유사한 어려움을 야기합니다. 전문가의 인간 주석조차 일관성이 부족하고, 특히 대규모로 진행할 때 학습 레이블에 내재적인 노이즈를 유발합니다.
    
- **프로세스 선호 모델 (PPM)의 도입**: rStar-Math는 단계별 양/음수 선호도 쌍을 구성하여 프로세스 선호 모델(PPM)을 훈련하는 새로운 훈련 방법을 도입합니다. PPM은 Q-값을 직접적인 보상 레이블로 사용하는 대신, 선호도 쌍 구성에 Q-값을 활용합니다.
    
- **PPM 훈련 방식**:
    
    - **선호도 쌍 구성**: 각 단계마다 MCTS 트리에서 가장 높은 Q-값을 가진 두 후보를 양성 단계로, 가장 낮은 Q-값을 가진 두 후보를 음성 단계로 선택합니다. 중요한 것은 선택된 양성 단계는 정답으로 이어져야 하고, 음성 단계는 오답으로 이어져야 한다는 점입니다. 중간 단계(마지막 답변 단계 제외)의 경우, 양성 및 음성 쌍은 동일한 이전 단계를 공유합니다. 마지막 답변 단계에서는 동일한 추론 궤적이 서로 다른 최종 답변을 거의 생성하지 않으므로 이 제한을 완화합니다. 평균 Q-값이 가장 높은 두 개의 올바른 궤적을 양성 예로, 평균 Q-값이 가장 낮은 두 개의 잘못된 궤적을 음성 예로 선택합니다.
        
    - **손실 함수**: [Ouyang et al., 2022]를 따라 쌍별 순위 손실을 사용하여 Bradley-Terry 모델을 정의합니다.
        
- **PPM의 장점**: PPM은 Q-값을 직접적인 보상 레이블로 사용하는 대신 선호도 쌍 학습을 통해 과 같은 장점을 제공합니다.
    
    - **정확한 단계별 점수 불필요**: Q-값이 단계별 품질을 정확하게 평가하기에 충분히 정확하지 않더라도, 긍정적(정확) 단계와 부정적(무관련/부정확) 단계를 안정적으로 구별할 수 있다는 사실을 활용합니다.
        
    - **노이즈 감소**: 단계별 점수 부여의 어려움과 인간 주석의 노이즈 문제를 완화하고, 더 안정적인 학습을 가능하게 합니다.
        
    - **더 효과적인 보상 신호**: PPM은 단계별 선호도를 직접적으로 학습하여, 더 효과적인 보상 신호를 제공하고, 모델의 추론 능력을 향상시킵니다.
        

**요약**: 3.3 섹션에서는 rStar-Math의 핵심 혁신 중 하나인 PPM을 상세히 설명합니다. PPM은 기존 PRM의 한계를 극복하고, 더 안정적이고 효과적인 보상 모델 학습을 가능하게 합니다. PPM의 훈련 방식과 장점을 제시하며, rStar-Math 방법론의 핵심적인 부분을 구성합니다.



## 3.4 자기 진화적 깊이 사고 (Self-Evolved Deep Thinking) 상세 요약: 4단계 자기 진화 레시피

rStar-Math는 SLM의 약한 능력으로 인해 4단계의 MCTS 깊이 사고를 수행하여 점진적으로 더 높은 품질의 데이터를 생성하고, 더 어려운 수학 문제로 학습 데이터셋을 확장합니다. 각 라운드는 점진적인 개선을 달성합니다.

**3.4.1 단계별 검증된 추론 궤적을 사용한 훈련 (Training with Step-by-Step Verified Reasoning Trajectory)**

- **수학 문제 수집 (Math Problems Collection)**
    
    - **데이터셋**: NuminaMath, MetaMath 등 공개적으로 사용 가능한 747k 개의 수학 단어 문제 데이터셋을 수집합니다.
        
    - **경쟁 수준 문제**: NuminaMath의 올림피아드 및 AIME/AMC와 같은 경쟁 수준 문제만 포함합니다.
        
    - **GPT-4 활용 문제 합성**: 제한된 경쟁 수준 문제를 보강하기 위해 GPT-4를 사용하여 7.5k MATH 학습 세트 및 3.6k AMC-AIME 훈련 분할의 시드 문제(seed problems)를 기반으로 새로운 문제를 합성합니다. GPT-4가 어려운 시드 문제에 대해 해결 불가능한 문제 또는 부정확한 솔루션을 생성하는 경우가 많으므로, GPT-4에 문제당 10개의 솔루션을 생성하도록 요청하고, 최소 3개의 일관된 솔루션만 유지합니다.
        
- **추론 궤적 수집 (Reasoning Trajectories Collection)**
    
    - **MCTS 롤아웃**: 747k 수학 데이터셋의 원래 솔루션 대신, 더 높은 품질의 단계별 검증된 추론 궤적을 생성하기 위해 광범위한 MCTS 롤아웃(섹션 3.2)을 수행합니다. 각 자체 진화 라운드에서 수학 문제당 16개의 롤아웃을 수행하여 16개의 추론 궤적을 생성합니다.
        
    - **문제 난이도 분류**: 생성된 궤적의 정답 비율에 따라 문제를 쉬움(모든 솔루션이 정답), 중간(정답과 오답 혼합), 어려움(모든 솔루션이 오답)으로 분류합니다. 정답 궤적이 없는 어려운 문제의 경우 16개의 롤아웃으로 추가 MCTS를 수행합니다. 그런  모든 단계별 궤적과 주석 처리된 Q-값을 수집하고 필터링하여 정책 SLM 및 프로세스 선호 모델을 훈련합니다.
        
- **정책 SLM의 지도 학습 미세 조정 (Supervised Fine-tuning the Policy SLM)**
    
    - **고품질 궤적 선택**: 고품질 추론 궤적을 선택하는 것이 frontier 수학 LLM을 미세 조정하는 데 핵심입니다. GPT 증류 및 Best-of-N과 같은 방법은 낮은 품질 또는 오류가 있는 중간 단계를 포함할 수 있지만, 더 효과적인 접근 방식은 궤적의 모든 단계가 고품질인지 확인하는 것입니다.
        
    - **단계별 Q-값을 활용한 최적 궤적 선택**: MCTS 롤아웃에서 단계별 Q-값을 사용하여 최적의 궤적을 선택합니다. 각 수학 문제에 대해 정답으로 이어지는 궤적 중에서 평균 Q-값이 가장 높은 상위 2개 궤적을 SFT 훈련 데이터로 선택합니다.
        
- **PPM 훈련 (Training PPM)**
    
    - **초기화**: PPM은 미세 조정된 정책 모델에서 초기화됩니다.  토큰 예측 헤드는 선형 레이어와 tanh 함수로 구성된 스칼라 값 헤드로 대체되어 출력을 [-1, 1] 범위로 제한합니다.
        
    - **선호도 쌍 학습**: 모든 솔루션 궤적이 완전히 정답이거나 완전히 오답인 수학 문제는 필터링합니다. 결과가 혼합된 문제의 경우 Q-값을 기반으로 각 단계에 대해 2개의 양성 및 2개의 음성 예제를 선택하고, 이를 훈련 데이터에 대한 선호도 쌍으로 사용합니다.
        

**3.4.2 자기 진화 레시피 (Recipe for Self-Evolution)**

SLM의 약한 능력으로 인해 더 높은 품질의 데이터를 점진적으로 생성하고 더 어려운 수학 문제로 훈련 세트를 확장하기 위해 4라운드의 MCTS 깊이 사고를 수행합니다.

- **라운드 1: 초기 강력한 정책 SLM-r1 부트스트래핑 (Bootstrapping an initial strong policy SLM-r1)**
    
    - **DeepSeek-Coder-V2-Instruct 활용**: SLM이 합리적으로 양질의 훈련 데이터를 자체 생성할 수 있도록 초기 강력한 정책 모델인 SLM-r1을 부트스트랩합니다. DeepSeek-Coder-V2-Instruct (236B)로 MCTS를 실행하여 SFT 데이터를 수집합니다.
        
    - **터미널 가이드 주석 및 제한된 롤아웃**: 보상 모델을 사용할 수 없으므로 터미널 가이드 주석을 Q-값에 사용하고 효율성을 위해 MCTS를 8개의 롤아웃으로 제한합니다.
        
    - **SFT 데이터 선택**: 정답 솔루션의 경우 평균 Q-값이 가장 높은 상위 2개 궤적을 SFT 데이터로 선택합니다. PPM-r1도 훈련하지만, 제한된 롤아웃으로 인해 신뢰할 수 없는 Q-값이 생성되어 PPM-r1의 효과가 저하됩니다 (표 4).
        
- **라운드 2: 신뢰할 수 있는 PPM-r2 훈련 (Training a reliable PPM-r2)**
    
    - **확장된 MCTS 롤아웃**: 정책 모델이 7B SLM-r1으로 업데이트됨에 따라, 더 신뢰할 수 있는 Q-값 주석을 위해 광범위한 MCTS 롤아웃을 수행하고 첫 번째 신뢰할 수 있는 보상 모델인 PPM-r2를 훈련합니다. 문제당 16개의 MCTS 롤아웃을 수행합니다.
        
    - **향상된 궤적 품질 및 Q-값 정확도**: 단계별 검증된 추론 궤적은 품질과 Q-값 정확도 모두에서 상당한 개선을 보여줍니다.
        
    - **PPM-r2의 효과**: PPM-r2는 부트스트랩 라운드보다 훨씬 더 효과적입니다 (표 4). 정책 SLM-r2도 예상대로 지속적으로 개선됩니다 (표 3).
        
- **라운드 3: PPM 증강 MCTS를 통한 데이터 품질 대폭 향상 (PPM-augmented MCTS to significantly improve data quality)**
    
    - **PPM-r2 증강 MCTS**: 신뢰할 수 있는 PPM-r2를 통해 PPM 증강 MCTS를 수행하여 데이터를 생성합니다.
        
    - **더 높은 품질의 궤적**: PPM 증강 MCTS는 훈련 세트에서 더 많은 수학 및 올림피아드 수준 문제를 다루는 훨씬 더 높은 품질의 궤적을 생성합니다 (표 2).
        
    - **정책 SLM-r3 및 PPM-r3 훈련**: 생성된 추론 궤적과 자체 주석 Q-값을 사용하여 새로운 정책 SLM-r3 및 PPM-r3를 훈련합니다. 둘 다 상당한 개선을 보입니다.
        
- **라운드 4: 어려운 수학 문제 해결 (Solving challenging math problems)**
    
    - **올림피아드 수준 문제 해결**: 3라운드 후, 초등학교 및 MATH 문제는 높은 성공률을 달성했지만, 올림피아드 수준 문제는 훈련 세트에 62.16%만 포함되었습니다. 이는 SLM의 약한 추론 능력 때문만은 아닙니다. GPT-4 또는 01에서도 해결되지 않은 올림피아드 문제가 많습니다.
        
    - **롤아웃 횟수 증가**: 적용 범위를 개선하기 위해 간단한 전략을 채택합니다. 해결되지 않은 문제에 대해 16개의 MCTS 롤아웃 후 추가로 64개의 롤아웃을 수행하고, 필요한 경우 128개까지 늘립니다. 또한 다양한 랜덤 시드로 여러 MCTS 트리 확장을 수행합니다. 이를 통해 올림피아드 수준 문제의 성공률이 80.58%로 향상됩니다.
        
- **자기 진화 종료**: 4라운드의 자체 진화 후, 747k 수학 문제 중 90.25%가 훈련 세트에 성공적으로 포함되었습니다 (표 2). 나머지 해결되지 않은 문제 중 상당 부분이 합성 질문으로 구성되어 있습니다. 20개 문제의 랜덤 샘플을 수동으로 검토한 결과 19개가 오답으로 잘못 레이블링된 것으로 확인되었습니다. 따라서 나머지 해결되지 않은 문제는 품질이 낮은 것으로 결론 내리고 4라운드에서 자체 진화를 종료합니다.
    

**요약**: 3.4 섹션에서는 rStar-Math의 핵심인 4단계 자기 진화 레시피를 상세하게 설명합니다. 각 라운드별 목표와 방법, 데이터 생성 및 모델 훈련 방식, 그리고 각 라운드를 통해 점진적으로 성능을 향상시키는 과정을 자세하게 제시합니다. 특히, 마지막 4라운드에서는 어려운 문제 해결을 위한 전략과 자기 진화 종료 기준을 설명하며, rStar-Math의 실제 작동 방식을 이해하는 데 중요한 정보를 제공합니다.





## 4. 평가 (Evaluation) 상세 요약: rStar-Math의 성능 검증 및 분석

4장에서는 rStar-Math의 성능을 다양한 벤치마크에서 평가하고, 주요 결과 및 분석을 제시합니다. rStar-Math의 효과와 일반화 능력을 입증하고, 핵심적인 발견 사항들을 논의합니다.

**4.1. 설정 (Setup)**

- **평가 데이터셋 (Evaluation Datasets)**
    
    - **다양한 수학 벤치마크**: rStar-Math의 성능을 다각적으로 평가하기 위해 다양한 수학 벤치마크를 사용합니다. GSM8K 외에도 경쟁 및 올림피아드 수준 벤치마크 (MATH-500, AIME 2024, AMC 2023, Olympiad Bench), 대학 수준 수학 문제 (College Math), 그리고 out-of-domain 벤치마크인 GaoKao (중국 대학 입학 시험) En 2023을 포함합니다.
        
    - **AIME 2024**: 미국 최고 고등학생들을 위한 시험으로, 2024년 데이터셋은 AIME I 및 II 시험의 30개 문제로 구성됩니다.
        
- **기본 모델 및 설정 (Base Models and Setup)**
    
    - **다양한 SLM**: rStar-Math의 일반화 가능성을 입증하기 위해 다양한 크기의 SLM을 기본 정책 모델로 사용합니다: Qwen2.5-Math-1.5B, Phi3-mini-Instruct (3B), Qwen2-Math-7B, Qwen2.5-Math-7B. Phi3-mini-Instruct는 수학 전문화되지 않은 범용 SLM입니다.
        
    - **4라운드 자기 진화**: 제한된 GPU 자원으로 인해 Qwen2.5-Math-7B에 대해서만 4라운드 자기 진화를 수행하여 4개의 진화된 정책 SLM과 4개의 PPM을 얻었습니다. 나머지 3개 정책 LLM은 Qwen2.5-Math-7B의 4번째 라운드에서 생성된 단계별 검증 궤적을 사용하여 미세 조정되었습니다. 최종 PPM은 3개 정책 SLM에 대한 보상 모델로 사용됩니다.
        
- **기준선 (Baselines)**
    
    - **Frontier LLM**: GPT-4o, Claude 3.5 Sonnet, OpenAI 01-preview, o1-mini 등 최첨단 LLM과 비교합니다. AMC 2023, Olympiad Bench, College Math, Gaokao, GSM8K 벤치마크에서 정확도를 측정하고, 다른 벤치마크는 공개 기술 보고서에서 가져옵니다.
        
    - **오픈 소스 우수 추론 모델**: DeepSeek-Coder-v2-Instruct, Mathstral, NuminaMath-72B, LLaMA3.1 등 오픈 소스 기반 우수 추론 모델들과 비교합니다.
        
    - **기본 모델 (Base Models)**: Instruct 버전 (e.g., Qwen2.5-Math-7B-Instruct), Best-of-N 버전 (e.g., Qwen2.5-Math-72B-Instruct+Qwen2.5-Math-RM-72B) 등 원래 모델 팀에서 개발한 System 1 및 System 2 성능을 포함합니다. Qwen 기본 모델에 사용된 보상 모델은 72B ORM으로, rStar-Math의 7B PPM보다 훨씬 큽니다.
        
- **평가 지표 (Evaluation Metric)**
    
    - **Pass@1 정확도**: 모든 기준선에 대해 Pass@1 정확도를 보고합니다. System 2 기준선의 경우 o1-mini 및 o1-preview의 기본 사고 시간과 같은 기본 평가 설정을 사용합니다. Best-of-N Qwen 모델의 경우 MATH-500, AIME/AMC 정확도를 재평가하고, 다른 벤치마크 결과는 기술 보고서에서 가져옵니다.
        
    - **공정한 비교**: rStar-Math는 Qwen과 동일한 수의 솔루션을 생성하도록 MCTS를 실행합니다. AIME/AMC의 경우 16개 궤적, 다른 벤치마크의 경우 8개 궤적을 생성하고, PPM을 사용하여 최상의 솔루션을 선택합니다. 테스트 시간 계산량 증가에 따른 성능도 rStar-Math64로 보고합니다 (64개 궤적 샘플링).
        

**4.2. 주요 결과 (Main Results)**

- **다양하고 어려운 수학 벤치마크 결과**: 표 5는 최첨단 추론 모델과 비교한 rStar-Math 결과를 보여줍니다. 세 가지 주요 관찰 사항이 있습니다.
    
    1. **SLM 수학 추론 능력 향상**: rStar-Math는 SLM의 수학 추론 능력을 크게 향상시켜, 훨씬 작은 모델 크기(1.5B-7B)로 OpenAI 01과 필적하거나 능가하는 성능을 달성했습니다. MATH 벤치마크에서 Qwen2.5-Math-7B는 rStar-Math를 통해 58.8%에서 90.0%로 향상되어 o1-preview 및 Claude 3.5 Sonnet을 능가하고 o1-mini와 동등한 수준을 달성했습니다. College Math 벤치마크에서는 o1-mini보다 2.7% 높았습니다. AIME 2024에서 rStar-Math는 53.3%를 기록하여 o1-mini 바로 아래 순위를 기록했으며, 7B 모델로 AIME I 및 II 모두에서 8/15 문제를 해결하여 상위 20% 안에 들었습니다. 주목할 만하게도 해결되지 않은 8개 문제는 기하학 기반 문제로, rStar-Math는 현재 지원하지 않는 시각적 이해 능력이 필요한 문제입니다.
        
    2. **SOTA System 2 능가**: 더 작은 정책 모델(1.5B-7B)과 보상 모델(7B)을 사용했음에도 불구하고, rStar-Math는 최첨단 System 2 기준선을 크게 능가합니다. 동일한 기본 모델(Qwen2-Math-7B, Qwen2.5-Math-1.5B/7B)을 사용하지만 10배 더 큰 보상 모델(Qwen2.5-Math-RM-72B)을 사용하는 Qwen Best-of-N 기준선과 비교했을 때, rStar-Math는 모든 기본 모델의 추론 정확도를 최첨단 수준으로 지속적으로 향상시킵니다. 10배 더 큰 Qwen2.5-Math-72B-Instruct 정책 모델을 사용하는 Best-of-N과 비교해도 rStar-Math는 GSM8K를 제외한 모든 벤치마크에서 능가합니다 (동일한 수의 샘플링된 솔루션 사용).
        
    3. **강력한 일반화 능력**: rStar-Math는 MATH, GSM8K, AIME와 같이 과최적화 위험이 있는 잘 알려진 벤치마크 외에도 Olympiad Bench, College Math, GaoKao 등 다른 어려운 수학 벤치마크에서도 강력한 일반화 능력을 보여주며 새로운 SOTA 점수를 달성했습니다. 훈련 세트는 주로 공개 데이터셋에서 가져왔으며, 이러한 벤치마크에 대한 특정 최적화는 없었습니다.
        
- **테스트 시간 계산량 확장 (Scaling up test-time computation)**
    
    - **MCTS 활용**: rStar-Math는 PPM의 안내를 받아 솔루션을 탐색하는 정책 모델을 보강하기 위해 MCTS를 사용합니다. 테스트 시간 계산량을 늘리면 더 많은 궤적을 탐색하여 성능을 향상시킬 수 있습니다.
        
    - **테스트 시간 계산량 확장 효과**: 그림 3은 4개의 어려운 수학 벤치마크에서 다양한 수의 샘플링된 궤적에 따른 공식 Qwen Best-of-N의 정확도 변화를 비교하여 테스트 시간 계산량 확장의 영향을 보여줍니다. 1개의 궤적만 샘플링하는 것은 정책 LLM의 Pass@1 정확도에 해당하며, System 1 추론으로의 폴백(fallback)을 나타냅니다.
        
    - **두 가지 주요 관찰**:
        
        1. **4개 궤적만으로도 SOTA 능가**: 4개의 궤적만으로도 rStar-Math는 Best-of-N 기준선을 크게 능가하고 o1-preview에 접근하며 o1-mini에 근접하는 성능을 보여줍니다.
            
        2. **지속적인 성능 향상**: 테스트 시간 계산량을 늘리면 모든 벤치마크에서 추론 정확도가 향상되지만, 경향은 다양합니다. Math, AIME, Olympiad Bench에서는 64개 궤적에서 포화 또는 느린 개선을 보이지만, College Math에서는 성능이 꾸준히 향상됩니다.
            

**요약**: 4장에서는 rStar-Math의 성능을 다양한 벤치마크에서 평가하고, 최첨단 모델들과 비교 분석합니다. rStar-Math가 작은 모델 크기에도 불구하고 SOTA 성능을 달성했으며, 특히 테스트 시간 계산량 확장을 통해 성능을 더욱 향상시킬 수 있음을 입증합니다. 또한, 다양한 벤치마크에서 강력한 일반화 능력을 보여주며, rStar-Math의 효과와 잠재력을 강조합니다.


---
## 5. 절삭 연구 및 분석 (Ablation Study and Analysis) 상세 요약: rStar-Math의 핵심 요소별 효과 분석

5장에서는 rStar-Math의 세 가지 핵심 혁신 요소 (자기 진화, 단계별 검증 추론 궤적, PPM) 의 효과를 분석하기 위한 절삭 연구(ablation study) 결과를 제시합니다. System 2 스타일 추론의 Pass@1 정확도는 AIME 및 AMC의 경우 16개 궤적, 다른 벤치마크의 경우 8개 궤적으로 측정되었습니다.

**5.1. 자기 진화의 효과 (The effectiveness of self-evolution)**

- **지속적인 성능 향상**: 표 5는 rStar-Math의 자기 진화적 깊이 사고의 각 라운드별 수학 추론 성능을 보여줍니다. 4라운드에 걸쳐 정확도가 지속적으로 향상되는 것을 확인할 수 있습니다.
    
- **라운드별 성능 향상**:
    
    - **라운드 1**: 기본 모델에 SFT를 적용하여 주요 개선이 이루어집니다.
        
    - **라운드 2**: MCTS에 더 강력한 PPM을 적용하여 System 2 깊이 사고의 잠재력을 최대한 발휘하면서 상당한 성능 향상을 가져옵니다. 주목할 만하게도 2라운드부터 rStar-Math는 GPT-4o를 능가합니다.
        
    - **라운드 3 & 4**: 더 나은 정책 SLM과 PPM을 통해 System 2 추론을 강화하여 추가적인 성능 향상을 보여줍니다.
    - 
**결론**: 자기 진화는 rStar-Math의 핵심 요소이며, 반복적인 개선을 통해 모델의 수학적 추론 능력을 지속적으로 향상시키는 데 효과적입니다.
---

**5.2. 단계별 검증된 추론 궤적의 효과 (The effectiveness of step-by-step verified reasoning trajectory)**

- **SFT 데이터셋 비교**: 4라운드에서 생성된 데이터를 SFT 훈련 데이터로 사용하여, 세 가지 강력한 기준선과 비교합니다.
    
    1. **GPT 증류**: MetaMath, NuminaMath-CoT와 같이 GPT-4로 합성된 CoT 솔루션 포함.
        
    2. **자체 생성 랜덤 샘플링**: 정책 SLM-r3를 사용하여 궤적을 랜덤하게 생성.
        
    3. **리젝션 샘플링**: 정책 모델에서 32개 궤적을 랜덤 샘플링하고, 훈련된 ORM으로 고품질 솔루션 순위를 매김 (부록 A.1).
        
- **표 7 결과**: Qwen2.5-Math-7B를 다양한 데이터셋으로 미세 조정한 결과입니다.
    
- **주요 관찰**:
    
    1. **SOTA 능가**: 단계별 검증된 궤적으로 미세 조정된 모델이 다른 모든 기준선을 크게 능가합니다. 이는 코드 증강 CoT 합성을 위한 PPM 증강 MCTS가 수학 솔루션 생성 중 더 조밀한 검증을 제공하기 때문입니다.
        
    2. **GPT-4 증류 능가**: 자체 SLM에서 랜덤 샘플링한 코드 증강 CoT 솔루션이 GPT-4 합성 NuminaMath 및 MetaMath 데이터셋보다 비슷하거나 더 나은 성능을 보입니다. 이는 자체 진화 라운드 후 정책 SLM이 고품질 수학 솔루션을 생성할 수 있음을 나타냅니다.
        

**결론**: 단계별 검증된 추론 궤적은 rStar-Math의 핵심 혁신 요소이며, 오류가 있는 중간 단계를 제거하고 더 어려운 문제로 훈련 세트를 확장하는 데 효과적입니다. PPM 증강 MCTS와 코드 증강 CoT 합성은 데이터 품질 향상에 중요한 역할을 합니다.

**5.3. PPM의 효과 (The effectiveness of PPM)**

- **보상 모델 비교**: PPM과 ORM, Q-value 기반 PRM (PQM)을 비교합니다. 공정한 비교를 위해 최고 품질의 훈련 데이터 (4라운드에서 생성된 단계별 검증 궤적) 를 사용하고, PPM 훈련에 사용된 문제와 일치하도록 수학 문제를 선택합니다. PPM과 유사하게 단계별 Q-값을 사용하여 각 수학 문제에 대한 양성 및 음성 궤적을 선택합니다. ORM은 쌍별 순위 손실로 훈련하고, PQM은 Q-값을 보상 레이블로 사용하여 MSE 손실로 최적화합니다.
    
- **표 8 결과**: 최종 라운드 정책 모델을 사용한 System 2 추론에 대한 ORM, PQM, PPM 성능 비교입니다.
    
- **주요 관찰**:
    
    1. **PRM > ORM**: PQM과 PPM 모두 ORM보다 성능이 뛰어납니다. 단계별 보상 신호를 제공하는 PRM이 복잡한 수학 추론 작업에서 더 효과적임을 나타냅니다.
        
    2. **PPM > PQM**: PPM이 PQM보다 MATH 및 Olympiad Bench와 같은 더 어려운 벤치마크에서 더 나은 성능을 보입니다. 이는 Q-값의 내재적 부정확성으로 인해 PQM이 어려움을 겪는 반면, PPM은 단계별 선호도 데이터를 활용하여 더 안정적인 학습을 가능하게 하기 때문입니다.
        

**결론**: PPM은 rStar-Math의 핵심 혁신 요소이며, ORM 및 PQM과 비교하여 더 효과적인 보상 신호를 제공하고, 특히 어려운 문제에서 모델 성능을 크게 향상시킵니다. PPM은 단계별 선호도 학습을 통해 Q-값의 부정확성 문제를 효과적으로 해결합니다.

**요약**: 5장에서는 rStar-Math의 핵심 요소별 효과를 절삭 연구를 통해 분석합니다. 자기 진화, 단계별 검증 추론 궤적, PPM 모두 rStar-Math 성능 향상에 기여하며, 각 요소의 중요성과 효과를 입증합니다. 특히 PPM은 기존 보상 모델의 한계를 극복하고, rStar-Math의 핵심 경쟁력임을 강조합니다.



## 6. 결론 및 논의 (Findings and Discussions) 상세 요약: rStar-Math의 주요 발견 및 시사점

6장에서는 rStar-Math 연구를 통해 얻은 주요 발견 사항과 논의를 제시합니다. rStar-Math의 핵심적인 특징과 시사점을 강조하며, 향후 연구 방향을 제시합니다.

**6.1. 내재적 자기-반성 능력의 출현 (The emergence of intrinsic self-reflection capability)**

- **OpenAI 01의 핵심 돌파구**: OpenAI 01의 핵심 돌파구 중 하나는 모델이 오류를 인식하고 스스로 수정할 수 있는 내재적 자기-반성 능력입니다. 하지만 오픈 소스 LLM에서는 효과가 미미했습니다.
    
- **rStar-Math의 자기-반성 능력**: rStar-Math의 MCTS 기반 깊이 사고 과정에서 예상치 못한 자기-반성 능력이 나타나는 것을 발견했습니다. 그림 4는 모델이 초기 몇 단계에서 SymPy를 사용하여 방정식을 잘못 공식화하여 오답으로 이어지는 경우(왼쪽 분기)를 보여줍니다. 흥미롭게도 4단계(오른쪽 분기)에서 정책 모델은 이전 단계의 낮은 품질을 인식하고 초기 문제 해결 경로를 따르지 않습니다. 대신, 백트래킹하여 더 간단한 새로운 접근 방식을 사용하여 문제를 해결하고 최종적으로 정답에 도달합니다. 부록 A.2에서 자기 수정의 추가 예시를 제공합니다.
    
- **자기-반성 능력의 의미**: 자기-반성 훈련 데이터나 프롬프트를 사용하지 않았음에도 불구하고, advanced System 2 추론은 내재적인 자기-반성 능력을 키울 수 있음을 시사합니다.
    

**6.2. PPM은 System 2 깊이 사고에서 추론 경계를 형성 (PPM shapes the reasoning boundary in System 2 deep thinking)**

- **정책 모델과 보상 모델의 중요성**: System 2 깊이 사고에서 정책 모델과 보상 모델은 모두 중요합니다. 실험 결과, 정책 모델이 충분히 강력한 수준에 도달하면 PPM이 상한 성능을 결정하는 핵심 요소가 됩니다 (부록 A.1 참조).
    
- **PPM의 결정적인 역할**: 그림 5는 다양한 크기의 정책 모델과 보상 모델을 사용하여 달성한 Pass@1 정확도를 요약합니다. 훈련 전략, 데이터셋, 모델 크기의 차이에도 불구하고, 보상 모델이 System 2 추론에서 지배적인 요소임이 입증되었습니다. rStar-Math-7B의 SFT 정확도가 Qwen2.5-Math-72B-Instruct보다 낮지만, 7B PPM과 결합하면 rStar-Math가 Qwen 72B ORM으로 72B 정책 모델을 능가합니다. 또한, 세 가지 정책 SLM 크기에서 Pass@1 정확도가 다양하게 나타나지만, PPM을 적용한 후 최종 추론 정확도는 수렴합니다.
    

**6.3. PPM은 정리-응용 단계 포착 (PPM spots theorem-application steps)**

- **정리-응용 단계의 중요성**: 어려운 수학 문제 해결에서 관련 정리 또는 핵심 결론을 식별하고 적용하는 것은 성공적인 문제 해결의 핵심 요소입니다 [Xin et al., 2024].
    
- **PPM의 역할**: rStar-Math 문제 해결 과정에서 PPM은 정책 모델의 깊이 사고 과정 내에서 중요한 정리-응용 중간 단계를 효과적으로 식별합니다. 이러한 단계는 높은 보상 점수로 예측되어 정책 모델이 올바른 솔루션을 생성하도록 안내합니다. 부록 A.2는 PPM이 페르마의 소정리, 비에타 공식, AM-GM 부등식, 피타고라스 정리, 신발끈 정리 등 핵심 정리를 성공적으로 식별하는 예시를 제공합니다.
    

**6.4. 일반화 논의 (Generalization discussions)**

- **다양한 영역으로 확장 가능**: rStar-Math는 다양한 영역에 적용 가능한 일반적인 방법론을 제공합니다.
    
    - **더 어려운 수학 문제**: 데이터셋 제한으로 인해 현재는 단어 문제에 초점을 맞추고 있지만, rStar-Math는 정리 증명과 같이 더 어려운 수학 과제로 일반화될 수 있습니다. 부록 A.2에서 페르마의 소정리를 포함하는 올림피아드 수준 문제를 성공적으로 증명하는 것을 보여줍니다.
        
    - **코드 및 상식 추론**: 일반적인 추론을 위한 단계별 검증된 훈련 궤적 합성은 MCTS 롤아웃의 끝에서 주어진 궤적이 원하는 출
        

- **피드백 메커니즘**: 일반적인 추론을 위한 단계별 검증된 훈련 궤적 합성은 MCTS 롤아웃의 끝에서 주어진 궤적이 원하는 출력에 도달했는지에 대한 피드백을 제공하는 메커니즘이 필요합니다. 예를 들어, 코드 추론에서는 광범위한 테스트 케이스 설계가 필요하고, 일반적인 추론에서는 인간 레이블링 또는 다른 LLM과의 상호 검증을 통해 피드백을 얻을 수 있습니다 [Qi et al., 2024].
    

**6.5. 결론 (Conclusion)**

- **rStar-Math의 핵심**: rStar-Math는 작은 SLM의 수학적 추론 능력을 크게 향상시키는 자기 진화적인 System 2 깊이 사고 접근 방식입니다. OpenAI 01 수준의 최첨단 성능을 달성했으며, SLM이 frontier 수준의 수학 추론을 위한 고품질 훈련 데이터를 자체 생성할 수 있음을 입증했습니다.
    
- **주요 성과**: 4가지 크기의 SLM과 어려운 수학 벤치마크에 대한 광범위한 실험을 통해 rStar-Math의 우수성을 입증했습니다. 기존 수학 추론 LLM 및 Best-of-N 기준선을 능가하는 선도적인 결과를 달성했습니다.
    
- **주요 발견**: 자기-반성 능력의 출현, 정리-응용 단계 식별에 효과적인 PPM의 역할 등 중요한 발견들을 제시했습니다.
    
- **향후 연구**: rStar-Math는 더 어려운 수학 문제 (정리 증명 등) 해결, 코드 및 상식 추론과 같은 다른 영역으로 확장될 수 있습니다. 더 어려운 수학 문제를 수집하여 rStar-Math를 더욱 개선할 수 있을 것입니다.
    

**요약**: 6장에서는 rStar-Math 연구의 주요 발견 사항들을 심층적으로 논의하고, rStar-Math가 제시하는 새로운 가능성과 향후 연구 방향을 제시합니다. 특히, 내재적 자기-반성 능력의 발견과 PPM의 중요성을 강조하며, rStar-Math가 LLM 연구 분야에 기여하는 바를 명확히 합니다.