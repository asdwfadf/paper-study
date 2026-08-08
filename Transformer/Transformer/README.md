# 1. Paper Information

- Title: Attention Is All You Need
- Paper URL: [https://arxiv.org/pdf/1706.03762]

---

# 2. Motivation

## Problem

- 기존 RNN의 한계

    ![alt text](image.png)

    - 역전파 시 멀수록 잊혀짐
        - tanh의 최대 미분값은 1이므로 멀수록 1 이하의 값이 계속 곱해짐
    
    - 순전파 시 갈수록 정보가 뭉개짐
        - tanh의 출력값 (-1 ~ 1) 때문에 마지막에 가면 첫 단어의 정보는 거의 남아있지 않음
        - 한/영 번역 문제라면 마지막 단어를 가장 집중해서 번역문을 생성함

    - LSTM과 GRU도 근본적인 문제를 해결하진 못함

## Goal

- 기존 RNN 기반의 인코더-디코더 모델이 가진 정보 압축의 한계(병목 현상)를 극복하고, 장거리 의존성 문제 해결

- 고정된 컨텍스트 벡터가 아닌 **어느 토큰에 집중해야 하는지를 모델이 스스로 학습**하도록 설계하여 기계 번역의 정확도와 연산 효율성을 극대화

---

# 3. Core Idea

- 순환 구조를 아예 없애고 '셀프 어텐션(Self-attention)' 메커니즘만을 사용하여 입력과 출력 간의 전역적 의존성을 직접적으로 계산함으로써 계산 효율성과 모델 성능을 동시에 극대화

---

# 4. Model Architecture & Forward Process

## Overall Architecture



## Components

### Encoder and Decoder Stacks
**Purpose**
- N=6개의 동일한 층을 쌓아 구성하며, 각 층은 '멀티 헤드 셀프 어텐션'과 '위치 기반 피드 포워드 네트워크'를 통해 시퀀스 내의 복잡한 의존성을 병렬적으로 학습함.

**Configuration**
- **Multi-Head Self-Attention**: 쿼리(Q), 키(K), 값(V)을 여러 표현 공간으로 투영하여 병렬 처리함으로써 시퀀스 내 토큰 간 관계를 입체적으로 파악함.
- **Position-wise Feed-Forward**: 각 위치별로 독립적인 2단계 선형 변환과 비선형 활성화 함수(ReLU)를 적용하여 특징을 정제함.
- **Residual Connection & Layer Norm**: 각 서브 레이어마다 잔차 연결을 수행하고 정규화를 적용하여 깊은 네트워크에서도 기울기 소실 문제를 방지함.

**Role**
- 잔차 연결을 통해 정보의 손실 없는 깊은 층의 학습을 지원하고, 병렬적 어텐션 구조를 통해 RNN의 순차적 한계를 극복함.

**Output**
- 인코더: (B, $n$, $d_{model}$)
- 디코더: (B, $m$, $d_{model}$)

### Positional Encoding
**Purpose**
- 모델 내에 순환(Recurrence)이나 합성곱(Convolution) 구조가 없으므로, 시퀀스의 상대적 또는 절대적 위치 정보를 모델에 주입함.

**Mechanism**
- 사인(Sine)과 코사인(Cosine) 함수를 사용하여 위치별 고유한 임베딩 벡터를 생성한 후, 이를 토큰 임베딩에 더해줌<BR>
($PE(pos, 2i) = \sin(pos/10000^{2i/d_{model}})$ 등).

**Role**
- 토큰이 문장 내 어디에 위치하는지에 대한 정보를 유지함으로써 모델이 순서 정보를 이해할 수 있게 함.

## Forward Process

### 1. Input Embedding
- 입력 토큰을 벡터($d_{model}=512$)로 변환하고 Positional Encoding을 합산함.

### 2. Encoder Stack
- 입력 시퀀스에 대해 Multi-Head Self-Attention을 수행하여 각 위치별 문맥 정보를 추출함.
- N개의 층을 통과하며 입력 토큰들 사이의 전역적인 의존성을 계산함.

### 3. Decoder Stack
- 이전 단계의 출력(Masked Self-Attention)과 인코더의 출력(Encoder-Decoder Attention)을 결합하여 다음 토큰을 예측함.
- **Masked Self-Attention**: 디코더가 미래의 토큰을 참조하지 못하도록 시점 $i$ 이후의 정보를 $-\infty$로 처리(Masking)함.

### 4. Final Output
- **Linear Layer**: 모델의 출력을 타겟 어휘 사전 크기만큼의 벡터로 투영함.
- **Softmax**: 투영된 벡터를 확률 분포로 변환하여 다음 토큰의 예측값을 생성함.

---

# 5. Mathematical Explanation (New Ideas)

- **Scaled Dot-Product Attention의 수학적 필요성**
  - **내적 값의 분산 증폭**: 
    입력 벡터 $q$와 $k$의 각 성분이 평균 0, 분산 1을 갖는 독립 확률 변수라고 가정할 때, 이들의 내적 $X = \sum_{i=1}^{d_k} q_i k_i$의 분산은 다음과 같이 차원 수 $d_k$에 비례하여 커짐.
    $$Var(X) = Var\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} Var(q_i k_i) = d_k \cdot 1 = d_k$$
  
  - **기울기 소실(Gradient Vanishing) 문제**:
    $d_k$가 커지면 내적 값의 범위가 극단적으로 넓어짐. 소프트맥스 함수는 입력이 매우 크거나 작을 때, 출력값을 0 또는 1로 강하게 편향시키며 이 구간의 기울기는 0에 수렴함. 결과적으로 모델 학습을 위한 역전파(Backpropagation) 시 기울기 소실 문제가 발생함.

  - **스케일링을 통한 분산 제어**:
    내적 값을 $\sqrt{d_k}$로 나누어주면(표준편차로 나눔), 새로운 변수 $X' = \frac{X}{\sqrt{d_k}}$의 분산은 항상 1이 됨.
    $$Var\left(\frac{X}{\sqrt{d_k}}\right) = \frac{1}{d_k} Var(X) = \frac{d_k}{d_k} = 1$$
    이 과정을 통해 소프트맥스의 입력 분포를 적절한 범위 내로 유지하여, 학습 과정에서 안정적인 기울기를 확보함.

- **시간 복잡도 및 효율성 분석**
  - 시퀀스 길이 $n$, 표현 차원 $d$, 커널 크기 $k$에 대하여, 층당 계산 복잡도를 다음과 같이 비교함.

  | Layer Type | Complexity per Layer | Sequential Operations | Max Path Length |
  | :--- | :--- | :--- | :--- |
  | **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
  | **Recurrent** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
  | **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |

- **복잡도 공식 상세 도출**
  1. **Self-Attention ($O(n^2 \cdot d)$)**: 
     - 어텐션 스코어 행렬($QK^T$) 연산이 $(n \times d) \times (d \times n) = n^2 \cdot d$의 복잡도를 가짐. 시퀀스 길이 $n$의 제곱에 비례하지만, 모든 토큰을 병렬로 처리하여 순차적 연산(Sequential Operations)이 $O(1)$임.
  2. **Recurrent ($O(n \cdot d^2)$)**: 
     - 각 타임스텝마다 $(d \times d)$ 행렬과 벡터의 곱을 $n$번 수행함. $d$의 제곱에 비례하며, 이전 상태에 의존해야 하므로 순차적 연산이 $O(n)$으로 병렬화가 불가능함.
  3. **Convolutional ($O(k \cdot n \cdot d^2)$)**: 
     - 커널 크기 $k$에 비례하여 계산량이 증가함. 병렬 처리는 가능하나 장거리 의존성을 위해 계층을 깊게 쌓아야 하며 이로 인해 정보 전달 경로가 길어짐.

- **핵심 통찰 (Transformer의 우위)**
  - **모델 확장성**: 모델의 성능을 높이기 위해 차원 $d$를 키울 때, RNN은 $O(d^2)$의 비용을 지불해야 하지만, Transformer는 $d$에 선형적인 비용($O(d)$)만 발생함. 이는 **Transformer가 RNN보다 훨씬 더 큰 규모의 모델을 효율적으로 학습할 수 있음을 수학적으로 시사함.**
  - **학습 효율성**: 일반적으로 $n < d$인 환경(대부분의 번역 태스크)에서 $O(n^2 \cdot d) < O(n \cdot d^2)$가 성립하므로, Transformer가 RNN보다 실질적인 학습 속도가 더 빠름.

---

# 6. Training Configuration

### 훈련 하이퍼파라미터 및 데이터 전처리
| 분류 | 항목 | 설정 및 상세 설명 |
| :--- | :--- | :--- |
| **Optimization** | Optimizer | Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) |
| | Learning Rate | Warmup (4,000 steps) 후 역제곱근 감쇠 적용 |
| | Regularization | Label Smoothing ($\epsilon_{ls}=0.1$), Dropout ($P_{drop}=0.1$) |
| **Data/Prep** | Datasets | WMT 2014 (EN-DE 4.5M / EN-FR 36M) |
| | Vocabulary | BPE (37K) / Word-piece (32K) 공유 어휘 |
| | Batching | 길이 기반 묶음 (배치당 약 25,000 토큰) |
| **Hardware** | GPU | 8 NVIDIA P100 GPUs |
| **Inference** | Beam Search | Size 4, Length Penalty ($\alpha=0.6$) |

## Notes
- **Learning Rate Schedule**: 학습 초기 안정성을 확보하기 위해 첫 4,000 스텝 동안 학습률을 선형적으로 증가(Warmup)시키고, 이후에는 스텝 수의 역제곱근($step\_num^{-0.5}$)에 비례하여 감소시켜 수렴 성능을 최적화함.
- **Regularization**: 
    - **Residual Dropout**: 모델의 과적합을 방지하기 위해 각 서브 레이어의 출력과 임베딩 합산 후 드롭아웃($P_{drop}=0.1$, 단 빅 모델 일부 0.3)을 적용함.
    - **Label Smoothing**: 모델이 특정 토큰에 대해 과잉 확신을 갖는 것을 억제하고, 학습 시 불확실성을 고려하여 BLEU 점수와 모델의 일반화 성능을 향상함.
- **Checkpoint Averaging**: 추론 시 더 안정적인 성능을 얻기 위해 마지막 체크포인트들을 평균화하여 사용함. 베이스 모델은 마지막 5개, 빅 모델은 마지막 20개를 사용함.
- **Inference Strategy**: 번역 시 입력 길이 + 50을 최대 출력 길이로 제한하며, 가능한 경우 조기 종료(Early termination)하여 불필요한 계산을 방지함.

---


# 7. Implementation


## Directory Structure

Transformer/
├── README.md
├── model.py
└── main.py

## Model Implementation

- EfficientNet-B0 Architecture를 PyTorch로 Scratch 구현
- Residual Connection은 `stride=1`이고 입력/출력 채널 수가 동일한 경우에만 적용
- Stochastic_depth는 Residual Connection 적용할 때만 적용
    - Residual Connection이 없을 때 적용하면 출력값이 0이 되기 때문에 학습이 진행은 되지만 망함
        - xW+b = 0W+b = b
            - b 값만 가지고 학습 진행됨 ..
- Weight Initialization
  - Convolution Layer : Kaiming Normal Initialization (`fan_out`)
  - Linear Layer : Uniform Initialization (`±1 / √out_features`)
  - Bias : 0으로 초기화

## Verification

| Item | Result |
|------|--------|
| Input Shape | [1, 3, 224, 224] |
| Output Shape | [1, 1000] |
| Total Parameters | 5,288,548 |
| FLOPs | 421,872,480 |

## Notes

- 랜덤값을 입력으로 사용
- 파라미터 수 원문과 동일

---

# 8. Analysis

## Merits

- **스케일링의 체계화**: 기존의 감에 의존한 모델 확장을 넘어, 복합 계수 $\phi$를 도입하여 깊이, 너비, 해상도를 수학적 공식으로 정량화함으로써 모델 확장 과정을 엔지니어링의 영역으로 정립함.

- **압도적인 효율성**: 기존의 대규모 모델(GPipe 등) 대비 훨씬 적은 파라미터와 FLOPS로 더 높은 정확도를 달성하여, 실용적 측면에서 자원 제약 환경에 최적화된 모델을 제시함.

- **범용적 전이 학습 성능**: ImageNet에서 검증된 복합 스케일링 기법이 CIFAR, Flowers 등 다양한 데이터셋의 전이 학습에서도 일관되게 높은 성능을 보임을 증명하여 아키텍처의 확장성과 범용성을 입증함.

## why?

- 왜 RNN에서 Sigmoid랑 ReLU 안쓰고 tanh 쓸까?
- Masked Self-Attention가 왜 필요할까? 미래 정보를 왜 못 보게 해야될까?

# 9. Personal Insights

- 기존 RNN의 문제로 Vanishing Gradient를 많이들 얘기하는데 멀리 있는 단어가 잘 고려가 안될 뿐 Vanishing Gradient는 아닌 것 같음
    - h1, h2, h3, h4, h5 가 있을 때 Gradient는 h1, h2, h3, h4, h5 의 미분값을 모두 더하기 때문에 h1의 미분값은 작은게 맞으나 Gradient는 작지 않음

---