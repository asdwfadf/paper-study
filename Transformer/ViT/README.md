# 1. Paper Information

- Title: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
- Paper URL: [https://arxiv.org/pdf/2010.11929]

---

# 2. Motivation

## Problem

- 컴퓨터 비전 분야에서 CNN 구조의 독점과 한계
    - 기존의 비전 모델들은 컨볼루션(convolutional) 신경망을 기반으로 한 구조가 지배적이며, 이는 현대의 표준으로 자리 잡았습니다.
    - 이러한 CNN 기반 접근은 지역적 특징 추출과 이동 불변성(translation equivariance)이라는 귀납적 편향(inductive bias)에 크게 의존합니다.
    - 최근 연구들에서는 CNN과 셀프 어텐션(self-attention)을 결합하거나 컨볼루션을 대체하려는 시도가 있었으나, 특수한 어텐션 패턴을 사용함에 따라 현대적인 하드웨어 가속기에서 효율적으로 확장되지 못했습니다.

## Goal

- CNN 의존성 탈피 및 순수 Transformer 적용
    - CNN에 대한 의존이 필수가 아님을 증명하고, 이미지 패치 시퀀스에 직접적으로 Transformer를 적용하여 우수한 이미지 분류 성능을 달성하고자 합니다.
    - 대규모 데이터셋으로 사전 학습(pre-training)된 모델을 여러 벤치마크 데이터셋으로 전이(transfer)했을 때, 최신 CNN 모델들과 비교하여 동등하거나 더 우수한 성능을 내면서도 학습에 필요한 연산 자원을 획기적으로 줄이는 것을 목표로 합니다.

## Core Idea

- 스케일링을 통한 귀납적 편향 극복
    - 이미지 데이터를 NLP의 단어 토큰처럼 패치(patch) 단위로 분할하고, 이를 Transformer의 표준 구조에 입력하여 모델이 이미지의 공간적 구조를 스스로 학습하도록 설계했습니다.
    - 소규모 데이터셋에서는 CNN이 가진 귀납적 편향이 중요할 수 있으나, 대규모 데이터셋(14M~300M 이미지)으로 학습할 경우 데이터에서 직접 패턴을 학습하는 방식이 귀납적 편향을 능가함을 제시했습니다.
    - 결과적으로, 이미지의 2차원적 구조에 대한 사전 지식을 최소화하고 대규모 데이터 스케일링을 통해 모델의 성능을 극대화하는 범용적인 구조를 실현했습니다.

---


# 3. Model Architecture & Forward Process

## Overall Architecture

![alt text](image.png)
![alt text](image-1.png)

## Components

### Transformer Encoder
**Purpose**
- 일련의 Transformer 인코더 블록을 쌓아 구성하며, '멀티 헤드 셀프 어텐션'과 'MLP 블록'을 통해 이미지 패치 간의 복잡한 의존성을 병렬적으로 학습함.

**Configuration**
- **Multi-Head Self-Attention (MSA)**: 패치 임베딩을 여러 헤드로 나누어 처리함으로써 이미지 내의 먼 거리 패치들 간의 관계를 입체적으로 파악함.
- **MLP Block**: 각 패치마다 독립적인 2단계 선형 변환과 GELU 활성화 함수를 적용하여 특징을 정제함.
- **Residual Connection & Layer Norm**: 각 서브 레이어마다 잔차 연결을 수행하고 레이어 정규화(Layer Normalization)를 적용하여 깊은 네트워크의 학습 안정성을 확보함.

**Role**
- 잔차 연결을 통해 정보 손실 없이 깊은 층의 학습을 지원하고, 병렬적 어텐션 구조를 통해 컨볼루션 연산의 지역적 한계를 극복함.

**Output**
- 인코더: (B, $N+1$, $D$)
  - (B: 배치 사이즈, N: 패치 개수, D: 임베딩 차원, +1은 [class] 토큰)

### Positional Embedding
**Purpose**
- 모델 구조 내에 고정된 공간적 순서 정보가 없으므로, 패치들의 위치 정보를 모델에 학습 가능한 형태로 주입함.

**Mechanism**
- 표준적인 1차원 학습 가능(learnable) 위치 임베딩을 패치 임베딩에 더해줌. 2차원 공간 정보를 위해 별도의 복잡한 인코딩을 시도했으나, 1차원 학습 가능 임베딩과 유의미한 성능 차이가 없음을 확인함.

**Role**
- 패치가 이미지 내 어디에 위치하는지에 대한 정보를 유지함으로써 모델이 공간적 구조를 이해할 수 있게 함.

## Forward Process

### 1. Patch Embedding
- 입력 이미지를 $P \times P$ 크기의 패치로 분할하고 평탄화하여 선형 투영을 수행함.
- [class] 토큰과 Position Embedding을 결합하여 Transformer 입력 형태로 변환함.
- **Output Shape**: $(B, N+1, D)$

### 2. Encoder Stack
- $L$개의 Transformer Encoder 블록을 통과하며 Self-Attention과 MLP를 수행함.
- 각 블록마다 입력과 동일한 크기의 시퀀스 정보를 유지함.
- **Output Shape**: $(B, N+1, D)$

### 3. Classification Head
- 인코더의 마지막 출력에서 맨 앞에 위치한 [class] 토큰($z_L^0$)만 추출함.
- 이 벡터를 최종 클래스 수($K$)에 맞춰 선형 변환함.
- **Output Shape**: $(B, K)$

### 4. Final Output
- Softmax를 적용하여 각 클래스에 대한 확률 분포를 계산함.
- **Output Shape**: $(B, K)$

---

# 4. Mathematical Explanation (New Ideas)

---

# 5. Training Configuration

### 훈련 하이퍼파라미터 및 데이터 전처리
| 분류 | 항목 | 설정 및 상세 설명 |
| :--- | :--- | :--- |
| **Optimization** | Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| | Learning Rate | 선형 Warmup(10,000 steps) 후 선형 감쇠(decay) 적용 |
| | Regularization | Weight Decay (0.1), Dropout (종료 시 0.1) |
| **Data/Prep** | Datasets | ImageNet (1k), ImageNet-21k, JFT-300M |
| | Preprocessing | 이미지 패치 분할 ($16 \times 16$ 또는 $14 \times 14$) |
| | Batching | 대규모 병렬 처리를 위해 배치 사이즈 4096 적용 |
| **Hardware** | TPU | Google TPUv3 가속기 사용 |
| **Fine-tuning** | Strategy | SGD (momentum 0.9), 고해상도 미세 조정(384) |

## Notes
- **Learning Rate Schedule**: 학습 초기 모델의 불안정을 방지하기 위해 10,000 스텝 동안 선형적으로 학습률을 증가(Warmup)시키며, 이후에는 선형 감쇠 방식을 사용하여 안정적인 수렴을 유도함.
- **Regularization**: 
    - **Weight Decay**: 데이터셋 규모가 클수록 높은 가중치 감쇠(0.1)를 적용하는 것이 모델 전이(transfer) 성능에 중요하게 작용함.
    - **Label Smoothing**: 사전 학습 단계에서 모델의 과잉 확신을 방지하고 일반화 성능을 높이기 위해 레이블 스무딩 기법을 활용함.
- **Resolution Adaptation**: 사전 학습(224 해상도) 후 미세 조정 단계에서 더 높은 해상도(384 해상도)로 조정하며, 이때 기존 위치 임베딩(position embedding)을 2D 보간(interpolation)하여 적용함.
- **Checkpointing & Averaging**: 최종 모델 성능 향상을 위해 체크포인트 평균화 기법을 적용할 수 있으며, 특히 ImageNet 미세 조정 시 Polyak-Juditsky 평균화를 사용하여 성능을 안정화함.
- **Inference Strategy**: 학습된 모델의 최종 [class] 토큰을 사용하여 클래스를 분류하며, 계산 효율을 위해 입력 해상도에 따른 배치 사이즈를 최적화하여 연산함.

---


# 6. Implementation

## Directory Structure

ViT/
├── README.md
├── model.py
└── main.py

## Model Implementation

- **Vision Transformer Architecture**
  - PyTorch를 사용하여 논문의 구조를 Scratch 구현.
- **Learnable Positional Embedding**
  - 논문에서 제시한 대로 `nn.Parameter`를 사용하여 위치 정보를 학습 가능한 가중치로 구현.
  - 고정된 `sin`, `cos` 함수가 아닌 데이터에 따라 최적화되는 방식 채택.
- **Multi-Head Self-Attention**
  - Query, Key, Value를 각각 Linear Layer로 투영(Projection).
  - `einops.rearrange`를 사용하여 병렬 헤드 처리를 위한 차원 재구성.
  - Scaled Dot-Product Attention 계산 후 Softmax 적용.
- **Encoder Block**
  - Pre-norm 구조 채택.
  - Multi-Head Self-Attention + Residual Connection.
  - MLP Block (GELU 활성화 함수 사용) + Residual Connection.
- **Feed-Forward Network (MLP Block)**
  - `Linear → GELU → Dropout → Linear → Dropout` 구조.
  - 중간 차원을 `mlp_size` (기본 3072)로 확장하여 표현력 확보.
- **Token Embedding**
  - `nn.Conv2d`를 사용하여 패치 분할 및 선형 투영(Linear Projection) 동시 수행.
  - [CLS] 토큰을 시퀀스 맨 앞에 결합하여 이미지 전체 문맥 정보 집약.
- **Weight Initialization**
  - `nn.init.trunc_normal_` (std=0.02)를 사용하여 가중치 및 위치 임베딩 초기화.
  - Layer Normalization의 bias는 0, weight는 1로 초기화하여 학습 안정성 확보.

## Verification

| Item             | Result |
| ---------------- | ------ |
| Input Shape      | `[1, 3, 224, 224]` |
| Output Shape     | `[1, 1000]` |
| Total Parameters | 86,566,120 |
| FLOPs | 17581983744 |

## Notes

- 랜덤 데이터를 입력으로 사용하여 순전파 검증

---

# 7. Analysis & Insights

## Merits

- **대규모 학습의 효율성**: CNN의 귀납적 편향(Inductive Bias)에 의존하는 대신, 대규모 데이터셋(JFT-300M 등)에서의 사전 학습을 통해 성능의 포화점을 극복함. 스케일링 법칙(Scaling Law)이 비전 태스크에서도 유효함을 입증함.
- **전역적 정보 통합**: 초기 레이어부터 이미지 전체를 조망하는 전역적 어텐션(Global Attention)을 통해, CNN이 수많은 층을 거쳐야 도달하는 수용장(Receptive Field)을 즉각적으로 확보함.
- **연산 자원 최적화**: 최신 가속기(TPU/GPU)에서 병렬 연산에 최적화된 Transformer 구조를 채택함으로써, 기존 SOTA CNN 모델들보다 상대적으로 적은 연산 비용으로 더 높은 정확도를 달성함.
- **범용적 구조**: 이미지 패치를 텍스트 토큰과 동일한 시퀀스로 처리함으로써, 향후 멀티모달(Vision + Language) 모델로의 확장 가능성을 열어줌.

## Demerits

- **데이터 갈증(Data Hunger)**: CNN이 가진 이동 불변성 등 강력한 귀납적 편향이 없기 때문에, 소규모 데이터셋(ImageNet 등)에서는 성능이 현저히 낮으며 학습 초기 파라미터 업데이트에 매우 민감함.
- **해상도 및 패치 크기 제약**: 입력 시퀀스 길이가 이미지 패치 크기의 제곱에 반비례하여 증가하므로, 고해상도 이미지를 입력할 경우 메모리 및 연산 복잡도가 급격히 상승함($O(n^2)$).
- **입력 구조의 이질성**: CNN과 달리 2차원 공간 정보를 내재하지 않아, 이를 보완하기 위한 위치 임베딩(Positional Embedding) 설계와 2D 보간(Interpolation) 기술이 모델의 핵심 성능을 좌우함.

## why?

- ViT는 왜 패치(Patch) 단위로 이미지를 처리해야 할까?
  - 픽셀 단위로 직접 어텐션을 적용할 경우 픽셀 개수의 제곱($O(H^2W^2)$)만큼 연산량이 폭증하여 현대 하드웨어에서 학습이 불가능하기 때문임.
  - 패치 분할은 이미지의 고해상도 정보는 유지하면서 연산량을 관리 가능한 시퀀스 길이로 압축하는 필수적인 타협점임.

- 어텐션 맵(Attention Map)의 해석 가능성 (왜 이 영역을 보고 있을까?)
  - ViT의 어텐션 맵을 시각화하면, 모델이 단순히 픽셀을 보는 것이 아니라 사물의 핵심적인 형태나 의미론적(Semantic)으로 중요한 영역을 스스로 탐색함을 알 수 있음.
  - 학습 초기에는 전역적인 정보를 흩뿌리듯 참조하지만, 학습이 진행될수록 물체의 특징적인 부분에 가중치를 집중시키는 정교함을 보임.
---