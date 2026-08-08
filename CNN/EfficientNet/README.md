# 1. Paper Information

- Title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- Paper URL: [https://arxiv.org/pdf/1905.11946]

---

# 2. Motivation

## Problem
- 기존의 모델들은 고정된 자원 예산 내에서 개발된 후 정확도를 높이기 위해 임의로 모델 크기를 확장해왔으며, 네트워크의 깊이(depth), 너비(width), 해상도(resolution)라는 세 가지 차원을 균형 있게 조절하는 체계적인 방법에 대한 이해가 부족함
    - 깊이(depth) = 층 수
    - 너비(width) = 채널 수
    - 해상도(resolution) = 입력 이미지의 width, height

## Goal
- 네트워크의 깊이, 너비, 해상도를 고정된 비율로 균형 있게 조절하는 원칙적인 '복합 스케일링(Compound Scaling)' 방법을 정립하여, 자원 제약 조건 내에서 정확도와 효율성을 동시에 극대화함

---

# 3. Core Idea

- **복합 스케일링(Compound Scaling) 방법론**
  - 네트워크의 깊이($d$), 너비($w$), 해상도($r$)를 개별적으로 확장하는 대신, 단일 복합 계수 $\phi$를 사용하여 세 차원을 고정된 비율로 동시에 확장함으로써 기존의 단일 차원 확장보다 훨씬 높은 효율성과 정확도를 달성함.
  - 관련 공식: $d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi$ (단, $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$)

- **효율적인 아키텍처 구현 전략**
  - **기반 네트워크 설계**: 다목적 신경망 구조 탐색(Multi-objective NAS)을 통해 최적의 모바일용 베이스라인인 **EfficientNet-B0**를 설계함.
  - **효율적 연산 블록**: 네트워크의 기본 단위로 모바일 역병목(MBConv) 블록을 채택하고, 여기에 Squeeze-and-Excitation(SE) 최적화를 결합하여 연산 효율을 극대화함.

---

# 4. Model Architecture & Forward Process

## Overall Architecture

![alt text](image.png)

## Components

### MBConv Block (Inverted Residual + SE)
**Purpose**
- 모바일 역병목 구조를 기본 단위로 사용하며, Squeeze-and-Excitation(SE) 모듈을 결합하여 채널 간 의존성을 모델링함으로써 모델의 표현력을 강화함

**Configuration**
- **Expansion (1×1 Conv)**: 입력 채널을 더 높은 차원으로 확장하여 특징 공간을 넓힘
- **Depthwise Conv (3×3 or 5×5)**: 확장된 특징 공간에서 공간적 특징을 효율적으로 추출
- **Squeeze-and-Excitation**: 전체 채널의 중요도를 재계산(Recalibration)하여 유의미한 특징을 강조
- **Pointwise Conv (1×1)**: 채널을 줄여 원래의 차원(또는 타겟 차원)으로 복원하며 잔차 연결(Residual Connection) 수행

**Role**
- 좁은 병목 계층(Bottleneck)을 연결하는 잔차 연결을 통해 학습 효율을 높이고, SE 블록을 통해 네트워크가 중요한 특징에 집중하도록 유도함

**Output**
- (B, $C_{out}$, $H/s$, $W/s$)

### Compound Scaling ($\alpha^\phi, \beta^\phi, \gamma^\phi$)
**Purpose**
- 네트워크의 깊이(Depth), 너비(Width), 해상도(Resolution)를 개별적으로 확장하는 대신, 복합 계수 $\phi$를 도입하여 세 차원을 고정된 비율로 균형 있게 확장함

**Mechanism**
- **Depth ($\alpha^\phi$)**: 계층 수 조정
- **Width ($\beta^\phi$)**: 채널 수 조정
- **Resolution ($\gamma^\phi$)**: 입력 해상도 조정
- **Constraint**: 연산량 증가를 제어하기 위해 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ 조건을 유지하며, 자원 예산 증가에 따라 $\phi$를 결정함

**Role**
- 단일 차원 확장 시 발생하는 정확도 포화 문제를 해결하고, 네트워크의 모든 차원이 상호 보완적으로 성장하도록 하여 자원 효율성 내에서 모델의 성능을 극대화함

## Forward Process

### 1. Input
(B, 3, $224 \cdot r$, $224 \cdot r$)

### 2. Stem Layer
(B, 3, $224 \cdot r$, $224 \cdot r$) <br>
↓ Conv3×3 (stride 2) + BN + Swish <br>
(B, 32 $\cdot w$, $112 \cdot r$, $112 \cdot r$)

### 3. MBConv Blocks (Scaling stages)
복합 스케일링 규칙에 따라 조정된 필터 수와 반복 횟수($L$)를 가진 MBConv 블록들을 순차적으로 통과함 <br>
(모든 MBConv 블록은 Squeeze-and-Excitation(SE)을 포함하며 Swish 활성화 함수를 사용)

### 4. Efficient Head
특징을 요약하고 최종 차원으로 투영하는 단계 <br>
↓ Conv1×1 (확장) + BN + Swish <br>
↓ Global Average Pooling (GAP) <br>
↓ Fully Connected(FC) Layer (최종 출력층) <br>
(B, 1280 $\cdot w$)

### 5. Classification
↓ Softmax <br>
(B, 1000)

*   **참고**: EfficientNet은 모든 레이어에서 활성화 함수로 Swish를 일관되게 사용

---

# 5. Mathematical Explanation (New Ideas)

---

# 6. Training Configuration

| Item | Value |
|------|-------|
| Optimizer | RMSProp |
| Decay | 0.9 |
| Momentum | 0.9 |
| Weight Decay | 1e-5 |
| Initial Learning Rate | 0.256 |
| Learning Rate Decay | 0.97 every 2.4 epochs |
| Batch Norm Momentum | 0.99 |
| Activation | SiLU(Swish-1) |

## Notes
- **Regularization**: 
    - 모델 규모에 따라 Dropout 비율을 선형적으로 증가시킴 (B0의 0.2에서 B7의 0.5까지).
    - Stochastic Depth(Survival probability 0.8)를 적용하여 과적합을 방지함.
        - 20% 확률로 Layer을 죽임
        - Layer가 살았을 때는 Residual Skip Connection 수행
            - y = F(x) + x
        - Layer가 죽었을 때는 Skip Connection 수행
            - y = x
- **Early Stopping**: 훈련 데이터셋 중 25K를 별도의 'minival' 세트로 구성하여 조기 종료(Early Stopping)를 수행함.

## Data Preprocessing
| Item | Description |
|------|-------------|
| Preprocessing | 학습 시 입력 해상도는 각 모델별 스케일링 설정에 따름 (예: B0는 224x224, B7은 600x600) |
| Augmentation | 학습 시 일반화 성능 향상을 위해 AutoAugment 정책을 기본적으로 적용함 |

# 7. Implementation


## Directory Structure

EfficientNet/
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

- **왜 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ 인가?**
    - 컨볼루션 연산량은 깊이($d$)에는 선형 비례하나, 채널 너비($w$)와 해상도($r$)에는 각각 제곱에 비례함($w^2, r^2$). 리소스가 $2^\phi$만큼 늘어날 때 연산량도 동일하게 $2^\phi$만큼 증가시키기 위해 너비와 해상도의 증가 비율을 제곱으로 제한한 것임.

- **깊이, 너비, 해상도를 왜 셋 다 키워야되는가?**
    - 해상도가 224*224 -> 1000*1000으로 커졌다면
        - 깊이도 커져야됨 -> 그래야 receptive field도 커져서 큰 해상도에 있는 많은 정보를 뽑아 먹을 수 있으니까
        - 너비도 커져야됨 -> 정보가 많은 만큼 필터의 갯수도 많아야 더 많은 특징을 뽑을 수 있으니까 

- **왜 Swish(SiLU) 활성화 함수를 사용하는가?**
    - ReLU와 달리 0 이하의 구간에서도 부드러운 기울기를 유지하여, 깊은 네트워크에서 기울기 소실(Gradient Vanishing)을 억제하고 표현력을 극대화하기 위해 채택됨.

# 9. Personal Insights

- 논문에서는 Stochastic Depth 확률을 0.8로 고정하지만, 최근 구현들에서는 Block이 깊어질수록 Stochastic Depth 확률을 선형적으로 증가시킴(0 ~ 0.8)
    - 모델의 앞 부분은 역전파 시 그래디언트 전달이 뒷 부분 보다 약한 것을 고려한 전략

---