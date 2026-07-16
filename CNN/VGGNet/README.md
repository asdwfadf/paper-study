# 1. Paper Information

- Title: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
- Paper URL: [https://arxiv.org/pdf/1409.1556]

---

# 2. Motivation

<!--
왜 이 논문이 등장했는가?
기존 연구의 문제점은 무엇이었는가?
이 논문이 해결하려는 문제는 무엇인가?
예를 들어
- Accuracy가 부족했다.
- 학습이 어려웠다.
- 계산량이 너무 컸다.
- 데이터가 부족했다.
-->

## Problem
- 기존 CNN은 큰 Convolution Filter와 큰 Stride를 사용하여 초기 단계에서 많은 정보 손실이 발생했다.
- 네트워크 깊이가 증가하면 성능이 향상되는지에 대한 체계적인 분석이 부족했다.

## Goal
- 네트워크 깊이가 성능에 미치는 영향을 분석한다.

---

# 3. Core Idea

<!--
논문의 핵심 아이디어를
3~5줄 정도로 요약
"이 논문의 가장 중요한 기여가 무엇인가?"
를 적는다.
-->
- 3×3 Convolution을 쌓아 큰 Receptive Field를 효율적으로 확보한다.
- Layer을 깊게 쌓아 표현력을 향상시킨다.
- Max Pooling을 통해 필요한 시점에서만 해상도를 줄인다.

---

# 4. Model Architecture & Forward Process

## Overall Architecture

- ![alt text](image.png)

---

## Components

### Conv3-out_c

**Purpose**
- 이미지의 지역적인 특징을 추출

**Configuration**
- Kernel Size : 3×3
- Stride : 1
- Padding : 1

**Role**
- 작은 Kernel을 반복적으로 사용하여 깊은 표현을 학습
- 이미지의 해상도는 유지

**Output**
- (B, C, H, W) → (B, out_c, H, W)

### maxpool

**Purpose**
- Feature Map의 해상도(Height, Width)를 감소시켜 연산량과 메모리 사용량을 줄임
- 중요한 특징을 유지하면서 불필요한 공간 정보를 제거

**Configuration**
- Kernel Size : 2×2
- Stride : 2
- Padding : 0

**Role**
- 각 2×2 영역에서 가장 큰 활성값만 출력
- Feature Map의 크기를 절반으로 줄임
- Receptive Field를 증가시킴

**Output**
- (B, C, H, W) → (B, C, H/2, W/2)

### FC Layer

**Purpose**
- 추출한 특징을 하나의 벡터로 변환
- 각 클래스에 대한 Logit을 계산하여 분류를 위한 입력을 생성

**Configuration**
- Input : 7 × 7 × 512 = 25,088
- FC1 : 4096
- FC2 : 4096
- FC3 : 1000 (ImageNet Classes)
- Activation : ReLU (FC1, FC2)
- Dropout : 0.5 (FC1, FC2)

**Role**
- Flatten된 Feature Vector를 입력받아 이미지의 의미를 학습
- FC1, FC2를 통해 복잡한 Feature 조합을 학습
- FC3에서 각 클래스에 대한 Logit 출력

**Output**
- (B, 512, 7, 7)
    ↓ Flatten
- (B, 25088)
    ↓ FC1
- (B, 4096)
    ↓ FC2
- (B, 4096)
    ↓ FC3
- (B, 1000)

### Softmax

**Purpose**
- FC Layer에서 출력된 Logits를 확률 분포로 변환

**Role**
- 각 클래스의 예측 확률 계산
- 확률의 총합은 1이 되며, 가장 높은 확률을 가진 클래스를 최종 예측으로 선택하게 됨

---

## Forward Process

1. Input
(B, 3, 224, 224)

2. conv3-64 + maxpool
(B, 3, 224, 224)
↓ conv + ReLU
(B, 64, 224, 224)
↓ conv + ReLU
(B, 64, 224, 224)
↓ maxpool
(B, 64, 112, 112)

3. conv3-128 + maxpool
(B, 64, 112, 112)
↓ conv + ReLU
(B, 128, 112, 112)
↓ conv + ReLU
(B, 128, 112, 112)
↓ maxpool
(B, 128, 56, 56)

4. conv3-256 + maxpool
(B, 128, 56, 56)
↓ conv + ReLU
(B, 256, 56, 56)
↓ conv + ReLU
(B, 256, 56, 56)
↓ conv + ReLU
(B, 256, 56, 56)
↓ maxpool
(B, 256, 28, 28)

5. conv3-512 + maxpool
(B, 256, 28, 28)
↓ conv + ReLU
(B, 512, 28, 28)
↓ conv + ReLU
(B, 512, 28, 28)
↓ conv + ReLU
(B, 512, 28, 28)
↓ maxpool
(B, 512, 14, 14)

6. conv3-512 + maxpool
(B, 512, 14, 14)
↓ conv + ReLU
(B, 512, 14, 14)
↓ conv + ReLU
(B, 512, 14, 14)
↓ conv + ReLU
(B, 512, 14, 14)
↓ maxpool
(B, 512, 7, 7)

7. FC-4096
(B, 512, 7, 7)
↓ flatten
(B, 25088)
↓ FC + ReLU + Dropout
(B, 4096)

8. FC-4096
(B, 4096)
↓ FC + ReLU + Dropout
(B, 4096)

9. FC-1000
(B, 4096)
↓ FC
(B, 1000)

10. soft-max
(B, 1000)
↓ softmax
(B, 1000)

---

# 5. Mathematical Explanation

<!--
논문의 수식을 하나씩 설명

단순히 수식을 적지 말고

왜 사용하는지

각 기호의 의미

직관적인 의미

Gradient는 어떻게 흐르는지

예시

L = CrossEntropy(...)

↓

왜 Cross Entropy를 사용하는가?

↓

수식이 의미하는 것은?
-->

---

# 6. Training Configuration

| Item | Value |
|------|-------|
| Dataset | ImageNet ILSVRC-2012 |
| Augmentation | Horizontal Flipping, RGB Color Shift |
| Input Size | 224 × 224 |
| Optimizer | mini-batch SGD |
| Batch Size | 256 |
| Epochs | 74 |
| Initial LR | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 5e-4 |
| Dropout | 0.5 |
| Loss | Cross Entropy |
| Initial weight | N(0, 0.01) |
| Initial bias | 0 |

- 논문에서는 Validation Accuracy가 더 이상 개선되지 않을 때 LR을 1/10으로 줄임

## Data Preprocessing

**Training**
- Resize (shorter side = 256)
- Random Crop (224 × 224)
- Random Horizontal Flip
- PCA-based RGB Color Augmentation

**Validation / Test**
- Resize (shorter side = 256)
- Center Crop (224 × 224)

**Multi-Scale Training**
- 학습 데이터의 다양성을 늘리기 위해 Resize Scale ∈ [256, 512] 로 변경한 실험도 했음

---

# 7. Implementation

<!--
논문를 Scratch로 구현한 내용을 작성합니다.

- 구현한 파일 구조
- 구현 과정
- 논문과 다르게 구현한 부분
- Random Input으로 Forward Pass 검증
- Tensor Shape 검증
- Parameter 개수 확인

코드 자체보다는 구현 과정과 검증 내용을 중심으로 작성합니다.
-->

## Directory Structure

VGGNet/
├── README.md
├── model.py
└── train.py

## Model Implementation

- extract_features와 classifier를 각각 `nn.Sequential`로 구현
    - extract_features: 13-Convolution Layer
    - classifier: 3-Fully Connected Layer
- 본 구현에서는 `nn.AdaptiveAvgPool2d(7, 7)`를 추가해 입력 크기와 관계없이 마지막 Feature Map의 크기를 7×7로 맞춘 후 Fully Connected Layer를 적용하도록 구현함
- Weight Initialization은 논문과 동일하게 Gaussian Distribution N(0,0.01) 적용함

## Verification

| Item | Result |
|------|--------|
| Input Shape | [2, 3, 224, 224] |
| Output Shape | [2, 1000] |
| extract_features params | 14,714,688 |
| classifier params | 123,642,856 |
| Total params | 138,357,544 |

## Notes

- VGGNet16(Configuration D) 구조를 PyTorch로 Scratch 구현
- 실제 데이터셋을 사용하지 않았으며, 랜덤값을 입력으로 사용
- 검증
    - Input/Output Shape
    - Forward pass
    - Total param: 138M으로 원 논문과 동일

---

# 8. Analysis

<!--
논문를 객관적으로 분석합니다.

- 장점
- 단점
- 왜 이러한 구조가 효과적인가?
- 어떤 상황에서 잘 동작하는가?
- 어떤 상황에서는 한계가 있는가?
- 후속 연구에서는 어떻게 개선되었는가?

논문를 비판적으로 분석하는 공간입니다.
-->

## merits
- 3*3 Convolution만 사용하여 구조가 매우 단순하고 구현이 쉽다.
- 네트워크가 깊어 표현력이 좋다.

## demerits
- Parameter가 매우 많다.
- FC Layer가 대부분의 Parameter를 차지한다.

## why
- 왜 7*7 필터를 사용안하고 3*3필터를 사용할까?:
    1. 3*3필터를 3번 사용하면 7*7과 같은 receptive field를 가지면서 parameter 수는 더 적음
        3*3*3 = 27개
        7*7 = 49개
    2. layer 사이에 활성화 함수를 사용해 비선형성 향상

---

# 9. Personal Insights

<!--
내가 논문를 읽고 느낀 점

새롭게 알게 된 점

인상 깊었던 부분

프로젝트에 적용할 수 있는 아이디어

다음에 공부할 내용

등을 자유롭게 작성
-->

요즘은 conv에 relu를 고려해서 분산을 두배로 늘리는 초기화기법을 적용하는데 해당 논문은 옛날거라서 그런지 그렇게하지않음 그래서 혁펜하임은 conv는 두배늘리고 mlp부분은 논문과 똑같이함



---