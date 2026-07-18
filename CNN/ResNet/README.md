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
- 이전의 연구 결과들을 보면 신경망의 깊이는 고수준 features를 추출하는데 매우 중요하지만 두 가지 문제가 있음
    1. vanishing/exploding gradients 문제
        - 정규화 기법들로 크게 해결
    2. degradation problem: 일정 수준 이상 깊어지면 얕은 모델보다 깊은 모델이 오히려 더 높은 training error가 발생
        - ![alt text](image-3.png)
        - Overfitting 아님

## Goal
- degradation problem을 해결해 깊은 신경망의 학습을 안정적으로 성공시킨다.

---

# 3. Core Idea

<!--
논문의 핵심 아이디어를
3~5줄 정도로 요약
"이 논문의 가장 중요한 기여가 무엇인가?"
를 적는다.
-->

![alt text](image.png)
- 목표값(H(x))을 학습하는 대신 입력과 출력의 차이인 잔차(F(x) = H(x) - x)를 학습하도록 재구성함
- 입력값을 출력층으로 바로 전달하는 Shortcut Connection을 도입하여, 추가적인 파라미터나 연산 비용 없이도 매우 깊은 신경망까지 안정적인 학습


---

# 4. Model Architecture & Forward Process

## Overall Architecture

![alt text](image-1.png)
![alt text](image-2.png)

---

## Components

### Conv7-64

**Purpose**<br>
- 입력 이미지의 저수준 특징(Edge, Texture 등)을 추출

**Configuration**<br>
- Kernel Size : 7×7<br>
- Stride : 2<br>
- Padding : 3<br>
- Activation : ReLU<br>
- Normalization : Batch Normalization

**Role**<br>
- 입력 이미지의 초기 특징을 빠르게 추출<br>
- Stride=2를 사용하여 Feature Map의 해상도를 절반으로 감소

**Output**<br>
- (B, 3, H, W) → (B, 64, H/2, W/2)

### MaxPool

**Purpose**<br>
- Feature Map의 해상도를 감소시켜 연산량과 메모리 사용량을 줄임

**Configuration**<br>
- Kernel Size : 3×3<br>
- Stride : 2<br>
- Padding : 1

**Role**<br>
- 중요한 특징을 유지하면서 Feature Map의 크기를 절반으로 감소<br>
- Receptive Field를 증가시킴

**Output**<br>
- (B, C, H, W) → (B, C, H/2, W/2)

### Basic Residual Block

**Purpose**<br>
- Residual Learning을 통해 degradation problem 해결

**Configuration**<br>
- Conv3×3<br>
- Batch Normalization<br>
- ReLU<br>
- Conv3×3<br>
- Batch Normalization<br>
- Shortcut Connection<br>
- ReLU

**Role**<br>
- Residual Learning을 통해 깊은 네트워크의 최적화를 쉽게 수행<br>
- Gradient Vanishing 문제를 완화하는 효과도 있음

**Output**<br>
- (B, C, H, W) → (B, C, H, W)

### Projection Shortcut

**Purpose**<br>
- Residual Learning을 적용 할 Feature Map의 크기 또는 Channel 수가 변경될 때 shape을 맞춘 후 적용

**Configuration**<br>
- Conv1×1<br>
- Stride : 2 (Downsampling 시)

**Role**<br>
- Shortcut의 Shape을 Main Branch와 동일하게 변환<br>

**Output**<br>
- (B, C, H, W) → (B, out_c, H/2, W/2)

### Global Average Pooling

**Purpose**<br>
- Feature Map의 공간 정보를 평균하여 하나의 Feature Vector로 변환

**Configuration**<br>
- Kernel Size : Feature Map Size

**Role**<br>
- Fully Connected Layer의 Parameter 수를 크게 감소<br>
- 채널별 특징을 요약

**Output**<br>
- (B, 512, 7, 7)<br>
  ↓ Global Average Pooling<br>
- (B, 512)

### FC Layer

**Purpose**<br>
- 최종 Feature Vector를 이용하여 각 클래스의 Logit을 계산

**Configuration**<br>
- Input : 512<br>
- Output : 1000 (ImageNet Classes)

**Role**<br>
- 각 클래스에 대한 Logit 생성

**Output**<br>
- (B, 512)<br>
  ↓ FC<br>
- (B, 1000)

### Softmax

**Purpose**<br>
- FC Layer에서 출력된 Logits를 확률 분포로 변환

**Role**<br>
- 각 클래스의 예측 확률 계산<br>
- 확률의 총합은 1이 되며, 가장 높은 확률을 가진 클래스를 최종 예측으로 선택

---

## Forward Process

### 1. Input

(B, 3, 224, 224)

### 2. Conv7×7-64

(B, 3, 224, 224)<br>
↓ Conv7×7 + BN + ReLU<br>
(B, 64, 112, 112)

### 3. MaxPool

(B, 64, 112, 112)<br>
↓ MaxPool<br>
(B, 64, 56, 56)

### 4. Layer1 (3 Residual Blocks)

(B, 64, 56, 56)<br>
↓ Basic Block ×3<br>
(B, 64, 56, 56)

### 5. Layer2 (4 Residual Blocks)

(B, 64, 56, 56)<br>
↓ Projection Shortcut<br>
(B, 128, 28, 28)<br>
↓ Basic Block ×3<br>
(B, 128, 28, 28)

### 6. Layer3 (6 Residual Blocks)

(B, 128, 28, 28)<br>
↓ Projection Shortcut<br>
(B, 256, 14, 14)<br>
↓ Basic Block ×5<br>
(B, 256, 14, 14)

### 7. Layer4 (3 Residual Blocks)

(B, 256, 14, 14)<br>
↓ Projection Shortcut<br>
(B, 512, 7, 7)<br>
↓ Basic Block ×2<br>
(B, 512, 7, 7)

### 8. Global Average Pooling

(B, 512, 7, 7)<br>
↓ Global Average Pooling<br>
(B, 512)

### 9. FC-1000

(B, 512)<br>
↓ FC<br>
(B, 1000)

### 10. Softmax

(B, 1000)<br>
↓ Softmax<br>
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

- H(x) = F(x) + x
    - x: 입력값
    - F(x): xW(학습 파라미터)
    - H(x): 원하는 최종 출력값
        - **F(x)는 입력 x에 대해 원하는 출력값(H(x))를 만들기 위해 얼마나 더해주거나 빼주어야 하는가(residual)를 학습하게 된다**
            - F(x) = H(x) - x



---

# 6. Training Configuration

| Item | Value |
|------|-------|
| Augmentation | Random Cropping(224*224), Horizontal Flipping, Color Augmentation |
| Input Size | 224 × 224 |
| Optimizer | mini-batch SGD |
| Batch Size | 256 |
| Initial LR | 0.1 |
| Momentum | 0.9 |
| Weight Decay | 1e-4 |
| Dropout | 0 |
| Loss | Cross Entropy |
| Initial weight | He Initialization |
| Initial bias | He Initialization |

- Validation Accuracy가 더 이상 개선되지 않을 때 LR을 1/10으로 줄임

## Data Preprocessing

- Resize (shorter side ∈ [256, 480])
- Per-pixel Mean Subtraction: 전체 데이터셋의 각 픽셀 위치별 평균값을 이미지에서 빼주어 데이터가 0을 중심으로 분포하게 하여 편향을 제거해줌으로써 최적화 과정이 원활해짐

## Evaluation Strategy

### 10-Crop Testing

- 입력 이미지의 네 모서리와 중심에서 224×224 크기로 Crop하여 5개의 이미지를 생성
- 각 Crop을 수평 뒤집기(Horizontal Flip)하여 추가로 5개의 이미지를 생성
- 총 10개의 이미지에 대해 각각 예측을 수행
- 10개의 예측 결과를 평균하여 최종 분류 결과를 생성

### Fully Convolutional Inference & Multi-scale Averaging

- Fully Connected Layer를 Convolution Layer로 변환하여 다양한 입력 크기를 처리
- 입력 이미지를 여러 Scale(224, 256, 384, 480, 640)로 Resize하여 각각 추론 수행
- 각 Scale에서 얻은 Prediction Score를 평균하여 최종 결과를 생성

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

ResNet/
├── README.md
├── model.py
└── main.py

## Model Implementation

- `conv1`, `layer1`~`layer4`, `fc`를 각각 모듈 단위로 구현
    - conv1 : 7×7 Convolution + Batch Normalization + ReLU
    - layer1~layer4 : Basic Residual Block을 각각 3, 4, 6, 3개씩 구성
    - fc : Global Average Pooling 이후 Fully Connected Layer
- Downsampling이 필요한 Residual Block에서는 `1×1 Convolution`을 사용한 Projection Shortcut을 구현하여 Feature Map의 크기와 Channel 수를 맞춤
- 입력 크기에 관계없이 마지막 Feature Map을 `(1, 1)`로 변환하기 위해 `nn.AdaptiveAvgPool2d((1, 1))`를 사용
- Weight Initialization은 논문과 동일하게 He(Kaiming) Initialization 적용

## Verification

| Item | Result |
|------|--------|
| Input Shape | [2, 3, 224, 224] |
| Output Shape | [2, 1000] |
| conv1 params | 9,408 |
| layer1 params | 221,952 |
| layer2 params | 1,116,416 |
| layer3 params | 6,822,400 |
| layer4 params | 13,114,368 |
| fc params | 513,000 |
| Total params | 21,797,672 |

## Notes

- ResNet34(BasicBlock) 구조를 PyTorch로 Scratch 구현
- Basic Residual Block을 이용하여 Residual Learning을 구현
- Downsampling이 필요한 Block에서는 Projection Shortcut(1×1 Convolution)을 사용
- Fully Connected Layer 이전에 Global Average Pooling을 적용하여 Feature Map을 하나의 Feature Vector로 변환
- 랜덤값을 입력으로 사용

- ResNet50 이상에서는 Bottleneck 구조 사용함
    - 1x1로 채널 수를 줄인 후 3x3을 통과시키고 다시 1x1로 채널 수를 키움
        - 3x3만 사용하면 파라미터 수가 너무 많아져서
        - 1x1 : 채널 수 조정
        - 3x3 : 공간 정보 보기

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
- 추가 파라미터나 연산 비용이 발생하지 않는다.
- 매우 깊은 신경망까지 안정적으로 학습해 성능 향상을 가능하게 한다.
- 구현이 무척 쉽다.

## demerits


## why
- H(x) = F(x) + x가 왜 최적화하기 더 쉬울까?
    - H(x) ≈ x 이라고 할 때
        - F(x) = 0 행렬
    - **Weight는 애초에 0 근처로 초기화하기 때문에 0을 만드는게 훨씬 쉽다**

- 왜 H(x) ≈ x?
    - 잘 학습된 신경망은 입력으로부터 차근차근 값을 바꿔나가 36층 -> 38층으로 갈 때 값의 변화가 그리 크지 않을 것임
        - 하나의 Block에서 많은 걸 하는 모델은 새로운 입력에 대해서 불안정함
            - Overfitting이라고 볼 수 있음
        - 하나의 Block에서 모든 걸 다 하려고 하지말고 차근차근 바꿔나가라고 AI에게 말해주는 셈

- 왜 ReLU 전에 x를 더할까?
    - ReLU의 음수를 없애는 연산으로 인한 정보 손실을 최대한 막기 위함
    - 역전파 과정에서 기울기를 구할 때 F(x) + x를 미분하면
        - dF(x)/dx + 1
            - 여기서 + 1이 기울기 소실 문제를 해결해줌

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

- Skip Connection은 흔히 Vanishing Gradient 문제를 해결하기 위한 기법으로 알려져 있지만, 논문를 직접 읽어보니 핵심 목적은 깊은 네트워크에서 발생하는 **Degradation Problem**을 해결하는 것이었다. 다만 Residual Learning 구조 덕분에 Gradient 전달이 쉬워지면서 Vanishing Gradient 완화에도 도움이 되었다는 점을 새롭게 알게 되었다.
- Skip Connection 하나가 추가되었을 뿐인데 이후 다양한 후속 모델에 영향을 줄 정도로 딥러닝 모델 설계에 큰 변화를 가져왔다는 점이 인상 깊었다.

---