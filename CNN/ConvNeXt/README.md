# 1. Paper Information

- Title: A ConvNet for the 2020s
- Paper URL: [https://arxiv.org/pdf/2201.03545]

---

# 2. Motivation

## Problem

- Swin의 sliding window(=shifted window)는 CNN과 비슷함
  - Swin의 W-MSA는 이미지의 local 정보에 집중하는 Convolution과 비슷함 즉, CNN은 좋은 방법임..!
    - 하지만 CNN은 `scaling behavior`가 트랜스포머 보다 안좋음
      - `scaling behavior` : 모델의 크기, 학습 데이터, 입력 해상도, 학습 계산량 등을 키웠을 때 성능이 효과적으로 증가하는가

## Goal

- pure CNN이지만 Swin의 구조를 수용해서 `scaling behavior` 약점을 극복해보자 !

## Core Idea

- stage ratio : 기존 (3,4,6,3) -> (3,3,9,3) or (3,3,27,3) 사용
  - swin의 (2,2,6,2) or (2,2,18,2) 처럼 3배 or 9배
  - 왜 (2,2,6,2) 안하고 (3,3,9,3)으로 함? -> transformer는 어텐션, feed-forward 등이 있어서 연산량이 더 많기 때문에 연산량을 비슷하게 맞추기 위함

- Patchify stem : 맨 처음 해상도 줄이는 과정을 4x4 conv(stride=4)로 대체
  - swin의 맨 처음이랑 동일하게 함

- 3 x 3 depthwise conv(ResNeXt 아이디어)를 사용해서 파라미터 수를 줄이고 이어서 1 x 1 separable conv(MobileNet V1 아이디어)로 채널 수를 늘림
  - ViT랑 비슷한 매커니즘임
    - depthwise conv : 공간(spatial) 정보만 봄 = self-attention
    - separable conv : 채널(channel) 정보만 봄 = Q,K,V 만들 때, MHA의 MLP, FeedForward

![alt text](image.png)
  - MHA로 공간 정보 연산 후 FeedForward에서 4배 늘렸다 줄이는 행위(c)

- Kernel size를 Swin의 Window size와 같은 7x7로 바꿈
  - 여러 사이즈를 테스트 해봤는데, 7x7이 가장 성능이 좋았음

![alt text](image-1.png)
- Activation 변경
  - ReLU -> GELU
  - 3번 -> 1번, Normalization도 1번
    - 정보 손실을 최소화 하기 위해서 채널 수를 늘린 후만 Activation 사용

- Downsampling을 Swin의 Patchmerging 처럼 블록 안에서 안하고 원할 때만 2x2 Conv(stride=2)를 사용
  - 훈련이 잘 안되서 앞에 LN 추가했더니 잘 됨
    - 이유는 명시해놓지 않았음

---

# 3. Model Architecture & Forward Process

## Overall Architecture

![alt text](image-1.png)
![alt text](image-2.png)

## Components

### ConvNeXt Block

**Purpose**
- 입력 feature map의 공간적 정보와 채널 정보를 효율적으로 변환합니다.
- Transformer block의 설계에서 영감을 받았지만, self-attention 없이 표준 convolution 연산만 사용합니다.

**Configuration**
- **Depthwise Convolution**: 각 채널별로 독립적인 $7 \times 7$ depthwise convolution을 적용하여 spatial mixing을 수행합니다.
- **Inverted Bottleneck**: 첫 번째 $1 \times 1$ convolution으로 채널 차원을 확장합니다.
- **GELU Activation**: 확장된 채널 feature에 GELU 활성화 함수를 한 번 적용합니다.
- **Pointwise Convolution**: 두 번째 $1 \times 1$ convolution으로 채널 차원을 원래 차원으로 축소합니다.
- **LayerNorm**: depthwise convolution 이후 채널 방향으로 LayerNorm을 적용합니다.
- **Residual Connection**: block의 입력을 변환된 출력에 더합니다.
- **Layer Scale**: residual branch에 초기값 $10^{-6}$의 Layer Scale을 적용합니다.

**Role**
- $7 \times 7$ depthwise convolution으로 주변 spatial 위치의 정보를 혼합합니다.
- $1 \times 1$ convolution으로 채널 간 정보를 혼합합니다.
- Spatial mixing과 channel mixing을 분리하여 Transformer의 block 구조와 유사한 연산 흐름을 구성합니다.
- Self-attention이나 shifted window와 같은 특수한 attention 모듈 없이 계층적 feature representation을 학습합니다.

**Output**
- 일반적인 ConvNeXt block은 입력과 동일한 해상도와 채널 수를 유지합니다.
- 입력과 출력의 형태는 다음과 같습니다.

$$
(B, H, W, C) \rightarrow (B, H, W, C)
$$

- Residual connection을 적용하기 위해 block 내부의 최종 채널 차원은 입력 채널 차원과 동일합니다.

### Downsampling Layer

**Purpose**
- Stage가 전환될 때 feature map의 spatial resolution을 낮춥니다.
- 해상도를 줄이는 동시에 채널 차원을 증가시켜 더 높은 수준의 특징을 추출합니다.

**Mechanism**
- $2 \times 2$ convolution과 stride $2$를 사용합니다.
- Downsampling layer를 residual block과 분리하여 독립적인 계층으로 구성합니다.
- 해상도가 변경되는 위치에 LayerNorm을 추가하여 학습을 안정화합니다.

**Role**
- ConvNet과 유사한 계층적 feature map을 생성합니다.
- 서로 다른 해상도의 feature map을 제공하여 object detection과 semantic segmentation 같은 downstream task에 활용할 수 있도록 합니다.

**Output**

$$
(B, H, W, C)
\rightarrow
(B, H/2, W/2, 2C)
$$

### Patchify Stem

**Purpose**
- 입력 이미지의 초기 spatial resolution을 줄이고, 첫 번째 feature map을 생성합니다.

**Mechanism**
- $4 \times 4$ convolution과 stride $4$를 사용합니다.
- 입력 이미지를 명시적인 token sequence로 변환하지 않고 convolution을 통해 feature map을 생성합니다.
- Stem 이후 LayerNorm을 적용합니다.

**Output**

$$
(B, H, W, 3)
\rightarrow
(B, H/4, W/4, 96)
$$

- ConvNeXt-T의 첫 번째 Stage에서는 채널 차원 $C=96$을 사용합니다.

## Forward Process

### 1. Patchify Stem

- 입력 이미지에 $4 \times 4$ convolution과 stride $4$를 적용합니다.
- 입력 이미지의 높이와 너비가 각각 $1/4$로 줄어듭니다.

$$
(B, H, W, 3)
\rightarrow
(B, H/4, W/4, 96)
$$

### 2. Stage 1

- 첫 번째 feature map에 ConvNeXt block을 반복적으로 적용합니다.
- ConvNeXt-T에서는 Stage 1에 $3$개의 block을 사용합니다.
- Feature map의 spatial resolution은 유지됩니다.

$$
(B, H/4, W/4, 96)
\rightarrow
(B, H/4, W/4, 96)
$$

### 3. Downsampling Layer 1

- Stage 1이 끝난 뒤 $2 \times 2$ convolution과 stride $2$를 적용합니다.
- 해상도는 절반으로 줄고 채널 수는 증가합니다.

$$
(B, H/4, W/4, 96)
\rightarrow
(B, H/8, W/8, 192)
$$

### 4. Stage 2

- ConvNeXt block을 반복적으로 적용하여 feature를 변환합니다.
- ConvNeXt-T에서는 Stage 2에 $3$개의 block을 사용합니다.

$$
(B, H/8, W/8, 192)
\rightarrow
(B, H/8, W/8, 192)
$$

### 5. Downsampling Layer 2

$$
(B, H/8, W/8, 192)
\rightarrow
(B, H/16, W/16, 384)
$$

### 6. Stage 3

- ConvNeXt-T에서는 Stage 3에 $9$개의 ConvNeXt block을 사용합니다.
- 네 개의 Stage 중 가장 많은 block을 배치하여 계산량을 집중합니다.

$$
(B, H/16, W/16, 384)
\rightarrow
(B, H/16, W/16, 384)
$$

### 7. Downsampling Layer 3

$$
(B, H/16, W/16, 384)
\rightarrow
(B, H/32, W/32, 768)
$$

### 8. Stage 4

- ConvNeXt-T에서는 Stage 4에 $3$개의 ConvNeXt block을 사용합니다.
- 최종 고수준 semantic feature를 생성합니다.

$$
(B, H/32, W/32, 768)
\rightarrow
(B, H/32, W/32, 768)
$$

### 9. Head

- 마지막 Stage의 feature map에 global average pooling을 적용합니다.
- Global average pooling 이후 최종 feature는 ConvNeXt-T의 마지막 채널 차원인 $768$이 됩니다.
- 이후 선형 분류기를 통해 ImageNet-1K의 $1{,}000$개 클래스에 대한 예측을 수행합니다.

$$
(B, H/32, W/32, 768)
\rightarrow
(B, 768)
\rightarrow
(B, 1000)
$$


## ConvNeXt-T 전체 구조 요약

| 단계 | 연산 | Block 수 | 출력 해상도 | 출력 채널 |
|---|---|---:|---:|---:|
| Stem | $4 \times 4$ convolution, stride $4$ | - | $H/4 \times W/4$ | $96$ |
| Stage 1 | ConvNeXt block | $3$ | $H/4 \times W/4$ | $96$ |
| Downsampling 1 | $2 \times 2$ convolution, stride $2$ | - | $H/8 \times W/8$ | $192$ |
| Stage 2 | ConvNeXt block | $3$ | $H/8 \times W/8$ | $192$ |
| Downsampling 2 | $2 \times 2$ convolution, stride $2$ | - | $H/16 \times W/16$ | $384$ |
| Stage 3 | ConvNeXt block | $9$ | $H/16 \times W/16$ | $384$ |
| Downsampling 3 | $2 \times 2$ convolution, stride $2$ | - | $H/32 \times W/32$ | $768$ |
| Stage 4 | ConvNeXt block | $3$ | $H/32 \times W/32$ | $768$ |
| Head | Global average pooling 및 linear classifier | - | - | $1{,}000$ classes |

---

# 4. Mathematical Explanation (New Ideas)

---

# 5. Training Configuration

![alt text](image-3.png)

---

# 6. Implementation

## Directory Structure

ConvNeXt/
├── README.md
├── model.py
└── main.py

## Model Implementation

- **ConvNeXt Architecture**
  - 논문의 계층적 ConvNet 구조를 하나의 `ConvNeXt_T` 클래스와 여러 개의 `CNBlock`으로 나누어 구현함.
  - 전체 ConvNeXt block 수는 다음과 같음.

  $$
  3 + 3 + 9 + 3 = 18
  $$

- **Patchify Stem**
  - 논문의 patchify stem을 `nn.Conv2d` 하나로 구현함.
  - 입력 이미지에 `kernel_size=4`, `stride=4`인 convolution을 적용하여 spatial resolution을 4배 축소함.
  - Stem 이후 `Permute`를 사용하여 채널 축을 마지막으로 이동한 뒤 `LayerNorm`을 적용하고, 다시 `NCHW` 형식으로 변환함.

- **ConvNeXt Block**
  - 논문의 ConvNeXt block을 `CNBlock` 클래스로 구현함.
  - 각 block은 `7 × 7` depthwise convolution, LayerNorm, 두 개의 `1 × 1` convolution, GELU, Layer Scale, Stochastic Depth, residual connection으로 구성함.
  - 입력과 출력의 feature map 크기는 동일하게 유지함.

- **Depthwise Convolution**
  - spatial mixing을 위해 `groups=in_c`로 설정한 `nn.Conv2d`를 사용함.
  - convolution kernel 크기는 `7 × 7`, stride는 $1$, padding은 $3$으로 설정함.
  - 채널 수와 group 수를 동일하게 설정하여 각 채널을 독립적으로 convolution함.

- **LayerNorm**
  - Depthwise convolution 이후 `Permute([0, 2, 3, 1])`를 사용하여 feature map을 `NCHW`에서 `NHWC` 형식으로 변환함.
  - `nn.LayerNorm(in_c)`를 적용하여 각 spatial 위치에서 채널 축을 기준으로 정규화함.
  - LayerNorm 이후에는 다시 `Permute([0, 3, 1, 2])`를 적용하여 `NCHW` 형식으로 복원함.

- **Channel Mixing**
  - 첫 번째 `1 × 1` convolution을 사용하여 채널 수를 $4$배로 확장함.

  $$
  C \rightarrow 4C
  $$

  - 확장된 feature에 GELU 활성화 함수를 적용함.
  - 두 번째 `1 × 1` convolution을 사용하여 채널 수를 원래 차원으로 되돌림.

  $$
  4C \rightarrow C
  $$

- **Layer Scale**
  - 각 `CNBlock`에 채널별 학습 가능한 Layer Scale parameter를 적용함.
  - Layer Scale parameter의 형태는 `(1, C, 1, 1)`이며, 초기값은 `layer_scale` 인자로 설정함.
  - 기본값은 $10^{-6}$으로 설정함.
  - 변환된 residual branch에 Layer Scale을 적용한 뒤 Stochastic Depth와 residual connection을 수행함.

  $$
  y = x + \operatorname{StochasticDepth}
  \left(
  \gamma \odot F(x)
  \right)
  $$

- **Stochastic Depth**
  - `torchvision.ops.StochasticDepth`를 사용하여 residual branch에 Stochastic Depth를 적용함.
  - `mode='row'`를 사용하여 batch의 각 샘플 단위로 residual branch를 확률적으로 제거함.
  - ConvNeXt-T의 최대 Stochastic Depth 확률은 $0.1$로 설정함.
  - 전체 $18$개 block에 대해 확률을 $0$에서 $0.1$까지 선형적으로 증가시킴.

- **Residual Connection**
  - `CNBlock`의 입력을 residual branch의 출력에 더함.
  - 입력과 residual branch의 채널 수와 spatial resolution이 동일하므로 별도의 projection layer를 사용하지 않음.

  $$
  \operatorname{output} = x + \operatorname{residual}
  $$

- **Separate Downsampling Layer**
  - Stage 사이의 downsampling을 `CNBlock`과 분리된 layer로 구현함.
  - Downsampling 직전에 `Permute`와 `LayerNorm`을 적용함.
  - 이후 `2 × 2` convolution과 `stride=2`를 사용하여 spatial resolution을 절반으로 줄임.
  - 다음 Stage의 채널 수에 맞게 출력 채널을 증가시킴.

  $$
  (B, C, H, W)
  \rightarrow
  (B, 2C, H/2, W/2)
  $$

- **Stage 구성**
  - `cfgs` 리스트를 사용하여 각 Stage의 채널 수, block 수, downsampling 여부를 지정함.
  - 각 Stage에서는 지정된 횟수만큼 `CNBlock`을 반복해서 추가함.
  - 마지막 Stage를 제외한 각 Stage 뒤에는 LayerNorm과 downsampling convolution을 추가함.

## Verification

| Item | 구현 방식 |
| :--- | :--- |
| Input Shape | `torch.randn(1, 3, 224, 224)`로 테스트 |
| Output Shape | `[1, 1000]` |
| Total Parameters | 28,580,968 |
| FLOPs | 4470433536 |

---

# 7. Analysis & Insights

![alt text](image-4.png)
  - 모델 및 데이터 규모를 키울수록 성능이 좋아짐
  - 전체적으로 Swin 보다 성능 우수
  - EfficientNet V2 보다도 성능은 우수하지만 속도는 뒤쳐짐

- CNN도 scalable 할 수 있다는 것을 보인 연구지만 Swin과 EfficientNet 보다 훨씬 좋은 것은 아님
- AI의 지속적인 발전을 생각했을 때(multi-modal) 역시 Transformer가 좋은듯..

---