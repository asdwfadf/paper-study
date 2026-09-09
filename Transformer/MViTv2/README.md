# 1. Paper Information

- Title: MViTv2: Improved Multiscale Vision Transformers for Classification and Detection
- Paper URL: [https://arxiv.org/pdf/2112.01526]

---

# 2. Motivation

1. ViT(Vision Transformer)는 이미지 분류에서는 성공적이었지만, **객체 탐지(Object Detection)**나 비디오 이해(Video Understanding) 같은 고해상도/고차원 태스크로 넘어가면 패치 개수(토큰)가 급격히 늘어나는데, 셀프 어텐션은 토큰 수의 **제곱(\(O(N^2)\))**에 비례하는 연산량을 소모함
저자들은 이 문제를 해결하기 위해 **'풀링 어텐션(Pooling Attention)'**이라는 효율적인 방식을 도입함

2. 절대적 위치 임베딩의 구조적 한계
기존의 Vision Transformer는 위치 정보를 줄 때 '절대적 위치 임베딩'을 사용했는데, 이는 비전 모델의 핵심 원리인 **'이동 불변성(Shift-invariance)'**을 깨뜨림
객체가 이미지 안에서 움직여도 모델은 그 객체를 동일하게 인식해야 하지만 절대 좌표를 학습하면 "객체가 어디에 있는가"에 모델이 과도하게 의존함
저자들은 "위치가 바뀌어도 객체 내 구성 요소(귀, 코, 꼬리 등)들 사이의 상대적 관계는 변하지 않는다"는 점에 주목하며 이동 불변성을 보장하는 상대적 위치 임베딩을 도입함

3. 저자들은 "이미지, 객체 탐지, 비디오 3가지 도메인에서 모두 최고의 성능을 내는 하나의 통합된 백본을 만들 수는 없을까?" 라는 고민을 함
즉, MViTv1에서 제안했던 풀링 어텐션 개념을 더 발전시켜, 아주 범용적으로 쓰일 수 있는 강력한 '시공간 백본'을 만드는 것이 이 연구의 궁극적인 동기

## Core Idea

- Pooling Attention
  - Attention 연산 전에 Q, K, V에 Pooling을 적용해서 해상도 및 연산량을 감소시킴
  - stride of Q : 해상도와 연산량을 줄임
  - stride of K and v : 연산량을 줄임
  - K와 V의 stride는 같아야 attention 연산을 수행할 수 있음

- Residual Pooling Connection
  - Attention 연산 전 Pooling 한 Q를 Attention 연산 후 결과 값에 더함
  - 새로운 아이디어라기 보다는 Pooling을 적용함으로써 해상도가 바꼇으니 당연히 맞춰서 하는 것

- Decomposed Relative Positional Embeddings
  - 객체의 위치와 상관없이 객체의 구성 요소(귀, 코, 꼬리 등)가 일정하면 동일한 패턴으로 인식하게 함.
  - 모든 3차원 조합($T \times W \times H$)을 학습하는 대신 축별로 분해하여 $O(T+W+H)$의 파라미터만 학습합니다.
  - 상대적 위치 정보를 $R$ 테이블에서 참조하여 어텐션 스코어에 더합니다.
  $$ \text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top + E(\text{rel})}{\sqrt{d}}\right)V
  $$
    - $QK^\top$ : 내용 유사도
    - $E(rel)$ : 위치 중요도

    - 여기서 위치 편향 $E(\text{rel})_{ij}$는 각 축의 거리 임베딩 합으로 계산됩니다.
    $$
    R_{p(i), p(j)} = R_{hh}(i),h(j) + R_{ww}(i),w(j) + R_{tt}(i),t(j)
    $$

### Decomposed Relative Positional Embeddings 계산 과정 예시

#### [가정 상황]
*   **격자 크기:** $14 \times 14$ ($224 \times 224$ 이미지, $16 \times 16$ 패치)
*   **토큰 $i$ 위치:** $(0, 0)$
*   **토큰 $j$ 위치:** $(0, 1)$ (가로로 바로 옆에 있는 토큰)
*   **임베딩 차원($D$):** 4차원 (예시를 위해 단순화)

#### Step 1: 거리 차이 계산
*   높이 거리 차이 $\Delta h = |0 - 0| = 0$
*   너비 거리 차이 $\Delta w = |0 - 1| = 1$

#### Step 2: $R$ 테이블에서 임베딩 벡터 참조 (Look-up)
* $R$ shape : 2L-1, D
  * L : 토큰 수
  * D : 모델 임베딩 차원과 같음<br>

모델이 이미 학습을 마친 $R$ 테이블에서 해당 거리 값을 가져옵니다.

*   $R_{hh}(0) = [0.1, 0.2, 0.1, 0.0]$
*   $R_{ww}(1) = [0.3, 0.1, -0.2, 0.1]$

#### Step 3: 분해된 임베딩 합산
$$
R_{p(i),p(j)} = [0.1, 0.2, 0.1, 0.0] + [0.3, 0.1, -0.2, 0.1] = [0.4, 0.3, -0.1, 0.1]
$$

#### Step 4: 어텐션 스코어에 적용
쿼리 벡터 $Q_i = [0.5, 1.0, 0.2, 0.5]$라면:
*   $E(\text{rel})_{ij} = Q_i \cdot R_{p(i),p(j)}$
*   $E(\text{rel})_{ij} = (0.5 \times 0.4) + (1.0 \times 0.3) + (0.2 \times -0.1) + (0.5 \times 0.1)$
*   $E(\text{rel})_{ij} = 0.2 + 0.3 - 0.02 + 0.05 = \mathbf{0.53}$

#### [최종 결과]
계산된 값 **0.53**을 어텐션 스코어($QK^\top$)에 더함 <br>
이 값은 모델이 $(0, 0)$ 위치의 토큰이 $(0, 1)$ 위치의 토큰을 얼마나 중요한 관계로 볼지 결정하는 '위치 편향(Position Bias)'으로 작용

---

# 3. Model Architecture & Forward Process

## Overall Architecture
- Three Visual Recognition Task
![alt text](image.png)

- Pooling Attention
![alt text](image-1.png)

- FPN for object detection
![alt text](image-2.png)

## Components

### MViT Block (Pooling Attention Block)

**Purpose**
- 효율적인 셀프 어텐션을 위해 쿼리(Q), 키(K), 값(V) 텐서에 풀링(Pooling) 연산을 적용하여 특징을 변환하고 시공간적 관계를 학습

**Configuration**
- **Pooling Attention**: Q, K, V 텐서에 선형 투영 후 풀링 연산을 적용하여 해상도와 시퀀스 길이를 유연하게 조절
- **Decomposed Relative Positional Embeddings**: 분해된 상대적 위치 편향을 더해 이동 불변성을 확보하고 시공간 구조를 학습합니다.
- **Residual Pooling Connection**: 풀링된 Q 텐서를 어텐션 출력에 잔차 연결하여 학습 안정성과 정보 흐름을 개선
- **MLP Block**: 각 토큰의 특징을 정제하는 2층 선형 변환과 비선형 활성화 함수로 구성

**Role**
- 쿼리와 키/값의 풀링을 통해 $O(N^2)$ 복잡도를 획기적으로 줄여 고해상도 처리를 가능하게 함
- 다중 스테이지에서 점진적으로 해상도를 낮추며 추상적인 정보를 학습함

**Output**
- 입력보다 줄어든 시퀀스 길이와 확장된 특징 차원을 가짐
- 출력 형태: $(B, \tilde{L}, D)$ (여기서 $\tilde{L}$은 풀링에 의해 줄어든 시퀀스 길이)

### Pooling Operator

**Purpose**
- 네트워크를 통과할수록 단계적으로 해상도를 줄이고 임베딩 차원을 확장

**Mechanism**
- 어텐션 블록 내에서 쿼리, 키, 값 텐서에 풀링(예: Stride를 가진 컨볼루션)을 적용
- Q의 풀링은 전체 해상도 변화를 결정하고, K/V의 풀링은 연산 복잡도를 직접적으로 조절함

**Role**
- CNN과 유사한 계층적 특징 구조를 Transformer 내부에 자연스럽게 생성함
- 고해상도 입력에서 시작하여 효율적으로 정보를 압축함으로써 객체 탐지와 비디오 인식에 최적화된 백본을 제공

## Forward Process

### 1. Patchification (Stem)

- 입력 이미지(또는 비디오 클립)를 작은 패치(또는 스페이스-타임 큐브)로 나눔
- 첫 번째 투영 층을 통해 입력 데이터를 $D$차원의 특징 벡터 토큰으로 변환

### 2. Linear Embedding

- 입력 데이터를 임베딩
$$
(B, L, D)
$$

### 3. MViTv2 Blocks

- 각 단계에서 Pooling Attention을 통해 토큰 개수를 줄이고 특징 차원을 늘려감
- 분해된 상대적 위치 임베딩을 사용하여 위치 정보를 명시적으로 학습
- Residual Pooling Connection을 통해 깊은 네트워크에서도 학습이 원활하도록 함

| Stage | 특징 맵 해상도 변화 |
|---|---|
| Stage 1 | $H_1 \times W_1$ |
| Stage 2 | $H_2 \times W_2$ |
| Stage 3 | $H_3 \times W_3$ |
| Stage 4 | $H_4 \times W_4$ |

### 4. Final Output

- 이미지 분류에서는 마지막 Stage의 출력 토큰들을 평균 내어(Global Average Pooling) 분류 헤드에 전달함

$$
(B, L_{final}, D_{final}) \rightarrow (B, D_{final}) \rightarrow (B, K)
$$

- 객체 탐지에서는 각 스테이지의 다중 해상도 출력 맵을 FPN(Feature Pyramid Networks)에 통합하여 사용함

---

# 4. Mathematical Explanation (New Ideas)

---

# 5. Implementation

## Directory Structure

MViTv2/
├── README.md
├── 
└── 

## Model Implementation

## Verification

| Item | 구현 방식 |
| :--- | :--- |
| Input Shape |  |
| Output Shape |  |
| Total Parameters |  |
| FLOPs |  |

---

# 6. Analysis & Insights

## Merits

- **기존 ViT보다 효율적인 고해상도 처리 및 스케일링**
  - 기존 ViT는 모든 패치에 전역 셀프 어텐션을 적용하여 연산량이 입력 토큰 수의 제곱으로 증가함.
  - MViTv2는 Pooling Attention을 통해 쿼리, 키, 값을 단계적으로 줄여나가며, 고해상도 입력에 대해서도 연산 효율성을 극대화하여 연산량을 획기적으로 낮춤.

- **범용적인 단일 아키텍처로의 적합성**
  - 기존 ViT는 단일 해상도 기반이라 dense prediction(객체 탐지 등)에 부적합함.
  - MViTv2는 자연스러운 다중 해상도 계층 구조(Hierarchical structure)를 통해 FPN과 쉽게 통합되며, 이미지, 비디오, 객체 탐지 등 서로 다른 비전 도메인에서 동일한 백본으로 SOTA 성능을 달성함.

- **이동 불변성(Shift-invariance)과 정교한 구조 학습**
  - 기존 ViT가 절대적 위치 임베딩을 사용해 절대 좌표에 의존하는 것과 달리, 분해된 상대적 위치 임베딩(Decomposed RPE)을 통해 공간적 위치 변화에 강건하고 객체의 구조적 패턴을 더 효과적으로 학습함.
  - Swin Transformer나 기타 기법들과 비교하여 Accuracy/Compute 효율성 측면에서 뛰어난 성능을 보임.

## Demerits

- **정보 병목 현상(Information Bottleneck)**
  - Pooling 연산은 본질적으로 정보를 압축하므로, 매우 정밀한 디테일이 필요한 태스크(예: 미세한 키포인트 검출)에서 공간적 정보 손실이 발생할 수 있음.

---