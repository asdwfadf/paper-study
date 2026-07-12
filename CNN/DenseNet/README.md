# Paper Title

> 논문 한 줄 요약

---

# 1. Paper Information
- Title: 
- Paper URL: 

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

---

# 3. Background

<!--
이 논문를 이해하기 위해 필요한 개념 정리

예를 들어

- CNN
- Self-Attention
- Batch Normalization
- Cross Entropy
- Diffusion
- Transformer

필요한 수식도 함께 정리
-->

---

# 4. Core Idea

<!--
논문의 핵심 아이디어를
3~5줄 정도로 요약
"이 논문의 가장 중요한 기여가 무엇인가?"
를 적는다.
-->

---

# 5. Model Architecture & Forward Process

## Overall Architecture

전체 모델 그림

---

### Components

- Encoder
- Decoder
- Residual Block
- Attention
...

각 Module의 역할 설명

---

## Forward Process

Input

↓

Layer 1

↓

Layer 2

↓

...

↓

Output

각 단계에서 Tensor Shape도 함께 작성

예)

Input : (1,3,224,224)

↓

Conv : (1,64,112,112)

↓

Residual Block

↓

...

↓

Output : (1,1000)

---

# 6. Mathematical Explanation

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

# 7. Training Strategy

<!--
논문에서 사용한 학습 방법

- Optimizer
- Scheduler
- Learning Rate
- Batch Size
- Epoch
- Weight Decay
- Augmentation
- Dataset
- Hardware

논문에서 특별히 사용한 Trick도 기록
-->

---

# 8. Scratch Implementation

<!--
직접 구현한 내용

구현한 파일 설명

model.py

train.py

dataset.py

등

구현하면서 어려웠던 점

논문와 달랐던 점

내가 수정한 부분

등을 작성
-->

---

# 9. Verification

<!--
GPU 없이도 검증 가능한 내용

Random Input

Output Shape

Tensor Shape

Parameter 개수

Forward Pass 성공 여부

논문의 Shape와 동일한지

등을 기록
-->

---

# 10. Strengths

<!--
논문의 장점

왜 성능이 좋아졌는가?

기존 방법보다

무엇이 개선되었는가?

실제로 어디에 많이 사용되는가?
-->

---

# 11. Limitations

<!--
논문의 단점

어떤 상황에서는 성능이 떨어지는가?

계산량

메모리

데이터 요구량

한계점

후속 연구가 등장한 이유
-->

---

# 12. Why? (Deep Dive)

<!--
가장 중요한 항목

논문의 "왜?"를 분석

예시

왜 Residual Learning을 사용했는가?

왜 Softmax를 사용했는가?

왜 LayerNorm인가?

왜 Multi-Head Attention인가?

왜 GELU인가?

왜 Positional Encoding이 필요한가?

왜 이 Loss를 선택했는가?

왜 이 구조가 성능이 좋아지는가?

논문를 읽으며 생긴 의문과
그에 대한 답을 정리
-->

---

# 13. Personal Insights

<!--
내가 논문를 읽고 느낀 점

새롭게 알게 된 점

인상 깊었던 부분

프로젝트에 적용할 수 있는 아이디어

다음에 공부할 내용

등을 자유롭게 작성
-->

---

# 14. Interview Questions

<!--
이 논문와 관련해서

면접에서 나올 법한 질문 정리

그리고 직접 답변 작성

예시

Q. 왜 ResNet은 깊어질수록 성능이 좋아질 수 있었나요?

Q. ResNet과 DenseNet 차이는?

Q. Self-Attention의 시간복잡도는?

등
-->

---