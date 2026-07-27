# Autoregressive Ranking: Bridging the Gap Between Dual and Cross Encoders

> **Venue**: arXiv preprint (arXiv:2601.05588v4, 2026-02-11)
> **Authors**: Benjamin Rozonoyer (UMass Amherst), Chong You, Michael Boratko, Himanshu Jain, Srinadh Bhojanapalli, Andrew McCallum, Felix Yu (Google DeepMind), Nilesh Gupta (UT Austin)
> **핵심 한 줄**: 생성형 랭킹(ARR)이 Dual Encoder보다 표현력에서 **이론적으로 우월**함을 증명하고, next-token prediction을 rank-aware하게 일반화한 손실 함수(STOICAL)를 제안.

---

## 1. Background

### Ranking / Retrieval 분야의 현황
- 현대 정보 검색(IR)은 보통 **2단계 파이프라인**으로 구성된다.
  - **1단계 (Retrieval)**: **Dual Encoder (DE)** — 쿼리와 문서를 각각 독립적으로 dense vector로 인코딩하고, ANN(근사 최근접 이웃) 검색으로 후보를 빠르게 선별. 효율적이지만 query-document 상호작용이 단일 벡터 유사도로 압축되어 표현력이 제한됨.
  - **2단계 (Reranking)**: **Cross Encoder (CE)** — 쿼리와 후보 문서를 함께 입력해 cross-attention으로 정밀한 relevance score를 산출. 정확하지만 모든 문서마다 별도 forward가 필요해 코퍼스 크기에 선형 비례하는 비용 → 1단계에는 부적합.
- LLM의 부상으로 **생성형 검색/랭킹**이 새 패러다임으로 등장. LLM이 문서 식별자(docID)를 토큰 단위로 autoregressive하게 생성하고, beam search로 정렬된 문서 리스트를 얻음 → 이를 **Autoregressive Ranking (ARR)** 이라 명명.

### 기존 방법의 한계

| 방법 | 문제점 |
|---|---|
| Dual Encoder (DE) | 단일 벡터 유사도 → 표현력 한계. 많은 relevant 문서를 정렬하려면 임베딩 차원이 커야 함(본 논문이 정량화) |
| Cross Encoder (CE) | 문서마다 개별 스코어링 → 코퍼스 크기에 선형 비용, 1단계 검색 불가능 |
| 기존 ARR/생성형 검색 | next-token prediction(NTP) 손실로 학습 → **rank-agnostic**. 문서 간 상대 순서를 명시적으로 모델링하지 못함 |

본 논문이 메우는 두 가지 공백:
1. ARR이 DE보다 표현력이 우월하다는 **형식적 이론 근거**의 부재.
2. 랭킹에 적합한 **rank-aware 학습 손실**의 부재.

---

## 2. Motivation

### 핵심 통찰 1: ARR은 DE의 차원 한계를 구조적으로 극복한다
- DE는 query/document를 같은 유클리드 공간에 임베딩하고 (음의 거리 또는 내적)을 유사도로 사용.
- **k개 문서의 임의 순서(ranking)** 를 표현하려면 DE의 임베딩 차원 n이 k에 대해 **선형적으로 증가**해야 함(본 논문 Proposition 3.1에서 증명).
- 반면 ARR은 docID를 여러 토큰으로 생성하므로, **고정(constant) hidden dimension**으로도 임의 개수의 문서를 랭킹할 수 있음.

### 핵심 통찰 2: 표준 NTP 손실은 랭킹에 부적합하다
- 기존 생성형 검색은 next-token prediction(또는 instruction tuning)으로 학습 → 토큰 단위 손실은 **문서들 간 상대적 순서를 명시하지 않는다(rank-agnostic)**.
- 그러나 beam search 같은 디코딩이 만드는 랭킹은 LLM의 확률값에 직접 좌우됨 → 확률을 rank-aware하게 만들면 랭킹 품질이 직접 개선됨.
- 핵심 아이디어: 생성 길이를 **하나의 docID 수준**(전체 리스트가 아님)으로 유지하면서, 학습 시 전체 ranked list를 인식하도록 손실을 설계 → 한 번의 모델 호출(single model call)로 랭킹 가능, snowballing 에러 완화.

<img src="./assets/arr_fig1_architectures.png" width="650">

> **Figure 1**: 랭킹을 위한 세 가지 아키텍처. (a) DE — 효율적이나 표현력 제한, (b) CE — 정확하나 고비용, (c) ARR — Causal Transformer가 docID 토큰을 생성하여 DE/CE 파이프라인을 단일 모델로 통합. 빨강=쿼리 토큰, 초록=문서 토큰.

---

## 3. Contributions
1. **ARR > DE 표현력 증명**: DE가 k개 문서의 complete ranking을 풀려면 임베딩 차원이 n ≥ ln(k!)/(2 ln k) ≈ k/(2 ln k) 로 **k에 선형 증가**해야 함을 증명. 반대로 ARR은 **상수 hidden dimension**으로 임의 개수 문서 랭킹이 가능(docID 토큰 임베딩 행렬의 rank 조건만 충족하면 됨).
2. **STOICAL (Simple Token-Item Calibrated Loss)**: NTP를 일반화한 rank-aware 학습 손실. **item-level reweighting**(문서 rank에 따른 가중치 λ(r))과 **prefix-tree marginalization**(유효한 docID 토큰에만 ground-truth relevance에 비례해 확률 질량 분배)으로 구성.
3. **실험 검증 (WordNet, ESCI)**: 유효하지 않은 docID 생성을 억제(constraint violation rate를 27.66% → ~0%로 격감)하고, top-1을 넘어선 ranking 지표(nDCG, R@k)를 개선. ARR이 WordNet에서 CE와 동등(on par)하면서 DE보다 월등함을 실증.

---

## 4. Method

### 4.1 문제 정의 (Ranking Task)
```
Ranking Task:
  각 query q ∈ Q 에 대해, relevant 문서들의 정렬 리스트
  L(q) = [d_1(q), ..., d_k(q)] 를 생성.
  순서가 중요: i < j 이면 d_i(q) ≻ d_j(q) (d_i 가 더 relevant).

Scoring-based Ranking Architecture:
  함수 f(q, d; θ): Q × D → R, (쿼리, 문서) 쌍에 스칼라 score 부여.
  score를 내림차순 정렬했을 때 ground-truth 순서와 일치하면 "랭킹을 푼다".
  DE와 ARR 모두 이 범주에 속함.

Complete Ranking Task (이상화된 분석용):
  - 모든 q에 대해 모든 문서가 relevant (D(q) = D)
  - 문서들의 모든 순열 π(D)에 대해, 해당 순서를 정답으로 갖는 q가 존재.
```

### 4.2 DE의 차원 한계 (Proposition 3.1)
DE의 유사도:
```
f_DE(q, d; θ) = -|| E_Q(q; θ_Q) - E_D(d; θ_D) ||_2     ... (1)
  E_Q, E_D : 쿼리/문서 인코더 (임의 아키텍처 허용, 표현력 무제한 가정)
  n = 임베딩 차원
```

**Proposition 3.1 (DE의 complete ranking 불가능 조건)**:
```
k := |D| 일 때, 다음이 성립하면 complete ranking을 푸는 θ가 존재하지 않는다:

  n < ln(k!) / (2 ln k)     ... (2)

증명 스케치:
  - N_{n,p}(k): R^n 의 L_p 메트릭에서 k개 점이 만드는 distinct distance permutation 최대 수.
  - Skala(2009) Corollary 8 에 의해 N_{n,2}(k) ≤ k^{2n}.
  - 부등식 (2)가 성립하면 k^{2n} < k! → n차원 임베딩으로 k! 순열을 모두 표현 불가.

Stirling 근사 ln(k!) = k ln k - k + O(ln k) 적용 시:
  n < k/2 - k/(2 ln k) + O(1)
  → k/ln k 가 k에 대해 sublinear 증가하므로,
    임베딩 차원 n은 문서 수 k에 선형으로 증가해야 함.
```
**핵심**: 이 결과는 인코더 아키텍처가 아무리 표현력이 좋아도 성립 → DE의 본질적 한계.

### 4.3 ARR의 표현력 (Propositions 3.2–3.3)
- ARR은 vocabulary V에 대한 next-token 확률을 `P(·|c) = σ(E φ(c))` 로 계산. docID는 `V_docIDs ⊆ V` 의 토큰 시퀀스.
- Constrained decoding으로 유효 docID 토큰만 생성:
```
P(v | c) = σ(E_docIDs · φ(c))   if v ∈ V_docIDs
         = 0                     otherwise
```
- 편의 행렬 `E' := [E_docIDs  1]` (docID 토큰 임베딩에 1열 추가).

**Proposition 3.2 / 3.3**: 무한 용량(infinite-capacity) ARR은 constrained decoding으로
- **임의의 (strictly positive) 확률 분포** 를 V_docIDs 위에 생성할 수 있다 ⟺ **rank(E') = |V_docIDs|**
- 동일 조건에서 **임의의 순열(permutation)** 도 생성 가능.

→ 즉 `rank(E') = |V_docIDs|`(E_docIDs가 거의 full-rank) 라는 **약한 조건**만 충족하면, **고정 차원 ARR**이 무제한 문서 수에 대해 임의 랭킹을 생성 가능. **이것이 DE보다 strictly superior한 지점**.

> **Remark (실무적 함의)**: WordNet처럼 docID가 영어 명사여서 |V_docIDs|가 |V|에 가까우면 (예: Mistral-7B는 |V|=32,768, d=4,096) rank 조건을 충족 못 함. 하지만 대부분의 시퀀스는 실제로 등장하지 않으므로 작은 부분집합만 학습하면 됨. ESCI에서는 **숫자형 docID**를 설계해 |V_docIDs|를 모델 차원에 가깝게 맞춰 rank 조건을 더 정확히 충족시킴.

### 4.4 STOICAL — Generalized Rank-Aware Training Loss

학습 데이터 포인트는 `(q, d_r, r)` 형태: 쿼리 q, docID d_r, 그리고 그 문서의 **true rank r**.

```
STOICAL Loss:

  L(q, d_r, r; θ) = λ(r) · Σ_{t=1}^{|T(d_r)|} CE( y(r,t),  p_θ( T(d_r)[t] | T(q̄), T(d_r)_{<t} ) )   ... (3)

  λ(r)     : rank r 에 대한 item-level 가중치 (reweighting function)
  T        : tokenizer,   q̄ : "augmented query" (쿼리 + 코퍼스 + system prompt)
  y(r,t)   : timestep t 에서의 target 분포 (rank r 에 의존 가능)
  CE       : cross entropy
```
- y(r,t)를 one-hot으로 두면 합산은 해당 docID의 **negative log-likelihood**로 환원 → 즉 STOICAL은 NTP의 일반화.

#### 4.4.1 Rank-Aware Item-Level Reweighting (λ(r))
rank r이 클수록(덜 relevant) 손실 기여를 줄이는 함수:
```
fractional :  λ(r) = 1 / r^α        (α: temperature 하이퍼파라미터)
stepwise   :  λ(r) = (n_q - r + 1) / n_q   (n_q: 쿼리의 총 문서 수)
top-1 only :  λ(r) = 1_{r=1}        (indicator; top-1 retrieval 학습 시)
```
- α → ∞ 이면 fractional은 top-1 학습(`1_{r=1}`)에 수렴.

#### 4.4.2 Prefix Tree Marginalization (y(r,t))
- 토큰화된 docID들로 **prefix tree(trie)** 를 구성. 각 timestep에서 부분 생성된 docID에 대해 **유효한(permissible) 연속 토큰**이 무엇인지 파악하고, 그에 맞춰 target 분포에 질량을 분배.
- 각 docID 노드에 score `μ(r) = 1/r^β` (β: temperature) 부여 후, 노드가 관할하는 모든 docID 점수를 marginalize.
- one-hot 대비, 같은 prefix를 공유하는 2등 이하 문서로도 부드럽게 질량을 분산 → rank-awareness 주입. β → ∞ 이면 one-hot으로 수렴.

<img src="./assets/arr_fig2_prefix_tree.png" width="680">

> **Figure 2**: (a) 랭킹된 docID 토큰들의 prefix tree (dog/cat/cats/deer/fish 예시), (b) one-hot supervision target, (c) marginalization target. (b)는 정답 토큰에만 1, (c)는 trie 구조에 따라 유효 토큰들에 1/r^β 비례로 분산(예: t1에서 d=1+1/4, c=1/2+1/3 등).

#### 학습 vs 추론
| 단계 | 과정 |
|---|---|
| **학습** | augmented query q̄ (쿼리 + shuffle된 docID 집합 + system prompt) 입력. STOICAL 손실로 docID 토큰 생성을 rank-aware하게 학습. 생성 길이는 단일 docID 수준. |
| **추론** | constrained decoding + beam search(beam size k)로 top-k docID를 직접 생성. score(d_j) = (1/|T(d_j)|) Σ_k log p_θ(...) — 토큰 log-prob 평균. 본 논문은 beam search의 proxy로 greedy decoding 사용. |

---

## 5. Experiments

### 5.1 Dataset

| | WordNet | ESCI (Shopping Queries) |
|---|---|---|
| 출처 | WordNet noun synset 계층 (Fellbaum 1998) | Reddy et al. (2022) Shopping Queries |
| 쿼리 | depth ≥ 1 의 noun synset 무작위 샘플 (예: "deer") | 영어 쿼리 (비영어 필터링) |
| Relevant docID | 쿼리의 hypernym(상위어) 경로 전체 (root까지) | Gecko(gecko-1b-en) 임베딩 내적 상위 product titles (1st, 100th, 200th, ... 10000th, step 100) |
| docID 설계 | 영어 명사 그대로 (taxonomy 기반) | sparse dictionary learning(OMP, dim 100, nonzero 3개)으로 만든 sparse vector의 nonzero 인덱스를 내림차순 concat (예: "25,36,39") → **trie-friendly** |
| 문서 길이 | 짧음 (~10 토큰) | 김 (수십 토큰) |
| 평가 규모 | 5,000 (query, targets) 예제 | 310 (query, targets) 예제 |

### 5.2 Implementation Details
- 모든 fine-tuning에 **Mistral-7B-v0.3-it** 사용 (Jiang 2024).
- DE 비교군(WordNet): synset 수 82,155 → 82,155×n 임베딩 테이블, 내적 유사도, weighted batch softmax loss, τ=0.05.
- CE 비교군: 임베딩을 concat(2n차원) → MLP(ReLU, hidden 3층 크기 2n, 출력 1). 배치 내 무작위 negative로 sigmoid 손실 학습.

### 5.3 Main Results

**WordNet (Table 1)** — CVR(constraint violation rate, ↓)과 nDCG·R@k(↑):

| 설정 | CVR ↓ | nDCG | R@1 | R@3 | R@5 |
|---|---|---|---|---|---|
| NTP baseline (λ=1_{r=1}, one-hot) | **27.66%** | 94.89 | **99.96** | 55.30 | 63.42 |
| fractional λ=1/r^α, α=1 | 0.0% | 99.60 | 91.10 | 88.81 | 93.94 |
| fractional, α=2 | 0.0% | **99.83** | 97.74 | 95.48 | 96.58 |
| fractional, α=3 | 0.0% | 99.81 | 99.22 | **96.16** | 96.53 |
| stepwise λ=(n_q-r+1)/n_q | 0.0% | 99.60 | 51.62 | 82.09 | 93.34 |
| trie marg. (β=1) | 1.48% | 96.54 | 98.10 | 63.06 | 64.25 |

- **핵심**: rank-aware reweighting이 **CVR을 27.66% → ~0%로 격감**(유효하지 않은 docID 생성 억제). nDCG와 top-1 이후 R@k(R@2~R@5)를 NTP 대비 크게 개선.
- **fractional > stepwise**. α → ∞ 이면 top-1 학습에 수렴.

**ESCI (Table 2)** — 쿼리당 재랭킹할 product title 수가 훨씬 많음:

| 설정 | nDCG | R@1 | R@2 | R@10 | R@50 |
|---|---|---|---|---|---|
| NTP (λ=1_{r=1}, one-hot) | 95.23 | **95.16** | 52.58 | 23.51 | 62.99 |
| trie marg. β=1 | **97.21** | 70.00 | 56.61 | 46.27 | **69.58** |
| trie marg. β=2 | 97.21 | 70.32 | **58.06** | **48.08** | 69.03 |

- ESCI는 문서 수가 많아 item-level 손실로 끝까지 학습하면 docID마다 별도 step이 필요 → 비현실적. 따라서 **NTP와 동일 budget**(쿼리당 single target, λ=1_{r=1})으로 두되 **trie-based marginalization**으로 rank 정보를 주입.
- 결과: R@1은 다소 희생하지만 **K>1의 모든 R@K와 nDCG 개선**. **Trie marginalization은 item-level reweighting의 경제적(economical) 대안**(무제한 학습 예제 가정 시엔 reweighting이 약간 우세).
- 학습한 **trie-friendly docID와 token-level loss 간의 시너지**도 확인.

### 5.4 Comparison with DEs and CEs (WordNet, Figure 3)
- **ARR은 CE와 동등(on par)**하면서 **DE보다 월등**.
- DE는 차원 n ∈ {4,8,16,32}을 키워도 ARR/CE를 못 따라감 (Proposition 3.1 실증).
- reweighting factor λ(r) = 1/r^α 의 α 증가가 **DE와 CE 모두에서** Recall@k(특히 작은 k)를 개선 → reweighting이 ARR을 넘어 랭킹 일반에 유용.

<img src="./assets/arr_fig3_de_ce_comparison.png" width="700">

> **Figure 3**: WordNet에서 DE/CE와 비교. (a) ARR vs DE(n∈{4,8,16,32})/CE: ARR이 CE와 거의 겹치며 DE를 압도. (b) DE의 α 효과, (c) CE의 α 효과 — α 증가가 작은 k에서 Recall 개선.

---

## 6. Key Takeaways
1. **이론적 분리(separation)**: DE는 complete ranking을 위해 임베딩 차원이 문서 수 k에 **선형 증가**(n ≳ k/(2 ln k))해야 하나, ARR은 **상수 hidden dimension**으로 임의 개수 문서를 랭킹 가능 → 표현력에서 **strictly superior**.
2. **충분조건이자 필요조건**: ARR이 임의 분포/순열을 생성하는 조건은 `rank(E') = |V_docIDs|`(docID 토큰 임베딩이 거의 full-rank)로, token-level과 sequence-level에서 동일하게 **필요충분**.
3. **STOICAL**: NTP를 일반화한 rank-aware 손실. **item-level reweighting**(λ(r)=1/r^α 등)과 **prefix-tree marginalization**(μ(r)=1/r^β)으로 ground-truth relevance에 비례해 유효 docID에 확률 질량 분배.
4. **Constraint violation 격감**: rank-aware 학습이 WordNet에서 **CVR 27.66% → ~0%**, 즉 유효하지 않은 docID 생성을 거의 제거하여 constrained decoding 부담을 완화.
5. **경제성**: ESCI처럼 문서 수가 많을 때 trie marginalization은 single-target budget으로도 K>1 ranking 지표(nDCG·R@K)를 개선하는 **reweighting의 경제적 대안**. 단, docID를 **trie-friendly하게 설계**할 때 시너지 발생.
6. **실무 시사점**: ARR은 DE+CE 2단계 파이프라인을 **단일 모델 + 단일 호출**로 통합 가능. 다만 t>1 timestep에서의 학습-추론 mismatch(teacher forcing)는 future work로 남김.
