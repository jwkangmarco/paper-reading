# Diffusion-Pretrained Dense and Contextual Embeddings (pplx-embed)

> **Venue**: arXiv preprint (arXiv:2602.11151v2, 2026-02-13) — Perplexity AI Technical Report
> **Authors**: Sedigheh Eslami*, Maksim Gaiduk, Markus Krimmel*, Louis Milliken*, Bo Wang*, Denis Bykov (Perplexity AI; *Equal contributions)
> **Platform**: Perplexity 프로덕션 웹 검색 (1B+ 웹페이지 규모)
> **핵심 한 줄**: **Diffusion 목적함수로 continued-pretrain한 양방향 인코더** 백본 위에 **다단계 contrastive learning**을 얹어, 양자화 내장(INT8/binary)·long-document 문맥 보존이 가능한 웹스케일 멀티링구얼 임베딩 모델 패밀리.

---

## 1. Background

### Text Embedding / Retrieval 분야의 현황
- Dense embedding은 텍스트를 연속 벡터 공간의 점으로 표현하여, ANN 검색으로 쿼리·문서를 공유 의미 공간에서 효율적으로 매칭 → 검색 시스템의 핵심.
- 최근 임베딩 학습은 사전학습된 **decoder-only LLM**(causal attention)을 백본으로 활용하는 방향으로 이동(기존 지식 활용, 품질 향상). 단, decoder-only는 **causal mask** 때문에 토큰이 미래 문맥을 못 봄 → 양방향 문맥 모델링에 불리.
- 본 논문은 대안 패러다임으로 **diffusion 기반 언어 모델(DLM)** 에 주목. DLM은 **양방향 attention** 을 쓰는 transformer encoder로, causal AR 모델보다 포괄적 문맥 모델링이 가능 → 글로벌 문서 문맥이 중요한 검색에 유리.

### 기존 방법의 한계

| 방법 | 문제점 |
|---|---|
| Decoder-only LLM 임베딩 (causal) | causal mask로 양방향 문맥 손실. 보통 **last-token pooling** 강제(mean pooling 부적합) |
| 일반 임베딩 모델 | full-precision(fp16/fp32) 출력 → 웹스케일에서 저장/검색 비용 큼 |
| Long-document 처리 | 청크로 쪼개면 각 청크가 문서 전역 문맥을 잃음 |
| 많은 LLM 임베딩 | instruction-tuned → 사용자가 instruction prefix 유지 부담 |

---

## 2. Motivation

### 핵심 통찰 1: Diffusion pretraining이 양방향 인코더를 만든다
- causal decoder-only 백본(Qwen3)을 **diffusion 목적함수로 continued pretraining**하면 → causal mask가 제거되고 **양방향 self-attention encoder**로 변환됨.
- 이로써 mean pooling 사용이 자연스러워지고, long document의 전역 문맥 보존(late chunking)에 유리.
- **실증**(§4): diffusion 백본 + mean pooling 조합이 causal Qwen3 백본보다 contrastive pair training에서 **훨씬 낮은 loss**와 retrieval 평균 **~1%p 향상** 달성.

### 핵심 통찰 2: 양자화를 학습에 내장(quantization-aware)하면 효율과 품질을 동시에
- INT8 양자화를 **추론뿐 아니라 모든 contrastive 학습 단계**에 적용하고 straight-through estimator로 역전파 → 양자화 손실을 학습이 직접 흡수.
- 결과: pplx-embed-v1-4B가 **390 docs/MB**(INT8) / **3,125 docs/MB**(binary) 저장 효율 — Qwen3-Embedding-4B(97 docs/MB), gemini-embedding-001(81)보다 압도적이면서 성능은 동등 이상.

### 핵심 통찰 3: 청크 임베딩에 문서 전역 문맥을 주입 (contextual training)
- 긴 문서를 청크로 나누되, 각 청크 임베딩이 **문서 전체의 문맥 정보**를 유지하도록 dual-objective(local + global) 손실로 학습 → ConTEB 벤치마크에서 SOTA.

---

## 3. Contributions
1. **Diffusion continued pretraining**: Qwen3-0.6B/4B를 diffusion 목적(absorbing state, continuous-time)으로 추가 학습하여 양방향 인코더로 전환. 250B 멀티링구얼 토큰.
2. **다단계 contrastive 커리큘럼**: Pair → (Triplet ∥ Contextual) → Merging & Selection 의 분기형 파이프라인으로 두 모델 산출 — **pplx-embed-v1**(표준 검색), **pplx-embed-context-v1**(문맥형).
3. **Native quantization-aware training**: mean pooling과 INT8 양자화를 결합한 pooling을 모든 학습 단계에 적용. INT8/binary 출력을 기본 제공, instruction-tuning 불필요.
4. **광범위한 벤치마크 검증 + 프로덕션 평가**: MTEB(Multilingual v2, Code), MIRACL, BERGEN, ToolRet에서 경쟁력. ConTEB SOTA. 1B 웹페이지 기반 자체 벤치마크(PPLXQuery2Query/2Doc)에서 우위.

---

## 4. Method

<img src="./assets/pplx_fig1_pipeline.png" width="640">

> **Figure 1**: pplx-embed-v1 / pplx-embed-context-v1 학습 파이프라인. Continued Pretraining → Pair Training → {Triplet Training, Contextual Training}으로 분기 → Merging & Selection. Contextual 분기 결과가 **pplx-embed-context-v1**, 병합 결과가 **pplx-embed-v1**.

전체 흐름:
1. **Continued Diffusion Pretraining** (§2.1) — decoder-only를 양방향 인코더로 변환
2. **Pair Training** (§2.3) — 쿼리-문서 기본 의미 정렬
3. **Contextual Training** (§2.4) — 청크 단위 임베딩에 문서 문맥 주입 → **pplx-embed-context-v1**
4. **Triplet Training** (§2.5) — hard negative로 유사하지만 비관련 문서 경계 정교화
5. **Merging & Selection** — contextual 체크포인트와 triplet 체크포인트를 **SLERP(Spherical Linear Interpolation)** 로 병합 → **pplx-embed-v1**

### 4.1 Continued Diffusion Pretraining
- Gong et al.(2025) 방법론을 따라 Qwen3-0.6B/4B-Base를 **양방향 diffusion LM**으로 추가 학습.
- causal attention mask **제거**. **absorbing state** 프로세스: 시점 t∈[0,1]에서 각 토큰이 확률 t로 `[MASK]` 상태로 붕괴(decay). `[MASK]`는 Qwen3 vocab의 희소 토큰 재활용.
```
학습 목표 (ELBO):
  t ~ U(0.001, 1) 샘플, 각 토큰을 확률 t로 마스킹.
  loss = (1/t) · Σ (마스킹된 위치의 token-wise cross entropy)
  → 표준 evidence lower bound (1/t 스케일링)
```
- 데이터: 절반은 FineWeb-Edu(영어), 절반은 FineWeb2/FineWeb2-HQ(29개 언어). 60,000 step, global batch 1024, seq len 4096 → 약 **250B 멀티링구얼 토큰**.
- 1% 시퀀스는 무작위 길이로 truncate(다양한 길이 노출). AdamW, warmup-stable-decay, peak LR 5e-4(0.6B)/3.16e-4(4B).
- causal mask annealing도 실험했으나 유의미한 향상 없어 미채택. left-shift 연산은 유지.

### 4.2 Pooling and Quantization
양방향 구조 덕분에 **mean pooling** 사용 (decoder-only의 last-token pooling 대비 이점). mean pooling과 INT8 양자화를 **native하게 결합**한 pooling:
```
INT8 양자화 mean pooling:
  v_l ∈ R^d (l=1..L) 토큰 임베딩에 대해,
  sequence embedding = floor( 127 · tanh( (1/L) Σ_{l=1}^L v_l ) + 1/2 )

  → 정수 entry {-127, ..., 127} = signed 8-bit
  → straight-through gradient estimation(Bengio 2013)으로 rounding 역전파
  → 양자화 임베딩은 cosine similarity로 비교

Binary 양자화 (옵션):
  bin(x) = 1   if x ≥ 0
         = -1  otherwise
  → training-free post-hoc binarization으로도 최소 손실
```
- 핵심: 양자화를 **추론뿐 아니라 모든 contrastive 학습 단계에 적용** → 양자화 인지 학습(QAT).

### 4.3 Pair Training (첫 contrastive 단계)
InfoNCE 변형. 쿼리를 배치 내 문서들과 **그리고 다른 배치 내 쿼리들과** 동시에 대조:
```
L_pair = -(1/N) Σ_i log [ e^{s(q_i,d_i)/τ}
          / ( e^{s(q_i,d_i)/τ} + Σ_{j≠i} m_i(d_j) e^{s(q_i,d_j)/τ}
                              + Σ_{j≠i} m_i(q_j) e^{s(q_i,q_j)/τ} ) ]

  s(q_i,d_i) = cosine similarity
  m_i(x) = 1_{ s(q_i,x) ≤ s(q_i,d_i)+0.1 }   ← false negative 마스킹
```
- **False negative 마스킹** `m`: in-batch negative의 유사도가 positive 유사도를 0.1 초과하면(= 잠재적 진짜 관련 문서) 기여를 마스킹 → 표현 공간 왜곡 방지(Zhang 2025b 영감).
- 3단계 점진 학습: ① 영어만 → ② 영어+cross-lingual → ③ 전체 멀티링구얼.

### 4.4 Contextual Training (청크 임베딩에 문서 문맥 주입)
문서 d_i가 C개 청크 {c_ik}로 구성. 청크 임베딩 c_ik가 문서 전역 문맥을 유지하도록 **dual-objective(local + global)** 손실. Conti et al.(2025) 영감.

**Local loss** (청크 수준 의미):
```
in-sequence loss (같은 문서 내 다른 청크를 negative로):
  L_seq = -(1/N) Σ_i log [ e^{s(q_i, c_{i*})/τ} / Σ_{k=1}^C e^{s(q_i, c_ik)/τ} ]
    c_{i*} = gold 청크 임베딩

in-batch loss (배치 내 모든 청크를 negative로, 같은 문서 포함):
  L_batch = -(1/N) Σ_i log [ e^{s(q_i, c_{i*})/τ} / Σ_{j=1}^N Σ_{k=1}^C e^{s(q_i, c_jk)/τ} ]

  L_local = α·L_seq + (1-α)·L_batch    (α = 0.2)
```

**Global loss** (쿼리-문서 수준, 중복 문서 마스킹):
```
중복 문서 마스킹 행렬 (해시 h(d), 예: MD5):
  M_ij^dup = 0  if h(d_i)=h(d_j) and i≠j   (중복 → false negative 방지)
           = 1  otherwise

  L_global = -(1/N) Σ_i log [ e^{s(q_i,d_i)/τ}
              / ( Σ_j M_ij^dup m_i(d_j) e^{s(q_i,d_j)/τ} + Σ_{j≠i} m_i(q_j) e^{s(q_i,q_j)/τ} ) ]
```

**Total contextual loss** (cosine 스케줄 β: 0.2 → 0.5):
```
L_context = β·L_global + (1-β)·L_local
  → 초기엔 local 청크 의미에 집중, 점차 document-level 학습 비중↑ (coarse 의미 망각 방지)
```

### 4.5 Triplet Training (hard negative)
명시적 hard negative로 미세한 relevance 경계 학습:
```
L_triplet = -(1/N) Σ_i log [ e^{s(q_i,d_i)/τ}
            / ( Σ_j e^{s(q_i,d_j)/τ} + Σ_j Σ_{k=1}^K e^{s(q_i, d^h_{jk})/τ} ) ]
  d^h_{jk} : 쿼리 q_i 에 대한 hard negative
```

### 4.6 Merging & Selection
- contextual 체크포인트 + triplet 체크포인트를 **SLERP(Shoemake 1985)** 로 구면 보간 병합 → **pplx-embed-v1**.

### 4.7 Datasets for Contrastive Learning
| 단계 | 데이터 구성 |
|---|---|
| Contrastive 전체 | 영어 65.6%, cross-lingual 6.7%, code 1%, 멀티링구얼 26.7% (60개 언어) |
| Contextual | ConTEB 학습 데이터 + MLDR 합성 데이터 |
| Triplet | 고품질 소량, 12개 데이터셋 (영어 92%, code 1%, 멀티링구얼 7%/15개 언어) |
- 합성 데이터: Qwen3-30B-A3B-Instruct-2507로 생성. **2-stage persona 기반**(top-5 relevant persona)으로 다양한 쿼리-문서 쌍 생성. Contextual용은 문서 내 passage에 대한 쿼리 합성.

### 학습 vs 추론
| 단계 | 과정 |
|---|---|
| **학습** | diffusion pretrain(양방향화) → pair → contextual/triplet 분기 → SLERP 병합. 모든 단계 INT8 양자화 QAT, mean pooling. |
| **추론** | mean pooling + INT8(또는 binary) 양자화 → cosine similarity. **instruction prefix 불필요**. contextual 모델은 late chunking으로 청크 인코딩. |

---

## 5. Experiments

### 5.1 평가 벤치마크
- **공개**: MTEB(Multilingual v2 — 131 tasks/18 retrieval/146 langs, Code — 12 tasks/15 langs), MIRACL(18 langs), ConTEB(8 datasets, contextual), BERGEN(RAG), ToolRet(35 tasks).
- **자체(프로덕션)**: PPLXQuery2Query(Q2Q), PPLXQuery2Doc(Q2D) — 1B+ 웹페이지에서 30M+ 문서 풀, 최대 115K 쿼리.

### 5.2 Main Results

**저장 효율 + MTEB/Code (Table 1)** — nDCG@10, Docs/MB:

| Model | Docs/MB | MTEB(Multi, v2) | MTEB(Code) |
|---|---|---|---|
| **pplx-embed-v1-4B (INT8)** | 390 | **69.66** | 78.73 |
| pplx-embed-v1-4B (BIN) | **3,125** | 68.22 | 78.11 |
| qwen3-embed-4B | 97 | 69.60 | **80.07** |
| gemini-embedding-001 | 81 | 67.71 | 76.00 |
| text-embedding-3-large | 81 | 59.27 | 66.54 |
| **pplx-embed-v1-0.6B (INT8)** | 976 | **65.41** | **75.85** |
| pplx-embed-v1-0.6B (BIN) | **7,812** | 61.44 | 73.91 |
| qwen3-embed-0.6B | 244 | 64.65 | 75.42 |
| embed-gemma-0.3B | 325 | 62.58 | 68.76 |

- **4B**: MTEB-Multi에서 gemini-001 능가, Qwen3-4B와 동등하면서 **저장 효율 4배(390 vs 97)**. Code는 Qwen3에 소폭 뒤짐.
- **0.6B**: 양 벤치마크 모두 Qwen3-0.6B 능가. binary는 docs/MB 7,812로 극단적 효율.

**MIRACL (Table 2)**: pplx-embed-v1-0.6B가 **모든 언어 subset에서 Qwen3-Embedding-0.6B 능가**. 심지어 0.6B가 자사 4B 평균(68.6 vs 70.4은 4B이 위지만 일부 역전)을 초과하는 강세. binary 변형도 Qwen3-0.6B 상회.

**ConTEB (Table 3)** — 문맥형 검색, nDCG@10:

| Model | Avg | (비고) |
|---|---|---|
| pplx-embed-v1-4B (INT8, 비문맥) | 58.83 | 비문맥 모델 중 최고 |
| **pplx-embed-context-v1-4B (INT8)** | **81.96** | **전체 SOTA** |
| pplx-embed-context-v1-4B (BIN) | 80.46 | |
| voyage-context-3 | 79.45 | |
| pplx-embed-context-v1-0.6B (INT8) | 76.53 | modernBERT-Large(75.6), anthropic contextual(72.4) 능가 |
| anthropic contextual* | 72.4 | |

- **pplx-embed-context-v1-4B가 voyage-context-3(79.45), Anthropic Contextual(72.4)를 제치고 SOTA**. 0.6B도 contextually-trained ModernBERT-Large와 Anthropic Contextual 능가(단 voyage-context-3에는 뒤짐).

**BERGEN (Table 4)** — RAG, Qwen2.5-32B-Instruct 생성, top-5 passage, match metric:

| Model | KILT-NQ | HotpotQA | TriviaQA | ASQA | PopQA |
|---|---|---|---|---|---|
| pplx-embed-v1-4B (INT8) | **67.7** | **51.9** | **91.9** | 71.5 | 68.7 |
| pplx-embed-v1-0.6B (INT8) | 67.2 | 51.6 | 91.0 | 72.6 | **70.0** |
| qwen3-embed-4B | 67.1 | 50.2 | 91.5 | **72.7** | 66.4 |
| bge-m3 | 66.8 | 49.3 | 89.4 | 69.4 | 68.5 |

- **pplx-embed-v1-0.6B이 더 큰 Qwen3-Embedding-4B를 5개 중 3개 task에서 능가** (작은 모델의 강세).

**ToolRet (Table 5)** — 도구 검색, nDCG@10 avg: pplx-embed-v1-4B **44.45%**로 전체 2위(NV-Embed-v1 42.71, GritLM-7B 41.13 능가). Web 카테고리 42.07로 특히 강함. INT8임에도 full-precision baseline과 경쟁력.

### 5.3 Internal Benchmarks (프로덕션, Tables 6–8)
- **PPLXQuery2Query**: 검색 로그에서 **같은 destination URL로 이어진 쿼리들은 의미적으로 유사**하다는 통찰로 수동 annotation 없이 query-to-query 데이터 구성. Large(2.4M) corpus에서 pplx-embed-v1-4B가 **R@10 73.46 / R@100 86.17** → Qwen3-Embedding-4B(67.90/81.96)를 **+5.56 / +4.21%p** 능가.
- **PPLXQuery2Doc**: 1B 웹페이지 corpus. 4-시스템(BM25, BGE-M3, mE5-large, Qwen3-0.6B) RRF로 relevance label. Large(30M) corpus에서 pplx-embed-v1-4B가 **R@1000 88.23%(영어)/91.66%(멀티링구얼)** → Qwen3-4B(83.13/88.58) 능가. 1단계 retriever로서 high recall 입증.

### 5.4 Ablation — Diffusion vs Autoregressive Pretraining (§4, Table 9)
동일 Qwen3-0.6B base에서 **diffusion(양방향) vs causal Qwen3** × **mean vs last pooling** 4개 구성을 적은 contrastive pair step으로 비교:

| Base | Pooling | Avg (영어 retrieval) |
|---|---|---|
| Qwen3 (causal) | last | 39.9 |
| Qwen3 (causal) | mean | 38.8 |
| Diffusion | last | 39.7 |
| **Diffusion** | **mean** | **40.6** |

- **Diffusion + mean pooling**이 최고. diffusion 백본이 causal보다 **학습 loss가 유의미하게 낮고**(Figure 2), retrieval 평균 **~1%p 향상**.
- mean pooling은 contextual training에 필수(한 문서에서 다수 청크 임베딩을 한 번에 계산 가능).

<img src="./assets/pplx_fig2_train_loss.png" width="560">

> **Figure 2**: EMA(α=0.02) 학습 loss. Diffusion 백본(특히 mean/last) 구성이 Qwen3 causal 대비 일관되게 낮은 loss 수렴.

### 5.5 Effect of Binary Quantization (§3.3)
- binary 양자화 성능 손실은 **0.6B(2~4.4%p 하락)가 4B(최대 1.6%p)보다 큼**.
- 이유: 4B의 출력 차원 **2560** vs 0.6B의 **1024** → 4B가 압축 표현에 더 많은 정보 보존 가능 → 양자화에 더 robust.

---

## 6. Key Takeaways
1. **Diffusion pretraining = 양방향화**: causal decoder-only(Qwen3)를 diffusion 목적(absorbing state)으로 250B 토큰 추가 학습하면 양방향 인코더로 전환 → mean pooling이 가능해지고, ablation에서 **diffusion+mean이 causal 대비 retrieval +~1%p, loss 더 낮음**.
2. **Native QAT로 압축+품질 동시 달성**: INT8 양자화를 모든 contrastive 단계에 적용(STE 역전파). pplx-embed-v1-4B는 **390 docs/MB**(Qwen3-4B 97의 4배), binary는 **3,125 docs/MB**로 극단적 저장 효율을 누리면서 MTEB-Multi **69.66**(Qwen3 69.60 동등 이상).
3. **Contextual SOTA**: local(in-seq/in-batch)+global dual-loss와 중복 문서 마스킹으로 청크 임베딩에 문서 문맥 주입 → ConTEB에서 **pplx-embed-context-v1-4B 81.96**로 voyage-context-3(79.45)·Anthropic Contextual(72.4) 능가, 신기록.
4. **작은 모델의 강세**: pplx-embed-v1-0.6B가 MIRACL 전 언어에서 Qwen3-0.6B 능가, BERGEN 5개 중 3개에서 **더 큰 Qwen3-4B까지 능가**. instruction prefix 불필요.
5. **프로덕션 검증**: 1B+ 웹페이지 기반 자체 Q2Q/Q2D 벤치마크에서 Qwen3-Embedding-4B 대비 R@10 **+5.56%p**, R@1000 영어 **88.23%** 등 1단계 retriever로서 우위 입증.
6. **양자화 robustness는 차원에 의존**: binary 손실이 0.6B(~2-4.4%p)가 4B(~1.6%p)보다 큰데, 출력 차원(2560 vs 1024)이 클수록 압축 표현에 정보를 더 보존하기 때문.
