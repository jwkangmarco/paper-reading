# DeepSeek-V4 — 핵심 질문 정리

> 논문: *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (2026-04-24, preview)
> 모델: V4-Pro (1.6T / 49B activated), V4-Flash (284B / 13B activated), 1M context

---

## 1. 핵심 연구질문

> **트랜스포머의 quadratic attention 병목을 깨고 1M 토큰 컨텍스트를 *루틴하게* 지원할 수 있는가?**

- Test-time scaling과 long-horizon agent 워크플로우 등장으로 ultra-long context 효율성이 결정적 제약
- 기존 오픈모델은 일반 능력은 따라잡았으나 long-context architectural inefficiency 미해결
- → **압축(compression) + 희소성(sparsity)** 결합 hybrid attention 으로 해소 시도

## 2. 언제 적용할 수 있는가

- **1M 토큰급 컨텍스트** (대규모 코드베이스, 멀티 문서 cross-analysis, 누적 reasoning trace agent)
- **Test-time scaling** 활용 시나리오 (Think Max 모드)
- **KV cache / FLOPs 가 서빙 병목**인 환경, shared-prefix 다 요청 챗봇·에이전트
- **자체 호스팅 오픈모델로 frontier 급 능력**이 필요할 때 (특히 코딩·수학·에이전트)
- 적용 약한 곳: 짧은 컨텍스트 단순 chat — 복잡도 대비 이득 적음

## 3. 그래서 어떤 메트릭이 좋아졌는가

### 3-1. 효율성 (vs DeepSeek-V3.2, 1M context)

| 지표 | V3.2 | V4-Flash | V4-Pro |
|---|---|---|---|
| Activated params | 37B | 13B | 49B |
| Single-token FLOPs | 100% | **10%** | **27%** |
| KV cache 크기 | 100% | **7%** | **10%** |

KV cache는 BF16 GQA8 baseline 대비 ~2% 수준.

### 3-2. Base 모델 품질 (Table 1, 동일 프레임)

| Benchmark | V3.2-Base | V4-Pro-Base | Δ |
|---|---|---|---|
| MMLU-Pro | 65.5 | **73.5** | +8.0 |
| SimpleQA-verified | 28.3 | **55.2** | +26.9 |
| FACTS Parametric | 27.1 | **62.6** | +35.5 |
| HumanEval | 62.8 | **76.8** | +14.0 |
| LongBench-V2 | 40.2 | **51.5** | +11.3 |
| BigCodeBench | **63.9** | 59.2 | −4.7 (회귀) |

→ 지식·long-context 차원에서 큰 개선, 추론·언어이해는 plateau 근처.

### 3-3. Post-trained, V4-Pro-Max vs 오픈모델 (Table 6)

| Benchmark | DS-V4-Pro-Max | K2.6-Thinking | GLM-5.1-Thinking |
|---|---|---|---|
| SimpleQA-Verified | **57.9** | 36.9 | 38.1 |
| LiveCodeBench | **93.5** | 89.6 | — |
| Codeforces ELO | **3206** | — | — |
| Apex Shortlist | **90.2** | 75.5 | 72.4 |

오픈모델 중 거의 전 항목 SOTA / 동률.

### 3-4. Frontier closed 와의 격차

| Benchmark | DS-V4-Pro-Max | Opus-4.6 | GPT-5.4 | Gemini-3.1-Pro |
|---|---|---|---|---|
| MMLU-Pro | 87.5 | 89.1 | 87.5 | **91.0** |
| GPQA Diamond | 90.1 | 91.3 | 93.0 | **94.3** |
| HLE | 37.7 | 40.0 | 39.8 | **44.4** |
| Codeforces | **3206** | — | 3168 | 3052 |
| MRCR 1M | 83.5 | **92.9** | — | 76.3 |

→ Reasoning은 3–6개월 뒤짐, Code 경시는 우위.

### 3-5. Gemma 4 / Qwen 3.5·3.6 비교 (외부 자료)

논문 자체에는 비교 없음. 외부 자료 종합 (출처 [E1]–[E10] 참조, 본 문서 끝 References 섹션):

| Benchmark | DS-V4-Pro-Max | Qwen3.5-Max (397B/17B) | Qwen3.6-35B-A3B | Gemma 4 31B |
|---|---|---|---|---|
| MMLU-Pro | **87.5** | ~85 [E5] | 85.2 [E2] | 85.2 [E1] |
| GPQA Diamond | 90.1 | 88.4 [E5] | **92.7** [E2] | 84.3 [E1] |
| LiveCodeBench v6 | **93.5** | 83.6 [E5] | — | 80.0 [E1] |
| Codeforces ELO | **3206** | — | — | 2150 [E7] |
| SWE-bench Verified | **80.6** | 76.4 [E5] | 78.8 [E8] | — |
| Terminal-Bench 2.0 | **67.9** | 52.5 [E10] | — | — |
| 컨텍스트 | **1M** | 256K [E10] | **1M** [E8] | 256K [E1] |

→ 1M context 지원은 DS-V4와 Qwen 3.6 양 진영뿐. Gemma 4는 dense 워크스테이션 카테고리.

⚠ **데이터 신뢰성 주의:** 외부 블로그·리뷰 출처가 섞여 있어 평가 셋업이 모델별로 다를 수 있고, 숫자에 ±오차가 있을 수 있음. 정확한 비교를 위해서는 각 모델의 공식 technical report 확인 필요.

## 4. Trade-offs

**근본:**
- Reasoning에서 frontier proprietary 대비 3–6개월 뒤짐
- 1M retrieval 품질이 128K 이후 가시적 저하 (MRCR 0.94@32K → 0.59@1M)
- 저자 스스로 "아키텍처가 상대적으로 복잡"임을 인정 (검증된 트릭들을 안전하게 다 넣음)

**아키텍처:**
- HCA heavy compression의 정보 손실 → sliding window + dense attention 보완
- CSA top-k 가 비선택 토큰 정보 잃음 → sliding window 보강
- Hash routing(첫 3 MoE 레이어)은 학습성↓ 안정성↑ trade

**학습 안정화 비용:**
- Anticipatory Routing: stale θ_{t-Δt} routing → ~20% wall-clock overhead (spike 시 active)
- mHC: 활성화 메모리 + pipeline 통신 증가 (recompute로 6.7%까지 억제)
- Determinism 위해 split-KV 포기 → wave-quantization 우회 위해 dual-kernel

**Post-training:**
- OPD가 mixed RL 대체 → 10개+ specialist teacher 운용·저장 부담
- FP4 QAT: indexer KV recall 99.7% (소량 손실 허용)

## 5. 추가 — DeepSeek은 효율성 방향으로 차별화하는가? 왜?

**Yes**, 일관된 패턴으로 "효율성 우선" 방향이 선명함:

- **V3 → V3.2 → V4 모든 세대**가 "FLOPs/token, KV cache 효율화"를 핵심 contribution으로 삼음
- V4 의 어휘 자체가 "**efficiency barrier**", "**break the inefficiency**" 등 효율을 내세움
- 새 attention(CSA/HCA), FP4 QAT, MegaMoE, 결정론적 커널 — 모든 레이어에서 cost 절감 누적
- Frontier 격차는 "3–6개월 정도"라고 직접 명시 — 절대 1위가 아닌 **"같은 성능을 1/10 cost로"**가 차별화 포인트

**왜 그 방향을 택했나 (추정):**

1. **하드웨어 제약** — US 칩 수출 통제로 H100/B200 직수입 제한. 같은 성능을 더 적은 FLOPs/메모리로 내는 것이 곧 경쟁력
2. **오픈웨이트 전략** — 자체 호스팅 사용자에게 inference cost / KV cache 가 직접적 비용. Frontier 절대 성능보다 효율성이 채택률 결정
3. **Test-time scaling 의 cost 폭발 대응** — Reasoning 모델은 토큰 수가 폭증. Long context + sparse/compressed attention 없이는 운영 불가
4. **MoE 일관 투자** — 1.6T 총 / 49B 활성으로 frontier 급 성능을 작은 활성 비용에 패키징
5. **인프라 자체 차별화** — TileLang, MegaMoE, FP4 QAT, on-disk KV cache, batch-invariant 커널 등 모델 외 시스템 레이어까지 fully open

**Frontier 진영과의 포지셔닝 차이:**

| 진영 | 추구 방향 | 경쟁 우위 |
|---|---|---|
| GPT / Claude / Gemini | **절대 성능 (특히 reasoning) 1위** | Closed weight + 자본 + 데이터 + scale |
| **DeepSeek** | **동급 성능을 최소 cost 로 + open weight** | 효율성 + 자체 호스팅 가능성 |

논문 conclusion 도 "low-latency architectures and system techniques to make long-context deployment more responsive" 라며 효율성/시스템 방향을 미래 과제로 명시.

→ **요약: "효율성을 통한 frontier 도달"이 DeepSeek 의 일관된 노선**이며, 이는 (a) 자원 제약, (b) 오픈웨이트 비즈니스 모델, (c) test-time scaling 시대의 inference cost 압력 이라는 구조적 조건에 대한 합리적 응답으로 보임.

---

## 6. 논문 목차별 한 줄 요약

- **§1 Introduction** — 1M context의 efficiency barrier를 깨기 위한 V4 시리즈 (Pro 1.6T/49B, Flash 284B/13B) 출시 동기
- **§2 Architecture** — V3 기반 + 새 attention(CSA/HCA) + mHC(residual 강화) + Muon optimizer 도입
  - **§2.1 Designs Inherited from V3** — DeepSeekMoE + MTP 유지, Sigmoid → Sqrt(Softplus) 변경, 일부 dense FFN을 Hash-routed MoE로 대체
  - **§2.2 Manifold-Constrained Hyper-Connections** — residual mapping을 doubly stochastic manifold(Birkhoff polytope)로 제한해 spectral norm ≤1 보장, 깊은 stack 안정화
  - **§2.3 Hybrid Attention (CSA + HCA)** — CSA는 m개 KV 압축 후 sparse top-k(Lightning Indexer), HCA는 m'≫m로 더 강하게 압축 후 dense — interleaved 사용
  - **§2.4 Muon Optimizer** — hybrid Newton-Schulz 10회로 orthogonalize, AdamW를 임베딩·헤드·norm 모듈에만 유지
- **§3 General Infrastructures** — 학습/추론 인프라 전반의 효율 최적화
  - **§3.1 Fine-Grained EP Overlap** — Expert를 wave로 쪼개 통신·연산 오버랩 (MegaMoE 커널, 1.5–1.96× 속도)
  - **§3.2 TileLang** — DSL + Z3 SMT solver + Host Codegen으로 host overhead를 µs 단위로 압축
  - **§3.3 Batch-Invariant Deterministic Kernels** — bitwise 재현성 위해 split-KV 포기, dual-kernel attention + 결정론적 reduction
  - **§3.4 FP4 QAT** — MoE 가중치와 CSA indexer QK path를 MXFP4로 양자화, 99.7% recall 유지
  - **§3.5 Training Framework** — Muon용 hybrid ZeRO, mHC fused kernel, 2-stage CP, tensor-level activation checkpointing
  - **§3.6 Inference Framework** — hybrid attention용 customized KV cache layout + on-disk shared-prefix 캐시 (3가지 SWA 전략)
- **§4 Pre-Training** — 32T(Flash)/33T(Pro) 토큰 학습 + 안정화 트릭
  - **§4.1 Data Construction** — V3 대비 long-document·multilingual·agentic 데이터 강화, 32T+ 토큰
  - **§4.2 Pre-Training Setups** — Flash 43L/d=4096, Pro 61L/d=7168, MoE 6/256~6/384 활성, 1M까지 단계적 sequence 확장; Anticipatory Routing + SwiGLU Clamping으로 spike 억제
  - **§4.3 Evaluations** — V3.2-Base 대비 V4-Flash가 더 적은 파라미터로 우수, V4-Pro는 거의 모든 항목 SOTA (Table 1)
- **§5 Post-Training** — mixed RL을 OPD(On-Policy Distillation)로 대체한 새 파이프라인
  - **§5.1 Post-Training Pipeline** — specialist(SFT+GRPO) → unified model을 OPD로 통합, Think Max 모드와 GRM(generative reward) 도입
  - **§5.2 RL/OPD Infrastructures** — FP4 rollout, full-vocab logit distillation, preemptible 토큰-단위 WAL, agentic sandbox(DSec)
  - **§5.3 Standard Benchmark Evaluation** — V4-Pro-Max가 오픈모델 SOTA, Codeforces 3206, frontier 대비 reasoning에서 3–6개월 뒤짐 (Table 6, 7)
  - **§5.4 Real-World Tasks** — 중국어 글쓰기·검색·white-collar·R&D 코드에서 Gemini-3.1/Opus-4.5/4.6와 win-rate 비교
- **§6 Conclusion, Limitations, Future** — 아키텍처 복잡성을 인정, 향후 simplify + multimodal + sparse embedding 방향 명시
- **§A Author List & Acknowledgment** — 저자 명단
- **§B Evaluation Details** — RAG vs Agentic Search 표, 중국어 functional/creative writing 세부 비교

---

## References

### 논문 (1차 자료)

- **[P1]** DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. 2026-04-24. https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
  - 본 문서 §1–§4 와 §5 의 직접 인용은 모두 이 논문의 Table 1, 6, 7, Figure 1 등에서 발췌

### 외부 자료 (Gemma 4 / Qwen 비교 — 2차 자료)

- **[E1]** Aurigait. *Gemma 4 by Google: Specs, Benchmarks, and How to Run It Locally (2026 Guide)*. https://aurigait.com/blog/gemma-4-features-benchmarks-guide/
- **[E2]** HuggingFace. *Qwen/Qwen3.6-35B-A3B model card*. https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- **[E3]** HuggingFace. *Qwen/Qwen3.6-27B model card*. https://huggingface.co/Qwen/Qwen3.6-27B
- **[E4]** Moksh S. *Gemma 4 Benchmarks: The Numbers That Actually Matter*. Medium, 2026-04. https://medium.com/@moksh.9/heres-a-tighter-benchmark-focused-blog-post-501c5ea829f4
- **[E5]** Techie007. *Qwen 3.5: The Complete Guide — Benchmarks, Local Setup*. Substack, 2026. https://techie007.substack.com/p/qwen-35-the-complete-guide-benchmarks
- **[E6]** Qwen Team. *Qwen3.5: Towards Native Multimodal Agents*. https://qwen.ai/blog?id=qwen3.5
- **[E7]** BenchLM. *Gemma 4 31B Benchmarks 2026*. https://benchlm.ai/models/gemma-4-31b
- **[E8]** AImadetools. *Qwen 3.6 vs 3.5: 1M Context, 78.8% SWE-bench*. https://www.aimadetools.com/blog/qwen-3-6-vs-3-5/
- **[E9]** Maniac.ai. *Qwen 3.5 vs Gemma 4: benchmark-by-size comparison*. https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size
- **[E10]** Digital Applied. *Qwen 3.5 Medium Models: Benchmarks, Pricing, and Guide*. https://www.digitalapplied.com/blog/qwen-3-5-medium-model-series-benchmarks-pricing-guide

### 추가 참조 (오픈모델 리더보드)

- VERTU. *Open Source LLM Leaderboard 2026: Rankings, Benchmarks*. https://vertu.com/lifestyle/open-source-llm-leaderboard-2026-rankings-benchmarks-the-best-models-right-now
- iternal.ai. *LLM Benchmarks 2026: 30+ Models Ranked*. https://iternal.ai/llm-selection-guide
- HuggingFace. *Welcome Gemma 4 (official blog)*. https://huggingface.co/blog/gemma4

### 4번 (Trade-offs) · 5번 (효율성 노선) 섹션

- 모두 **[P1]** 의 §6 "Conclusion, Limitations, and Future Directions" 와 §3.5–§3.6 (training/inference framework) 에서 도출
- "frontier 대비 3–6개월 뒤짐" 표현은 **[P1]** §1 의 Reasoning 항목 직접 인용
- 미국 칩 수출 통제 등 외부 거시 컨텍스트는 일반 공개 정보 (별도 출처 인용 없이 필자 해석)
