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

논문 자체에는 비교 없음. 외부 자료 종합:

| Benchmark | DS-V4-Pro-Max | Qwen3.5-Max (397B/17B) | Qwen3.6-35B-A3B | Gemma 4 31B |
|---|---|---|---|---|
| MMLU-Pro | **87.5** | ~85 | 85.2 | 85.2 |
| GPQA Diamond | 90.1 | 88.4 | **92.7** | 84.3 |
| LiveCodeBench v6 | **93.5** | 83.6 | — | 80.0 |
| Codeforces ELO | **3206** | — | — | 2150 |
| SWE-bench Verified | **80.6** | 76.4 | 78.8 | — |
| Terminal-Bench 2.0 | **67.9** | 52.5 | — | — |
| 컨텍스트 | **1M** | 256K | **1M** | 256K |

→ 1M context 지원은 DS-V4와 Qwen 3.6 양 진영뿐. Gemma 4는 dense 워크스테이션 카테고리.

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
