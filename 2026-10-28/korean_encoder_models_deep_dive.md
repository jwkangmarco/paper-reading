# 한국어 Encoder 4종 상세 비교: Kiwi COCO-LM · A.X-Encoder · KF-DeBERTa · KoELECTRA-v3

> **작성일**: 2026-10-28

---

## 0. 한눈에 비교

### 0.1 아키텍처

| 항목 | Kiwi COCO-LM | A.X-Encoder-base | KF-DeBERTa-base | KoELECTRA-base-v3 |
|---|---|---|---|---|
| 개발 | kiwi-farm (bab2min, 개인) | SKT AI Model Lab | 카카오뱅크 + 에프엔가이드 | monologg (개인) |
| 공개 | 2026-09 | 2025-07 | 2023-12 (논문 2023-10) | 2020-10 |
| 계열 | ModernBERT | ModernBERT | DeBERTa-v2 | ELECTRA (BERT 구조) |
| 레이어 / hidden / heads | 22 / 768 / 12 | 22 / 768 / 12 | 12 / 768 / 12 | 12 / 768 / 12 |
| FFN | GeGLU, 1152 | GeGLU, 1152 | GELU, 3072 | GELU, 3072 |
| 위치 표현 | RoPE (global θ=160K / local θ=10K) | RoPE (global θ=160K / local θ=10K) | 상대 위치 (disentangled attention, c2p+p2c, bucket 256) | 학습된 절대 위치 |
| Attention | 3층마다 global, 나머지는 **local 256** | 3층마다 global, 나머지는 **local 128** | full | full |
| 최대 길이 | **8,192** | **16,384** | 512 (학습 기준) | 512 |
| 토크나이저 | Byte-level BPE (kiwipiepy 형태소 경계 반영) | WordPiece + NFC 정규화 | WordPiece (cased) | Mecab + WordPiece |
| vocab | 64,000 | 50,000 | 130,000 | 35,000 |
| 파라미터 (임베딩 포함) | 161M (임베딩 49M) | 149M (임베딩 38M) | ≈186M (임베딩 100M) | ≈113M (임베딩 27M) |
| 기본 pooling (`classifier_pooling`) | CLS | mean | (pooler) CLS | CLS |
| 라이선스 | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 |

### 0.2 사전학습 레시피

| 항목 | Kiwi COCO-LM | A.X-Encoder-base | KF-DeBERTa-base | KoELECTRA-base-v3 |
|---|---|---|---|---|
| 초기화 | **ModernBERT-base 가중치 재사용** (vocab 교체) | from scratch (SKT 자체 개발) | 비공개 | from scratch |
| 목적함수 | **COCO-LM**: RTD + Corrective LM + Seq Contrastive | 비공개 (MLM 헤드 공개, `<mask>` 예측 가능) | **MLM** (DeBERTa-v2). v3의 RTD는 성능이 떨어져 기각 | **ELECTRA RTD** |
| 보조 모델 | Generator 6층, 80M | 없음 | 없음 | Generator (hidden 256, 4 heads, 크기 비율 0.33) |
| 데이터 | 한국어 12.1B + 영어·다국어 15.0B = **27.2B 토큰** | 비공개 | 범용 + 금융 말뭉치 (규모 비공개) | **34GB** (v1/v2의 14GB + 모두의 말뭉치 20GB) |
| 학습량 | 5 epoch, **≈136B 토큰** | 비공개 | 비공개 | 1.5M step × 256 × 512 |
| 배치 | 1,024 seq / 512K 토큰 (16 micro-step, 길이 혼합) | 비공개 | 비공개 | 256 seq |
| 하드웨어 | **RTX 4070 Ti SUPER 1장, 51일** | SKT TITAN 슈퍼컴퓨터 | 비공개 | TPU v3-8, 14일 |
| 옵티마이저 | Muon + stochastic rounding (BF16 전체) | 비공개 | 비공개 | ELECTRA 기본 (LR 2e-4) |

### 0.3 벤치마크 

| 벤치마크 (출처)                                             | Kiwi      | A.X       | KF-DeBERTa | KoELECTRA-v3 |
| ----------------------------------------------------- | --------- | --------- | ---------- | ------------ |
| KLUE 6과제 평균 (Kiwi 카드)                                 | 82.34     | **82.91** | 82.89      | 77.79        |
| GLUE 9과제 평균 (Kiwi 카드)                                 | **83.02** | 74.99     | 71.34      | 69.49        |
| KoBEST 4과제 평균 (A.X 카드)                                |           | **85.50** | 83.25      |              |
| KLUE 3과제 평균 (A.X 카드)                                  |           | **86.10** | 85.80      |              |
| KLUE 8과제 평균 (KF-DeBERTa 카드)                           |           |           | **82.83**  | 77.34        |
| 범용 4과제 평균 (KF-DeBERTa 카드: NSMC, PAWS, KorNLI, KorSTS) |           |           | **87.01**  | 85.71        |
| 금융 6과제 평균 (KF-DeBERTa 카드)                             |           |           | **95.27**  | 93.90        |

---

## 1. Kiwi COCO-LM (kiwi-farm/kiwi-coco-lm-base)

### 1.1 개요

ModernBERT-base를 **한·영 이중언어 모델로 이어서 학습**(continued pretraining)한 encoder다. 한국어 형태소 분석기 Kiwi(kiwipiepy) 개발자가 만들었다. 원 모델의 영어 능력을 유지하면서 한국어 성능을 A.X·KF-DeBERTa 수준으로 올린 것이 핵심이다. 그 결과 **KLUE와 GLUE 모두에서 상위권**에 드는 유일한 모델이다. 공개된 것은 Discriminator이고, Generator는 [별도 저장소](https://huggingface.co/kiwi-farm/kiwi-coco-lm-base-generator)에 있다.

### 1.2 아키텍처

ModernBERT-base 구조를 그대로 쓴다 (`ModernBertForMaskedLM`).

| 항목 | 값 |
|---|---|
| 레이어 | 22 (Discriminator) / 6 (Generator) |
| hidden / heads / FFN | 768 / 12 / 1152 (GeGLU) |
| Attention | 3층마다 1층은 full attention, 나머지는 sliding window. `local_attention=256` (원본 ModernBERT-base는 128) |
| RoPE | full θ = 160,000 / sliding θ = 10,000 |
| 정규화 · bias | pre-norm LayerNorm, bias 없음 (`norm_bias=false`, `mlp_bias=false`) |
| 최대 길이 | 8,192 |
| 임베딩 | 입출력 임베딩 공유 (`tie_word_embeddings=true`) |
| 파라미터 | Discriminator 161M (임베딩 64K × 768 ≈ 49M, 사전학습 전용 RTD/SCL 헤드 ≈ 1.3M 포함), Generator 80M |
| SCL 투영 차원 | 128 (`scl_dim`) |

### 1.3 토크나이저

- ModernBERT-base vocab에서 **잘 쓰이지 않는 외국어 토큰을 빼고, 그 자리에 한국어 토큰을 넣어** 64K를 만들었다. 겹치는 토큰은 원 가중치를 그대로 물려받는다.
- 한국어 토큰은 kiwipiepy의 **형태소 경계를 고려한 Byte-level BPE trainer**로 학습했다.
- byte-level이라 **UNK가 없다.** 희귀 음절 16개 중 0개가 UNK로 처리된다.

모델 카드의 토크나이저 효율 비교 (한국어 문서 534어절, 영어 문서 693단어):

| 모델 | 방식 | vocab | 한글 포함 vocab 비율 | 한국어 어절당 토큰 | 영어 단어당 토큰 | 희귀 음절 UNK (/16) |
|---|---|---:|---:|---:|---:|---:|
| Kiwi COCO-LM | Byte-level BPE | 64,000 | 41.97% | 2.37 | 1.31 | **0** |
| A.X-Encoder | WordPiece | 50,000 | 48.17% | 2.54 | 1.89 | **0** |
| KF-DeBERTa | WordPiece | 130,000 | 80.40% | **2.19** | 1.62 | 10 |
| KoELECTRA | WordPiece | 35,000 | 82.92% | 2.44 | 2.48 | 10 |
| KLUE RoBERTa | WordPiece | 32,000 | 88.89% | 2.41 | 2.54 | 14 |
| ModernBERT | Byte-level BPE | 50,368 | 0.05% | 7.12 | 1.30 | 0 |
| mmBERT | BPE (byte fallback) | 256,000 | 0.90% | 3.30 | **1.24** | 0 |

→ 한국어 효율은 한국어 전용 WordPiece와 비슷하고(2.37), 영어 효율은 원본 ModernBERT와 거의 같다(1.31). KF-DeBERTa는 한국어 효율이 가장 좋지만, vocab이 2배이고 영어 효율과 UNK에서 불리하다.

### 1.4 사전학습 목적함수: COCO-LM

ELECTRA의 RTD는 "이 토큰이 바뀌었는가"라는 이진 신호만 준다. COCO-LM은 여기에 두 가지 학습 신호를 더한다.

1. **RTD (Replaced Token Detection)**: Generator(6층)가 마스크 위치를 그럴듯한 토큰으로 바꾸고, Discriminator가 각 토큰이 바뀌었는지 판별한다.
2. **CLM (Corrective Language Modeling)**: 판별에 더해 **원래 토큰을 복원**한다. 토큰 수준의 학습 신호를 강화한다.
3. **SCL (Sequence Contrastive Learning)**: 원본의 일부를 자른(crop) 시퀀스와 Generator가 바꾼 시퀀스를 positive 쌍으로, 배치 안의 다른 시퀀스를 negative로 삼아 [CLS] 표현을 가깝게 만든다. 시퀀스 수준 표현을 학습한다.

**Kiwi가 SCL을 바꾼 부분**

| 변경 | 내용 | 이유 |
|---|---|---|
| whole-word crop | 서브워드가 아니라 어절 경계 단위로 crop | 단어가 중간에 잘리지 않게 |
| 긴 시퀀스는 구간 crop | 1,024 토큰을 넘는 시퀀스는 최대 1,024 토큰 구간을 무작위로 고른 뒤 그 안에서 crop | 길이 정보만으로 positive를 찾는 지름길(shortcut) 학습 방지 |
| crop 쪽 stop-gradient | crop 시퀀스는 forward만 하고 gradient를 흘리지 않음 | 배치 전체(1,024개)의 crop 표현을 activation 저장 없이 미리 계산 → **GradCache 없이 16GB GPU 1장에서 배치 1,024 이상 대조학습** |

### 1.5 학습 설정

**배치 구성 (길이 혼합)**: optimizer 1회 업데이트 = 16 micro-step = 1,024 시퀀스 = **512K 토큰**. micro-step마다 길이를 다르게 하되, 토큰 수는 32K로 일정하게 맞춘다.

| Seq Length | Batch | micro-step 수 |
|---|---|---|
| 8,192 | 4 | 4 |
| 4,096 | 8 | 2 |
| 2,048 | 16 | 2 |
| 1,024 | 32 | 2 |
| 512 | 64 | 2 |
| 256 | 128 | 2 |
| 128 | 256 | 2 |

**최적화**
- [Muon optimizer + Stochastic Rounding](https://github.com/bab2min/muon_sr): weight, activation, gradient, optimizer state를 **모두 BF16**으로 둔다.
- Liger Kernel과 Flash Attention 2를 적용했다.
- **NVIDIA RTX 4070 Ti SUPER (16GB) 1장으로 약 1,239시간(51일)** 학습했다.

**데이터 (27.2B 토큰 × 5 epoch ≈ 136B 토큰)**. 매 epoch 마스킹 seed를 바꿔 같은 문서라도 입력이 달라지게 했다.

| 언어 | 데이터 | 토큰 (M) |
|---|---|---:|
| 한국어 | 한국어 위키백과 | 601.4 |
| 한국어 | KcBERT 댓글 | 2,747.8 |
| 한국어 | 웹 커뮤니티 댓글 | 133.7 |
| 한국어 | Koreascience.kr 초록 | 214.9 |
| 한국어 | 나무위키 | 1,149.8 |
| 한국어 | AIHub 대규모 구매도서 | 1,118.4 |
| 한국어 | AIHub 전문분야 | 1,608.4 |
| 한국어 | AIHub 뉴스 | 439.8 |
| 한국어 | AIHub 대화 | 71.4 |
| 한국어 | 국립국어원 구어 | 244.7 |
| 한국어 | 국립국어원 문어 | 1,452.2 |
| 한국어 | 국립국어원 신문 | 2,280.2 |
| 한국어 | 국립국어원 온라인 게시자료 | 80.4 |
| | **한국어 소계** | **12,143.3** |
| 영어 | English Wikipedia | 8,191.2 |
| 다국어 | FLAN Zero-shot (train, 무작위 샘플) | 6,833.1 |
| | **영어·다국어 소계** | **15,024.3** |

→ 136B 토큰 ÷ 512K 토큰/step으로 계산하면 **약 26만 optimizer step**이다 (카드에 명시된 값은 아니고 역산한 값).

### 1.6 벤치마크 (모델 카드)

**KLUE** (base급)

| Model            | Avg       | YNAT (Macro-F1) | STS (Pearson / F1) | NLI (Acc) | NER (Entity / Char F1) | RE (Micro-F1 / AUC) | MRC (EM / ROUGE-W) |
| ---------------- | --------- | --------------- | ------------------ | --------- | ---------------------- | ------------------- | ------------------ |
| **Kiwi COCO-LM** | 82.34     | 86.15           | 92.23 / 86.90      | **87.87** | 84.44 / 93.02          | 65.36 / 69.67       | 71.38 / 77.00      |
| A.X-Encoder      | **82.91** | 85.77           | 92.58 / 86.33      | 87.37     | **87.63 / 93.29**      | 65.74 / 68.40       | **74.97 / 79.70**  |
| KF-DeBERTa       | 82.89     | **86.53**       | 92.63 / **87.45**  | 87.73     | 85.26 / 92.62          | **67.16 / 70.38**   | 72.92 / 77.73      |
| KLUE RoBERTa     | 80.44     | 85.99           | 92.68 / 85.89      | 85.63     | 83.47 / 90.77          | 63.11 / 63.54       | 68.98 / 73.60      |
| KoELECTRA        | 77.79     | 86.05           | **93.13** / 86.23  | 87.23     | 85.59 / 92.61          | 59.09 / 53.44       | 55.18 / 61.68      |
| ModernBERT       | 62.36     | 77.05           | 76.63 / 69.62      | 67.40     | 74.96 / 86.15          | 39.40 / 29.13       | 39.51 / 44.02      |
| mmBERT           | 79.33     | 84.24           | 90.51 / 81.58      | 83.30     | 85.49 / 92.42          | 61.95 / 69.06       | 65.26 / 70.65      |

**GLUE**

| Model            | Avg       | SST2      | CoLA      | MRPC (Acc / F1)   | STS-B (P / S)     | QQP (Acc / F1)    | MNLI      | QNLI      | RTE       | WNLI      |
| ---------------- | --------- | --------- | --------- | ----------------- | ----------------- | ----------------- | --------- | --------- | --------- | --------- |
| **Kiwi COCO-LM** | **83.02** | 93.01     | 61.16     | **90.93 / 93.47** | **91.00 / 90.79** | 91.12 / 88.10     | 87.64     | 93.26     | **83.03** | **56.34** |
| A.X-Encoder      | 74.99     | 90.48     | 24.23     | 85.05 / 89.32     | 87.55 / 87.41     | 90.28 / 86.90     | 83.53     | 91.05     | 66.07     | 56.34     |
| KF-DeBERTa       | 71.34     | 86.47     | 9.25      | 85.05 / 89.57     | 86.81 / 86.54     | 88.34 / 84.10     | 78.46     | 88.19     | 63.18     | 56.34     |
| KLUE RoBERTa     | 70.75     | 87.27     | 4.45      | 83.82 / 88.26     | 85.24 / 84.92     | 89.44 / 85.50     | 79.78     | 86.78     | 63.54     | 56.34     |
| KoELECTRA        | 69.49     | 81.65     | 11.84     | 86.03 / 89.69     | 84.48 / 84.24     | 86.35 / 80.87     | 72.73     | 84.94     | 62.09     | 56.34     |
| ModernBERT       | 80.30     | **95.30** | **61.92** | 86.52 / 90.30     | 90.00 / 89.95     | 90.83 / 87.74     | **88.73** | **93.34** | 66.43     | 49.30     |
| mmBERT           | 81.01     | 93.69     | 60.79     | 87.75 / 91.38     | 90.81 / 90.79     | **91.35 / 88.42** | 87.46     | 93.06     | 80.14     | 43.66     |

**관찰**
- KLUE에서 A.X·KF-DeBERTa와 0.5점 이내다. NLI는 1위, NER과 MRC는 A.X보다 낮다.
- GLUE에서는 ModernBERT 원본(80.30)보다 **오히려 높다**(83.02). 한국어를 추가로 학습했는데도 영어 성능을 잃지 않았다. 특히 RTE가 66→83으로 크게 올랐는데, COCO-LM 목적함수나 FLAN 데이터의 영향으로 추정된다(카드에 분석은 없음).
- 한국어 전용 모델들은 CoLA가 4~24로 무너진다. 영어 문법성 판단은 이중언어 모델만 할 수 있다.

### 1.7 한계 (카드 명시)

- 한국어·영어 외 언어는 성능을 보장하지 않는다 (외국어 토큰을 제거했기 때문).
- 정제되지 않은 웹 댓글(KcBERT 댓글 등)이 들어 있어 편향과 부적절한 표현이 있을 수 있다.
- COCO-LM으로 학습했기 때문에 **`[MASK]` 빈칸 채우기에는 적합하지 않다.** 미세조정해서 써야 한다.
- 8K 입력이 가능하지만 평가는 대부분 짧은 문장이나 문단 단위로만 했다.

---

## 2. A.X-Encoder-base (skt/A.X-Encoder-base)

### 2.1 개요

SKT가 아키텍처 설계, 데이터 큐레이션, 학습까지 **모두 자체 개발**한 한국어 encoder다. 학습은 SKT의 슈퍼컴퓨터 **TITAN**에서 했다. 원래 용도는 **A.X LLM의 사전학습 데이터 큐레이션**으로, 교육적 가치, 도메인, 난이도를 판별하는 문서 분류기 역할을 한다. 그래서 **긴 문서 처리(16K)와 추론 속도**를 우선했다. 2026년 기준으로 한국어 임베딩·리랭커 모델 대부분이 이 모델을 백본으로 쓴다(KURE-v2, colbert-ko 등, 조사 문서 참조).

### 2.2 아키텍처 (config 기준)

| 항목 | 값 |
|---|---|
| 구조 | ModernBERT (`ModernBertForMaskedLM`) |
| 레이어 / hidden / heads / FFN | 22 / 768 / 12 / 1152 (GeGLU) |
| Attention | 3층마다 global, 나머지 **local window 128** (ModernBERT-base와 동일) |
| RoPE | global θ = 160,000 / local θ = 10,000 |
| 최대 길이 | **16,384** (ModernBERT 원본 8,192의 2배) |
| vocab | 50,000 WordPiece, NFC 정규화, BERT 사전 토큰화 |
| 파라미터 | 149.4M (임베딩 38.4M) |
| 기본 pooling | **mean** (`classifier_pooling: mean`) |
| 기타 | Flash Attention 2 권장, BF16 배포 |

### 2.3 사전학습 레시피


### 2.4 벤치마크 (모델 카드) 

**KoBEST (F1)**

| Method | BoolQ | COPA | SentiNeg | WiC | Avg |
|---|---|---|---|---|---|
| klue/roberta-base | 72.04 | 65.14 | 90.39 | 78.19 | 76.44 |
| kakaobank/kf-deberta-base | 81.30 | 76.50 | 94.70 | 80.50 | 83.25 |
| **skt/A.X-Encoder-base** | **84.50** | **78.70** | **96.00** | **80.80** | **85.50** |

**KLUE**

| Method | NLI (Acc) | STS (F1) | YNAT (Acc) | Avg |
|---|---|---|---|---|
| klue/roberta-base | 84.53 | 84.57 | 86.48 | 85.19 |
| kakaobank/kf-deberta-base | 86.10 | 84.30 | **87.00** | 85.80 |
| **skt/A.X-Encoder-base** | **87.00** | **84.80** | 86.50 | **86.10** |

**종합 평균 (카드 그림)**: klue/roberta-base 80.19, **klue/roberta-large 84.09**, kf-deberta-base 84.34, **A.X-Encoder-base 85.47**. 위 두 표의 7과제 단순 평균과 일치한다(검산함). roberta-large의 개별 과제 점수는 카드에 없다. → **base 크기로 RoBERTa-large를 넘는다**는 것이 카드의 주장이다.

**추론 속도 (A100, KF-DeBERTa와 비교, 카드 그림)**
- 512~1,024 토큰에서는 비슷하다 (약 0.015~0.017초).
- 2,048 토큰부터 차이가 벌어진다. 16,384 토큰에서 KF-DeBERTa는 약 2초대, A.X는 약 0.08초다(로그축 그림을 읽은 값).
- 그림 속 표기는 "x3 Faster"지만, 로그축 값을 그대로 읽으면 16K에서 차이가 그보다 훨씬 크다. 배치 크기, 정밀도, attention 구현 같은 측정 조건은 공개되지 않았다. KF-DeBERTa는 512 토큰으로 학습된 모델이라 16K 입력은 성능 비교로서 의미가 없고, 속도 비교로만 봐야 한다.


---

## Sources

- [kiwi-farm/kiwi-coco-lm-base](https://huggingface.co/kiwi-farm/kiwi-coco-lm-base) · [Generator](https://huggingface.co/kiwi-farm/kiwi-coco-lm-base-generator) · [muon_sr](https://github.com/bab2min/muon_sr) · [korean-tokenizer-analysis](https://github.com/bab2min/korean-tokenizer-analysis)
- [COCO-LM (arXiv 2102.08473)](https://arxiv.org/abs/2102.08473)
- [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) · [PyTorchKR landscape issue #882](https://github.com/PyTorchKR/landscape/issues/882)
- [ModernBERT (arXiv 2412.13663)](https://arxiv.org/abs/2412.13663)
- [kakaobank/kf-deberta-base](https://huggingface.co/kakaobank/kf-deberta-base) · [GitHub](https://github.com/kakaobank/KF-DeBERTa) · [카카오뱅크 개발기](https://ai.kakaobank.com/f982c5b8-9cbd-4ef1-8ebd-d7359a70284b) · [HCLT 2023 논문 (koreascience)](https://koreascience.kr/article/CFKO202306643316013.page)
- [monologg/KoELECTRA (GitHub)](https://github.com/monologg/KoELECTRA) · [koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
