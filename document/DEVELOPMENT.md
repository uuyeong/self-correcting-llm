# 개발 가이드 (Self-Correcting LLM)

> 혼자 진행하는 6주 프로젝트의 브랜치·커밋·작업 흐름 규칙.

---

## 브랜치 전략

### 구조

```
main
 └─ phase/1-data-hidden-states
 └─ phase/2-probing-exp1
 └─ phase/3-regen-pipeline-exp2-3
 └─ phase/4-comparison-exp4-5-6
 └─ phase/5-visualization-demo
 └─ phase/6-report-submission
```

### 규칙

| 규칙 | 내용 |
|------|------|
| `main` 보호 | 항상 실행 가능한 상태 유지. 직접 push 금지 (PR만) |
| 브랜치 단위 | Phase 하나 = 브랜치 하나. Phase 내 소작업은 같은 브랜치에서 commit으로 구분 |
| 머지 방식 | `git merge --no-ff` (merge commit으로 phase 경계 명확히) |
| 완료 조건 | 해당 phase 실험 결과 파일(`results/exp*.json`)과 그래프(`results/figures/`)가 존재할 때 main 머지 |

### 브랜치 생성 예시

```bash
git checkout main
git pull origin main
git checkout -b phase/1-data-hidden-states
```

---

## 커밋 메시지 규칙

### 형식

```
<type>(<scope>): <한 줄 요약>

[선택] 본문 — 왜 이 변경이 필요했는지, 어떤 결정을 내렸는지
```

### type 목록

| type | 사용 시점 |
|------|----------|
| `feat` | 새로운 기능·모듈 추가 |
| `fix` | 버그 수정 |
| `exp` | 실험 실행 및 결과 기록 |
| `refactor` | 기능 변화 없는 코드 정리 |
| `data` | 데이터 전처리·로더 변경 |
| `viz` | 시각화 코드·그래프 추가/수정 |
| `docs` | 문서 수정 (README, 구체화문서 등) |
| `chore` | 설정 파일, 의존성, .gitignore 등 |

### scope 목록

`data` · `llm` · `probing` · `detector` · `regen` · `plots` · `exp1` ~ `exp6` · `demo` · `config` · `notebook`

### 예시

```
feat(llm): add DoLa-style logit contrast generation

상위/하위 레이어 logit 차이를 최종 logit에 합산하는 방식으로 구현.
전체 디코딩 루프 수정 없이 근사 가능.

feat(probing): implement per-layer LogisticRegression probing classifier

exp(exp1): run layer-wise AUROC — best layer=18, AUROC=0.71

결과: results/exp1_aurocs.npy, figures/exp1_layer_auroc_heatmap.pdf
→ config.BEST_LAYER = 18 로 업데이트

fix(detector): correct threshold boundary condition (MID upper bound)

data(loader): add domain keyword heuristic for PopQA domain split

viz(plots): add t-SNE for 3B vs 8B hidden state separability (exp4)

chore(config): set BEST_LAYER = 18 after exp1 results
```

### 커밋 단위 기준

- 함수/클래스 하나 완성 → 커밋
- 실험 하나 실행·결과 확인 → 커밋 (`exp` type)
- 여러 파일에 걸친 기능은 연관 파일 모두 완성 후 하나의 커밋

---

## 주간 Phase & 브랜치 매핑

| Week | 기간 | 브랜치 | 핵심 완료 조건 |
|------|------|--------|--------------|
| 1 | 5/11~5/17 | `phase/1-data-hidden-states` | TruthfulQA/HaluEval/PopQA 로드 성공 + 8B hidden states shape 확인 |
| 2 | 5/18~5/24 | `phase/2-probing-exp1` | `exp1_aurocs.npy` 생성, AUROC 히트맵 PDF, `config.BEST_LAYER` 설정 |
| 3 | 5/25~5/31 | `phase/3-regen-pipeline-exp2-3` | `exp2_strategy_results.json`, `exp3_mid_comparison.json` 생성 |
| 4 | 6/1~6/7 | `phase/4-comparison-exp4-5-6` | `exp4~6` JSON + 그래프 6종 완성 |
| 5 | 6/8~6/14 | `phase/5-visualization-demo` | `precomputed_results.json` 채움, HF Spaces 배포 URL 확보 |
| 6 | 6/15~6/23 | `phase/6-report-submission` | CVPR LaTeX PDF 2페이지, 코드 zip 제출 완료 |

---

## 결과물 저장 규칙

```
results/
├── exp1_aurocs.npy
├── exp1_probing_result.pkl      # ProbingResult 객체 (후속 실험에서 로드)
├── exp2_strategy_results.json
├── exp3_mid_comparison.json
├── exp4_size_comparison.json
├── exp5_domain_aurocs.json
├── exp6_baseline_comparison.json
└── figures/
    ├── exp1_layer_auroc_heatmap.pdf
    ├── exp2_strategy_comparison.pdf
    ├── exp3_mid_ab.pdf
    ├── exp4_auroc_3b.pdf
    ├── exp4_auroc_8b.pdf
    ├── exp4_tsne.pdf
    ├── exp5_domain_heatmap.pdf
    └── exp6_baseline_comparison.pdf
```

- `results/` 폴더의 **JSON과 PDF는 git에 포함** (결과 재현 증거)
- `.npy` / `.pkl` (수백 MB 이상) → `.gitignore`로 제외, Google Drive에 별도 보관
- `hidden_states_cache/` → git 제외, Drive 동기화

---

## Colab 작업 흐름

```bash
# 1. Colab에서 repo clone
!git clone https://github.com/uuyeong/self-correcting-llm.git
%cd self-correcting-llm

# 2. 작업할 브랜치 checkout
!git checkout phase/2-probing-exp1

# 3. 실험 실행
!python experiments/exp1_layer_analysis.py

# 4. 결과 파일 확인 후 commit & push
!git add results/ config.py
!git commit -m "exp(exp1): layer-wise AUROC — best_layer=18, AUROC=0.71"
!git push origin phase/2-probing-exp1
```

---

## PR(Pull Request) 규칙

제목: `[Phase N] <한 줄 완료 요약>`

예:
- `[Phase 2] Probing classifier training + Exp 1 layer AUROC complete`
- `[Phase 3] 3-tier regeneration pipeline + Exp 2 & 3 results`

본문 포함 항목:
- 완료된 실험 결과 수치 (AUROC, Accuracy 등)
- 다음 Phase에서 필요한 정보 (예: `BEST_LAYER = 18`)
- 주요 결정 사항 또는 리스크 대응 내용

---

## 파일 명명 규칙

| 대상 | 규칙 | 예시 |
|------|------|------|
| 실험 스크립트 | `exp{N}_{설명}.py` | `exp1_layer_analysis.py` |
| 결과 JSON | `exp{N}_{설명}.json` | `exp2_strategy_results.json` |
| 그래프 PDF | `exp{N}_{설명}.pdf` | `exp4_tsne.pdf` |
| 캐시 npy | `exp{N}_{모델태그}_hidden_states.npy` | `exp4_3b_hidden_states.npy` |
| 브랜치 | `phase/{N}-{kebab-case}` | `phase/3-regen-pipeline-exp2-3` |
