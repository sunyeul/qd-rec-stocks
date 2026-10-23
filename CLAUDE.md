# qd-rec-stocks

Sakana AI의 [QD-Rec](https://github.com/SakanaAI/QD-Rec)을 미국 주식 아이디어 발굴에 응용하는 개인 연구 프로젝트.
"종목 + 투자논리" 아카이브를 진화시키고, 그 논리가 실제 수익과 분산으로 이어지는지 포워드 테스트로 검증한다.
자동매매나 매매 권유가 아니라 검토 후보를 만드는 도구다.

- 전체 설계 문서: `docs/design.md` (설계 판단은 여기를 기준으로 하고, 설계가 바뀌면 이 파일을 고친다)
- 원 논문: arXiv 2610.23470

## 환경

- Windows, PowerShell, `uv`, Python 3.10+
- 리모트: `origin` = 비공개 저장소(sunyeul/qd-rec-stocks), `upstream` = SakanaAI/QD-Rec
- 원본 업데이트: `git fetch upstream && git merge upstream/main`
- LLM은 OpenRouter 경유(`.env`의 `OPENROUTER_API_KEY`). `.env`는 커밋하지 않는다.

## 확정된 결정

- 시장: 미국. PoC 유니버스: S&P 100.
- 예산: 저예산 PoC 우선. 100종목, 실행당 300회 평가, LLM 변형 3종(Novelty / QD-Rec 숫자만 / QD-Rec 숫자+뉴스), 반복 2회. 검증은 과거 시점 백테스트(2026-05~09 월말 5개, 1개월 보유, 실행당 120회)로 하고, 포워드 테스트는 후속(아래 "백테스트로 전환").
- 모델: 전부 저가 모델. 단 제안자와 심사자는 서로 다른 모델 계열.
- 평가는 두 층: 개별 종목(IC, 분위 격차, 논리 적중률과 "실력/운" 4분면)과 포트폴리오(비용 차감 초과수익, 샤프, MDD, 종목 간 상관, 팩터 회귀).
- 커뮤니티 데이터: PoC에서는 제외(RQ4 확장 실험용). Reddit은 앱 등록이 번거로워 보류. StockTwits 공개 스트림을 1순위로 검토 중.

## 미정

- 포트폴리오 N과 테마당 상한 k (추천값 N=20, k=3)
- 구체적 모델 조합, 월 예산 상한(B단계에서 단가 실측 후), 포워드 테스트 시작일

## 코드 규칙

- `shared/`(원본 코어)는 최소한으로만 수정한다. 새 코드는 `apps/stocks/`, `ingest/`, `portfolio/`, `evaluation/`에 둔다.
- **누출 방지가 최우선이다.**
  - 스냅샷 날짜 t의 정보 기준 시각은 t일 뉴욕 16:00(장 마감).
  - 종목 카드 등 생성 쪽 코드는 `ingest.reader`(`load_asof`, `load_news_asof`)로만 데이터를 읽는다.
  - t 이후 가격이 필요한 평가 코드는 `evaluation/`에만 두고, 수익률은 t 종가부터 잰다.
  - 재무 지표(시가총액·PER은 실시간 주가 기반)는 t 마감~다음 개장 사이에 받은 것만 t의 값으로 인정한다.
- 누출 관련 로직을 바꾸면 `tests/test_ingest.py`에 테스트를 추가한다.
- 테스트: `uv run python -m pytest tests -q`

## 현재 상태 (2026-10-24)

- [x] 저장소 세팅, 의존성(yfinance 1.7.0, lxml, pytest dev)
- [x] M1 데이터 수집 `ingest/` 구현·커밋. 사용법과 저장 구조는 `ingest/README.md`
  - 뉴스: 종목 스트림이 비면 Yahoo 검색으로 대체(`source` 열). 2026-10-22 실행 결과 `stream raw 0, search raw 95, kept 95`로 전부 검색 경로에서 왔다. 컷오프 이후 기사도 저장하지만 `load_news_asof`가 걸러내며, 다음 스냅샷에서 쓰기 위해 의도적으로 남긴다
- [ ] 작업 스케줄러: PoC 검증 단계라 등록하지 않음(2026-10-24 해제). 수집은 필요할 때 `uv run python -m ingest.run_daily`로 수동 실행. 포워드 테스트 시작 시 `scripts/run_ingest.ps1`로 다시 등록
- [x] StockTwits 접근 확인(2026-10-24): 403(Cloudflare "Just a moment..." 차단 페이지). 커뮤니티 데이터는 Wikipedia 페이지뷰(관심도)로 대체한다(RQ4 확장용, PoC 범위 밖)

## 다음 단계

1. **M2 종목 카드** — 구현됨(2026-10-24). `apps/stocks/items.py`(카드), `summaries.py`(뉴스 요약 LLM, `data/cards/`에 캐시), `create_embeddings.py`(요약 → 임베딩 빌드). 스냅샷과 모드는 환경변수 `SNAPSHOT_DATE`, `ANONYMIZE=1`
   - 카드는 영어. 요약 문장에 원문에 없는 숫자가 있으면 그 문장을 버린다(`keep_grounded`)
   - 익명화: 요약 LLM이 이름·제품명을 지우고, `mask_names`가 회사명·티커를 규칙으로 한 번 더 지운다. 제품명은 LLM에만 의존
   - 남은 일: 재무 스냅샷이 아직 하나도 없어 사업 요약·밸류에이션 칸은 실데이터로 확인 못 함. 전체 100종목 빌드도 안 함(10-22 가격은 5종목 시험분뿐). 요약 모델은 `google/gemini-2.5-flash-lite`
2. **M3 생성·심사** — 구현됨(2026-10-24), 5종목 시험분으로 시드 3개 + QD-Rec 5회 실행까지 확인. `profiles.py`(기본 프로필 `base`, 실제 보유는 `profiles_private.py`), `prompts.py`, `proposer.py`, `verifier.py`, `evaluation.py`, `seed.py`, `qd_rec.py`
   - 기본 모델: 제안자 `google/gemini-3.5-flash-lite`(마감 2026-03), 심사자 `openai/gpt-6-luna`(마감 2026-05-18), 요약 `google/gemini-2.5-flash-lite`(계열 분리, 추론 강도 low). `PROPOSER_MODEL`, `JUDGE_MODEL`로 변경
   - 검증기: thesis·evidence·catalyst·risks의 숫자가 카드나 프로필에 없으면(반올림은 허용) 0점. 루프에서는 아카이브에 넣지 않고 버리고, 시드에서는 다시 생성한다. falsifier·horizon은 기준값을 새로 정하는 칸이라 검사하지 않는다
   - `PER_SECTOR=3`이면 섹터별 시가총액 상위 3개(같은 회사의 다른 주식 종류는 하나만)로 아이템을 줄인다. 아카이브 파일명 태그에 스냅샷·모드·풀이 들어간다(예: `base-2026-10-23-named-top3`)
   - B단계 실행 순서: `$env:PER_SECTOR="3"` → `create_embeddings` → `seed` → `qd_rec --iterations 100`(세 방식: `--method novelty|novelty_evo|qd_rec`)
   - 설계 기준: 제안자 JSON 양식(thesis, evidence, catalyst, risks, falsifier, horizon), 규칙 기반 숫자 인용 검증기(`verifier.py`, 실패 시 0점), 심사 기준(근거 일치 3, 프로필 적합 3, 리스크·반증 2, 촉매 구체성 2). PoC B단계: 종목 30개·100회 평가로 품질 점검, 통과 기준은 검증기 실패율 5% 미만 & 다양성이 Novelty 이상
3. **백테스트로 전환(2026-10-24 결정)** — 블로그용으로 빨리 결론을 내기 위해 PoC의 주 검증을 포워드 3개월에서 과거 시점 백테스트로 바꿨다. 근거와 한계는 `docs/design.md`의 "백테스트로 전환"
   - 누출 방지: 모든 모델의 학습 마감일 이후 스냅샷만 쓴다(제안자 gemini-2.5-flash-lite 2025-01-31, 심사자 gpt-6-luna 2026-05-18은 제3자 자료 기준). 모델을 바꾸면 `scripts/backtest.py`의 DATES도 다시 정한다
   - `ingest/backfill.py`: 과거 t의 유니버스·가격·재구성 재무(연간 재무제표, 회계연도 종료 + 60일 이후만)·뉴스 → `data/backtest/`
   - `ingest/news_history.py`: 과거 뉴스는 Google News RSS 검색(제목·매체·날짜, t 전날까지만). GDELT는 429로 막힘
   - 변형 3종: `novelty`(숫자+뉴스), `qd_rec_numbers`(`WITH_NEWS=0`), `qd_rec_news`. 아카이브 태그에 뉴스 여부가 들어간다(예: `base-2025-03-31-named-nonews-all`)
   - `portfolio/build.py`: 종목별 최고 점수 상위 N=20, 섹터당 k=3
   - `evaluation/returns.py`, `evaluation/report.py`: t 종가부터 1·3개월 수익, SPY 초과(비용 0.2% 차감), IC, 아카이브 다양성, 보유 기간 상관. 비교 대상은 균등 비중과 12-1 모멘텀
   - 전체 실행: `uv run python -m scripts.backtest` (월말 5개 × 3변형 × 120회, 끝난 단계는 건너뜀, 키 잔액이 $0.5 미만이면 멈춤) → `reports/backtest.md`
   - 시험(2025-03-31, 10회)에서 본 점: 심사자 gpt-5-nano의 점수가 후하다(평균 8.8, 10점 다수). 변별력이 약하면 심사 기준이나 모델을 손봐야 한다
   - **결과(2026-10-24, 중립 프로필, 날짜 3개, QD-Rec 120회, `reports/backtest.md`)**: 1개월 SPY 대비 +0.2 / +2.4 / +0.1%p(3번 모두 SPY 이김, 평균 +0.9%p), 균등 비중은 평균 +1.2%p. IC(1M) −0.05 / +0.18 / −0.06. 숫자 검증 실패 3~7%. 실행당 $0.33~0.39, 4~5분. 첫 날짜 Novelty 대조는 +1.3%p
   - 알게 된 점: (1) 옛 "기술주 과다" 프로필은 수익을 +4.6%p로 부풀렸다(비기술주 쏠림). (2) 검증기가 "6 to 12 month" 같은 기간을 지어낸 숫자로 오판했다 → 수정. (3) 진화 연산자 꼬리표는 모델의 자기 보고라 gemini-3.5는 99% "변형", 2.5는 90% "탐색". 꼬리표는 기록용이라 결과에 영향 없음. (4) 점수가 6점에 몰려 변별력이 약하다
   - 다음 후보: 심사자를 gpt-5.6-terra(마감 2026-02-16, 논문의 심사자)로 올려 첫 날짜 60회 비교(약 $2.5~3)
4. **포워드 테스트**: 실수집 데이터(`data/snapshots/`)로 후속 진행
