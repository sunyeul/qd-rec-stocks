# qd-rec-stocks

Sakana AI의 [QD-Rec](https://github.com/SakanaAI/QD-Rec)을 미국 주식 아이디어 발굴에 응용하는 개인 연구 프로젝트.
"종목 + 투자논리" 아카이브를 진화시키고, 그 논리가 실제 수익과 분산으로 이어지는지 포워드 테스트로 검증한다.
자동매매나 매매 권유가 아니라 검토 후보를 만드는 도구다.

- 전체 설계 문서: https://claude.ai/artifact/M1W76ZES5xESSbLq5AcFaL (설계 판단은 여기를 기준으로 한다)
- 원 논문: arXiv 2610.23470

## 환경

- Windows, PowerShell, `uv`, Python 3.10+
- 리모트: `origin` = 비공개 저장소(sunyeul/qd-rec-stocks), `upstream` = SakanaAI/QD-Rec
- 원본 업데이트: `git fetch upstream && git merge upstream/main`
- LLM은 OpenRouter 경유(`.env`의 `OPENROUTER_API_KEY`). `.env`는 커밋하지 않는다.

## 확정된 결정

- 시장: 미국. PoC 유니버스: S&P 100.
- 예산: 저예산 PoC 우선. 100종목, 실행당 300회 평가, LLM 변형 3종(Novelty / QD-Rec 숫자만 / QD-Rec 숫자+뉴스), 반복 2회, 3개월 포워드, 월 1회 리밸런싱.
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

1. **M2 종목 카드** (`apps/stocks/`, `apps/companies/`를 참고해 복사·수정)
   - `items.py`: `load_items`, `item_text`(숫자 블록 + 사업 요약 + 최근 30일 뉴스 요약), `archive_text`, `embedding_text`, `load_item_embeddings`
   - 뉴스 요약 LLM: 종목·스냅샷당 1회만 만들고 캐시. "사실 / 해석이 갈리는 점"으로 나누고 숫자를 새로 만들지 않게 제한
   - 익명화 모드: 회사명·티커·제품명을 `C-127` 같은 ID로, 시가총액은 구간으로
   - 카드당 약 1,500토큰 상한, 512차원 임베딩
2. **M3 생성·심사**: 제안자 JSON 양식(thesis, evidence, catalyst, risks, falsifier, horizon), 규칙 기반 숫자 인용 검증기(`verifier.py`, 실패 시 0점), 심사 기준(근거 일치 3, 프로필 적합 3, 리스크·반증 2, 촉매 구체성 2). PoC B단계: 종목 30개·100회 평가로 품질 점검, 통과 기준은 검증기 실패율 5% 미만 & 다양성이 Novelty 이상
3. **M4 포트폴리오**(`portfolio/build.py`), **M5 평가 하니스**(`evaluation/`), **M6 포워드 테스트**
