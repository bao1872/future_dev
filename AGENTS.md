# future_dev Governance

## 0. Purpose

`future_dev` is an **experimental offline quantitative research project** for a fixed futures instrument.

Current research instrument:

- Source: TqSdk
- Symbol: `KQ.m@SHFE.ag`
- Research timeframes: `15m`, `1h`, `4h`
- UI: Streamlit
- Research data: offline CSV only

Primary development instrument:
`KQ.m@SHFE.ag`

Offline robustness research may additionally use:

- `KQ.m@SHFE.cu`
- `KQ.m@SHFE.al`
- `KQ.m@SHFE.sn`
- `KQ.m@DCE.i`
- `KQ.m@INE.sc`
- `KQ.m@DCE.m`
- `KQ.m@CZCE.CF`

All remain TqSdk-only.
This does NOT expand Streamlit/UI/current product scope.

This repository is **not** a production trading system. The current objective is to make strategy research fast, observable, reproducible enough for research, and resistant to silent data / indicator errors.

---

## 1. Authority order

When instructions conflict, follow this order:

1. **User's explicit instruction in the current task**
2. This `AGENTS.md`
3. Existing code contracts / README statements
4. AI inference

Never reinterpret a clear user restriction as permission to bypass it.

**Explicit instruction > inferred intent.**

---

## 2. Project mode

Current mode:

`PROJECT_MODE = RESEARCH`

In RESEARCH mode, prioritize:

1. Data and causal correctness
2. Preservation of canonical indicator semantics
3. Fast strategy iteration
4. Clear visual inspection
5. Minimal necessary code quality

Production architecture is not a goal.

Do **not** introduce infrastructure merely because it may be useful later.

---

## 3. Hard architecture boundaries

### 3.1 Streamlit is the research UI

- Streamlit is the default and only research application UI.
- Plotly may be used inside Streamlit for charts.
- Do not create a separate frontend application.
- Do not introduce React / Next.js / Dash / Flask UI unless explicitly requested.

### 3.2 TqSdk is a market-data source, not a strategy dependency

TqSdk belongs only to the market-data acquisition layer.

Allowed:

`TqSdk -> download/validate -> offline files -> research/strategy -> Streamlit`

Forbidden:

`strategy -> TqSdk`

`backtest -> TqSdk`

`indicator -> TqSdk`

A strategy must be runnable when the network is unavailable, as long as valid offline data exists.

### 3.3 One current offline dataset; no dataset versioning

The project researches a fixed instrument and maintains **one current valid offline market-data set**.

Do not create:

- dataset IDs
- historical snapshot catalogs
- dataset registries
- data version databases
- date-stamped duplicate data directories

unless the user explicitly requests them later.

For research reproducibility, record the actual `data_start` and `data_end` used by an experiment. That is sufficient in the current phase.

### 3.4 Raw/current data must not be silently corrupted

A refresh must follow:

`download -> validate -> replace current offline files`

Do not intentionally replace known-good current files with failed or partially validated output.

### 3.5 Canonical Panji indicators are frozen semantics

`panji_indicators.py` is the canonical indicator bundle extracted from Panji / `market_dev`.

It includes the validated definitions for DSA, SMC, BOS/CHoCH, Order Block lifecycle, and Momentum/SQZMOM.

Default rule: **read-only semantics**.

Without explicit user authorization, do not:

- rewrite formulas
- change default parameters
- change causal transitions
- change BOS / CHoCH meaning
- change Order Block creation / entry / mitigation semantics
- change Momentum / SQZMOM semantics
- split the file merely for code cleanliness
- refactor it into new abstractions
- replace it with a new implementation

Adapters and visualization code may consume its outputs.

### 3.6 Strategy code owns trading hypotheses, not indicator definitions

Indicators answer: "what is the market state?"

Strategies answer: "under what combination of states do we act?"

Strategy modules may combine canonical outputs, but must not silently modify canonical algorithms.

---

## 4. Scope discipline: minimum necessary change

For every task, make the smallest change that satisfies the requested objective.

Forbidden behavior:

- "while I am here" refactors
- unrelated cleanup
- speculative abstractions
- folder reshuffles for aesthetics
- renaming unrelated APIs
- adding frameworks for hypothetical future needs
- converting a local task into a repository-wide modernization

Rule:

`User asks for X -> implement X -> validate X -> stop.`

---

## 5. Change levels and required validation

### L0 - Documentation / text / layout only

Examples:

- README wording
- Streamlit labels
- chart layout
- comments

Validation:

- inspect the changed surface only
- no data refresh
- no full tests

### L1 - Research UI / local research tooling

Examples:

- Streamlit page
- Plotly rendering
- experiment result display
- local adapter that does not alter semantics

Validation:

- import / syntax check
- targeted Streamlit smoke or target script run

Do not trigger unrelated data downloads or full repository validation.

### L2 - Strategy / offline data interface logic

Examples:

- strategy conditions
- offline loader
- research result calculation
- data-range filtering

Validation:

- target strategy / function run
- basic result sanity
- relevant data validation only

### L3 - Core semantic change

Examples:

- canonical indicator changes
- downloader time-window semantics
- bar-close definition
- cross-timeframe aggregation definition
- continuous-contract adjustment algorithm
- future execution / order logic

Requirements:

- explicit user authorization
- state the semantic delta before changing it
- run focused invariants / PIT / baseline comparison appropriate to the change

L3 changes are not routine cleanup.

---

## 6. STOP rules

The agent must STOP and report the conflict instead of improvising when any of the following occurs:

1. The requested implementation appears to require violating an explicit user restriction.
2. A change would alter canonical indicator semantics without explicit authorization.
3. A market-data validation fails in a way that could affect research correctness.
4. Full-history and prefix-only calculations disagree where causality requires equality.
5. The agent cannot determine whether a proposed refactor is semantically neutral.
6. Existing verified behavior would have to be discarded only to make a new architecture cleaner.
7. A task unexpectedly requires production trading / order execution assumptions not supplied by the user.

STOP means:

- do not invent permission
- do not silently weaken validation
- do not hide the failure
- report the exact delta / evidence needed

---

## 7. Validation governance

Validation must match the risk of the change.

### UI-only change

Use:

- import check
- page smoke

Do not run:

- market-data refresh
- PIT suite
- unrelated full calculations

### Strategy change

Use:

- target strategy on the requested offline range
- signal/trade sanity checks
- compare only directly affected metrics if needed

### Offline data change

Use the existing relevant checks:

- timestamp monotonicity / duplicates
- OHLC sanity
- non-negative volume / OI
- closed-bar handling
- 15m -> 1h aggregation
- 1h -> 4h aggregation

### Canonical indicator change

Only with explicit authorization. Use relevant:

- invariant checks
- prefix PIT checks
- baseline comparison

**Do not default to full-regression behavior.**

---

## 8. Data governance

Current market data lives in the existing `silver_main_data/` layout unless the user explicitly authorizes migration.

The offline market-data interface is the research entrypoint.

Strategy / research code should call the offline loader rather than hard-code CSV paths.

Record for each experiment:

- strategy name
- parameters
- data start
- data end
- result summary
- optional Git commit SHA

Do not add dataset version IDs.

Derived data must never overwrite raw/current market files.

The existing `adjusted/` / rollover products must not be treated as valid merely because files exist; follow the current README / validation state.

---

## 9. Strategy governance

Each strategy is a self-contained research hypothesis.

A strategy module should expose a small stable surface:

- `NAME`
- `DESCRIPTION`
- `DEFAULT_PARAMS`
- `run(data, params)`

Where `data` is offline DataFrame data supplied by the research layer.

A strategy must not:

- authenticate to TqSdk
- download data
- write into raw market data
- modify canonical indicator defaults globally
- perform deployment / order execution

Prefer one meaningful strategy concept per module. Do not create `final_v2_best_new.py` style files.

Use Git history for code evolution.

---

## 10. Market-data interface governance

The market-data layer has three responsibilities only:

1. Source acquisition (`TqSdk`)
2. Offline storage / loading
3. Validation

Do not build provider factories or generalized multi-provider frameworks before a real second provider exists.

Do not add a database while CSV is sufficient.

---

## 11. Streamlit governance

Streamlit is a research workbench, not a production dashboard.

Pages should support:

- offline data inspection
- indicator / structure visualization
- strategy parameter experiments
- result comparison

Keep business / strategy logic out of page files when it would make the logic impossible to reuse or test. Page files may orchestrate calls to thin research / market-data modules.

Do not create a second visualization stack for the same purpose without a concrete need.

---

## 12. Prohibited engineering by default

Unless explicitly requested, do not add:

- Docker
- Docker Compose
- Kubernetes
- CI/CD
- GitHub Actions
- FastAPI
- Flask backend
- databases
- Redis
- queues
- microservices
- service/repository/domain architecture
- dependency injection frameworks
- production observability stacks
- deployment scripts
- staging/prod environments
- broad pytest regression suites
- benchmark infrastructure
- plugin architectures
- generalized provider factories

The absence of these is intentional, not technical debt in the current project phase.

---

## 13. Git discipline

Prefer one logical change per commit.

A commit for a Streamlit chart must not also change downloader semantics or canonical indicators.

Do not require PR / release / staging workflows in the current phase unless explicitly requested.

---

## 14. Existing core assets

Treat these current repository assets as established components, not invitations to rewrite:

- `download_silver_main_tqsdk.py` - validated TqSdk offline downloader and cross-TF checks
- `build_continuous.py` - continuous-contract / adjustment research utility; use only when its current data assumptions are valid
- `panji_indicators.py` - canonical Panji indicator SSOT
- `visualize_smc_momentum_tqsdk.py` - existing SMC/Momentum visual/PIT validation utility
- `silver_main_data/` - current offline market-data location

New structure should integrate around these assets.

---

## 15. Task execution checklist

Before changing code:

1. What exact user goal is being solved?
2. Which ownership area is affected: UI, market data, strategy, canonical indicator, continuous contract?
3. Is the change L0/L1/L2/L3?
4. Does it cross a hard boundary?
5. What is the minimum targeted validation?

After changing code:

1. Verify only what the risk requires.
2. Report material assumptions / failures.
3. Do not add unrelated follow-up work automatically.
