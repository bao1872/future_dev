# future_dev Governance

## 0. Purpose

`future_dev` is an **experimental offline quantitative research project** for a fixed futures instrument.

Current research instruments:

- **Market-data source: PyTDX 1.72r2 customized build.**
  No second provider is maintained.
- Symbol form: PyTDX `*L8` continuous main series
- Research bar scales: `5m` source bar, `15m` aggregated locally
- UI: Streamlit
- Research data: offline CSV only

Primary development instrument:
`AGL8`

Offline robustness research may additionally use:

- `CUL8`
- `ALL8`
- `SNL8`
- `IL8`
- `SCL8`
- `ML8`
- `CFL8`

All remain PyTDX-only.
This does NOT expand Streamlit/UI/current product scope.

### Source selection is closed

PyTDX has passed targeted semantic validation and has been selected
by the user. Agents must use it directly for experiments.

Do **not** continue provider comparison, provider abstraction, or
replacement research unless:

- a concrete data-correctness failure occurs, or
- the user explicitly requests it.

Data-interface investigation is not an open-ended task. Investigate
a data source only to remove a concrete, blocking limitation.

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
2. **Statistical validity**
3. Fast hypothesis iteration
4. **Reproducible experiment outputs**
5. Visual inspection

Production architecture is not a goal.

Do **not** introduce infrastructure merely because it may be useful later.

---

## 3. Hard architecture boundaries

### 3.1 Streamlit is the research UI

- Streamlit is the default and only research application UI.
- Plotly may be used inside Streamlit for charts.
- Do not create a separate frontend application.
- Do not introduce React / Next.js / Dash / Flask UI unless explicitly requested.

### 3.2 PyTDX is a market-data source, not a strategy dependency

PyTDX belongs only to the market-data acquisition layer.

Allowed:

`PyTDX -> acquire 5m L8 bars -> normalize verified TDX timestamp semantics -> offline CSV -> research/strategy -> Streamlit`

Forbidden:

`strategy -> PyTDX`

`model -> PyTDX`

`indicator -> PyTDX`

A strategy must be runnable when the network is unavailable, as long as valid offline data exists.

The only acquisition module is `market_data/pytdx_source.py`. Do not
add provider factories, `DataSourceProtocol`, client managers or
server pools. The server is fixed; if it does not connect, the run
fails.

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

`panji_indicators.py` remains frozen / read-only, but it is
**currently inactive in the statistical research path unless
explicitly re-enabled**. Do not delete the file and do not build
anything new around it.

### 3.5b Validated TDX bar semantics -- data contract

These were verified against the historical transaction stream and
cross-validated against an independent vendor source on real
contracts (AG2610 / I2701 / SC2610, one full trading day). Treat
them as the fixed contract; do not re-derive them.

Time:

    TDX bar label = INTERVAL END

    bar_start_time      = corrected_bar_end_time - period
    availability_time   = corrected_bar_end_time

`bar_start_time` is the research index, so it matches conventional
K-line semantics. `availability_time` is the moment a bar's close /
volume / OI actually become known, so it is the correct key for
causal joins, purge/embargo cutoffs and target availability.

TDX datetime is partly trading-day based. The night session that
precedes a trading day carries that trading day's date. Correction
must use the previous observed trading day from the exchange-level
trading calendar, never a naive `hour >= 21 -> minus one day`, which
breaks across weekends and holidays.

Payload fields:

    trade     = bar volume
    position  = bar-end open interest

Therefore:

    delta OI = position[t] - position[t-1]

Never trust these fields:

    amount        UNUSABLE, protocol decode artifact (~1e-40)
    zengcang      UNTRUSTED
    nature        UNTRUSTED
    nature_name   NOT a current feature
    direction     NOT a current feature

Historical transaction stream: available and paginates to a true
end, shares the same underlying tick source as the bars, but it is
**incomplete in high-activity periods** (roughly 85%-95% of bar
volume, with the deficit concentrated in the highest-volume
minutes). It is **not an approved feature source in the current
experiment phase**.

### 3.5c L8 series role

`*L8` is the current continuous-market research series.

It is a vendor-defined continuous main series. It is **not** treated
as exact executable-contract history, and it is **not** used for
contract-roll PnL accounting.

Do not attempt to reverse-engineer the vendor's roll or adjustment
algorithm. Delisted contracts are not retrievable from PyTDX, so
historical actual-contract reconstruction is not available.

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

- authenticate to PyTDX
- download data
- write into raw market data
- modify canonical indicator defaults globally
- perform deployment / order execution

Prefer one meaningful strategy concept per module. Do not create `final_v2_best_new.py` style files.

Use Git history for code evolution.

---

## 10. Market-data interface governance

The market-data layer has three responsibilities only:

1. Source acquisition (`PyTDX`)
2. Offline storage / loading
3. Validation

There is no second provider. Do not build provider factories or generalized multi-provider frameworks.

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

- `market_data/pytdx_source.py` - the only acquisition module; holds the validated TDX bar-time contract
- `market_data/validation.py` - offline structural and cross-timeframe aggregation checks
- `scripts/check_tdx_data.py` - minimal post-download structural check
- `panji_indicators.py` - canonical Panji indicator SSOT, frozen and currently inactive
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
