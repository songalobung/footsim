# Architectural & Modelling Decisions

This document records modelling choices, priors, constraints, optimizers, and placeholders as required by `SPEC.md`.

---

## Milestone 1: Dixon-Coles Goals Model

### 1. Identifiability Constraint
- **Choice**: Attack and defence log-parameters each sum to zero across all teams ($\sum a_i = 0$, $\sum d_i = 0$), with an explicit free intercept parameter $c$ capturing the baseline league scoring level.
- **Formulation**:
  $$\log \lambda_{ij} = c + a_i + d_j + \text{home\_adv}$$
  $$\log \mu_{ij} = c + a_j + d_i$$
- **Reparameterisation**: For $n$ teams, the optimizer searches over $n-1$ attack parameters and $n-1$ defence parameters. The $n$-th parameters are set deterministically as $a_n = -\sum_{i=1}^{n-1} a_i$ and $d_n = -\sum_{j=1}^{n-1} d_j$. Gradients for $a_1 \dots a_{n-1}$ propagate through $a_n$ via the chain rule ($\frac{\partial f}{\partial a_i} - \frac{\partial f}{\partial a_n}$).
- **Equivalence**: This is mathematically identical to the spec's formulation where defence absorbs the intercept ($d'_j = c + d_j$). Having a symmetric zero-sum centered scale makes attack and defence parameters directly comparable across teams.

### 2. Time Decay Weighting
- **Weight function**: $w_k = \exp(-\xi \cdot \Delta t_k)$, where $\Delta t_k$ is the elapsed time in days from the match date to `as_of`.
- **Tuning**: Evaluated via walk-forward weekly blocks (predicting Tuesday-to-Monday fixture rounds using only data prior to each round) over the 2022–23 and 2023–24 seasons (avoiding the last two completed seasons reserved for Milestone 2 evaluation).
- **Metric**: Multiclass 1X2 log loss on the full $(P(H), P(D), P(A))$ distribution, alongside Ranked Probability Score (RPS).
- **Result**: Grid search over $\xi \in [0, 0.0065]$ per day showed a minimum log loss at $\xi = 0.0035$ (log loss 0.96313, RPS 0.19931), with $\xi = 0.0030$ performing virtually identically (log loss 0.96314). Default is set to $\xi = 0.0030$.

### 3. Promoted-Team Prior & Shrinkage
- **Classification**: A team in season $S$ is considered "promoted" if it did not participate in season $S-1$. Only training matches before `as_of` are used for this classification.
- **Empirical Offsets**: For all historical promoted teams in the training sample, total goals scored and conceded are compared against expected goals under the league average rate for those seasons:
  $$\text{attack\_offset} = \log\left(\frac{\sum \text{GF}_{\text{promoted}}}{\sum E[\text{GF}]}\right) \approx -0.35$$
  $$\text{defence\_offset} = \log\left(\frac{\sum \text{GA}_{\text{promoted}}}{\sum E[\text{GA}]}\right) \approx +0.28$$
- **Prior Penalties (L2 / Ridge)**:
  - Established teams: shrunk toward $0$ with prior standard deviation $\sigma_{\text{est}} = 1.0$.
  - Promoted teams: shrunk toward $(\text{attack\_offset}, \text{defence\_offset})$ with prior standard deviation $\sigma_{\text{prom}} = 0.25$.
- **New teams without history**: If a newly promoted team appears in fixtures before it has played any match in the historical training sample, passing it in `new_teams` assigns it the empirical promoted offset $(\text{attack\_offset}, \text{defence\_offset})$ directly.

### 4. Bounded $\rho$ and Positivity of $\tau$
- **Theoretical bounds**: To keep $\tau(x, y; \lambda, \mu, \rho) > 0$ for all four low-scoring cells:
  $$\max\left(-\frac{1}{\lambda}, -\frac{1}{\mu}\right) < \rho < \min\left(\frac{1}{\lambda\mu}, 1\right)$$
- **Optimization bounds**: In the likelihood optimization, $\rho$ is box-bounded to $[-0.2, 0.2]$.
- **Prediction clipping**: When computing score matrices via `dc_score_matrix(lam, mu, rho)`, $\rho$ is dynamically clipped into $[0.999 \cdot \text{lower\_bound}, 0.999 \cdot \text{upper\_bound}]$ for the specific fixture's $(\lambda, \mu)$, ensuring strictly positive probabilities under any extreme scoring expectation.
- **Renormalisation**: The $11 \times 11$ score matrix (0 to 10 goals each) is divided by its sum so all probabilities sum to exactly 1.0.

### 5. Optimizer and Tolerances
- **Algorithm**: `scipy.optimize.minimize` with `L-BFGS-B`.
- **Gradients**: Exact vectorised analytic gradients are provided for all parameters ($c, \text{home\_adv}, \rho, a_{1..n-1}, d_{1..n-1}$). Verified against finite-difference numerical gradients (`scipy.optimize.check_grad`).
- **Tolerances**:
  - `ftol = 1e-12`
  - `gtol = 1e-7`
  - `maxiter = 2000`
  - `maxcor = 20`
- **Fit Time**: Fits 6 complete seasons (2,280 matches, 40+ iterations) in under 0.10 seconds.

### 6. Lookahead Prevention
- When `fit(matches, as_of="YYYY-MM-DD")` is called, the dataset is strictly filtered to `Date < as_of` before any team index, design matrix, or prior estimation is calculated.
- Tested by intentionally mutating and corrupting future data (and match dates equal to `as_of`); parameter estimates remain completely bitwise identical.

### 7. Data Cleaning & Season Handling
- **Seasons**: The last 6 completed EPL seasons (2020–21 through 2025–26) total 2,280 matches (380 per season).
- **Date parsing**: Explicit handling of both 8-character (`%d/%m/%y`) and 10-character (`%d/%m/%Y`) formats found in `football-data.co.uk` CSVs.
- **Team name normalization**: All team variations and aliases map to canonical names via `footsim/data/team_names.csv`.
- **Missing values**: Rows missing `FTHG` or `FTAG` are dropped. Rows missing secondary event statistics (cards, corners, fouls) are retained and flagged with `events_missing = True`.

---

## Milestone 2: Walk-Forward Evaluation & Benchmarking

### 1. Matchday Fixture Grouping
- **Method**: Date gaps $> 2$ days delineate distinct fixture blocks (`matchday_blocks`).
- **Rationale**: Weekend rounds typically run Friday through Monday (gaps $\le 1$ day), while midweek rounds run Tuesday through Thursday (gaps $\le 1$ day). Any gap of 3 or more days indicates a new gameweek block. Across a 38-round season, this produces 31–34 blocks (accounting for occasional double gameweeks).
- **Refitting policy**: For each block $B$, the model is fitted strictly once with `as_of = B.Date.min()`. Every fixture in $B$ is predicted using this model. No future matches or matches from within block $B$ are accessible during fitting.

### 2. Pinnacle Benchmark & Margin Removal
- **Source columns**: Closing odds (`PSCH, PSCD, PSCA`) are preferred. Where closing odds are absent, opening odds (`PSH, PSD, PSA`) are used. In the two-season test evaluation (2024–25 and 2025–26), Pinnacle odds are present for 590 of 760 matches (380/380 in 2024–25; 210/380 in 2025–26).
- **Margin removal choice**: **Shin's method** (1992, 1993; Jullien & Salanié 1994) is the default (`margin_method="shin"`).
  - Solves for insider trader fraction $z \in [0, 1)$ via Brent root finding on $[0, 0.9999]$.
  - Naturally accounts for the favorite-longshot bias, where bookmakers load higher margins on longshots.
  - Proportional margin removal ($p_i = \frac{1/O_i}{\sum 1/O_k}$) is also implemented and selectable via `--margin-method proportional`.

### 3. Evaluation Metrics
- **1X2 Markets**:
  - **Multiclass Log Loss**: $-\frac{1}{N} \sum_{i=1}^N \log p_{i, y_i}$
  - **Brier Score**: $\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^3 (p_{ik} - y_{ik})^2$
  - **Ranked Probability Score (RPS)**:
    $$\text{RPS} = \frac{1}{N} \sum_{i=1}^N \frac{1}{2} \left[ (p_{i,\text{Home}} - y_{i,\text{Home}})^2 + ((p_{i,\text{Home}} + p_{i,\text{Draw}}) - (y_{i,\text{Home}} + y_{i,\text{Draw}}))^2 \right]$$
  - Compared on the exact subset where Pinnacle odds are available.
- **Goals Markets**:
  - **Over/Under 2.5 Goals**: Binary log loss on $P(\text{Over 2.5}) = \sum_{x+y \ge 3} P(x, y)$.
  - **Both Teams To Score (BTTS)**: Binary log loss on $P(\text{BTTS}) = \sum_{x \ge 1, y \ge 1} P(x, y)$.

### 4. Calibration Analysis
- Evaluated across 10 equal-width bins in $[0, 1]$ ($[0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$).
- Computed separately for Home Win, Draw, Away Win, Over 2.5 Goals, and BTTS.
- Plotted with 45-degree diagonal reference line and saved as `reports/calibration_plot.png`.

### 5. CLI Command
- `footsim backtest --seasons 2425,2526` produces all four report artifacts in `reports/` in ~10 seconds.

---

## Milestone 3: Event Rate Models (Cards, Corners, Fouls)

### 1. Model Formulation & Likelihood
- **Model structure**: Count rates are modelled per team-match (two observations per match: home side and away side):
  $$\log m_k = c + \text{home\_adv} \cdot \mathbb{I}_{\text{home}} + \text{for}_{i} + \text{against}_{j} + \text{referee}_{r}$$
  where $i$ is the team, $j$ is the opponent, and $r$ is the match referee.
- **Distribution family**:
  - **Negative Binomial (NB2)**: $\text{Var}(Y) = m + \alpha m^2$, parameterized via log size $\log r = \log(1/\alpha) \in [-3, 8]$. Used for yellow cards, corners, and fouls.
  - **Poisson**: $\alpha = 0$, used for red cards.
- **Time decay weighting**: Observations receive time decay weight $w_k = \exp(-\xi \cdot \Delta t_k)$. High-frequency statistics (yellows, corners, fouls) use $\xi = 0.003$ per day. Rare statistics (red cards) use $\xi = 0.0$ to avoid excessive variance across small annual card counts.
- **Analytic gradients**: Analytic gradients for $c, \text{home\_adv}, \log r, \text{for}_{1..n}, \text{against}_{1..n}, \text{referee}_{1..m}$ are derived and verified against numerical differentiation (`check_grad` error $< 10^{-5}$).
- **Optimization**: `scipy.optimize.minimize` using `L-BFGS-B`.

### 2. Regularisation & Shrinkage Priors
- **Priors**: Zero-mean Gaussian priors (ridge penalties) $\frac{1}{2 \sigma^2} \theta^2$ are applied to team for/against effects and referee effects.
- **Referee shrinkage**: Referees with few matches are naturally shrunk toward 0 (league average), preventing small-sample noise from dominating. Missing or unrecorded referees receive an effect of 0.
- **Red cards handling**:
  - Empirical data across 2,280 EPL matches shows no statistically significant home advantage (127 home reds vs 138 away reds, $p = 0.54$, and in 2024–25 exactly 26 vs 26). Fitting an unpenalised home advantage overfits sample noise.
  - Red cards therefore omit `home_adv` (`use_home=False`) and apply strong Gaussian shrinkage on team effects ($\sigma_{\text{team}} = 0.15$) and referee effects ($\sigma_{\text{ref}} = 0.01$).
- **Hyperparameter tuning**: Prior standard deviations were tuned by walk-forward Poisson deviance on the 2022–23 and 2023–24 seasons (avoiding the 2024–26 test seasons):
  - **Yellow cards**: $\sigma_{\text{team}} = 0.20$, $\sigma_{\text{ref}} = 0.08$, $\xi = 0.003$ (NB2)
  - **Corners**: $\sigma_{\text{team}} = 0.20$, $\sigma_{\text{ref}} = 0.03$, $\xi = 0.003$ (NB2)
  - **Fouls**: $\sigma_{\text{team}} = 0.10$, $\sigma_{\text{ref}} = 0.08$, $\xi = 0.003$ (NB2)
  - **Red cards**: $\sigma_{\text{team}} = 0.15$, $\sigma_{\text{ref}} = 0.01$, $\xi = 0.0$ (Poisson)

### 3. Walk-Forward Acceptance Results
Evaluated on the last two full EPL seasons (2024–25 and 2025–26, 760 matches, 1,520 team-matches) refitting strictly per matchday block prior to kickoffs. Each model beats the league-average baseline on overall Poisson deviance:

| Statistic | Family | Model Deviance | Baseline Deviance | Improvement | Model Log Loss | Baseline Log Loss | Dispersion $\alpha$ | Beats Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Yellow Cards** | NB2 | 1.05190 | 1.12083 | **+6.15%** | 1.64788 | 1.68232 | 0.0003 | **True** |
| **Corners** | NB2 | 1.54328 | 1.77099 | **+12.86%** | 2.37057 | 2.44072 | 0.0776 | **True** |
| **Fouls** | NB2 | 1.00823 | 1.12447 | **+10.34%** | 2.59912 | 2.65587 | 0.0003 | **True** |
| **Red Cards** | Poisson | 0.34266 | 0.34291 | **+0.07%** | 0.22983 | 0.22996 | 0.0000 | **True** |

- Generated reports: `reports/event_rates_predictions.csv` and `reports/event_rates_summary.csv`.
- CLI command: `footsim backtest-events --seasons 2425,2526`.

---

## Milestone 4: Match Simulator

### 1. Vectorised Minute-by-Minute Architecture
- **Simulations**: $N$ matches (default 100,000–200,000) simulated simultaneously as NumPy arrays across minutes $1 \dots T$ (~95–100 minutes total, looping over minutes, not simulations).
- **Runtime Performance**: 100,000 simulations complete in ~10.16 seconds on a standard CPU (well within the < 15-second acceptance requirement).
- **Stoppage Time Distributions**:
  - First Half extra minutes: $S_1 \sim \text{Poisson}(\mu=2.0)$, clamped to $[0, 10]$.
  - Second Half extra minutes: $S_2 \sim \text{Poisson}(\mu=4.5)$, clamped to $[0, 15]$.
  - Default placeholder: empirical averages from recent Premier League seasons with modern extended stoppage rules.

### 2. Minute Profiles (Empirical Placeholders)
- Minute profiles define the baseline probability density $w_m$ of an event occurring in minute $m$ (summing to 1.0 across 90 regulation minutes):
  - **Goals**: $w_m \propto 1.0 + 0.4 \cdot (m / 90)$ (higher scoring rate in late second half due to fatigue and open play).
  - **Yellow Cards**: $w_m \propto 1.0 + 1.2 \cdot (m / 90)$ (tactical fouls and late cards escalate strongly).
  - **Red Cards**: $w_m \propto 1.0 + 0.8 \cdot (m / 90)$.
  - **Corners**: Flat uniform profile $w_m = 1/90$.
  - **Fouls**: Flat uniform profile $w_m = 1/90$.
- Extra stoppage minutes inherit the intensity of minute 45 (for 1H stoppage) and minute 90 (for 2H stoppage).

### 3. Game-State Multipliers (Placeholders in `SimConfig`)
- When a team receives a red card:
  - **Scoring intensity multiplier**: $0.70$ (scores 30% less).
  - **Conceding intensity multiplier**: $1.35$ (concedes 35% more).
- Trailing teams pushing late (minutes $\ge 75$, score deficit $\ge 1$):
  - **Scoring intensity multiplier**: $1.20$ (pushes forward).
  - **Conceding intensity multiplier**: $1.25$ (vulnerable on counter-attacks).
- Multipliers are configurable in `footsim/sim/config.py` (`SimConfig`). Neutral mode sets all multipliers to $1.0$.

### 4. Dixon-Coles Low-Score Neutral Coupling
- Independent minute-level Poisson draws converge to independent Poisson ($\rho = 0$).
- To match the analytic Dixon-Coles bivariate distribution when $\rho \neq 0$, an exact minimal transition coupling was derived:
  - For $\rho < 0$: transfer $(1,0) \to (0,0)$ with probability $|\rho|\mu$ and $(0,1) \to (1,1)$ with probability $|\rho|\lambda$.
  - For $\rho > 0$: transfer $(0,0) \to (1,0)$ with probability $\rho\mu$ and $(1,1) \to (0,1)$ with probability $\rho\lambda$.
- Acceptance verification: 1,000,000 neutral simulations converge to the analytical Dixon-Coles probability matrix with maximum absolute error $< 0.002$ across all cells with $p \ge 0.5\%$.

### 5. Second Yellow to Red Conversion
- Second yellow cards for any simulated team are tracked dynamically. A second yellow converts into a red card sending off, updates red card counts, and activates the red card game-state penalty for the remainder of the match.

### 6. Player Layer Omission
- `football-data.co.uk` CSVs provide match-level team totals (goals, cards, corners, fouls, referee) but do not record starting lineups or individual player events.
- Per `spec.md` ("Needs lineups. If no lineup data is available, skip it and say so"), the optional player layer is skipped until a lineup data provider is attached.

---

## Milestone 5: Market Query Layer

### 1. Unified Simulation Filter Engine
- Every market query is evaluated as a boolean filter expression over the simulation DataFrame (`matching_sims / total_sims`).
- Evaluated via `df.eval(..., engine="python")`, supporting arbitrary complex logical combinations (e.g. `"home_goals == 2 and away_goals == 1 and home_reds >= 1"`).
- Ready-made helpers: `market_1x2`, `market_correct_score`, `market_correct_scores`, `market_over_under_goals`, `market_btts`, `market_asian_handicap`, `market_clean_sheet`, `market_total_cards`, `market_total_corners`, `market_red_card`, `market_first_scorer_team`, `market_ht_ft`.

### 2. 200-Sample Size Rule & Wilson Score Confidence Intervals
- Wilson score interval (1927) is used for all 95% confidence intervals:
  $$\text{center} = \frac{\hat{p} + \frac{z^2}{2N}}{1 + \frac{z^2}{N}}, \quad \text{margin} = \frac{z}{1 + \frac{z^2}{N}} \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}$$
- When fewer than 200 simulated matches satisfy the query:
  - The `reliable` flag is strictly set to `False`.
  - The CLI and `MarketResult.summary()` display a prominent warning forbidding point estimate presentation.
  - Recommends an increased simulation count $N \ge \lceil 200 / \hat{p} \rceil$ required to achieve 200 matching occurrences.

---

## Milestone 6: Command-Line Interface (CLI)

### 1. Command Suite
- Built with `typer`:
  - `footsim fit --league E0 --as-of 2025-01-11`: fits goals and event models, serializes models to disk.
  - `footsim predict --home Arsenal --away Chelsea --referee "M Oliver" --sims 200000`: runs full simulation and prints compact table (1X2, top 10 correct scores, O/U 1.5/2.5/3.5, BTTS, cards, corners, red card probability).
  - `footsim market "<expression>" --home Arsenal --away Chelsea`: queries arbitrary boolean conditions.
  - `footsim backtest --seasons 2023,2024`: walk-forward backtest evaluated against Pinnacle closing odds.
  - `footsim backtest-events --seasons 2425,2526`: walk-forward deviance benchmark on event rate models.

---

## Milestone 7: Engine Enhancements & Advanced Analytics

### 1. Shot-Conversion Expected Goals Proxy (`xg_blend`)
- **Rationale**: Real goals exhibit high stochastic variance in small samples (~2.7 goals/match). Shots on target and dangerous shot volume provide a far lower variance signal of attacking and defensive quality.
- **Formulation**:
  $$\text{xG}_{\text{home}} = 0.30 \cdot \text{HST} + 0.05 \cdot \max(\text{HS} - \text{HST}, 0)$$
  $$\text{xG}_{\text{away}} = 0.30 \cdot \text{AST} + 0.05 \cdot \max(\text{AS} - \text{AST}, 0)$$
  Blended objective target:
  $$x = (1 - \alpha) \cdot \text{FTHG} + \alpha \cdot \text{xG}_{\text{home}}$$
  $$y = (1 - \alpha) \cdot \text{FTAG} + \alpha \cdot \text{xG}_{\text{away}}$$
  where $\alpha = \text{xg\_blend} \in [0.0, 1.0]$. The Poisson negative log-likelihood applies seamlessly to continuous targets via $\Gamma(x+1)$ (`scipy.special.gammaln`).

### 2. Hierarchical Per-Team Home Advantage (`home_adv_mode="team"`)
- **Rationale**: Travel distance, pitch geometry, and crowd atmosphere create team-specific home advantages (e.g. Newcastle vs Fulham) rather than a uniform league scalar.
- **Empirical Bayes Shrinkage**:
  $$h_i = \bar{h} + \frac{N_{i, \text{home}}}{N_{i, \text{home}} + \tau_h} \left( \Delta \bar{g}_{i, \text{home-away}} - \bar{h} \right)$$
  with prior weight $\tau_h = 19.0$ (one full home season). Prevents small-sample noise from overfitting promoted or volatile teams.

### 3. Frank Copula Bivariate Scoreline Distribution (`method="copula"`)
- **Limitation of Dixon-Coles**: The classic Dixon-Coles model only introduces score dependence on low-scoring cells $(0,0), (1,0), (0,1), (1,1)$ via $\rho$.
- **Bivariate Copula Solution**: The Frank Copula CDF:
  $$C(u, v; \theta) = -\frac{1}{\theta} \ln\left(1 + \frac{(e^{-\theta u} - 1)(e^{-\theta v} - 1)}{e^{-\theta} - 1}\right)$$
  discretized over Poisson marginal CDFs $F_X, F_Y$. Provides smooth tail dependence across all scorelines, accurately modeling open high-scoring shootouts ($2-2, 3-3$) with parameter $\theta \approx 0.25$.

### 4. Secondary Event In-Play Game-State Mechanics
- **Trailing Corner Urgency**: Teams trailing in the second half cross frequently into crowded penalty areas, boosting corner hazard by $+30\%$ (`trailing_corner_mult = 1.30`) while leading teams drop into a low block generating $-15\%$ corners (`leading_corner_mult = 0.85`).
- **Late-Game Card Escalation**: In competitive close fixtures ($|\text{score\_diff}| \le 1$) after minute 70, tactical stopping fouls and time-wasting escalate yellow/red hazard by $+25\%$ (`close_game_card_mult = 1.25`).
- **Derby / Rivalry Modifiers**: Canonical Premier League derbies (North London, Merseyside, Manchester, London derbies) receive an empirical card hazard multiplier of $1.25$ and foul multiplier of $1.15$.

### 5. Player-Level Simulation Layer (`footsim/sim/players.py`)
- **Architecture**: Starting lineups define positional profiles (FW, MF, DF, GK) with normalized individual xG shares ($\sum \text{xG\_share} = 1.0$) and foul/card shares.
- **Attribution Engine**: Per simulated match, goals and cards are attributed to players via weighted categorical sampling.
- **Outputs**: Generates anytime goalscorer probabilities, first goalscorer probabilities, and card hazard tables per player.

### 6. Post-Hoc Probability Calibration (`PlattCalibrator`)
- **Multinomial Logistic Scaling**: Corrects for bookmaker margin asymmetries and model probability overconfidence across $1\text{X}2$ outcomes:
  $$z'_k = a \cdot \ln(p_k) + b_k, \quad p'_k = \text{softmax}(z')$$
  Evaluated on walk-forward predictions to produce calibrated probability outputs.

---

## Milestone 8: Squad Availability, VORP & Schedule Congestion

### 1. Value Over Replacement Player (VORP)
- **Problem**: In naive Poisson or team-aggregate models, player absences are either ignored completely or treated as losing 100% of their goal contribution.
- **Solution**: Team attacking output degrades by the difference between the absent starter and their bench replacement:
  $$\lambda_{\text{adj}} = \lambda_{\text{base}} \times \left[ 1 - \sum_{p \in \text{Absences}} \text{xG\_share}_p \cdot (1 - R_{\text{tier}}) \right]$$
  where $R_{\text{tier}}$ is the empirical bench replacement factor ($0.76$ for Tier 1 elite squads like Man City down to $0.28$ for Tier 4 shallow squads like Ipswich).

### 2. Positional Asymmetry ("The Rodri Effect")
- **Asymmetric Impact**:
  - Missing Strikers/Wingers (FW): Primarily suppresses attacking expected goals $\lambda$.
  - Missing Defensive Midfielders (DM / CDM): Central transition anchors destabilize the team structure. Concession $\mu$ jumps by $+22\% \times (1 - R_{\text{tier}})$ and attack drops $-6\%$.
  - Missing Center-Backs (CB) and Goalkeepers (GK): Concession jumps by $+18\%\text{--}+24\%$.

### 3. Rest Days & Schedule Congestion
- **Short Turnaround Degradation**:
  - When days of rest between matches $\le 3$, physical energy and sprint intensity decay:
    $$\Delta \lambda = -0.08 \cdot \frac{4 - \text{rest\_days}}{4} \cdot (1 - 0.5 \cdot D_{\text{team}})$$
    $$\Delta \mu = +0.09 \cdot \frac{4 - \text{rest\_days}}{4} \cdot (1 - 0.5 \cdot D_{\text{team}})$$
  - Squad depth index $D_{\text{team}}$ buffers deep teams against congestion decay.

---

## Milestone 9: Live In-Play Match Simulation (`footsim live`)

### 1. Time-Proportional Arrival & Active Red Card Hazard
- **Time Fraction**: For in-play minute $m \in [0, 90]$, remaining expected minutes $t_{\text{rem}} = \max(1, 94 - m) / 90.0$.
- **Active Red Cards**:
  - Scoring penalty: $\lambda_{\text{rem}} = \lambda \cdot t_{\text{rem}} \cdot (1 - 0.32 \cdot r_{\text{home}})$.
  - Concession penalty: $\mu_{\text{rem}} = \mu \cdot t_{\text{rem}} \cdot (1 + 0.38 \cdot r_{\text{home}})$.
- **Analytical Next Goal**:
  $$P(\text{No more goals}) = e^{-(\lambda_{\text{rem}} + \mu_{\text{rem}})}$$
  $$P(\text{Home scores next}) = (1 - P(\text{No more goals})) \cdot \frac{\lambda_{\text{rem}}}{\lambda_{\text{rem}} + \mu_{\text{rem}}}$$

---

## Milestone 10: Rolling Form Momentum & Tactical Head-to-Head

### 1. Rolling 5-Match Form Momentum
- Calculates points per game (PPG) over the last 5 fixtures compared against the baseline league expectation ($1.35$ PPG).
- Applies Empirical Bayes shrinkage with a conservative scaling factor ($0.15$), capping total form momentum adjustment to $\pm 8\%$ to prevent overfitting to short-term variance.

### 2. Tactical Head-to-Head (H2H) Window (24 Months)
- Multi-year head-to-head records are confounded by managerial turnover, tactical shifts, and squad renewal.
- H2H records are strictly filtered to the last 730 days (24 months). If fewer than 2 matches exist in that window, the edge is shrunk to zero. If $\ge 2$ matches exist, goal differential edge is shrunk by $0.08$ with bounds $[-0.12, +0.12]$ goals.

---

## Milestone 11: Pre-Match Inspection Dashboard (`footsim inspect`)

### 1. Process Metrics Over Raw Scores
- In small rolling windows (5 matches), actual match scorelines are heavily contaminated by variance (deflections, referee calls, finishing luck).
- The inspection dashboard audits **underlying process metrics**:
  - Expected Goals created ($\text{xGF}$) vs conceded ($\text{xGA}$) using the continuous shot-conversion model ($0.30 \cdot \text{SOT} + 0.05 \cdot \text{OffTarget}$).
  - Shots on target ratio ($\text{SOT}_{\text{ratio}} = \text{SOT}_{\text{for}} / (\text{SOT}_{\text{for}} + \text{SOT}_{\text{against}})$).
  - Goalkeeper saves made ($S = \max(0, \text{SOT}_{\text{against}} - \text{GA})$) and rolling save percentage.

### 2. Post-Shot Expected Goals (PSxG) Goalkeeper Profiling
- Starting goalkeepers are mapped to empirical PSxG +/- ratings derived from Opta and FBref shot-stopping tracking.
- Top-tier shot stoppers (e.g. Alisson Becker $+0.32$, Emiliano Martinez $+0.28$, David Raya $+0.24$) consistently prevent goals above the baseline, while volatile or backup keepers leak $+0.08\text{--}+0.20$ goals per match.

### 3. Referee Disciplinary Index
- Referees exhibit persistent individual card tolerance thresholds. The inspection dashboard calculates each official's historical cards-per-match ratio against the league average rate:
  $$\text{strictness\_index} = \frac{\bar{Y}_{\text{ref}} + \bar{R}_{\text{ref}}}{\bar{Y}_{\text{league}} + \bar{R}_{\text{league}}}$$
  Classifying officials as *Lenient* ($< 0.88$), *Neutral* ($0.88\text{--}1.15$), or *Strict* ($> 1.15$).

---

## Milestone 12 (Phase A): Economic & Edge Engine Decisions

### 1. Priority 0 Data Leakage Invariant
- **Rule**: No feature, rating, or parameter estimation may use any match played at or after $t_{\text{kickoff}}$, and no closing odds may influence pre-match selections.
- **Assertion**: `assert_feature_timestamp(feature_name, feature_ts, kickoff_ts)` is enforced throughout the backtesting pipeline.

### 2. Opening vs Closing Odds and Closing Line Value (CLV)
- Football-data CSVs provide `PSH, PSD, PSA` (opening odds) and `PSCH, PSCD, PSCA` (closing odds).
- Wagers are placed at opening prices; CLV is calculated against the closing line: $\text{CLV} = O_{\text{open}} / O_{\text{close}} - 1.0$.
- CLV is the gold standard for separating statistical skill from short-term outcome luck in efficient betting markets.

### 3. Fractional Kelly Staking & Bankroll Bounds
- Full Kelly staking ($f^* = (bp - q) / b$) is excessively volatile given parameter estimation error.
- Default staking is set to Fractional Kelly ($0.25 \times f^*$) with an absolute maximum stake ceiling of 3.0% of bankroll per fixture to avoid ruin during drawdowns.

---

## Milestone 13 (Phase B): Statistical Significance & Segmented Calibration Decisions

### 1. Paired Non-Parametric Bootstrap
- Because football match outcomes are discrete and clustered across matchdays, standard independent $t$-tests underestimate variance.
- $B=2{,}000$ paired bootstrap resamples compute empirical differences $\Delta = \text{Score}_{\text{model}} - \text{Score}_{\text{Pinnacle}}$, yielding exact empirical $p$-values and percentile 95% confidence intervals for RPS, Brier, and Log Loss.

### 2. Diebold-Mariano Test with Newey-West Autocovariance
- Evaluates sequential predictive accuracy over time with Bartlett kernel weighting across lag $h=1$ to account for weekend/midweek clustering.

### 3. Segmented Calibration Archetypes
- Rather than assuming uniform calibration error across all fixtures, matches are segmented into:
  - Home Favorites ($p_H \ge 0.50$)
  - Away Favorites ($p_A \ge 0.38$)
  - Balanced / Contested Draws ($p_H < 0.50$ and $p_A < 0.38$)
  - Extreme Longshots ($p < 0.15$)
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) are evaluated per segment to guide segment-aware shrinkage.

---

## Milestone 14 (Phase C): Opponent-Adjusted EWMA Ratings & Shot-Level xG Decisions

### 1. Shot-Level Calibrated xG Proxy
- Combines shots on target, off-target shots, and corners using empirical EPL conversion weights:
  $$\text{xG}_{\text{proxy}} = 0.31 \cdot \text{SoT} + 0.045 \cdot (\text{Shots} - \text{SoT}) + 0.035 \cdot \text{Corners}$$
- Calibrated to single-team match bounds $[0.05, 6.0]$.

### 2. Opponent Adjustment via Rating Ratios
- Performance is scaled relative to the opponent's pre-match defensive strength:
  $$\text{Performance}_{\text{att}} = \frac{\text{xG}_{\text{scored}}}{\max(R_{\text{def, opp}} \cdot 1.35, 0.40)}$$
- Updated sequentially using EWMA with $\alpha = 0.15$ (~10 match half-life), vectorized using pre-extracted numpy arrays for sub-second execution across 10+ seasons.





