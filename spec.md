# Football Match Simulation Engine: Build Spec

Build a Python package that predicts football matches by simulating them minute by minute. Every market (result, correct score, cards, corners, scorers, combinations) is read off the same set of simulated matches, so all probabilities agree with each other.

Start with one league (English Premier League). Get it calibrated before adding anything else.

## Rules for you

- Build in the milestone order below. Do not start a milestone until the previous one passes its acceptance checks.
- Do not predict markets with separate models. One simulator, many queries.
- Time-based splits only. Never random train/test splits. No lookahead: a prediction for a match may only use matches played before it.
- Seed every random generator. Same seed, same output.
- Type hints, docstrings on public functions, pytest tests for each module.
- Keep it simple. No web app, no database, no deep learning. Files and pandas are enough.
- If something here is ambiguous, pick the simpler option and write the choice in `docs/decisions.md`.

## Stack

Python 3.11+, numpy, pandas, scipy, statsmodels, lightgbm (only if needed for the cards model), matplotlib, pytest, typer (CLI). Use pyarrow/parquet for stored sims.

## Repo layout

```
footsim/
  data/            loaders and cleaning
  goals/           Dixon-Coles model
  events/          cards, corners, fouls rate models
  sim/             match simulator
  markets/         query layer over simulations
  eval/            walk-forward backtest and metrics
  cli.py
tests/
docs/decisions.md
```

## Data

Primary source: football-data.co.uk season CSVs, for example `https://www.football-data.co.uk/mmz4281/2425/E0.csv`. Load at least the last 6 completed seasons.

Columns to use: `Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, Referee`, plus Pinnacle odds (`PSH, PSD, PSA`, and closing columns `PSCH, PSCD, PSCA` where present).

Rules:
- Normalise team names across seasons in one mapping file.
- Parse dates explicitly (formats differ between seasons).
- Cache downloads locally. Do not re-download on every run.
- Drop rows with missing goals. Keep rows with missing cards/corners but mark them.

Optional later: xG from Understat or FBref, and minute-level event data (StatsBomb open data) to estimate when cards and goals happen. The football-data CSVs have no minute information, so until minute data is added, use a documented default minute profile for cards and goals.

## Milestone 1: Dixon-Coles goals model

Reference: https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/

Model:
- Home goals ~ Poisson(lambda), lambda = exp(attack_home + defence_away + home_adv)
- Away goals ~ Poisson(mu), mu = exp(attack_away + defence_home)
- Dixon-Coles tau correction on scores 0-0, 1-0, 0-1, 1-1, with parameter rho.
- Time-decay weight on each match in the likelihood: exp(-xi * days_since_match).
- Identifiability: constrain attack parameters to average 1 (or sum to zero in log space). State which one you chose.

Requirements (the reference article gets these wrong or leaves them slow):
- Vectorise the negative log-likelihood with numpy. No Python loops over matches.
- Use `scipy.optimize.minimize` with L-BFGS-B or SLSQP and analytic gradients if you can. If not, the vectorised likelihood alone must fit 6 seasons in under 10 seconds.
- Bound rho so all tau values stay positive.
- Handle promoted teams (no history): shrink their attack/defence toward a promoted-team average, not the league average. Document the prior.
- Tune xi by walk-forward log loss on the full 1X2 distribution, not only on the probability of the observed result. Search a grid, for example 0 to 0.006 per day. Report the curve.
- Output a scoreline probability matrix up to 10 goals each, renormalised to sum to 1.

Public API:

```python
model = DixonColes(xi=0.003).fit(matches_df, as_of="2025-01-11")
matrix = model.score_matrix("Arsenal", "Chelsea")      # 11x11 numpy array
lam, mu = model.expected_goals("Arsenal", "Chelsea")
```

Acceptance checks:
- With rho=0 and xi=0 the model matches a plain Poisson GLM within tolerance.
- Score matrix sums to 1.
- Fitting with `as_of` never touches matches on or after that date (test it).
- Draw probability from Dixon-Coles is higher than from plain Poisson on average over a season.

## Milestone 2: Walk-forward evaluation

Build `eval/` before adding more models, so every later change is measured.

- Walk forward by matchday: for each block of fixtures, fit using only earlier matches, predict the block, store predictions, move on. Refit at most once per matchday block.
- Metrics on 1X2: log loss, Brier score, ranked probability score (RPS).
- Also report over/under 2.5 and BTTS log loss.
- Benchmark against Pinnacle closing odds converted to probabilities with the margin removed (proportional or Shin method, say which).
- Calibration table and plot: predicted probability bins vs observed frequency.
- Save results to `reports/` as CSV and PNG.

Acceptance: a single command produces the report for the last two full seasons.

## Milestone 3: Event rate models

Fit these on the football-data columns. Each one produces an expected count per team per match, plus an overdispersion parameter.

- Yellow cards, red cards, corners, fouls: negative binomial (statsmodels GLM or direct likelihood).
- Features: team tendency for and against, home flag, referee effect (shrunk toward average for referees with few matches), time decay like in Milestone 1.
- Red cards are rare. Model them as a rate with strong shrinkage. Do not try to classify them.
- Optional: LightGBM for cards if it beats the negative binomial on walk-forward log loss. If it does not, drop it.

Public API:

```python
ev = EventRates().fit(matches_df, as_of="2025-01-11")
ev.expected("Arsenal", "Chelsea", referee="M Oliver")
# {"home_yellows": ..., "away_yellows": ..., "home_reds": ..., "corners_home": ..., ...}
```

Acceptance: each model beats a league-average baseline on walk-forward Poisson/NB deviance. Report it.

## Milestone 4: Match simulator

Simulate N matches (default 100,000) with numpy, vectorised across simulations. Loop over minutes (about 95-100 steps), not over simulations.

Per minute, per simulation:
- Goal intensity for each team = base intensity from Milestone 1 (lambda or mu spread over the minute profile) times a game-state multiplier.
- Game-state multiplier depends on score difference, minutes remaining, and red cards on the pitch. Put the multipliers in a config file. Start with simple defaults (a team with a red card scores less and concedes more; a trailing team pushes late) and fit them from data when minute-level data is available. Mark every default as a placeholder in `docs/decisions.md`.
- Card hazards (yellow, red), corner hazards, and foul hazards from Milestone 3, spread over a minute profile.
- A second yellow converts to a red and a sending off.
- Stoppage time: draw extra minutes at the end of each half from a simple distribution.

Optional player layer (do it last, behind a flag):
- Assign each goal to a player by that player's share of the team's expected goals.
- Assign cards to players by foul share.
- Needs lineups. If no lineup data is available, skip it and say so.

Output: a DataFrame with one row per simulated match: final score, half-time score, goal minutes, cards by team and minute, red-card minutes, corners by team, first scorer team, and so on. Save as parquet.

Acceptance checks:
- With game-state multipliers set to 1, the simulated score distribution converges to the Dixon-Coles matrix. Test with 1,000,000 sims and a tolerance on each cell with at least 0.5% probability.
- 100,000 simulations of one match finish in under 15 seconds on a laptop.
- Same seed gives identical output.

## Milestone 5: Market query layer

A market is a boolean filter over the simulation DataFrame. Probability = matching sims / total sims.

```python
sims = simulate("Arsenal", "Chelsea", n=200_000, seed=1)
p = market(sims, "home_goals == 2 and away_goals == 1 and home_reds >= 1")
```

Every result returns:
- probability
- number of matching simulations
- 95% Wilson confidence interval
- a `reliable` flag, false when fewer than 200 sims match

Provide ready-made helpers: 1X2, correct score, over/under goals, BTTS, handicaps, clean sheets, total cards, total corners, red card in match, first goalscorer team, half-time/full-time, and an arbitrary expression.

Sample-size rule: never present a probability from fewer than 200 matching sims as a point estimate. Show the interval and the flag. If the user needs a rare combination, the tool suggests a higher `n`.

## Milestone 6: CLI

```
footsim fit --league E0 --as-of 2025-01-11
footsim predict --home Arsenal --away Chelsea --referee "M Oliver" --sims 200000
footsim market "home_goals == 2 and away_goals == 1" --home Arsenal --away Chelsea
footsim backtest --seasons 2023,2024
```

`predict` prints a compact table: 1X2, top 10 correct scores, over/under 1.5/2.5/3.5, BTTS, expected cards and corners, red-card probability.

## Milestone 7: Advanced Engine Enhancements

- **Frank Copula Discrete Bivariate Distribution**: Full-range score dependence across all 0-10+ goals beyond Dixon-Coles 0-0/1-0/0-1/1-1 tau limitation.
- **Hierarchical Empirical Bayes Home Advantage**: Team-specific home advantage parameters shrunk toward league baseline.
- **Shot-Conversion xG Proxy**: Blending raw goals with shots on target and total shots.
- **Secondary Event Urgency Multipliers**: Late-game trailing corner urgency (+30%), close-match card escalation (+25%), and derby rivalry modifiers (+20% cards/fouls).
- **Player-Level Simulation Layer**: Lineups, individual xG/card shares, anytime & first goalscorer attribution.
- **Post-Hoc Probability Calibration**: Multinomial logistic Platt calibrator.

## Milestone 8: Squad Availability, VORP & Schedule Congestion

- **Squad Depth Tiers & Bench Replacement Quality**:
  - Clubs partitioned into Depth Tiers (Tier 1: City/Chelsea, Tier 2: Arsenal/Liverpool/etc., Tier 3: Mid-table, Tier 4: Shallow).
  - Bench replacement factor $R_{\text{tier}} \in [0.28, 0.76]$ based on bench market value and wage bill.
- **Positional Asymmetry & VORP**:
  - Missing Strikers/Wingers (FW): Attacking $\lambda$ degrades by $-\text{xg\_share} \times (1 - R_{\text{tier}}) \times 1.15$.
  - Missing Playmakers (AM/MF): Decreases attacking $\lambda$ and increases opponent $\mu$.
  - Missing Defensive Midfield Anchors (CDM, "The Rodri Effect"): Increases opponent concession $\mu$ by $+22\% \times (1 - R_{\text{tier}})$ and decreases $\lambda$ by $-6\%$.
  - Missing Center-Backs (CB) and Goalkeepers (GK): Increases concession $\mu$ by $+18\%\text{--}+24\%$.
- **Rest Days & Schedule Congestion**:
  - Turnaround $\le 3$ days penalizes attack and increases late concession inversely scaled by squad depth index.

## Milestone 9: Live In-Play Match Simulation

- **Command**: `footsim live --home Arsenal --away Chelsea --minute 65 --score 1-0 --home-reds 0 --away-reds 1 --sims 50000`
- **Dynamic Time & State Scaling**:
  - Scales remaining baseline intensities by $(94 - \text{minute}) / 90$.
  - Active Red Card Hazard: $-32\%$ scoring and $+38\%$ concession per active red card.
  - Urgency & Desperation: Trailing teams push for equalizers with $+35\%$ corner frequency and vulnerability to counters.
  - In-Play Markets: Full-time 1X2 from current state, Next Goal (Home / Away / No More Goals), In-play Over/Under totals, and top projected final scores.

## Milestone 10: Rolling Form Momentum & Tactical H2H

- **Rolling 5-Match Form**: Points-per-game and goal difference shrunken via Empirical Bayes ($\pm 8\%$ maximum multiplier).
- **Tactical 24-Month Head-to-Head**: Head-to-head records strictly bounded to the last 730 days under modern managerial setups.

## Milestone 11: Pre-Match Inspection Dashboard (`footsim inspect`)

- **Command**: `footsim inspect --home "Arsenal" --away "Chelsea" --referee "M Oliver"`
- **Detailed Process Metrics**:
  - Rolling 5-match audit with Date, Venue, Opponent, Scoreline, Result, xG Created vs Conceded, Shots on Target, Goalkeeper Saves, and Fouls.
  - Starting Goalkeeper Shot-Stopping Profiles with Post-Shot xG (PSxG +/- goals prevented expectation per game).
  - Tactical 24-month H2H record with individual matches and net goal differential edge.
  - Referee Disciplinary Strictness Index evaluated against historical league card distributions.

## What not to build

- No betting or staking logic.
- No claims of profitability. The benchmark is closing-line probabilities, and matching them is already a good result.
- No markets that cannot be validated and need fewer than 200 matching sims at the default `n`, unless clearly flagged unreliable.

## Definition of done

1. `pytest` passes across all test modules (91+ tests).
2. `footsim backtest` reports RPS and log loss for the last two seasons next to Pinnacle closing odds.
3. Simulated score distribution with neutral game state matches the analytic Dixon-Coles matrix.
4. `footsim predict`, `footsim live`, and `footsim inspect` run end to end for any fixture.
5. `docs/decisions.md` lists every default, prior, squad tier, and placeholder.