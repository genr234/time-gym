# Time Gym v2

**When an AI agent gets more time, does it deliver a better report? Can it estimate how long a fixed task will take?** Time Gym is a tool-use benchmark and a reinforcement learning environment for a Hack Club hackathon application.

**Offline result:** On the held-out family, the fair reference scored **1.00** for budget efficiency. Its mean report value rose from **2.42 to 9.83** as the budget went from 15 to 60 minutes. The fixed-15 control scored **0.00** and the informed oracle scored **1.39**. These are results for scripts. No model was tested live or trained.

## Run the benchmark

Install [uv](https://docs.astral.sh/uv/) and use these commands from the repository root. They do not need an API key.

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m time_gym.report_baseline > dev.jsonl
uv run --locked python -m time_gym.report_baseline --split held-out > held-out.jsonl
uv run --locked --extra figures python -m time_gym.analysis dev.jsonl held-out.jsonl --output figures
```

Use the default split to develop policies and prompts. Reserve `--split held-out` for a final check. The development split has 36 cases from three families (seeds 0–11). The held-out split has 12 cases from a fourth family (seeds 100–111). The benchmark and policies were committed before the held-out run. The JSONL files record tool calls, draft scores, final drafts, and outcome types.

## Two tasks, one simulated clock

- **Fixed time.** Given 15, 30, or 60 minutes, deliver the highest-value verified report. Each case has 12–20 fictional topics with visible weights from 1 to 5. Even with perfect information, a full report took at least 212 minutes in the offline matrix. A read costs a hidden number of minutes within the catalog range. A failed read still costs time. The latest saved draft counts. A save costs one minute. A read that exceeds the budget gives no evidence.
- **Fixed scope.** Give an 80% ETA interval before reading. Then verify the three listed topics with no simulated deadline. The target is the time a fixed procedure would use on that case: read stages in order, try the accessible record with the lowest upper cost first, and save once. The target includes failed attempts in that procedure. The agent cannot change it by delaying a save or choosing other reads. This tests a forecast for a stated procedure, not for the agent's own delivery time.

A finding needs one current record from each stage, read before the save. Some records disagree. The last stage gives the deciding value. Some records require a link from an earlier read. The held-out family adds a choice of links and a final confirmation after the chosen link. A failed or archived record cannot support a finding. Equivalent dates such as `18 May` and `May 18` earn the same credit; a separate counter records nonstandard format.

**Report value** is the sum of verified weights minus the sum of unsupported weights. **Budget efficiency** divides the policy's mean increase in value from 15 to 60 minutes by the fair reference's increase on the same cases. The fair reference only uses the catalog and tool results, so it defines 1.00. The oracle knows hidden costs and picks the highest-value set that fits. The ratio can exceed 1.00. Report it with absolute value: an unchanged good draft and an empty draft can both have an efficiency of 0.00.

For ETA, report coverage, mean interval width, and mean midpoint error. Positive error means the forecast was late. The interval loss is its width plus ten times the distance to the nearest bound if the target lies outside. The terminal ETA reward is `20 / (20 + loss)` if the report verifies all targets, and zero otherwise.

## What the offline results show

There were **1,200 episodes**: 900 for development and 300 for the held-out family. Each case ran seven budget policies at three budgets and four ETA policies.

| Budget policy | Development value, 15 → 60 min | Held-out value, 15 → 60 min | Efficiency, dev / held-out |
| --- | ---: | ---: | ---: |
| Informed oracle | 3.25 → 14.92 | 2.42 → 12.75 | 1.12 / 1.39 |
| Fair reference | 2.72 → 13.11 | 2.42 → 9.83 | 1.00 / 1.00 |
| Fixed-15 control | 2.72 → 2.72 | 2.42 → 2.42 | 0.00 / 0.00 |
| Random plan | 0.19 → 7.67 | 0.00 → 5.58 | 0.72 / 0.75 |

![Mean report value by budget and split](figures/budget-value.png)

The three budget attack scripts earn zero or negative value. On 60-minute held-out runs, the re-read script spends 54.3 minutes on repeated reads on average. Fixed-15 leaves 47.4 minutes unused on average while another finding can still fit.

![Episode flags by type and split](figures/failure-types.png)

The fair ETA covers 100% of cases in both splits. On the held-out split its mean interval is 44.2 minutes wide and its midpoint is 3.25 minutes late. Coverage above the 80% target at this width does not show good calibration. The 999-minute interval also covers every target but earns just 0.020 reward. The padding script saves exactly at its predicted minute (150) in all 48 cases, misses all 48 independent targets, and earns 0.028 mean reward on the held-out split.

![ETA coverage against mean interval width](figures/eta-calibration.png)

## Use the environment or the model runner

```python
from time_gym import ReportEnv

env = ReportEnv(family="chain", budget=60)  # Use budget=None for the ETA task.
brief, _ = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step('{"tool":"finish","args":{}}')
```

The reset observation gives instructions and the catalog. Actions are JSON text for `read_source`, `save_draft`, `finish`, and (ETA only) `estimate`. Keep prior tool responses in the agent's context. The budget task gives one terminal reward: final report value divided by total available weight. Terminal `info` gives the individual metrics. No Tinker integration or training is included.

After approval for a specific spend, the Inkling 2 Mini runner requires `--policy inkling`, `--allow-paid`, and `OPENROUTER_API_KEY`. Use `--reasoning on` or `off`, and `--samples` and `--sample-seed` for repeats. A budget episode allows 17, 32, or 62 requests at 15, 30, or 60 minutes. The ETA default is 120. These are request limits, not price limits. An API error, request limit, plain-text response, or output that cannot be parsed gets its own row. Later episodes still run. Provider support for the seed and reasoning settings is not verified live.

**Claim boundary:** Tool actions use simulated minutes; model inference uses none. These results check the scorer and controls on fictional tasks. They do not show that a model can use a real hour well or improve with training. Next, measure an untuned model on development cases. Test a prompt change if it shows a repeatable weakness. Keep held-out cases out of those decisions and test real-clock behavior separately.
