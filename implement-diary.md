# TA-RWARE Implement Diary

> 按時間軸記錄每筆 implementation / 修改。每筆標示對應 [Implementation Plan](../Research-Note/wiki/experiments/implementation-plan.md) 的階段與 indicator，並簡述「做了什麼、為什麼、結果是什麼」。

---

## 2026-05-17 (Phase 1 IAC)

### [12] `tarware/algos/` — Phase 1: IAC non-hierarchical baseline

**對應 Plan**: Phase 1 — IAC Training Indicators (episode_return, pick_rate, clash_rate, policy_entropy, mask_behavior, utilization)

---

#### `tarware/algos/ac_nets.py` (45 lines)

獨立的 `Actor` / `Critic` MLP，各自 2×FC(64)+ReLU + 輸出層，orthogonal init（hidden: sqrt(2)、policy head: 0.01、value head: 1.0）。`Actor.forward()` 回傳 raw logits；`Critic.forward()` 回傳 scalar（`squeeze(-1)`）。

---

#### `tarware/algos/rollout.py` (74 lines)

`Transition` dataclass + `NStepRollout`。`compute_gae()` 實作標準 GAE backward pass，支援 bootstrap value 與 done-mask 遮蔽。通過 `tests/test_rollout.py` 3 個數值測試（包含 done boundary）。

---

#### `tarware/algos/picker_policy.py` (116 lines)

**關鍵元件**。`PickerHeuristicPolicy` 複製 `heuristic.py` 的 picker 分配邏輯，讓 AGV RL policy 能搭配確定性 picker baseline。

實作難點：
1. **狀態持久化**：heuristic 每 step 掃描整個 `assigned_agvs` dict（包含 busy 中的 AGV），而非只看當步 action。使用 `agv.target` 推斷進行中的任務，不只看傳入的 action。
2. **執行順序**：heuristic 是先 assign pickers（lines 117-124）、再 pop 已到達的 pickers（lines 125-129），才建立 actions。若順序反過來，picker 在「已抵達、同步被 assign、同步被 pop」的邊界 case 會輸出錯誤 action（step 15 bug）。

通過 `tests/test_picker_policy.py` parity test：5 個 seed × 500 steps 完全吻合 heuristic_episode 輸出。

---

#### `tarware/algos/iac.py` (136 lines)

`HyperParams` dataclass + `IACAgent`。`act()` 對 logits 做 masked_fill（-1e9）再 sample，返回 (action, log_prob, value, entropy)。`update()` 標準化 advantage、計算 policy loss + value loss、clip grad norm、clear buffer。

---

#### `tarware/algos/trainer.py` (186 lines)

`IACTrainer` 封裝完整訓練 loop。關鍵細節：使用 `env.unwrapped` 存取 warehouse 內部狀態（`agents`, `compute_valid_action_masks()`, `num_agvs`, `num_pickers`）；env.reset() 回傳 obs tuple（非 gymnasium 標準的 (obs, info)）。n-step update 在 `rollout_step >= n_step or done` 時觸發，bootstrap 在 episode 結束時為 0.0。checkpoint 每 500 ep 存一次（從 ep 1000 開始）。

---

#### `scripts/run_iac.py` (74 lines)

Mirror `run_heuristic.py` 架構，支援所有 hyperparameter CLI flags，W&B logging，checkpoint_dir 自動建立。

---

#### `tests/` (3 files, 366 lines)

| 檔案 | 測試 | 結果 |
|------|------|------|
| `test_rollout.py` | GAE 數值正確性（3 cases）| PASS |
| `test_picker_policy.py` | 5 seed × full episode parity | PASS |
| `test_iac_smoke.py` | 10 episodes no-crash + schema check | PASS |

全部 5 tests PASS (`pytest tests/ -v`)。

---

#### Sanity run

```
env: tarware-tiny-2agvs-1pickers-partialobs-v1
episodes: 1500, seed=0
```

Pick rate trend（待 run 完成後補填）：ep 0 ≈ 1.44, ep 500, ep 1000, ep 1500 見 `logs/iac_tiny_sanity.log`。

---

## 2026-05-16

當天工作分三個時段：

1. **並行 implementation**：拆兩個 agent（Agent A 寫 code、Agent B 做 correctness review），同時推進 Phase 0-C 所需的 baseline profiling 基礎建設（項目 [1][2][3]）。
2. **文件補完**：更新 CLAUDE.md，建立 conda env 與 PYTHONPATH 的操作說明（項目 [4]）。
3. **Code-review 後續修正**：根據 codex review 找出兩個 bug，逐一修（項目 [5][6]）。

---

### [1] `scripts/run_heuristic.py` — Schema fix: 保留 raw `delivery_travel_times`

**對應 Plan**: Phase 0-C Baseline Profiling — **Travel time 分佈** indicator
**做了什麼**: `build_episode_metrics` 回傳 dict 加入 `"delivery_travel_times": all_travel_times`（raw list of int），per-episode summary `travel_time_p50/p95/...` 保留不變。

**為什麼**:
原本 raw list 算完 per-episode summary 就丟掉，沒進 jsonl。10000 episodes 跨 8 個 parallel runs merge 後**無法**算全域 p95——percentile 是非線性運算，`mean(per-episode p95) ≠ global p95`。Phase 0-C 要的 heavy-tail profiling 直接 break。

**結果**:
- jsonl 每 row 多 ~40 個 int；10000 episodes 全集 < 10 MB
- W&B `wandb.log` 對 list silent handle，不影響既有行為
- 修正後 jsonl 才有後續 merge 全域 percentile 的依據

---

### [2] `scripts/merge_runs.py` — 新增：8 個 parallel runs → 1 merged run

**對應 Plan**: Phase 0-B Heuristic Baseline Run（10000 episodes 並行執行基建）+ Phase 0-C 全域 percentile 計算
**做了什麼**: 新增 ~205 行的 merge tool；validation + 排序 + 全域 metric 重算 + 可選 W&B 上傳；source dir 絕不修改。

**CLI**:
```bash
scripts/merge_runs.py --run_glob "runs/heuristic_*_${TIMESTAMP}" [--wandb]
```

**Validation**（任一失敗 abort）:
1. 每 source dir 必須有 `config.json` + `episodes.jsonl`
2. `env_id`、`algo` 一致
3. `[seed, seed+actual)` 跨 runs 不重疊
4. 全域 `episode` index 無重複
5. row count ≠ `config.num_episodes` → WARN 繼續（partial run）

**Output**:
- `episodes.jsonl` — 依 episode 升序
- `merged_config.json` — `sources`、`seed_ranges`、`source_actual_counts`、`total_episodes`、`merged_at`、`global_metrics`（含 `travel_time_*` 如果 jsonl 有 raw list）

**W&B 模式**:
- `wandb.init(group="merged__{env_id}", tags=["merged", algo, env_id])`
- 依 episode 順序 `wandb.log(row, step=episode)`，最後 log `global/*` 並寫入 `summary`

**Sanity check（在 0510_0025 舊 runs 上跑，無 `--wandb`）**:
- matched 8 dir、merged 10000 rows、`seed coverage [0, 10000)`、`pick_rate_mean=54.29`
- 舊 runs 早於 [1] schema fix，`delivery_travel_times` raw list 不存在 → `global travel_time_*` silent skip（merge 不報錯）

---

### [3] `Research-Note/wiki/research/parallel-heuristic-correctness.md` — 新增：並行正確性審查

**對應 Plan**: Phase 0-A Codebase 理解（深入 step() RNG / metric 計算邏輯）+ Phase 0-C 確保 profiling 資料可信
**做了什麼**: 完整 per-metric 表 + RNG 分析 + seed 不重疊性證明 + merge tool contract，給 Phase 0-C 的 profiling pipeline 背書。

**結論摘要**:

| 項目 | 結果 |
|------|------|
| Parallel episode determinism | **PASS** — `Warehouse.reset(seed)` 完整 reseed `np.random` 與 `random` |
| Seed 不重疊性 | **PASS** — worker k 用 `[k*1250, (k+1)*1250)`，完整 tile `[0, 10000)` |
| 所有 scalar metric concatenation safety | **PASS** — 全部 per-episode 獨立計算 |
| Global travel-time percentile 可還原性 | **FAIL → 已修**（[1]） |
| W&B step 唯一性 | **PASS** — `step=episode_offset+i` 全域不重疊 |
| `fps` 的可 aggregate 性 | **WARN** — per-process wall-clock，僅 diagnostic |
| RNG leak across episodes | **PASS** — `reset()` 每 episode 第一個 call |

---

### [4] `CLAUDE.md` × 2 — Infra：補上 conda env 與 PYTHONPATH 操作說明

**對應 Plan**: 不對應特定 Phase（infra / dev docs）；所有 Phase 都會用到
**做了什麼**: 把 onboarding 必查的環境資訊寫進兩份 CLAUDE.md。

**TA-RWARE/CLAUDE.md**:
- Installation 後新增 "Conda environment" 區段，說明必須用 `/mnt/sda/home/r147250250916/.conda/envs/tarware/bin/python`、要設 `PYTHONPATH`、理由（only this env has `gymnasium`, `tarware`, `pyastar2d`）
- Running 範例改成完整 conda env 路徑；加 `merge_runs.py` 範例

**MARL/CLAUDE.md**:
- 新增 "Environment" 區段一行指向 tarware conda env

---

### [5] `scripts/merge_runs.py` — Code-review fix: partial-run interval bug

**對應 Plan**: Phase 0-C tooling 正確性（接續 [2]）
**做了什麼**: interval 改用 `seed + actual` 而非 `seed + nominal`，避免 partial run 誇大 coverage、誤判 overlap fail。

**問題**:
原本 line 111 用 `seed + nominal` 建 interval。反例：
- Run A: `seed=0, nominal=1250, actual=500`，實際只有 `[0, 500)`
- Run B: `seed=500, nominal=1250, actual=1250`，實際 `[500, 1750)`
- 舊邏輯（nominal）: `[0, 1250)` vs `[500, 1750)` → **誤報 overlap**
- 新邏輯（actual）: `[0, 500)` vs `[500, 1750)` → 正確 pass

**改了什麼**:
- interval tuple 改成 `(seed, seed+actual, name, actual, nominal)`
- overlap 檢查使用 actual
- `merged_config.json` 加 `source_nominal_counts`，與 `source_actual_counts` 並列保留
- end-of-merge print 加 `(span N, rows M)`，不一致時印 `[warn] X episode(s) missing within span`

**驗證**:
- 完整 8 runs 行為不變（`pick_rate_mean=54.292`）
- Fake partial scenario `seed=0/actual=500 + seed=500/nominal=750`：修前 abort、修後正確合併 1250 rows

---

### [6] `tarware/warehouse.py` — Code-review fix: `delivery_travel_times` off-by-1

**對應 Plan**: Phase 0-A Codebase 理解（review env step 順序時發現）→ 直接影響 Phase 0-C Travel time 分佈 indicator
**做了什麼**: `delivery_travel_times` 公式加 `+1`，採用定義 A：「AGV 花了幾個 environment steps 完成 delivery」（包頭包尾）。

**問題**（step N 內時序）:

| 行 | 動作 | `_cur_steps` |
|----|------|--------------|
| L451 | `task_start_step = _cur_steps` | N |
| L901 | `delivery_travel_times.append(_cur_steps - task_start_step)` | N |
| L1004 | `_cur_steps += 1` | → N+1 |

最小反例：AGV 在分派任務的同 step 就 deliver，記錄到的 `travel_time = 0`，但實際消耗 1 個 environment step。

**改了什麼**:
- L901: `self._cur_steps - agent.task_start_step` → `self._cur_steps - agent.task_start_step + 1`
- 上方加註解說明 fence-post 邏輯

**影響**:
- 所有未來 `delivery_travel_times` 整體 +1
- 0510_0025 舊批 10000 episodes 的值較新版少 1（跨版本比較時要心理 +1）

---

## Open risks（未修，列為操作守則）

| 風險 | 決策 | 操作守則 |
|------|------|---------|
| `np.random` 用 module-global 而非 private `Generator` | 不修 | 目前 `reset()` 完整 reseed safe；未來若引入會在 import 時 reseed 的 module 要重新評估 |
| Run dir timestamp 解析度只到分鐘 (`%m%d_%H%M`) | 不修 | 重跑前換 seed 或等過 1 分鐘；8-core 並行 seed 不同自然不撞，只有手動立即重跑 + 忘換 seed 才會 silent overwrite |

---

## 修改的檔案總覽

| 檔案 | 類型 | 對應 Plan | 項目 |
|------|------|-----------|------|
| `scripts/run_heuristic.py` | 修改 | Phase 0-C | [1] `build_episode_metrics` 加 raw `delivery_travel_times` |
| `scripts/merge_runs.py` | 新增 + 修改 | Phase 0-B/C 基建 | [2] 8 runs → 1 merged；[5] partial-run interval fix |
| `tarware/warehouse.py` | 修改 | Phase 0-A → 0-C | [6] off-by-1 +1 修正 |
| `TA-RWARE/CLAUDE.md` | 修改 | Infra | [4] Conda env / Running 範例 / `merge_runs.py` 範例 |
| `MARL/CLAUDE.md` | 修改 | Infra | [4] Environment 區段 |
| `Research-Note/wiki/research/parallel-heuristic-correctness.md` | 新增 | Phase 0-A | [3] 並行正確性審查 |

---

## 標準操作流程

**Step 1 — 8-core parallel heuristic**（schema fix + off-by-1 fix 都已套用）:
```bash
cd /mnt/sda/home/r147250250916/research/MARL/TA-RWARE
mkdir -p logs

for i in $(seq 0 7); do
  seed=$((i * 1250))
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/mnt/sda/home/r147250250916/research/MARL/TA-RWARE \
  nohup /mnt/sda/home/r147250250916/.conda/envs/tarware/bin/python \
    scripts/run_heuristic.py \
    --num_episodes=1250 --seed=$seed --episode_offset=$seed \
    --log_dir=runs --wandb \
    > logs/heuristic_${seed}.log 2>&1 &
done
wait
```

**Step 2 — Merge + W&B 上傳**:
```bash
# TIMESTAMP 換成 Step 1 跑完看到的 mmdd_HHMM
TIMESTAMP=mmdd_HHMM

/mnt/sda/home/r147250250916/.conda/envs/tarware/bin/python scripts/merge_runs.py \
  --run_glob "runs/heuristic_tarware-medium-8agvs-4pickers-partialobs-v1_*_${TIMESTAMP}" \
  --wandb
```

產出 merged run 含 `global_metrics.travel_time_p95`（從 raw list 重算）、`pick_rate_mean`、`clash_rate_mean`、`stuck_rate_mean`，對應 Phase 0-C 要回填到 baseline 表格的 indicators。
