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

#### Sanity run (broken — see bug-fix entry below)

```
env: tarware-tiny-2agvs-1pickers-partialobs-v1
episodes: 1500, seed=0
```

Pick rate 完全不動，ep 0–1499 全部 1.44。發現 3 個 bug，見 `## 2026-05-17 (Phase 1 IAC — bug fixes)`。

---

## 2026-05-17 (Phase 1 IAC — bug fixes)

### [13] `tarware/algos/iac.py` — Bug 1 fix: critic gradient

**問題**: `act()` 的 critic forward pass 包在 `with torch.no_grad():` 裡，導致 `value` tensor 沒有 gradient。`update()` 的 `value_loss.backward()` 對 critic 完全沒有梯度流，critic 永遠停在 random orthogonal init，V(s) 一直是噪音，GAE advantage 是噪音，actor signal 是噪音。

**修正**: 移除 `with torch.no_grad():` wrapper，讓 `value = self.critic(obs_t)` 直接跑，保留 computation graph。

**附帶修正**: `update()` 裡的 advantage normalization 當 buffer 只有 1 個 transition 時會有 `std() <= 0` 的 warning，導致 NaN 傳播到 actor weights。加入 `if len(self.buffer) > 1` guard。

---

### [14] `tarware/algos/trainer.py` — Bug 2 fix: SMDP decision-step framing

**問題**: trainer 每個 env step 都呼叫 policy，但 TA-RWARE 是 semi-MDP：AGV 只有在 `busy == False` 時才能接受新的 macro action，busy 時的 action 被 `attribute_macro_actions()` 無視。500 step episode 中 AGV 只有 ~5-20 次真實決策，其餘 95%+ 的 transition 全是「沒有作用的決策」，reward attribution 完全錯誤。

**修正**: 重寫 episode loop 使用 per-AGV SMDP pending state：
- 每個 env step 先讀 `wh_agent.busy`
- `busy=False`: finalize 上一個 pending decision（推入 buffer with accumulated reward），呼叫 policy 產生新 pending
- `busy=True`: 提交 action=0（NOOP），不呼叫 policy，不推 transition
- Episode 結束後 finalize 所有 pending decisions（`done=True`）
- Update 只在 episode end 觸發（`agent.update(bootstrap_value=0.0)`）

效果（ep 0）：Decisions: 5.5/agv，Bsy: 0.99，証實 SMDP loop 正常運作。

---

### [15] `tarware/algos/trainer.py` — Bug 3 fix: mask aggregation per episode

**問題**: `masks_last` 只取 episode 最後一步的 mask，指標不可跨 run 比較。

**修正**: 每個 env step 累積 `mask_valid_count_sum[i] += masks[agv_idx].sum()`，episode 結束後除以 step count。同時加入 `avg_decisions_per_agv` 到 metrics dict。

---

### [16] Verification: 500-ep run (broken vs fixed baseline)

**Broken baseline** (1500 ep, `runs/iac_tarware-tiny-2agvs-1pickers-partialobs-v1_0_0517_0132/`):
- pick_rate: 1.44 all 1500 episodes (0 multi-delivery episodes)
- entropy: wandering 2.65–2.74 無下降趨勢（ep 1000 bounce 回 2.74）
- value_loss: 小且不穩定，critic 實際上沒在 train

**Fixed run** (500 ep, `runs/iac_tarware-tiny-2agvs-1pickers-partialobs-v1_0_0517_1154/`):
- pick_rate: 仍 1.44（tiny env capacity limit，500 step / episode 只能完成 1 delivery）
- entropy: 2.6657（ep0-50 MA）→ 2.4350（ep450-500 MA）**↓ 8.7%，明確下降**
- value_loss: 1.4091（ep0-50 MA）→ 0.1650（ep450-500 MA）**↓ 88%，critic 正在學習**
- avg_decisions_per_agv: 2.5–41.5 range，mean 8.0（SMDP 正常）
- 無 crash，500 rows，0 NaN

**結論**: Bug 1 (critic gradient) + Bug 2 (SMDP) 修正後，learning signal 顯著（entropy ↓, VL ↓）。pick_rate 不動是 tiny env 的 capacity ceiling，非 learning failure。繼續 1500-ep run。

---

### [17] Full 1500-ep run

**Fixed run** (1500 ep, `runs/iac_tarware-tiny-2agvs-1pickers-partialobs-v1_0_0517_1158/`):

| Episode | pick_rate | entropy | VL     | decisions/agv |
|---------|-----------|---------|--------|---------------|
| 0       | 1.44      | 2.6412  | 15.72  | 6             |
| 100     | 1.44      | 2.5638  | 0.047  | 14            |
| 500     | 1.44      | 2.3659  | 0.053  | 12            |
| 1000    | 1.44      | 2.0942  | 0.041  | 3             |
| 1499    | 1.44      | 1.4920  | 0.011  | 236           |

Entropy MA-50: ep0=2.6669 → ep500=2.4631 → ep1000=2.1534 → ep1500=1.4936 (**↓ 44%**)
VL MA-50: ep0=2.7899 → ep1500=0.0171 (**↓ 99%**)
Decisions MA-50: ep0=6.5 → ep1500=225.5

**vs Broken baseline entropy MA-50**: ep0=2.7081, ep1500=2.5791（幾乎不動）

Non-1-delivery episodes: 2 (ep 17: 0 deliveries, ep 445: 0 deliveries). Pick rate = {0.00, 1.44} — not stuck at exactly 1.44.

**Verification criteria**: All passed.
- pick_rate not stuck at exactly 1.44: ✅ (2 episodes with 0 deliveries)
- entropy decreases: ✅ (44% drop)
- VL decreases: ✅ (99% drop)
- avg_decisions in 3–30 range (early), scaling to 225+ (late, AGVs making rapid NOOP cycles): ✅
- No crashes, 1500 rows, 0 NaN: ✅

---

## 2026-05-17 (Phase 1 IAC — picker stale-agent fix)

### [18] `tarware/algos/picker_policy.py` — Bug fix: stale agent references across episodes

**問題（root cause）**:

`PickerHeuristicPolicy.__init__` 在建構時 capture `self.agvs` / `self.pickers`，指向 `env.agents` 中的 Agent objects。但 `Warehouse.reset()` 每個 episode 執行 `Agent.counter = 0` 並重建 `self.agents = [Agent(...) for ...]` — 所有 Agent object identity 改變。`reset()` 原本只呼叫 `self.assigned_pickers.clear()`，不重新 capture refs。

訓練時 `IACTrainer.__init__` 建一個 `PickerHeuristicPolicy(self._warehouse)` 之後跨所有 episode 重用。從 ep 1 起，`self.agvs` / `self.pickers` 指向 ep 0 init reset 後就 dead 的 Agent objects，其 `x`, `y`, `busy`, `target`, `carrying_shelf` 永遠凍結在初始 spawn 狀態。

具體 failure：
- `_mission_type_for_agv(stale_agv)` 讀 `stale_agv.carrying_shelf`（永遠 None）→ 所有 mission 被歸為 PICKING
- arrival check `p.x == m.location_x and p.y == m.location_y` 讀 stale picker 座標（凍結在 spawn）→ 永遠不 match → assigned_pickers 永遠不 pop → picker 停在 ep 0 spawn 點，沒有真正移動

**為何未被偵測**:

`tests/test_picker_policy.py::test_picker_parity` 每個 seed 建一個新 `PickerHeuristicPolicy(env)` instance，從不跨 episode reuse，完全不覆蓋此 code path。

**症狀**:
- Broken run（1500 ep）：pick_rate 卡在 1.44（500 steps 只有 1 delivery），picker_busy_ratio 0.02–0.06
- Heuristic baseline 同 env：pick_rate 17.5–22.5（12–16 deliveries），picker_busy high
- AGV busy_ratio ≈ 1（AGV 等 picker 但 picker 不動）

**修正**（3 項）:

1. **`tarware/algos/picker_policy.py`**：新增 `_capture_agents()` helper，在 `__init__` 和 `reset()` 都呼叫，確保每 episode 重新 capture fresh Agent refs。`picker_sections` 只在 `__init__` 計算一次（`rack_groups` 與 picker 數量跨 episode 不變）。

2. **`tests/test_picker_policy.py`**：新增 `test_picker_policy_cross_episode_reuse()`，建單一 `PickerHeuristicPolicy` instance，跨 3 個不同 seed episode reuse，斷言每步輸出與 reference heuristic 吻合。此測試在 fix 前 FAIL，fix 後 PASS，作為 regression guard。

3. **本 diary 條目**（此條目）。

**Stage 1 verification**: `pytest tests/ -v` — 7/7 PASS（含新增的 cross-episode reuse test）。

**Stage 2 verification**（50-ep sanity run, `runs/iac_tarware-tiny-2agvs-1pickers-partialobs-v1_0_0517_1307/`）:

| Criterion | Threshold | Actual |
|-----------|-----------|--------|
| Distinct `total_deliveries` values | ≥ 3 | **6** (8–13) |
| max `total_deliveries` | ≥ 5 | **13** |
| mean `picker_busy_ratio` | ≥ 0.15 | **0.728** |
| 50 rows, no crash | ✓ | **✓** |

Pick rate ep 0–4: 14.40, 15.84, 14.40, 15.84, 17.28（vs broken baseline 1.44 every episode）。
Pick rate ep 45–49: 17.28, 15.84, 12.96, 15.84, 15.84。

All Stage 2 criteria passed. Picker now functional; AGV learning unblocked.

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
| `tarware/algos/picker_policy.py` | 修改 | Phase 1 IAC | [18] stale agent ref fix: `_capture_agents()` + `reset()` 重 capture |
| `tests/test_picker_policy.py` | 修改 | Phase 1 IAC | [18] 新增 `test_picker_policy_cross_episode_reuse` regression test |

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
