## Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers — Paper 筆記

### 1. Paper 基本資訊

- Title: Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers
- Authors: Aleksandar Krnjaic, Raul D. Steleac, Jonathan D. Thomas, Georgios Papoudakis, Lukas Schäfer, Andrew Wing Keung To, Kuan-Ho Lao, Murat Cubuktepe, Matthew Haley, Peter Börsting, Stefano V. Albrecht
- Institution / affiliation: Dematic, University of Edinburgh
- arXiv version: arXiv:2212.11498v3, 30 Aug 2024
- 核心主題: 使用 hierarchical MARL 解決 warehouse order-picking problem，讓 robotic workers 與 human workers 在 warehouse 中協作，提高 pick rate。

---

### 2. 研究問題與核心目標

這篇 paper 處理的問題是 order-picking problem。

Order-picking 指的是：warehouse 接收到 orders 後，需要把 order-lines 中要求的 items 從 warehouse 內取出，並送到指定位置進行後續處理。

Paper 中的 performance objective 是 maximising pick rate。

- pick rate: completed order-lines per hour
- 目標: 對給定 warehouse configuration 與 order profile，學出一個 joint policy $\pi$，使 average pick rate $K$ 最大化。

形式化目標：

$$
\pi \in \arg\max_{\pi} K(W, \pi)
$$

Paper 強調，若使用單一 centralized decision-making entity 控制所有 workers，joint action space 會隨著 workers 數量呈 exponential growth，因此不可行。因此作者使用 MARL，把 pickers 與 AGVs 建模成 individual agents。

---

### 3. 研究動機

Paper 的主要 motivation 是：

1. 傳統 industry heuristic methods 需要大量 engineering effort。
   - 因為 warehouse configuration 本身變異很大，例如 warehouse size、layout、workers 數量與種類、item replenishment frequency、order profiles 等。
   - 每個 warehouse 都需要重新調整 heuristic，缺乏 general-purpose flexibility。

2. Order-picking 本質上是 multi-agent problem。
   - 多個 AGVs、robots、human pickers 需要 cooperation。
   - 因此適合用 MARL。

3. MARL 的優勢是可以透過 experience 自動學習 cooperation strategy。
   - 不需要為每種 warehouse layout 或 picking paradigm 手工設計 rule。
   - 可以適用於不同 order-picking paradigms，例如 Person-to-Goods 與 Goods-to-Person。

---

### 4. 本文主要 contribution

Paper 的 contribution 可以整理為四點：

1. 提出一個 general-purpose and scalable MARL solution，用於 heterogeneous warehouse workers 的 order-picking problem。

2. 提出 hierarchical MARL architecture。
   - 使用 multi-layer hierarchy。
   - manager agent 負責指派 tasks 給 worker agents。
   - worker agents 包含 pickers 與 AGVs。
   - task 表示 warehouse 中的 section / zone，例如 aisle。

3. 將 hierarchical architecture 套用在既有 MARL algorithms 上。
   - Independent Actor-Critic (IAC)
   - Shared Network Actor-Critic (SNAC)
   - Shared Experience Actor-Critic (SEAC)
   - 對應 hierarchical versions：HIAC、HSNAC、HSEAC

4. 在兩種 warehouse simulator 中驗證：
   - Dematic PTG simulator: high-performance simulator，可表示 real-world warehouse operations。
   - TA-RWARE: 作者提出的 open-source RWARE adaptation，用於 GTP warehouse task assignment。

---

### 5. Order-picking paradigms

#### 5.1 Person-to-Goods (PTG)

PTG 中，human workers 會在 warehouse 中移動，手動 pick required items。

本文考慮的是 AGV-assisted PTG：

- human pickers 負責 picking。
- AGVs 負責 transportation。
- 傳統 picker role 被拆成兩部分：
  - order transportation: AGVs
  - item picking: human or robotic pickers

PTG 在本文中的流程：

1. AGV 被分配一個 order $z^v$。
2. human picker 到 item location $l \in L_{item}$ 從 storage medium 取出 order-line $(u,q)$。
3. picker 把 item 放進 AGV。
4. AGV 收到完整 order 中所有 order-lines 後，把 order 送到 delivery station $l \in L_{delivery}$。
5. PTG 中 $|L_{delivery}| = 1$。

#### 5.2 Goods-to-Person (GTP)

GTP 中，storage mediums 被 transport robots 移動到 stationary human pickers。

本文的 GTP 流程：

1. 多個 AGVs 會搬運包含 requested items 的 storage mediums。
2. picker robot 把 storage medium 從 item location $L_{item}$ 移到 AGV 上。
3. AGV 把 storage medium 送到 picking station $l \in L_{delivery}$。
4. human operator 在 picking station 從 storage medium 中 pick order-line。
5. 當 order $z$ 中所有 order-lines 都被 picked，order 完成。

在 TA-RWARE 中，paper 特別說明：human picker 不被建模在 simulator 內，而是在 delivery locations 外部。當 paper 在 TA-RWARE context 中提到 picker，指的是 picking robot，負責把 storage mediums 載上 / 卸下 AGV。

---

### 6. Warehouse definition

Paper 將 warehouse 定義為三元組：

$$
W = \{L, Z, W\}
$$

其中：

#### 6.1 Locations $L$

$$
L = L_{item} \cup L_{delivery} \cup L_{other}
$$

- $L_{item}$: 存放 items 的 locations，items 位於 storage mediums 中。
- $L_{delivery}$: completed orders 或 storage mediums 被送達的位置。
- $L_{other}$: 其他位置，例如 idle locations 或 charging locations。

#### 6.2 Order distribution $Z$

$Z$ 是 order distribution，受 warehouse supplier 與 customer behaviour 影響，paper 假設 $Z$ 是 known。

一個 order 定義為：

$$
z = \{(u_0, q_0), \ldots, (u_n, q_n)\}
$$

其中：

- $(u_k, q_k)$: 一個 order-line。
- $u$: item。
- $q$: required quantity。
- item $u$ 存放在某個 item location $l \in L_{item}$ 的 storage medium 中。

#### 6.3 Workers $W$

$$
W = V \cup P
$$

- $V$: homogeneous set of AGVs。
- $P$: homogeneous set of pickers。
- AGVs $v \in V$ 可以 visit locations $l \in L$。
- pickers $p \in P$ 可以 visit locations $l \in L_{item}$。

---

### 7. POSG formulation

Paper 將 multi-agent interaction 建模為 partially observable stochastic game (POSG)。

POSG 定義為：

$$
(I, S, \{A_i\}_{i \in I}, \{O_i\}_{i \in I}, P, \Omega, \{R_i\}_{i \in I})
$$

其中：

- $I = \{1, \ldots, N\}$: agent set。
- $S$: state space。
- $A_i$: agent $i$ 的 action space。
- $A = A_1 \times \cdots \times A_N$: joint action space。
- $O_i$: agent $i$ 的 observation space。
- $P(s_{t+1} \mid s_t, a_t)$: transition probability。
- $\Omega(o^1_{t+1}, \ldots, o^N_{t+1} \mid s_{t+1}, a_t)$: observation probability。
- $R_i(s_t, a_t, s_{t+1})$: agent $i$ 的 reward function。

At timestep $t$：

1. agent $i$ 只能看到 partial observation：

$$
o^i_t \in O_i
$$

2. agent 根據 observation history 選 action：

$$
h^i_t = (o^i_1, \ldots, o^i_t)
$$

$$
\pi^i(a^i_t \mid h^i_t)
$$

3. joint action：

$$
a_t = (a^1_t, \ldots, a^N_t)
$$

4. environment transition 到 $s_{t+1}$。

5. 每個 agent 收到自己的 reward：

$$
r^i_t = R_i(s_t, a_t, s_{t+1})
$$

Agent 的目標是最大化 expected discounted return：

$$
G^i = \sum^T_{t=1} \gamma^{t-1} r^i_t
$$

對每個 agent：

$$
\forall i \in I: \pi^i \in \arg\max_{\pi'^i} \mathbb{E}[G^i \mid \pi'^i, \pi^{-i}]
$$

其中：

$$
\pi^{-i} = \pi \setminus \{\pi^i\}
$$

---

### 8. General action space design

Paper 的 action space 是 high-level location selection，而不是 primitive movements。

Completion of orders 需要：

- pickers 能 visit all item locations $l \in L_{item}$。
- AGVs 能 visit all locations $l \in L$。

因此作者定義：

$$
A^p = L_{item}
$$

$$
A^v = L
$$

也就是：

- picker action: 選擇一個 item location。
- AGV action: 選擇一個 warehouse location。

Agent 被視為 busy，直到它 transit 到選定的 action location。

這個設計的意義：

- worker policy 專注在 coordinating item location selection。
- low-level navigation 交給 predefined controller。
- 在 experiments 中，low-level controller 使用 A* algorithm 找 shortest path。

這個設計的問題：

- action space 會隨 $|L|$ 或 $|L_{item}|$ 變大。
- action completion duration 較長，因為一次 action 可能代表移動到遠方 location。

Paper 使用兩種方法處理：

1. invalid action masking
2. hierarchical MARL architecture

---

### 9. Invalid action masking

Paper 使用 invalid action masking 來降低 effective action space。

方法是調整 logits，把明顯 sub-optimal actions mask out。

以 PTG 為例：

- AGV 在 fulfilling an order $z$ 時，應該只移動到包含 current order 中 requested items 的 item locations。
- 因為期望上 $|z| \ll |L|$，所以把 AGV action space 從 $|L|$ 降到 $|z|$ 對 learning 有幫助。

Paper 也明確指出 action masking 的 trade-off：

- 優點: early training stages 有幫助，簡化 exploration 與 coordination。
- 缺點: invalid action masking introduces bias，可能限制 policy expressiveness。
- 例子: pickers 可能不能 pre-emptively move 去等待 AGVs。
- 作者說 complete exception from large action space training regime 留給 future work。

---

### 10. Hierarchical MARL architecture

Paper 的主要方法是 3-layer manager / worker / low-level hierarchy。

Architecture：

1. Manager agent
   - 觀察 warehouse state and orders。
   - 為每個 worker agent 指派 task，也就是 target zone。

2. Worker agents
   - 接收 local observation 與 manager 指派的 task。
   - 從 assigned target zone 中選一個 item location / target location。

3. Low-level controller
   - 根據 worker 選的 target location，計算 path。
   - 在 experiments 中使用 A* algorithm。

Paper 表示這是對 Feudal Multi-Agent Hierarchies (FMH) 的 3-layer adaptation。

但與 FMH 不同：

- manager goals 不會影響 worker reward functions。
- manager goals 是用來 partition worker action spaces。
- worker agents 不執行 primitive actions，而是把決策交給 lower-level controllers。

---

### 11. Zone partition 與 manager action space

Paper 將 warehouse locations 分割成 disjoint zones。

$$
L = \bigcup_{y \in Y} y
$$

其中：

- $Y$: set of zones。
- 每個 $y$ 是一個 warehouse zone，例如 aisle 或 section。
- zones 是 disjoint partition。

Manager 的 action 是：對每個 worker agent 指派一個 zone。

$$
A^m = Y^{|I|}
$$

若 manager 指派 zone $y^i$ 給 worker agent $i$，worker policy $\pi^i$ 只在 assigned zone 內選 target location：

$$
l^i_t \in y^i
$$

這使每個 worker 的 effective action space 變成：

$$
\max_{y \in Y} |y| \ll |L|
$$

因此 hierarchy 的核心作用是：

- 降低 action space complexity。
- 讓 agent 在較低 spatial resolution 上解決 task assignment。
- 透過 single manager policy 形成 high-level central coordination。

---

### 12. Manager reward

Manager reward $r^m_t$ 是所有被指派 goals 且 non-busy workers 在 timestep 中得到的 rewards 加總。

Paper 定義：

$$
r^m_t = \sum_{i \in I} r^i_{t:t+k_i}
$$

其中：

$$
r^i_{t:t+k_i} =
\begin{cases}
\sum^{t+k_i}_{\tau=t} r^i_{\tau}, & \text{if } i \text{ received a goal at } t \\
0, & \text{otherwise}
\end{cases}
$$

$k_i$ 表示 worker $i$ 到達 goal 前經過的 number of steps。

直觀來說：

- worker 接到 manager goal 後，在完成該 goal 期間得到的 rewards 會被加總。
- manager 的 reward 是這些 worker accumulated rewards 的總和。

---

### 13. MARL algorithms

Paper 比較三種 MARL data-sharing mechanisms，並提出其 hierarchical versions。

#### 13.1 Independent Actor-Critic (IAC)

- 每個 picker 與 AGV 都有自己的 independent networks。
- 優點: 可以學到 specialised behaviours。
- 缺點: 同類型 agents 之間沒有 shared experience。

Hierarchical version: HIAC

#### 13.2 Shared Network Actor-Critic (SNAC)

- pickers 與 AGVs 分別共享 networks。
- 目的是提高 training efficiency。
- 但在 GTP experiments 中，SNAC 表現較差，paper 說原因是所有 worker agents 使用 identical policies，可能造成 frequent collisions 或 deadlocks。

Hierarchical version: HSNAC

#### 13.3 Shared Experience Actor-Critic (SEAC)

- 每個 picker 與 AGV 仍有自己的 independent networks。
- 但同類型 agents 之間有 additional shared gradient update。

Hierarchical version: HSEAC

---

### 14. Training configuration 與 neural network architecture

Paper appendix 給出的 training configuration：

#### 14.1 Manager network

- manager policy 與 value network 是 multi-headed neural networks。
- 包含三層 fully-connected layers。
- 每層 128 neurons。
- activation: ReLU。

#### 14.2 Worker agent network

- 每個 agent 使用 value and critic network。
- represented by two fully connected layers。
- 每層 64 neurons。
- activation: ReLU。

#### 14.3 Hyperparameters

- learning rate: 0.0003
- network update frequency:
  - PTG: 100 steps
  - GTP: 250 steps
- Adam optimiser epsilon: 0.001
- GAE lambda parameter: 0.96
- discount factor: 0.99
- training episodes: 10,000 episodes

---

### 15. Environment 1: Dematic PTG Simulator

#### 15.1 Environment description

Dematic PTG simulator 是 high-performance PTG warehouse simulator，可以表示 real-world warehouses 與 PTG picking operations。

在此 simulator 中：

- pickers 是 human workers。
- AGVs 負責運送 picked items。
- AGVs 會到多個 storage locations 依序接收 items。
- human pickers 將 items pick 到 AGVs。
- 當 AGV 收集完 order 中所有 items 後，AGV 將 order 送到 single delivery location。

#### 15.2 Order-picking dynamics

- Episodic task。
- 一個 episode 包含 $N$ orders。
- orders 隨機分佈在 $L_{item}$ locations。
- episode termination condition: all orders are completed。

#### 15.3 Warehouse configurations

| Configuration | Small | Medium | Large | Disjoint |
|---|---:|---:|---:|---:|
| Aisles | 2 | 10 | 22 | 12 + 12 |
| Item Locations $|L_{item}|$ | 200 | 400 | 1276 | 1392 |
| Partitions $|Y|$ | 4 | 10 | 22 | 24 |
| Pickers $|P|$ | 4 | 6 | 8 | 4 |
| AGVs $|V|$ | 8 | 12 | 16 | 16 |
| Avg. order-lines per order $\mathbb{E}(|z^v|)$ | 5 | 5 | 5 | 2 |
| Orders $|Z|$ | 80 | 80 | 80 | 80 |

Paper 註解：Disjoint warehouse 被分成兩個 sub-warehouses，例如 regular and frozen goods，中間由 passage 連接。

#### 15.4 PTG observation space

Manager observation：

$$
O^m = \{(l^i_c, l^i_t) \mid i \in I\} \oplus \{z^v \mid v \in V\}
$$

Picker observation：

$$
O^p = \{(l^i_c, l^i_t) \mid i \in I\} \oplus \{z^v \mid v \in V\}
$$

AGV observation：

$$
O^v = \{(l^i_c, l^i_t) \mid i \in I\} \oplus z^v
$$

其中：

- $\oplus$: concatenation operator。
- $l^i_c \in L$: agent $i$ 的 current location。
- $l^i_t \in L$: agent $i$ 的 target location。
- manager、pickers、AGVs 都觀察所有 agents 的 current locations 與 target locations。
- manager 與 pickers 觀察所有 AGVs 的 orders $z^v, v \in V$。
- AGV 只觀察自己的 order $z^v$。

#### 15.5 PTG reward function

PTG reward：

- Picker:
  - picking an item onto an AGV: $+0.1$
  - per timestep penalty: $-0.01$

- AGV:
  - receiving a picked item: $+0.1$
  - delivering the order: $+0.1$
  - per timestep penalty: $-0.01$

#### 15.6 PTG invalid action masking

PTG 中的 masking：

- AGV:
  - order-specific masking。
  - 不屬於 current order 的 item locations 會從 action space 移除。

- Picker:
  - picker 的 invalid action mask 讓 picker 在 AGVs 的 target item locations 之間選擇，以促進 coordination。
  - pickers 不能選擇 other pickers already in transit to 的 locations。

---

### 16. Environment 2: TA-RWARE (GTP)

#### 16.1 Environment description

TA-RWARE 是作者提出的 open-source simulator，是 RWARE 的 adaptation，用於 GTP paradigm。

TA-RWARE 的設計目的是建立 cooperative task 並研究 task assignment optimisation。

TA-RWARE 包含 heterogeneous agents：

- AGVs
- picker robots

Agents 的 action 是 select target locations。

Map traversal 由 predefined heuristic 處理。

在 TA-RWARE 中：

1. AGVs 走到 single warehouse location 取 storage medium。
2. picking robot 把 storage medium 轉移到 AGV。
3. AGV 把 storage medium 送到 delivery location，也就是 human pick station。
4. human picker 不被建模在 simulator 中，而是在 delivery locations 外部。

#### 16.2 Order-picking dynamics

- Agents 從 dynamic request queue 中選擇 storage medium 來 pick。
- request queue 有固定長度，依 warehouse layout 而定。
- 當某個 storage medium 被 delivered 後，新的 storage medium 會變成 requested。

#### 16.3 Warehouse configurations

| Configuration | Small | Large |
|---|---:|---:|
| Rack Rows | 2 | 4 |
| Rack Columns | 5 | 7 |
| Column Length | 8 | 8 |
| Column Width | 2 | 2 |
| Item Locations $|L_{item}|$ | 160 | 448 |
| Partitions $|Y|$ | 10 | 28 |
| Pickers $|P|$ | 4 | 7 |
| AGVs $|V|$ | 8 | 14 |
| Concurrent requested items | 20 | 60 |
| Delivery Locations $|L_{delivery}|$ | 10 | 14 |

#### 16.4 TA-RWARE observation space

Manager observation：

$$
O^m = \{(l^i_c, l^i_t) \mid i \in I\} \oplus \{(cr^v, re^v, ld^v) \mid v \in V\} \oplus \{(oc^l, re^l) \mid l \in L_{item}\}
$$

Picker observation：

$$
O^p = \{(l^i_c, l^i_t) \mid i \in I\} \oplus \{(cr^v, re^v, ld^v) \mid v \in V\}
$$

AGV observation：

$$
O^v = \{(l^i_c, l^i_t) \mid i \in I\} \oplus (cr^{own}, re^{own}, ld^{own}) \oplus \{(oc^l, re^l) \mid l \in L_{item}\}
$$

符號意義：

- $l^i_c$: agent $i$ 的 current location。
- $l^i_t$: agent $i$ 的 target location。
- $cr^v$: AGV $v$ 是否 carrying shelf。
- $re^v$: carried shelf 是否 requested。
- $ld^v$: AGV 是否 waiting for load/unload。
- $cr^{own}, re^{own}, ld^{own}$: AGV 自己的 status。
- $oc^l$: shelf location $l$ 是否 occupied by shelf。
- $re^l$: shelf location $l$ 的 shelf requested state。

Observation access：

- manager、pickers、AGVs 都觀察所有 agents 的 current locations 與 target locations。
- manager 與 pickers 觀察所有 AGV statuses。
- AGVs 只觀察 own status。
- manager 與 AGVs 觀察所有 shelf locations 的 occupied / requested state。

#### 16.5 TA-RWARE reward function

TA-RWARE reward：

- Picker:
  - loading/unloading a storage medium onto an AGV: $+0.1$
  - per timestep penalty: $-0.001$

- AGV:
  - delivering the storage medium: $+1$
  - per timestep penalty: $-0.001$

#### 16.6 TA-RWARE invalid action masking

TA-RWARE 中的 masking：

- AGV:
  - practical action space 被 reduce 到 shared pool of requested storage medium locations and delivery locations。

- Picker:
  - 採用與 PTG 類似的 masking。
  - picker 可以 travel to load/unload from current AGV target locations that are not already serviced。

---

### 17. Heuristic baselines

Paper 使用三個 human-engineered heuristic baselines。

#### 17.1 Follow Me (FM)

適用於 PTG。

- 多個 AGVs 被 assigned to each picker，形成 group。
- 每個 AGV 的 order 被 concatenate。
- 使用 travelling salesman problem (TSP) solution 決定 items picking order。
- TSP path 最小化 worker group 的 distance，限制是 orders 完成前 group 要 stay together。

特性：

- 優點: minimize idle time for pickers，因為 pickers 一直在 travelling 或 picking。
- 缺點: 可能造成 pickers travel more than needed。

#### 17.2 Pick, Don’t Move (PDM)

適用於 PTG。

- Pickers 被分配到 zones，例如一個 picker per aisle。
- AGVs 可以 travel through entire warehouse。
- AGVs 使用 TSP solution 走訪 current order 中所有 item locations。
- Pickers 在自己的 zones 中與 AGVs 會合，並把 items pick 到 AGV。
- Pickers 依照 AGV 與 picker 到 target locations 的 relative proximity 來 prioritize service。

特性：

- 優點: minimize travel distance for pickers。
- 缺點: 若 current orders 中很少 items 位於某 picker 的 zone，可能造成 picker under-utilisation。

#### 17.3 Closest Task Assignment (CTA)

適用於 GTP。

- AGVs 前往 single storage location。
- 將 storage medium 從該 location 送到 delivery locations。
- requested storage mediums 被 assigned 給 closest AGV。
- AGV 將 storage medium 送到 closest delivery location。
- delivery 後，AGV 把 storage medium 返回 closest empty shelf location。
- closest 定義為 A* algorithm 找到的 minimum distance path。
- Pickers 固定在 allocated zones，類似 PDM。
- Pickers 根據 FIFO queue prioritize AGVs，也就是哪個 AGV 先被 assigned pick 或 drop 到該 zone，就先服務。

特性：

- 優點: minimize travel distance for pickers。
- 缺點: 可能造成 picker under-utilisation。

---

### 18. Experiments

#### 18.1 Evaluation environments

Paper 在六種設定中評估：

- PTG: Small, Medium, Large, Disjoint
- GTP: Small, Large

#### 18.2 Compared methods

Heuristics：

- PTG: FM, PDM
- GTP: CTA

Non-hierarchical MARL baselines：

- IAC
- SNAC
- SEAC

Hierarchical MARL proposed methods：

- HIAC
- HSNAC
- HSEAC

#### 18.3 Evaluation metric

Primary performance measure：

- pick rate measured in order-lines per hour
- 表示每個 episode 中平均 picks 的頻率。

Table I 使用 final 50 training episodes 的 average pick rates，並報告 mean ± 95% CI。

---

### 19. Main experimental results

#### 19.1 Table I results

| Method | PTG Small | PTG Medium | PTG Large | PTG Disjoint | GTP Small | GTP Large |
|---|---:|---:|---:|---:|---:|---:|
| FM | 901.3 ± 1.9 | 1098.1 ± 3.8 | 1230.2 ± 5.1 | 568.4 ± 1.7 | - | - |
| PDM | 783.6 ± 2.8 | 982.2 ± 4.0 | 1123.9 ± 4.9 | 677.4 ± 2.1 | - | - |
| CTA | - | - | - | - | 52.7 ± 0.9 | 67.1 ± 0.8 |
| IAC | 1053.0 ± 2.8 | 1206.4 ± 4.2 | 1263.9 ± 5.8 | 733.2 ± 2.7 | 65.2 ± 0.5 | 80.4 ± 0.6 |
| SNAC | 990.9 ± 2.8 | 1142.7 ± 4.3 | 1235.0 ± 5.7 | 688.7 ± 2.7 | 60.8 ± 0.7 | 72.1 ± 0.9 |
| SEAC | 1019.7 ± 2.9 | 1185.1 ± 5.1 | 1262.9 ± 5.7 | 739.8 ± 2.4 | 64.8 ± 0.4 | 82.2 ± 0.5 |
| HIAC | 1025.9 ± 4.3 | 1232.1 ± 4.8 | 1354.2 ± 5.9 | 794.1 ± 2.7 | 66.7 ± 0.3 | 86.0 ± 0.5 |
| HSNAC | 1030.8 ± 3.8 | 1232.8 ± 5.1 | 1363.8 ± 6.0 | 796.9 ± 2.4 | 66.0 ± 0.7 | 85.0 ± 0.5 |
| HSEAC | 1028.2 ± 3.9 | 1242.1 ± 5.0 | 1370.9 ± 5.7 | 803.5 ± 2.6 | 64.6 ± 0.4 | 84.8 ± 0.6 |

#### 19.2 PTG findings

Paper 的觀察：

- 不同 warehouse configurations 可能 favor 不同 heuristics。
  - Large 中 FM 較好。
  - Disjoint 中 PDM 較好。

- hierarchical algorithms 在所有 PTG settings 中都比 FM 與 PDM 達到更高 pick rates。

- hierarchical versions 相較 non-hierarchical versions 有更好的 sample efficiency。
  - HIAC vs IAC
  - HSNAC vs SNAC
  - HSEAC vs SEAC

- warehouse complexity 增加時，hierarchical architecture 的優勢更明顯。

具體數字：

- Small PTG：IAC 與 hierarchical algorithms 接近，只差約 2.2%。Paper 認為原因是 task relatively low difficulty，因此 hierarchy 沒有帶來太大額外優勢。
- Medium PTG：HSEAC 比 FM 高 13.1%，比 PDM 高 26.5%。
- Large PTG：HSEAC 比 FM 高 11.4%，比 PDM 高 22.0%。
- Disjoint PTG：HSEAC 比 FM 高 41.3%，比 PDM 高 18.6%。

#### 19.3 GTP findings

Paper 的觀察：

- 所有 MARL methods 在 GTP Small 與 GTP Large 都超過 CTA heuristic。
- SNAC 在 GTP 中最差，paper 說原因是所有 worker agents 使用 identical policies，容易造成 frequent collisions 或 deadlocks。
- HSNAC 相較 SNAC 有改善，因為 manager 會 conditioning worker policies on assigned target goals，將 agents 分散到 warehouse 中，避免 deadlocks。
- hierarchical models 在 Large configuration 特別顯示 scaling benefit。

具體數字：

- GTP Small：HIAC 比 CTA 高 26.6%。
- GTP Large：HIAC 比 CTA 高 28.2%。

---

### 20. Figures 重點

#### 20.1 Figure 1

Figure 1 顯示兩個 simulator：

- Left: Dematic PTG simulator，包含 human pickers 與 AGVs。
- Right: TA-RWARE GTP simulator，包含 picking bots 與 AGVs。

圖中也標示了：

- delivery locations
- item locations
- AGV agent
- picker agent
- AGV agent carrying storage medium
- pick requested

#### 20.2 Figure 2

Figure 2 顯示 proposed 3-layer hierarchy：

- Manager 接收 manager observation $o^m$，輸出 manager action $y^1, \ldots, y^N$，也就是每個 worker 的 assigned zone。
- Worker $i$ 接收 worker observation $o^i$ 與 manager action $y^i$，輸出 worker action $a^i$。
- Low-level controller 根據 worker action 產生 path。

#### 20.3 Figure 3

Figure 3 顯示 PTG simulator 中各方法的 training curves。

- X-axis: Episodes
- Y-axis: Orderlines / hour
- Configurations: Small, Medium, Large, Disjoint
- Compared methods: FM, PDM, IAC, SNAC, SEAC, HIAC, HSNAC, HSEAC
- Shaded area: 95% stratified bootstrap confidence interval
- smoothing: 300 episode average smoothing

主要訊息：hierarchical methods 在多數設定中收斂更快、final pick rate 更高，尤其 warehouse 較大或較複雜時。

#### 20.4 Figure 4

Figure 4 顯示 TA-RWARE GTP simulator 中各方法的 training curves。

- X-axis: Episodes
- Y-axis: Orderlines / hour
- Configurations: Small, Large
- Compared methods: CTA, IAC, SNAC, SEAC, HIAC, HSNAC, HSEAC
- Shaded area: 95% stratified bootstrap confidence interval
- smoothing: 300 episode average smoothing

主要訊息：所有 MARL methods 都超過 CTA；hierarchical methods 尤其在 Large configuration 展現更好的 scaling 與 sample efficiency。

---

### 21. Paper 的核心結論

Paper 結論如下：

1. MARL algorithms 可以為 PTG 與 GTP order-picking paradigms 學出 effective solutions。

2. Hierarchical MARL architecture 能改善 baseline MARL algorithms。
   - 原因是 hierarchical decomposition 降低 large action spaces 的難度。
   - 方法是在 lower spatial resolution 上處理 order-picking problem。

3. Manager 提供 high-level central coordination。
   - Manager 為所有 agents 選擇 goals。
   - 這使 worker agents 更容易分散到 warehouse 中，減少 coordination 問題。

4. Proposed MARL solutions outperform multiple engineered industry heuristics。
   - 在不同 warehouse configurations 下皆有效。
   - 在 PTG 與 GTP paradigms 中皆有效。

---

### 22. Paper 提到的 future work

Paper 提到幾個 future work directions：

1. 加入其他 optimisation objectives 到 objective function。
   - travel distance
   - energy usage
   - maintenance costs
   - operational costs
   - human employee welfare

2. 為了讓 methods 更能在 agent 數量與 warehouse size 上 scale，可發展：
   - unsupervised environment design
   - sub-task decomposition

3. 關於 invalid action masking，paper 也提到目前 masking 可能限制 policy expressiveness，complete exception from large action space training regime 留給 future work。

---

### 23. 與 task assignment / MAPF 的關係

Paper 在 related work 中區分 order-picking 與 MAPF / MAPD。

#### 23.1 與 MAPF 的區別

MAPF 的設定通常是：

- agents 已經有 individually allocated targets。
- 目標是讓 agents reach targets while avoiding collisions。

Paper 的 setting 不同：

- task assignment 是主要問題。
- path-finding 由 predefined methods 處理。
- 因此本文將 task assignment 與 path-finding 區分開來，並視為 complementary。

#### 23.2 與 MAPD 的區別

MAPD problems 通常是 agents 被 sequentially assigned tasks，包括 pickup and delivery locations。

Paper 認為 MAPD approaches 的限制是：

- 依賴 hand-engineered heuristics。
- 通常假設 homogeneous agent architectures。
- agent diversity 通常只到不同 velocities。
- cooperation 主要被簡化成 collision avoidance，並由 MAPF module 處理。

本文 setting 更複雜，因為：

- pickers 與 AGVs 是 heterogeneous workers。
- 不同 worker types 的 task assignment 具有 interdependencies。
- pickers 與 AGVs 需要同步在某些 item locations 會合。
- 同類型 agents 之間也需要 coordination，以避免 cramming at same item location。

---

### 24. 這篇 paper 的方法摘要

整體方法可以用以下流程理解：

1. Warehouse 被定義為 locations、order distribution、workers。

2. Workers 包含 AGVs 與 pickers。

3. 將 order-picking problem 建模成 POSG。

4. Worker action 不是 primitive movement，而是 target location selection。

5. Low-level navigation 由 A* algorithm 負責。

6. 由於 action space 很大，paper 使用：
   - invalid action masking
   - hierarchical manager-worker architecture

7. Manager 為每個 worker 指派 zone。

8. Worker 在 assigned zone 內選 target location。

9. Manager 與 workers 透過 MARL jointly trained。

10. 方法套用到 IAC、SNAC、SEAC，形成 HIAC、HSNAC、HSEAC。

11. 在 PTG 與 GTP simulator 中，hierarchical methods 整體超過 non-hierarchical MARL baselines 與 industry heuristics。

---

### 25. 可以直接記住的重點

- Paper 不是讓 MARL 學 primitive path planning，而是讓 MARL 學 task assignment / target location selection。
- Low-level controller 使用 A* algorithm。
- Hierarchy 的 manager 不改變 worker reward，而是用 assigned zones partition worker action space。
- Worker action space 原本很大：picker 是 $L_{item}$，AGV 是 $L$。
- Hierarchical decomposition 後，worker 只需要在 assigned zone 中選 target location。
- Reward 是 individual reward functions，不是只用一個 shared team reward。
- Manager reward 是被分配 goal 的 non-busy workers 在完成 goal 期間 rewards 的加總。
- PTG 中 human pickers 被建模為 pickers；TA-RWARE GTP 中 picker 指的是 picking robot，不是 simulator 外部的人類 operator。
- TA-RWARE 是 RWARE 的 GTP adaptation，重點是 heterogeneous agents 與 task assignment optimisation。
- Hierarchical methods 的優勢在 warehouse 複雜度增加時更明顯。
- SNAC 在 GTP 中表現較差的原因是 identical policies 容易導致 frequent collisions 或 deadlocks。
- Paper 的 primary metric 是 pick rate，也就是 completed order-lines per hour。

