根據本次對話中實際做過的 implementation / code 修改，把它們追加到 `implement-diary.md`。

規則：
- 分類：[Implement-Diary]
- 檔案路徑：`/mnt/sda/home/r147250250916/research/MARL/TA-RWARE/implement-diary.md`
- **本檔僅追加新內容，絕不覆蓋既有條目**
- 對話中只是「討論」「規劃」「假設修改」但**沒有真的動 code / 寫檔**的事項不寫進來；只有實際的 code change / 檔案新增刪除才入帳
- 每筆 entry 必須對應到 [Implementation Plan](../../Research-Note/wiki/experiments/implementation-plan.md) 的某個 Phase + indicator；如果完全不對應（純 infra / 工具），標 `Infra`
- 沿用既有 entry 的編號規則：當天時間軸上的第 N 個改動就標 `[N]`（接續最後一筆編號繼續往下）

---

## 文件結構

```
# TA-RWARE Implement Diary

> 引言（保留既有，不動）

---

## YYYY-MM-DD

當天時段摘要（如有跨時段工作可分段；單一時段可省略此段）

### [N] <檔案名 or 主題> — <一句話標題>

...（見下方 entry 格式）

---

### [N+1] ...

---

## Open risks（未修，列為操作守則）

| 風險 | 決策 | 操作守則 |
| ... | ... | ... |

---

## 修改的檔案總覽

| 檔案 | 類型 | 對應 Plan | 項目 |
| ... | ... | ... | ... |

---

## 標準操作流程

（如有變動或新增則更新）
```

---

## Entry 格式（每筆一個 `[N]`）

```
### [N] <檔案路徑 or 主題> — <一句話標題>

**對應 Plan**: Phase X-Y <stage name> — <indicator 名稱>  ← 或 `Infra`
**做了什麼**: <一句話總結這次動作，動詞開頭>

**為什麼**:
<為何要動。如果是 bug，描述原問題；如果是新功能，描述空缺。可帶反例或最小 reproduce>

**改了什麼**:
- <條列實際 code-level 變更，能精確到 function/line 最好>
- <若多檔案就分檔列>

**結果 / 驗證**:
- <Sanity check 跑過什麼、輸出什麼數字>
- <若有影響範圍（破壞既有資料等）也列在這>
```

可選欄位：
- **CLI / 介面**：新增 script 才寫
- **Validation 規則**：新增 tool 有檢查邏輯時才寫
- **影響**：跨版本資料對照、其他 module 需要同步更新時才寫

---

## 追加流程

執行命令時：

1. **讀取 `implement-diary.md` 末尾**，找出今天日期段是否存在、最後一筆 `[N]`
2. **若今天日期段不存在**：在最後一筆日期段後加分隔線 + `## YYYY-MM-DD` 起始；第一筆是 `[1]`
3. **若今天日期段已存在**：接著最後一筆編號續寫 `[N+1]`、`[N+2]`...
4. **盤點本次對話實際動過的 code / 檔案**：每一筆獨立的修改寫一個 entry。同一檔案的多次連續微調合併為一個 entry；同一檔案在會話中相隔很遠的兩次大改視為兩個 entry
5. **更新「修改的檔案總覽」表格**：新增 row 對應新 entry；同檔多次修改在同一 row 內以 `;` 分隔項目編號
6. **如有新發現但未修的風險**：補進「Open risks」表格，明確標 Decision 與操作守則
7. **如標準操作流程有變動**（新 script、CLI 改了）：對應更新 SOP 區塊

---

## 撰寫風格

- **動詞開頭、終句省略主詞**：「`build_episode_metrics` 加 raw list」，不是「我加了 raw list 到 ...」
- **精確引用**：檔案用 markdown link `[warehouse.py:901](TA-RWARE/tarware/warehouse.py#L901)`；函式名 / 行號要對得上
- **數字證據**：sanity check 要有具體數字（rows / pick_rate / ...），不是「跑過了 OK」
- **連結 Plan**：Phase 名稱用 plan 裡的原文，避免自造別名
- **不要重複既有 entry 的內容**：如果新 entry 是接續修同個檔案，引用前一筆編號（`[2]`）即可，不要把背景重貼一次
- **無關脈絡不寫**：對話中的 review / 討論 / 「我們考慮過 X 但沒做」這類不入帳；只有實際動到的事才寫
