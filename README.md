# 6人桌 100bb GTO 解算（CFR，翻前 + 翻牌）

## 設定
| 項目 | 設定 |
|---|---|
| 人數 / 籌碼 | 6-max，所有人 100bb，盲注 0.5/1，無前注 |
| 演算法 | Discounted CFR (α=1.5, β=0, γ=2)，翻前 169 類手牌、翻牌 1326 組合向量化 |
| 精度 | 每個翻牌解算到剝削度 (exploitability) ≤ **10% pot** 為止 |
| 翻牌下注尺度 | 33%、50%、75%、100% pot、ALLIN；加注也用同一組比例（加注到 = 跟注 + x% × 跟注後底池），每條街最多 3 次全尺度加注，之後只剩 ALLIN |
| 翻牌範圍 | 49 個代表性牌面（1755 種同構翻牌做加權聚類，權重見 `results/flop_subset.json`） |
| 終點 | 翻牌行動結束（過牌-過牌或跟注）即以 turn+river 全部發完的勝率分底池；不模擬 turn/river 的行動 |

## 翻前樹（為了讓翻後都是單挑底池而做的簡化）
- 不 limp；開池 2.5bb（SB 3bb）
- 面對開池：棄牌 / 3bet；只有 BB 可以平跟
- 3bet：IP 3x（7.5bb）、盲注位 OOP 3bet 到 11bb、BB 對 SB 3bet 到 9bb
- 開池者面對 3bet：棄 / 跟 / 4bet（OOP 2.5x、IP 2.2x）/ ALLIN；其他人一律棄牌（不含冷 4bet、squeeze、冷跟）
- 3bettor 面對 4bet：棄 / 跟 / ALLIN

產生 35 個翻牌場景：5 個單加注底池 (SRP_x_vs_BB)、15 個 3bet 底池 (3BP_開池者_vs_3bettor)、15 個 4bet 底池 (4BP_...)。

## 翻前與翻牌的連結（多輪迭代）
1. 第 1 輪：翻前終點用「全下勝率 × 底池」估值 → 解翻前 → 用得到的範圍解 49 個翻牌。
2. 由翻牌解的每手牌 EV 計算各場景、各位置、每類手牌的**權益實現率** R = EV / (底池 × 勝率)。
3. 第 2、3 輪：翻前終點改用 E·R₁ / (E·R₁ + (1−E)·R₂) 分配底池（保持零和）→ 重解翻前 → 重解翻牌。

## 輸出檔案（`results/`）
| 路徑 | 內容 |
|---|---|
| `preflop/preflop_charts.txt` | 每個翻前決策點的 13×13 頻率表（%） |
| `preflop/preflop_strategy.json` | 翻前完整策略（每節點、每類手牌、每動作頻率 + 到達機率） |
| `flop/<場景>/<翻牌>.json` | 該翻牌所有決策節點：路徑、行動者、底池、整體動作頻率 |
| `flop/<場景>/<翻牌>.npz` | 每個節點 1326 組合的策略 `s{i}`、範圍權重 `r{i}`、每手 EV `ev_oop/ev_ip` |
| `summary/spots.csv` | 每個場景的到達頻率、範圍組合數、平均 EV、最大剝削度、翻前路線 |
| `summary/flop_root_strategies.csv` | 每個場景×翻牌，前兩層節點的動作頻率 |
| `summary/realization.json` | 權益實現率 |
| `flop_subset.json` | 49 個代表牌面、權重、各自代表的同構牌面 |
| `pass1/`, `pass2/` | 前幾輪迭代的翻前結果與摘要（供比較） |
| `solve_log.txt` | 解算紀錄 |

動作代號：`X` 過牌、`B33/B50/B75/B100` 下注 % pot、`R33…R100` 加注、`A` ALLIN、`C` 跟注、`F` 棄牌。

## 查詢
```
python src/query.py                                   # 列出場景
python src/query.py SRP_BTN_vs_BB                     # 各翻牌的 OOP 第一動作頻率
python src/query.py SRP_BTN_vs_BB KcTc6c              # 所有節點的整體頻率
python src/query.py SRP_BTN_vs_BB KcTc6c "X B50"      # BB 過牌、BTN 下注 50% 後，BB 每手牌的策略
```

## 圖形介面（互動檢視 + 匯出圖片）
```
python src/gui.py        # 或雙擊 GUI.bat，會自動開啟 http://127.0.0.1:8765
```
只需要 numpy，不需額外套件。分頁：
- **翻前範圍**：80 個翻前決策點的 13×13 圖（顏色＝動作頻率、高度＝範圍權重），點動作按鈕沿著樹往下走，走到跟注會連到對應的翻牌場景；點格子可比較三輪迭代。
- **翻牌策略**：選場景＋翻牌，逐節點瀏覽；13×13 依範圍加權彙整，右側列出每個組合的策略與 EV；可切換 EV 熱圖。
- **牌面總覽**：某場景某節點在 49 個牌面上的頻率長條圖（含加權平均），可依權重/高牌/下注率/材質/剝削度排序。
- **場景總表**：spots.csv 可排序表格、到達頻率圖、各場景 OOP 第一動作與 IP 面對過牌的平均策略。
- **權益實現率**：realization.json 的 13×13 熱圖。

右上「迭代輪次」可切換 pass1 / pass2 / 最終結果。每個圖都有「匯出 PNG」，「匯出全部圖片」會存到 `results/images/`：
| 路徑 | 內容 |
|---|---|
| `images/preflop/NN_<行動者>_<路線>.png` | 每個翻前節點的範圍圖 |
| `images/flop/<場景>/<翻牌>_root.png`、`_X.png` | 每個翻牌：OOP 第一動作、IP 面對過牌 |
| `images/boards/<場景>_root.png`、`_X.png` | 49 牌面頻率總覽 |
| `images/overview/spots_*.png` | 場景到達頻率、各場景平均策略 |
| `images/realization/<場景>_<位置>.png` | 權益實現率熱圖 |

## 重新執行
```
python src/run_all.py --flops 49 --target 10 --passes 3 --workers 6
```
`--target` 是翻牌剝削度上限（% pot），改小（例如 1）會更精確但更慢；`--flops` 可改 25 / 49 / 184 等。
`--pool` 預設 `thread`（多執行緒，numpy 矩陣運算時會釋放 GIL）；`--pool process` 改用多進程，但在部分 Windows 環境子進程無法複製管道句柄（`PermissionError: [WinError 5]`），Pool 會不斷重開 worker 而卡死。
參考時間：4 個翻牌 × 2 輪約 10 分鐘（6 workers）；49 個翻牌每輪約 40–50 分鐘。
需求：Python 3 + numpy + ziglang（`pip install numpy ziglang`，用來編譯 `src/eval.c`）。

## 限制（請注意）
- 翻牌結束後直接用勝率結算，不含 turn/river 的下注，因此翻牌策略偏向「把勝率帶到攤牌」的打法，與完整三條街的解算器結果會有差異。
- 翻前樹移除了 limp、冷跟、squeeze、冷 4bet，所以沒有多人底池。
- 剝削度 10% pot 屬於很粗的精度，相近尺度（如 33% vs 50%）之間的分配可能不穩定，看整體傾向較可靠。
- 49 個牌面是代表性子集，其他牌面請看 `flop_subset.json` 對應到哪個代表牌面。
