Plan: Constraint-Based Logic Gate Selection with Null Option
1. 任務目標 (Objective)
修改 GateSelectionEnv，將元件庫（Cell Library）依據邏輯功能分組，並讓 Agent 在每個分組中選擇一個元件或選擇「不選取（None）」。此舉旨在優化搜尋空間，同時允許 Agent 透過刪除不必要的邏輯類型來優化面積與延遲。

2. 執行步驟 (Execution Steps)
Phase 1: 邏輯分組與「空選項」初始化 (Grouping & Null Option)
功能分類：

解析 .genlib 檔案，提取每一行 GATE 的 Boolean function（Y=... 部分）。

將 Gate 依據正規化後的邏輯表達式歸類至 self.gate_groups（Dictionary）。

過濾與保留：

排除： 移除所有 DFF、DHL 等時序元件，僅保留組合邏輯（Combinational Logic）。

強制保留： 將 INV、BUF 與 CONST 類型的元件存入 self.f_keep。這些元件不參與 Action Space 的選擇，而是預設加入每一回合的映射中。

加入 Null 選項：

在邏輯上，為每一個 gate_groups 中的類別增加一個索引代表 None。例如：若某類型有 N 個元件，則該類型的選擇肢增加到 N+1 個，最後一個索引表示該功能類型不選用任何元件。

Phase 2: Action Space 與 狀態表示 (Action Space & State)
多維 Action Space：

使用 gym.spaces.MultiDiscrete 定義 Action Space。

維度定義為：[len(group) + 1 for group in self.gate_groups.values()]。

每個維度的索引 i∈[0,N−1] 對應具體的 Gate，i=N 對應「不選取」。

Observation Space：

狀態向量應反映每個類型目前的選取索引或狀態，提供給 DQN 作為決策參考。

Phase 3: Step 邏輯與元件庫生成 (Step Logic & Genlib Generation)
執行 Action：

Agent 在每一回合中一次性或循序對所有類型做出決定。

在 technology_mapper 階段，遍歷所有類型的 Action：

若 Action 索引指向具體 Gate，則將該行加入待映射的清單。

若 Action 索引為 N（None），則該功能類型在本次映射中不包含任何 Gate。

組合生成：

將 f_keep（固定保留的元件）與 Agent 選中的元件合併，生成暫時性的 .genlib 檔案。

Phase 4: 終止條件與獎勵機制 (Termination & Reward)
終止狀態 (Terminal State)：

當所有邏輯類型的選擇（包含選取具體 Gate 或選擇 None）皆完成時，該 Episode 結束。

獎勵計算：

調用 ABC 進行 Mapping。

若因選取太少元件（如缺少基礎邏輯）導致 Mapping 失敗（Delay/Area 為 inf），給予極大的負獎勵。

若成功 Mapping，則依據 delay 與 area 的縮減量計算回饋。

3. 實作規範 (Implementation Guidelines)
Library Parsing: 使用 re（正規表達式）精確抓取 GATE 行的內容，並確保 f_keep 包含所有必需的驅動元件（INV/BUF）。

Action Mapping: 建立一個查找表（Lookup Table），將 MultiDiscrete 的索引精確對應回 self.gate_groups 中的 GATE 字串。

Code Quality: 程式碼中使用英文註解，邏輯結構需清晰，特別是在處理 Y=... 正規化（Normalization）的部分。

4. 驗證標準 (Verification)
功能性測試： 隨機選擇 Action（包含多個 None 索引），檢查生成的 .genlib 是否如預期般缺少特定類型。

ABC 整合測試： 確保即使某些功能類型被設為 None，只要基礎邏輯完備（如含有 NAND 或其組合），ABC 仍能完成技術映射（Technology Mapping）。

模型收斂性： 觀察 Agent 是否學會透過選擇「None」來剔除冗餘、效率低下的 Cell。

Agent 指令 (Instructions for AI)
修改 GateSelectionEnv.__init__ 以建立 self.gate_groups 並計算各組大小 +1。

在 step 函數中，解析 Action 向量，過濾掉索引為 N 的選擇，僅將索引 <N 的元件寫入暫存的 .genlib。

確保 DFF 與 DHL 在解析階段就被完整排除。