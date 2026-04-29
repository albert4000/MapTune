# Batched Agent Scripts: Usage, stdout, and generated files

這份文件專門整理以下四支腳本的實際執行方式、stdout 內容、以及執行期間會產生/覆寫的檔案：

- `batched_MAB_EP.py`
- `batched_MAB_UCB.py`
- `batched_DQN.py`
- `batched_DDQN.py`

## Scope

本文內容是直接根據腳本原始碼整理，不是泛泛而談。

四支腳本都假設你在 `MapTune/` 目錄下執行，並且：

- `abc` 已經在 `PATH` 裡
- 輸入的 `<genlib>` 對應的 `.lib` 檔案存在
- `temp_blifs/` 目錄存在
- `gen_newlibs/` 目錄存在

## Shared CLI contract

這四支腳本都沒有 `argparse`，也沒有 `--help` / `-h`。

它們都是直接吃 3 個 positional arguments：

```bash
python <script>.py <num_sampled_gate> <design> <genlib>
```

參數意義：

- `<num_sampled_gate>`: 要從原始 `.genlib` 中抽出的 gate 數量，不包含程式固定保留的 `BUF` / `INV` 類 gate
- `<design>`: 電路設計檔，通常是 `benchmarks/*.bench`
- `<genlib>`: 原始 genlib 檔案，例如 `7nm.genlib`

實際在程式中的解析方式都是：

```python
sample_gate = int(sys.argv[-3])
design = sys.argv[-2]
genlib_origin = sys.argv[-1]
```

所以：

- 必須剛好把最後三個 argument 依序放成 `sample_gate design genlib`
- 沒有 named arguments
- 少參數會直接噴 `IndexError`
- `sample_gate` 不是整數會噴 `ValueError`

## Quick commands

```bash
cd /home/r14_shih/data1_4TB/LibSel/MapTune
```

### `batched_MAB_EP.py`

```bash
python batched_MAB_EP.py 65 benchmarks/s838a.bench 7nm.genlib
```

### `batched_MAB_UCB.py`

```bash
python batched_MAB_UCB.py 65 benchmarks/s838a.bench 7nm.genlib
```

### `batched_DQN.py`

```bash
python batched_DQN.py 65 benchmarks/s838a.bench 7nm.genlib
```

### `batched_DDQN.py`

```bash
python batched_DDQN.py 65 benchmarks/s838a.bench 7nm.genlib
```

## Shared pre-run behavior

四支腳本一開始都會先跑一次 baseline mapping：

1. 用原始 `<genlib>` 映射 `<design>`
2. 產生對應的暫存 `.blif`
3. 再讀回對應的 `.lib` 和剛產生的 `.blif`
4. 用 `ps; topo; upsize; dnsize; stime;` 算 baseline `Delay` / `Area`

也就是說，**還沒開始 RL/MAB 搜尋之前，就已經會先寫一個 temp blif 檔**。

## Generated files by script

### `batched_MAB_EP.py`

命令：

```bash
python batched_MAB_EP.py <num_sampled_gate> <design> <genlib>
```

固定參數：

- `batch_size = 10`
- `num_iterations = 100`
- `epsilon = 0.2`

執行時會產生/覆寫的檔案：

1. baseline / 每次 mapping 都會覆寫：
   - `temp_blifs/<design_without_.bench>_bs_ep_temp.blif`
2. 每次抽樣都會覆寫同一個 sampled genlib：
   - `gen_newlibs/<design>_<N>_bs_ep_samplelib.genlib`

其中：

- `<design_without_.bench>` 例如 `s838a`
- `<design>` 是原字串，例如 `benchmarks/s838a.bench`
- `<N>` 是 `sample_gate + 保留的 BUF/INV gate 數量`

注意：

- 這個檔名直接把 `design` 字串拼進路徑，所以如果 `design=benchmarks/s838a.bench`，實際輸出會長成：
  - `gen_newlibs/benchmarks/s838a.bench_<N>_bs_ep_samplelib.genlib`
- 代表 `gen_newlibs/benchmarks/` 這層路徑必須存在，否則 `open(..., 'w')` 會失敗
- 同一輪執行中，sampled genlib 和 temp blif 都是**反覆覆寫**，不是每次 iteration 留一份新檔

stdout 內容：

```text
Baseline Delay: <baseline_delay>
Baseline Area: <baseline_area>
Batch iteration: 0
Current best reward:  <best_reward_after_iteration_0>
Current best result:  (<best_delay_after_iteration_0>, <best_area_after_iteration_0>)
Batch iteration: 1
Current best reward:  <best_reward_after_iteration_1>
Current best result:  (<best_delay_after_iteration_1>, <best_area_after_iteration_1>)
...
Batch iteration: 99
Current best reward:  <best_reward_after_iteration_99>
Current best result:  (<best_delay_after_iteration_99>, <best_area_after_iteration_99>)
Best Delay: <final_best_delay>
Best Area: <final_best_area>
Best Reward: <final_best_reward>
Total time: <runtime_seconds>
```

stdout 條列：

- 一開始固定印：
  - `Baseline Delay: ...`
  - `Baseline Area: ...`
- 每個 iteration 印 3 行，共 `100 * 3 = 300` 行：
  - `Batch iteration: i`
  - `Current best reward: ...`
  - `Current best result: (...)`
- 最後固定印 4 行：
  - `Best Delay: ...`
  - `Best Area: ...`
  - `Best Reward: ...`
  - `Total time: ...`

## `batched_MAB_UCB.py`

命令：

```bash
python batched_MAB_UCB.py <num_sampled_gate> <design> <genlib>
```

固定參數：

- `batch_size = 10`
- `num_iterations = 100`
- `c = 2`

執行時會產生/覆寫的檔案：

1. baseline / 每次 mapping 都會覆寫：
   - `temp_blifs/<design_without_.bench>_bs_ucb_temp.blif`
2. 每次抽樣都會覆寫同一個 sampled genlib：
   - `gen_newlibs/<design>_<N>_bs_ucb_samplelib.genlib`

同樣注意：

- 如果 `design=benchmarks/s838a.bench`，實際檔案路徑會是：
  - `gen_newlibs/benchmarks/s838a.bench_<N>_bs_ucb_samplelib.genlib`
- 需要 `gen_newlibs/benchmarks/` 已存在
- 檔案是覆寫，不會保留每一步歷史版本

stdout 內容：

```text
Baseline Delay: <baseline_delay>
Baseline Area: <baseline_area>
Batch iteration: 0
Current best reward:  <best_reward_after_iteration_0>
Current best result:  (<best_delay_after_iteration_0>, <best_area_after_iteration_0>)
...
Batch iteration: 99
Current best reward:  <best_reward_after_iteration_99>
Current best result:  (<best_delay_after_iteration_99>, <best_area_after_iteration_99>)
Best Delay: <final_best_delay>
Best Area: <final_best_area>
Best Reward: <final_best_reward>
Total time: <runtime_seconds>
```

stdout 條列：

- baseline 2 行
- 每個 iteration 3 行，總共 300 行
- 收尾 4 行

## `batched_DQN.py`

命令：

```bash
python batched_DQN.py <num_sampled_gate> <design> <genlib>
```

固定參數：

- `num_episodes = 5000`
- `batch_size = 10`
- `buffer_size = 10000`
- `epsilon = 0.2`（寫死在 `select_action()` 預設值）
- network:
  - `Linear(state_size, 64)`
  - `Linear(64, 128)`
  - `Linear(128, action_size)`

執行時會產生/覆寫的檔案：

1. baseline / 每次 mapping 都會覆寫：
   - `temp_blifs/<design_without_.bench>_dqn_temp.blif`
2. 每次 episode 完成一個有效選擇後，會覆寫 sampled genlib：
   - `gen_newlibs/<design>_<N>_dqn_samplelib.genlib`

同樣注意：

- `design` 直接拼進檔名，所以可能需要先建立 `gen_newlibs/benchmarks/`
- sampled genlib 與 temp blif 都只保留最後一次覆寫結果
- 腳本**不會**把 model checkpoint 存檔
- 腳本**不會**把 replay buffer 存檔
- 腳本**不會**輸出 best gate list 到檔案

stdout 內容：

```text
Baseline Delay:  <baseline_delay>
Baseline Area:  <baseline_area>
Current Best Result:  (<delay_when_new_best_found>, <area_when_new_best_found>)
Episode 1, Highest Reward = <highest_reward_after_episode_1>
Episode 2, Highest Reward = <highest_reward_after_episode_2>
Episode 3, Highest Reward = <highest_reward_after_episode_3>
...
Episode 5000, Highest Reward = <highest_reward_after_episode_5000>
Total time:  <runtime_seconds>
```

stdout 條列：

- 一開始印 baseline 2 行
- 每個 episode 結束印 1 行：
  - `Episode <k>, Highest Reward = ...`
- 當某個 episode 產生新的全域最佳 reward 時，額外插入：
  - `Current Best Result:  (<delay>, <area>)`
- 最後印：
  - `Total time:  ...`

補充：

- `Current Best Result: ...` 出現次數不固定，取決於訓練過程中 best reward 被刷新幾次
- 最終 stdout **沒有**像 MAB 那樣額外印 `Best Delay` / `Best Area` / `Best Reward`

## `batched_DDQN.py`

命令：

```bash
python batched_DDQN.py <num_sampled_gate> <design> <genlib>
```

固定參數：

- `num_episodes = 5000`
- `batch_size = 10`
- `buffer_size = 10000`
- `epsilon = 0.2`（寫死在 `select_action()` 預設值）
- `tau = 0.01`（target network soft update）
- network:
  - `Linear(state_size, 64)`
  - `Linear(64, 128)`
  - `Linear(128, action_size)`

執行時會產生/覆寫的檔案：

1. baseline / 每次 mapping 都會覆寫：
   - `temp_blifs/<design_without_.bench>_ddqn_temp.blif`
2. 每次 episode 完成一個有效選擇後，會覆寫 sampled genlib：
   - `gen_newlibs/<design>_<N>_ddqn_samplelib.genlib`

另外這支和 `batched_DQN.py` 一樣：

- 不會寫 checkpoint
- 不會寫 best result summary file
- 不會寫 reward log csv

stdout 內容：

```text
Baseline Delay:  <baseline_delay>
Baseline Area:  <baseline_area>
Current Best Result:  (<delay_when_new_best_found>, <area_when_new_best_found>)
Episode 1, Highest Reward = <highest_reward_after_episode_1>
Episode 2, Highest Reward = <highest_reward_after_episode_2>
...
Episode 5000, Highest Reward = <highest_reward_after_episode_5000>
Total time:  <runtime_seconds>
```

stdout 條列：

- baseline 2 行
- 每個 episode 1 行
- best reward 被刷新時額外印 `Current Best Result: ...`
- 最後印 `Total time: ...`

## Exact arguments and examples

### Minimal positional form

```bash
python batched_MAB_EP.py 65 benchmarks/s838a.bench 7nm.genlib
python batched_MAB_UCB.py 65 benchmarks/s838a.bench 7nm.genlib
python batched_DQN.py 65 benchmarks/s838a.bench 7nm.genlib
python batched_DDQN.py 65 benchmarks/s838a.bench 7nm.genlib
```

### Recommended full form

如果你的環境不是預設 `python` 指到正確版本，建議明確用：

```bash
python3 batched_MAB_EP.py 65 benchmarks/s838a.bench 7nm.genlib
python3 batched_MAB_UCB.py 65 benchmarks/s838a.bench 7nm.genlib
python3 batched_DQN.py 65 benchmarks/s838a.bench 7nm.genlib
python3 batched_DDQN.py 65 benchmarks/s838a.bench 7nm.genlib
```

## Hidden assumptions and failure points

這四支腳本的使用上有幾個實際上很重要的點：

1. `genlib_origin[:-7] + '.lib'`
   - 代表它假設輸入檔名真的以 `.genlib` 結尾
   - 例如 `7nm.genlib -> 7nm.lib`

2. `design[:-5]`
   - 代表它假設 design 檔名真的以 `.bench` 結尾
   - 例如 `benchmarks/s838a.bench -> benchmarks/s838a`
   - 所以 temp blif 會是 `temp_blifs/benchmarks/s838a_*.blif`
   - 這也代表 `temp_blifs/benchmarks/` 必須存在

3. 這些腳本都會呼叫 `abc`
   - 如果 `abc` 不在 `PATH`，會直接失敗

4. `batched_DQN.py` / `batched_DDQN.py` 需要額外 Python 套件：
   - `gymnasium`
   - `numpy`
   - `torch`

5. `batched_MAB_EP.py` / `batched_MAB_UCB.py` 雖然有 `import os` / `PIPE`，但程式內其實沒用到

## Per-script summary table

| Script | Search style | Fixed outer loop | Key stdout lines | Generated sampled genlib | Generated temp blif |
|---|---|---:|---|---|---|
| `batched_MAB_EP.py` | Epsilon-greedy MAB | `100` iterations, batch size `10` | `Batch iteration`, `Current best reward`, `Current best result`, final best summary | `gen_newlibs/<design>_<N>_bs_ep_samplelib.genlib` | `temp_blifs/<design_without_.bench>_bs_ep_temp.blif` |
| `batched_MAB_UCB.py` | UCB MAB | `100` iterations, batch size `10` | `Batch iteration`, `Current best reward`, `Current best result`, final best summary | `gen_newlibs/<design>_<N>_bs_ucb_samplelib.genlib` | `temp_blifs/<design_without_.bench>_bs_ucb_temp.blif` |
| `batched_DQN.py` | DQN | `5000` episodes | `Current Best Result`, `Episode k, Highest Reward`, `Total time` | `gen_newlibs/<design>_<N>_dqn_samplelib.genlib` | `temp_blifs/<design_without_.bench>_dqn_temp.blif` |
| `batched_DDQN.py` | DDQN | `5000` episodes | `Current Best Result`, `Episode k, Highest Reward`, `Total time` | `gen_newlibs/<design>_<N>_ddqn_samplelib.genlib` | `temp_blifs/<design_without_.bench>_ddqn_temp.blif` |

## What these scripts do not provide

如果你是想拿完整實驗紀錄，這四支腳本目前**沒有**直接提供下面這些東西：

- 沒有 `--help`
- 沒有 named arguments
- 沒有自訂 output directory
- 沒有 log file
- 沒有 CSV / JSON metric dump
- 沒有 checkpoint save/load
- 沒有最後最佳 gate index list 的文字檔輸出
- 沒有 random seed argument

如果你之後要把這四支腳本變成真正可重跑、可收集結果的實驗介面，最值得先補的是：

1. `argparse`
2. `--output-dir`
3. `--episodes` / `--iterations` / `--batch-size`
4. `--seed`
5. best-result / reward-history 存檔
