# 🔬 LLM 參數調優框架 — 中醫舌診（TCM 舌診）

自動化搜尋最佳推理參數（Prompt + Temperature + Top‑P），專為中醫舌診場景設計。本專案為參數調優工具，通常搭配主應用系統使用（例如：https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis）。本框架產出之最佳參數可匯回主應用的配置或 prompt 資料夾。

## 🎯 核心功能
- 網格搜尋：測試多組 Prompt × Temperature × Top‑P 組合
- 穩定性評估：重複推論並以 Jaccard 相似度量一致性
- 混合評分：規則檢查（術語覆蓋 / 格式）與 LLM 評分（語言適切性 / 因果邏輯）
- 實驗追蹤：自動輸出 CSV 報告與原始輸出

## 📁 資料與目錄位置（注意）
默認情況下，範例資產與歷史實驗資料位於 `archive/`：

- 範例圖片與 prompts：`archive/Docs_and_Assets/assets/`
- 歷史實驗原始輸出：`archive/Configs_and_Data/experiment_data/`

建議：若要在主分支直接運行流程，請考慮將必要的 `assets/` 與 `experiment_data/` 移至專案根目錄，或在 README 中說明實際路徑（如上）。

## 🚀 快速開始

### 1) 環境需求
- Python 3.11+

### 2) 安裝依賴（兩種常見選法）
- 使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

- 使用 `pyproject.toml`（可用 `pip` 安裝本套件或使用 Poetry）：

```bash
pip install .
# 或（開發模式）
pip install -e .
```

> 若要使用 `uv`（task runner），請先安裝 `uv`：

```bash
pip install uv
# 範例：執行同步工作流
uv run tuning_workflow_sync.py
```

### 3) 設定 API Key

```bash
cp .env.example .env
# 編輯 .env，填入必要變數（例如 GEMINI_API_KEY）
```

請勿將 `.env` 或 `archive/Docs_and_Assets/assets/secrets/` 提交到版本控制（已在 `.gitignore` 中排除）。

### 4) 準備測試影像
- 若你把影像放在 repo 根的 `assets/`，請於 `tuning_workflow_sync.py` 中設定路徑：

```python
IMAGE_PATH = "assets/MyTongue.jpg"
```

若使用目前的 archive 位置，請改為 `archive/Docs_and_Assets/assets/MyTongue.jpg`。

### 5) 執行實驗

```bash
# 直接用 Python
python tuning_workflow_sync.py

# 或使用 uv（如上）
uv run tuning_workflow_sync.py
```

## 📊 Prompt 版本說明（`prompts/`）

- **V1 醫生原版**：完整專業術語，涵蓋所有診斷維度
- **V2 CoT**：要求輸出思考過程，提高可解釋性
- **V3 精簡版**：去除冗詞、條列輸出，一致性較高

## 範例輸出
- `outputs/`：CSV 實驗報告、`best_config.yaml`
- `experiment_data/`：每次實驗的原始 LLM 輸出

注意：`outputs/` 與 `experiment_data/` 為執行時生成的資料夾，通常會加入 `.gitignore`，不一定會出現在遠端倉庫。

## 建議（可選）
- 在 `.env.example` 補上各欄位說明與是否必填
- 新增 `CONTRIBUTING.md`（PR / Issue 指引）

## 📜 授權
MIT License
