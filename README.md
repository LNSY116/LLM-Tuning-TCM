# 🔬 LLM Tuning TCM (中醫舌診參數調優框架)

這是一個自動化搜尋最佳 LLM 推理參數（Prompt、Temperature、Top-P）的調優工具，專為中醫舌診場景設計。
調優產出的最佳參數，可直接匯回 [Tongue-Diagnosis 主應用](https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis) 的設定中使用。

## 🎯 核心功能
- **網格搜尋 (Grid Search)**：自動交叉測試不同 Prompt、Temperature 與 Top-P 組合。
- **混合評分機制**：結合規則檢查 (術語覆蓋率、格式) 與 LLM 裁判 (因果邏輯、語言適切性)。
- **穩定性評估**：針對高溫參數進行多次推論，透過 Jaccard 相似度計算一致性。
- **自動化報告**：自動產生所有測試組合的 CSV 評分結果與 `best_config.yaml`。

## 📁 專案結構
# 🔬 LLM Tuning TCM (中醫舌診參數調優框架)

這是一個自動化搜尋最佳 LLM 推理參數（Prompt、Temperature、Top-P）的調優工具，專為中醫舌診場景設計。
調優產出的最佳參數，可直接匯回 [Tongue-Diagnosis 主應用](https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis) 的設定中使用。

## 🎯 核心功能
- **網格搜尋 (Grid Search)**：自動交叉測試不同 Prompt、Temperature 與 Top-P 組合。
- **混合評分機制**：結合規則檢查 (術語覆蓋率、格式) 與 LLM 裁判 (因果邏輯、語言適切性)。
- **穩定性評估**：針對高溫參數進行多次推論，透過 Jaccard 相似度計算一致性。
- **自動化報告**：自動產生所有測試組合的 CSV 評分結果與 `best_config.yaml`。

## 📁 專案結構
```text
.
├── assets/                  # 放置公用測試影像 (如 MyTongue.jpg) 與提示詞
├── archive/                 # 歷史實驗數據、機敏檔案與私人影像 (不進版控)
├── outputs/                 # 執行腳本後產生的分析報告與最佳參數配置
├── tuning_workflow_sync.py  # 參數調優核心工作流腳本
├── requirements.txt         # 依賴套件清單
└── pyproject.toml           # 專案配置 (支援 uv 等工具)
```

## 🚀 快速上手

### 1. 安裝環境
需搭配 Python 3.11+。
```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 uv (推薦)
pip install uv
uv pip install -r requirements.txt
```

### 2. 環境設定
複製範例環境檔並填入你的 API Key：
```bash
cp .env.example .env
# 編輯 .env 填入 GEMINI_API_KEY
```

### 3. 準備測試資料
請將用來測試的舌象照片放置於 `assets/MyTongue.jpg`。
*(若要使用個人機敏照片，請放入 `archive/Docs_and_Assets/assets/` 內，腳本會自動 fallback 尋找。)*

### 4. 執行調優
```bash
# 標準執行
python tuning_workflow_sync.py

# 或使用 uv 執行
uv run tuning_workflow_sync.py
```

### 5. 查看結果
執行完畢後，結果會儲存於 `outputs/` 目錄：
- `doctor_sync_tuning_results.csv`：詳細的網格搜尋與評分紀錄 (支援中斷後接續執行)。
- `best_config.yaml`：系統根據綜合評分選出的最佳參數配置，可直接給主應用使用。
