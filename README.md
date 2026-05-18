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
├── prompts/                 # 提示詞存放區 (存放如 system_prompt.md)
├── config/                  # 其他設定檔
├── experiment_data/         # 實驗原始輸出紀錄
├── outputs/                 # 產生的分析報告與最佳參數配置 (best_config.yaml)
├── reports/                 # 綜合分析報告
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
1. **圖片**：請在專案根目錄建立 `assets/` 資料夾，並將測試用的舌象照片放置於 `assets/MyTongue.jpg`。
2. **提示詞**：請確認 `prompts/` 目錄內包含 `system_prompt.md` 作為基礎提示詞。

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
- `best_config.yaml`：系統根據綜合評分選出的最佳參數配置，可直接提供給主應用使用。

## 最新進度 (2026-05-18)

- **實驗報告**：已完成實驗報告，請參閱 [reports/20260515_experiment_report.md](reports/20260515_experiment_report.md)。主要結論：最佳配置為使用原始 `醫生提示詞參考.txt`、Temperature = 0.0、Top-P = 0.9（對應實驗 EXP02）。
- **結果檔案**：完整數據已儲存在 `outputs/doctor_sync_tuning_results.csv`，另有彙整表 [archive/Reports_and_Results/tuning_results.md](archive/Reports_and_Results/tuning_results.md)。
- **完成項目**：已執行 15 組實驗（EXP01–EXP15），完成資料彙整與分析並撰寫報告。
- **風險與限制**：目前僅使用少量測試圖片進行參數驗證，術語覆蓋率仍有提升空間，評分機制依賴 LLM Judge，建議加入人工盲測以驗證結果。
- **下一步建議**：
	- 擴充測試圖片集（建議 20–30 張，涵蓋不同病理類型）。
	- 引入 Human Evaluation（中醫師盲測）與自動指標交叉驗證。
	- 建立「專業版」與「大眾版」兩套輸出模板，分別優化術語覆蓋率與可讀性。

- **目前狀態**：實驗與報告已完成；下一階段為擴充資料與進行部署前的驗證。
