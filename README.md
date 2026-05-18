# 🔬 LLM Tuning TCM (中醫舌診參數調優框架)

這是一個自動化搜尋最佳 LLM 推理參數（Prompt、Temperature、Top-P）的調優工具，專為中醫舌診場景設計。
調優產出的最佳參數，可直接匯回 [Tongue-Diagnosis 主應用](https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis) 的設定中使用。

## 🎯 核心功能
- **非同步高併發 (Async Workflow)**：使用 `asyncio` 實作多 Worker 併發，大幅提升參數搜索效率，並支援斷點續傳。
- **混合評分與 PRF1 指標**：結合實體級 Jaccard 穩定性、Precision/Recall/F1 證素評分，以及 Rule-based LLM 裁判。
- **雙提示詞架構**：將「專業診斷推導」與「大眾白話翻譯」拆分，確保評分的科學性與客觀性。
- **自動化報告**：自動產生所有測試組合的 CSV 評分結果。

## 📁 專案結構
```text
.
├── assets/                  # 測試圖片與 ground_truth.json
├── prompts/                 # 提示詞存放區
│   ├── system_prompt_professional.md  # 專業診斷專用 (調優目標)
│   └── system_prompt_layman.md        # 白話翻譯專用
├── experiment_data/         # 實驗原始輸出紀錄
├── outputs/                 # 產生的分析報告與最佳參數配置
├── reports/                 # 綜合分析報告
├── tuning_workflow_sync.py  # 舊版同步工作流腳本
├── tuning_workflow_async.py # 新版非同步高效工作流
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
1. **圖片**：請在專案根目錄建立 `assets/test_images/` 資料夾，並放入測試照片。若無，會 Fallback 使用 `assets/MyTongue.jpg`。
2. **黃金標準**：建立 `assets/ground_truth.json` 來定義每張圖片的正確證素標註。

### 4. 執行調優
```bash
# 執行最新非同步版本
python tuning_workflow_async.py

# 或使用 uv 執行
uv run tuning_workflow_async.py
```

### 5. 查看結果
執行完畢後，結果會儲存於 `outputs/` 目錄：
- `doctor_async_tuning_results.csv`：詳細的網格搜尋與評分紀錄 (支援中斷後接續執行)。

## 最新進度 (2026-05-19)

- **架構重構完成**：完成 `tuning_workflow_async.py` 實作，大幅提升併發效率與中斷接續的可靠性。
- **評分機制升級**：
  - 放棄單純字元級 Jaccard，改採**實體級 Jaccard**，確保穩定性分數能真實反映醫學邏輯一致性。
  - 引入 `ground_truth.json` 提供標準答案，以此計算 **Precision / Recall / F1 Score**。
  - LLM Judge 升級為 **Rule-based Judge**，強制注入對應表與特徵，消滅預訓練幻覺偏差。
- **提示詞架構拆分**：將原 `system_prompt.md` 拆分為 `professional` (調優專用) 與 `layman` (翻譯專用)，徹底解決「要求白話文卻懲罰術語覆蓋率」的邏輯矛盾。
- **下一步建議**：
  - 擴充測試圖片集（建議 20–30 張，涵蓋不同病理類型）與對應的 Ground Truth 標註。
  - 運行新版非同步腳本，產生更具統計與科學意義的最佳參數配置。
