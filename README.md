# 👅 TCM Tongue Diagnosis Model Tuning Framework

> 結合傳統中醫理論與現代大語言模型 (LLM)，透過結構化的參數微調與自動化批次驗證，打造專業級中醫舌診 AI 系統。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-Fast-red.svg)](https://github.com/astral-sh/uv)

## 🌟 核心理念與方法論創新

本專案不僅僅是簡單的 API 呼叫，而是建立了一套嚴謹的 **AI 診斷模型評估框架**：
- **LLM-as-a-Judge 評估維度**：採用動態推論 (N=3) 機制，藉由自動化交叉比對模型輸出的穩定性、中醫辨證的準確度，以及建議的一致性，科學化評估不同 Prompt、Temperature 與 Top-P 參數組合的表現。
- **Batch API 自動化工作流**：實作非同步批次處理架構，大幅降低大規模實驗成本，適合處理大量舌診影像與多維度參數矩陣測試。
- **領域知識融合**：將中醫舌象特徵（如舌色、苔色、齒痕等）轉化為結構化的 AI 評分指標。

## 🧠 Prompt Engineering 與參數實驗 (Team Focus)

本專案的核心重點在於**中醫舌診 Prompt 的迭代與 LLM 參數最佳化**。為了達到高一致性與高準確度，我們進行了多維度的實驗。

👉 **[查看最新實驗報告：20260515_實驗報告 (含 Temperature 死亡交叉分析)](reports/20260515_experiment_report.md)**

### 我們測試的 Prompt 版本 (`prompts/` 目錄)：
1. **醫生原版 (`v1_doctor.txt`)**：涵蓋最完整的專業術語，但一致性較難控制。
2. **思維鏈 CoT 版 (待整理)**：引導模型逐步思考（Step-by-step），試圖平衡專業度與穩定度。
3. **極度精簡版 (待整理)**：大幅削減背景設定，專注於特徵抽取，取得極高的一致性。

### 給組員的協作指南：
- **若你要查看不同參數 (Temperature/Top-P) 的影響**：請直接閱讀 `reports/` 下的實驗報告。
- **若你要新增或測試新的 Prompt**：
  1. 請在 `prompts/` 建立新的 `.txt` 檔案。
  2. 修改專案腳本以載入你的 Prompt。
  3. 執行小規模測試確認後，再使用 Batch API 進行驗證。

## 🛠 核心功能

- `tuning_workflow_batch.py`：基於 Batch API 的低成本參數微調與自動化測試腳本。
- `tuning_workflow_sync.py`：用於即時開發與小規模驗證的同步推理腳本。
- `retrieve_results.py`：自動解析與格式化實驗結果的工具。

## 🚀 快速開始

### 1. 環境需求
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (超快速 Python 套件管理器)

### 2. 安裝與初始化
```bash
# 克隆專案
git clone https://github.com/[你的GitHub帳號]/[專案名稱].git
cd [專案名稱]

# 使用 uv 同步環境與依賴
uv sync
```

### 3. 環境變數設定
請複製 `.env.example` 檔案並填入您的 API Key（專案中 `.env` 已被設定為 git ignore，請放心）：
```bash
cp .env.example .env
# 請編輯 .env 填寫您的 API 憑證
```

### 4. 執行測試
```bash
# 執行同步測試工作流
uv run tuning_workflow_sync.py
```

## 📁 目錄說明

```text
├── src/                  # 核心原始碼與執行腳本 (待整理移動)
├── prompts/              # 🎯 [新增] 各版本的提示詞庫 (讓組員可重用與迭代)
│   └── v1_doctor.txt     # 醫生原版 Prompt
├── reports/              # 📊 [新增] 實驗報告與數據分析
│   └── 20260515_experiment_report.md # 最新提示詞實驗報告
├── experiment_data/      # 基準測試用的影像與資料 (不含個資)
├── pyproject.toml        # 專案與依賴配置 (uv)
├── MyTongue.jpg          # 專案視覺與示範素材
└── README.md             # 專案首頁 (重點指引組員去哪看實驗結果)
```

> **關於 `experiment_data/`**：本資料夾僅存放去識別化後的基準測試（Benchmark）影像。真實臨床數據請務必遵循 HIPAA 等醫療隱私規範，請勿上傳至本公開倉庫。

## 🤝 貢獻指南 (Contribution)

我們歡迎任何形式的貢獻，特別是：
1. 中醫臨床醫師的 Prompt 優化建議。
2. 針對 LLM-as-a-Judge 評估標準的擴充。
3. 程式碼架構優化與 Bug 修復。

請先發起 Issue 討論您的想法，再提交 Pull Request。

## 📜 授權條款

本專案採用 [MIT License](LICENSE) 授權。

## 📧 聯絡方式
如有合作需求或問題回報，歡迎透過 Issue 系統或 Email 與我聯繫：`[你的Email]`
