# 👅 TCM Tongue Diagnosis — LLM Parameter Tuning Framework

> 結合傳統中醫理論與 Google Gemini，透過結構化的參數微調與自動化評分，找出最佳的舌診 Prompt 與推理參數組合。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-Fast-red.svg)](https://github.com/astral-sh/uv)

## 🧠 核心方法論

- **系統化參數網格搜尋**：測試 3 種 Prompt 版本 × 5 組 Temperature/Top-P 組合。
- **動態多重推論 (N=3)**：T > 0 時重複推論 3 次，計算 Jaccard 穩定性分數。
- **混合評分機制 (Rule + LLM Judge)**：規則型評分（關鍵字覆蓋 + 格式）+ Gemini 自評（因果邏輯 + 語言適切性）。

## 📁 目錄結構

```
├── prompts/
│   ├── v1_doctor.txt           # 醫生原版 Prompt
│   └── 醫生提示詞參考.txt       # 本地實驗用提示詞
├── reports/                    # 實驗報告與分析
├── outputs/                    # 腳本輸出的 CSV 報告
├── experiment_data/            # 各實驗逐筆原始文字輸出
├── .env.example                # 環境變數範本
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── archive_old_data.py         # 清理工具
└── tuning_workflow_sync.py     # 主程式（同步推理）
```

## 🚀 快速開始

### 1. 環境需求
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

### 2. 安裝依賴
```bash
git clone https://github.com/LNSY116/Tongue-Diagnosis.git
cd Tongue-Diagnosis
uv sync
```

### 3. 設定 API Key
```bash
cp .env.example .env
# 編輯 .env，填入你的 GEMINI_API_KEY
```

### 4. 準備測試影像
將你的舌診照片放入專案根目錄，並修改 `tuning_workflow_sync.py` 頂部的：
```python
IMAGE_PATH = "MyTongue.jpg"  # ← 改成你的檔名
```

> ⚠️ **隱私提醒**：真實臨床舌診影像請勿上傳至公開 repo，已於 `.gitignore` 排除常見圖片格式。

### 5. 執行實驗
```bash
uv run tuning_workflow_sync.py
```

輸出結果將儲存於 `outputs/doctor_sync_tuning_results.csv`。

## 📊 Prompt 版本說明 (`prompts/`)

| 版本 | 描述 |
|---|---|
| **V1 醫生原版** | 完整專業術語，涵蓋所有診斷維度 |
| **V2 CoT 版** | 要求輸出思考過程，提升可解釋性 |
| **V3 精簡版** | 去除客套話，直接條列輸出，一致性最高 |

## 📜 授權

MIT License

## 📧 聯絡

[GitHub Issue](https://github.com/LNSY116/Tongue-Diagnosis/issues)
