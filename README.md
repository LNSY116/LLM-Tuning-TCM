# 🔬 LLM Parameter Tuning Framework for TCM Tongue Diagnosis

> 自動化搜尋最佳 LLM 推理參數（Prompt + Temperature + Top-P）的實驗框架，專為中醫舌診場景設計。
> 
> ⚠️ **本專案為「參數調優工具」，需搭配主應用系統使用**：
> - 主應用：[FJCU-AI-APPLICATION/Tongue-Diagnosis](https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis)
> - 本框架輸出的最佳參數，可回填至主應用的 `assets/config/` 或 `prompts/` 目錄

## 🎯 核心功能
- ✅ **網格搜尋**：自動測試多組 Prompt × Temperature × Top-P 組合
- ✅ **穩定性評估**：動態重複推論 + Jaccard 相似度計算
- ✅ **混合評分**：規則檢查（術語覆蓋/格式）+ LLM-as-a-Judge（因果邏輯/語言適切）
- ✅ **實驗追蹤**：自動產出 CSV 報告 + 原始輸出存檔

## 🔄 與主應用系統的協作流程
```mermaid
graph LR
A[主應用: 收集黃金 Prompt] --> B[本框架: 參數網格搜尋]
B --> C[本框架: 產出最佳參數組合]
C --> D[回填主應用: 更新 config/prompt]
D --> E[主應用: 上線驗證]
