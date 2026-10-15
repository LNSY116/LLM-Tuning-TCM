# **中醫舌診 AI 診斷模型優化策略：參數調優與提示詞工程的實驗設計全指南**

在人工智能與傳統醫學交匯的當前階段，構建一個可靠、精準且具備臨床解釋性的中醫舌診辨證系統，已成為計算醫學領域的核心挑戰。中醫舌診作為「望、聞、問、切」四診之首，其觀測數據的客觀化與邏輯化是實現自動化診斷的先決條件 1。大型語言模型（LLM）的興起，為舌象特徵到病機證素的複雜映射提供了強大的計算框架，但其內在的隨機性與生成式特徵也為診斷的嚴謹性帶來了挑戰。本報告旨在針對開發者提供的 temperature（溫度）與 top\_p（核採樣）參數，結合其特定的專家系統提示詞（System Prompt），設計一套系統化的實驗尋優流程，以期在閉環知識庫的約束下，找出達成最佳診斷一致性與邏輯準確性的技術參數組合。

## **第一章：LLM 生成參數與醫學邏輯的底層關聯**

在優化任何診斷模型之前，必須深刻理解 temperature 與 top\_p 如何在數學層面改寫模型對中醫證素的篩選行為。語言模型的輸出本質上是基於概率分布的離散隨機過程，而參數調優的本質是重新調整這個分布的「形狀」3。

### **1.1 Temperature（溫度）的物理與統計學意義**

Temperature 參數（![][image1]）在 softmax 函數中扮演著縮放因子的角色。對於給定的 logits 向量 ![][image2]，模型生成下一個標記（token）的概率 ![][image3] 由以下公式決定：

![][image4]  
當 ![][image1] 設定在較低範圍（例如 0 到 0.5）時，模型會放大高概率標記與低概率標記之間的差距。在診斷場景中，這意味著模型將極度依賴於提示詞中【一、對應表】所定義的最高權重路徑 3。低溫環境會抑制模型的「聯想能力」，強制其執行確定性的邏輯映射，這對於醫學診斷中排除幻覺（Hallucination）至關重要 5。

然而，如果 ![][image1] 設定過高（如接近 2.0），概率分布會趨於平坦，導致模型可能會從概率極低的標記中進行採樣。在中醫語境下，這可能導致 AI 忽視高特異性的「膩苔」特徵，轉而輸出與提示詞表不符的、來自外部預訓練數據的診斷結果 3。因此，診斷模型的實驗設計應將重心放在低溫區間的穩定性測試上。

### **1.2 Top-P（核採樣）與動態詞庫縮減**

Top-P 或稱核採樣（Nucleus Sampling），與固定數量的 Top-K 不同，它根據標記的累積概率質量來動態決定候選池的大小 5。其數學定義為找出最小集合 ![][image5] 使得：

![][image6]  
在處理中醫證素時，Top-P 的優化具備深遠意義。由於診斷指令限制了僅能輸出九種特定證素（氣虛、陽虛等），合理的 Top-P 值應能確保模型只在符合這九種標籤的標記空間中遊走 3。若 ![][image7] 設置過大，模型可能在邏輯推理的中間環節引入非專業或提示詞外的術語；若過小（如 ![][image8]），則可能因候選詞庫過於狹窄而在面對複雜舌象組合時發生邏輯「斷裂」或生成失敗 7。

## **第二章：現有專家提示詞的邏輯架構分析**

開發者提供的系統提示詞是一個典型的「基於規則的封閉世界模型」（Rule-based Closed-world Model）。這種架構試圖將模糊的中醫辨證轉化為確定性的算法過濾流程 9。

### **2.1 閉環知識庫與唯一依據原則**

提示詞中的【一、對應表】被定義為系統的「唯一依據（Single Source of Truth, SSOT）」。這要求 LLM 在執行 {{PREDICTIONS}} 替換時，必須主動抑制其在海量醫學書籍中學到的外部知識 9。實驗設計必須檢測在不同參數下，模型是否能維持這種「知識屏蔽」的能力。若參數導致模型提及「心火旺」或「腎精虧損」等未包含在九大證素中的術語，則應判定為參數或提示詞邏輯崩潰 9。

### **2.2 權重層級與特異性抑制**

該提示詞最精妙之處在於建立了特異性權重機制（規則四與五）。這模擬了資深中醫師的臨床思維：並非所有症狀都具備同等診斷價值 10。

| 權重等級 | 舌象特徵示例 | 診斷邏輯 |
| :---- | :---- | :---- |
| 高特異性 | 裂紋、厚膩苔、舌下怒張 | 一錘定音，優先成立相關證素 9。 |
| 低特異性 | 齒痕、胖大、嫩舌 | 基礎指標，易被高特異性證據壓制 9。 |
| 衝突處理 | 舌淡（虛） \+ 苔厚（實） | 根據規則八，判斷是否可解釋或需降級 9。 |

實驗設計的核心挑戰之一，就是驗證在不同的 temperature 設定下，模型是否能正確執行「痰濕證據強時壓制氣虛」這類精細的權重運算，而非僅僅做簡單的特徵疊加 9。

### **2.3 病機推導與組合辨證的線性化**

提示詞中的規則六與七為模型設定了明確的線性演化路徑（如氣虛 ![][image9] 陽虛）。這種鏈式推導旨在確保輸出格式（九）中的「體質說明」具備因果邏輯 9。開發者要求的輸出不僅是標籤，更是一段讓一般大學生能理解的「故事」，這對 LLM 的文本生成連貫性提出了要求，而這正是參數優化的關鍵平衡點：既要邏輯嚴密，又要語言自然 9。

## **第三章：系統化尋優實驗設計方法論**

為了找出最佳參數與提示詞修正方案，建議採用「三階段演進式尋優」框架。這套框架結合了機器學習中的超參數搜索技術與臨床醫學的對標測試。

### **3.1 構建金標準數據集（Golden Dataset）**

沒有精確的基準數據，優化將淪為「感覺驅動」。開發者需要構建一個包含至少 100 至 150 個案例的數據集 11。

1. **特徵組合覆蓋**：應包含「典型案例」（如淡紅舌、薄白苔的正常案例）、「單一證素案例」以及「複雜衝突案例」（如紅舌與厚白苔並存）1。  
2. **專家共識標註**：每個案例的輸入（舌象特徵）與預期輸出（九大證素之一或多個、白話體質說明）應由至少三位資深中醫師進行雙盲標註，並僅保留標註達成一致的「純淨案例」15。  
3. **邊界壓力測試**：特別設計一些處於對應表邊緣的案例，例如舌苔「微厚」與「厚」之間的臨界點，測試模型的特異性權重閾值 17。

### **3.2 參數網格搜索（Grid Search）方案**

針對 $T \\in $ 與 ![][image10]，我們不建議進行無差別的全量搜索，而是聚焦於高穩定性區間 19。

| 實驗組 | Temperature (T) 取值 | Top-P (P) 取值 | 預期目的 |
| :---- | :---- | :---- | :---- |
| **A 組 (嚴謹組)** | 0.0, 0.2, 0.4 | 0.1, 0.3, 0.5 | 測試模型在極低隨機性下的規則執行度。 |
| **B 組 (均衡組)** | 0.5, 0.7, 0.8 | 0.7, 0.8, 0.9 | 測試白話文輸出的流利度與因果推導的自然感。 |
| **C 組 (極端組)** | 1.0, 1.5 | 1.0 | 觀察模型何時開始脫離提示詞對應表產生幻覺。 |

開發者應使用自動化腳本（如 promptfoo 或 Python API 調用）遍歷上述網格中的所有組合，並對每個組合重複執行 3 至 5 次，以計算其「診斷一致性」20。

### **3.3 提示詞邏輯的消融實驗（Ablation Study）**

除了參數，提示詞本身的結構也需要微調。建議測試以下變體：

1. **變體 V1 (基準)**：當前上傳的版本。  
2. **變體 V2 (思維鏈強化)**：在【三、候選建立】之前，加入「請先一步步思考每個舌象特徵對應的證素權重，並寫下理由」的指令 23。  
3. **變體 V3 (負面約束強化)**：顯式列出常見錯誤，如「禁止將胖大舌直接等同於陽虛，除非有水濕特徵支持」25。

## **第四章：基於 LLM-as-a-Judge 的評測體系**

在完成網格搜索後，將產生數千條輸出數據。人工審核這些數據不具備可行性，必須引入「裁判模型」評分機制 27。

### **4.1 裁判模型的選擇與校準**

建議使用目前最強性能的模型（如 GPT-4o 或 Claude 3.5 Opus）作為裁判。為了確保裁判不產生偏見，需先提供一組經過人工精確評分（金標準）的樣本，讓裁判模型學習開發者的診斷標準 27。

### **4.2 多維度評分量表設計**

裁判模型應針對每條輸出，從以下五個維度給出 1 至 5 分的評分 29：

* **規則遵從度（Mapping Fidelity）**：模型是否嚴格執行了【一、對應表】？是否出現了表外證素？  
* **權重合理性（Weight Logic）**：在存在衝突的舌象中，模型是否正確識別並應用了「高特異性優先」原則？  
* **推導因果性（Causal Coherence）**：體質說明中的因果關係是否符合規則六的演化鏈？  
* **語言適切性（Linguistic Register）**：是否使用了「大學生能理解」的語言？專業術語轉譯是否準確？  
* **格式完整性（Output Compliance）**：是否包含免責聲明？五大部分是否齊全？

### **4.3 統計分析指標**

實驗最終應輸出各參數組合的彙總得分表，關鍵指標包括：

1. **平均準確率 (Mean Accuracy)**：對應金標準標籤的比例。  
2. **結果方差 (Result Variance)**：同一輸入多次運行的診斷穩定性 22。  
3. **格式崩潰率 (Format Failure Rate)**：輸出不符合五段式結構的頻率。

## **第五章：提示詞優化深度解析與 {{PREDICTIONS}} 標記策略**

開發者提到的 {{PREDICTIONS}} 標記是系統的邏輯支點。這部分提示詞的編排直接影響模型在推理時的「注意力分配」32。

### **5.1 PREDICTIONS 標記的上下文定位**

目前的提示詞將 {{PREDICTIONS}} 放置在中間位置。研究表明，將輸入數據放置在推理指令之後、格式要求之前，有利於 LLM 在長上下文中保持對原始特徵的對焦 23。此外，應確保傳入 {{PREDICTIONS}} 的數據格式是高度結構化的 JSON 或鍵值對，這能顯著提升模型的解析精準度。

### **5.2 針對閉環映射的提示詞強化建議**

目前的提示詞在【一、對應表】部分非常詳盡，但在「負權重」處理上較為模糊。建議在【四、特異性權重】中增加「排他性規則」。例如：

* 「若苔質為『剝』，即便舌質『嫩』，也必須優先診斷為陰虛，並視嫩舌為次要特徵。」 這種硬化的條件語句能減少 LLM 在低溫模式下的猶豫 9。

### **5.3 體質說明的「轉譯」演算法**

體質說明是面向大眾的關鍵環節。為了優化這一點，可以在系統提示詞中嵌入一個「轉譯字典示例」：

* 「氣滯 ![][image9] 身體能量運行不順暢，像堵車一樣。」  
* 「血瘀 ![][image9] 血液循環動力不足，容易產生沉澱。」 通過這種「少樣本提示（Few-shot Prompting）」技術，模型在任何參數設定下都能維持一致的語言風格 35。

## **第六章：診斷模型在複雜場景下的壓力測試**

中醫診斷的難點在於「複合證候」與「寒熱錯雜」。實驗設計必須驗證系統在極端數據下的魯棒性。

### **6.1 虛實夾雜場景**

當 {{PREDICTIONS}} 傳入「舌色：淡（虛）；舌質：老（實）；舌下：怒張（實）」時，模型面臨規則一與規則八的直接衝突。

* **低 ![][image1] 表現**：模型可能會機械地輸出氣虛與血瘀。  
* **中 ![][image1] 表現**：模型可能會試圖解釋為「氣虛無力推動血液，導致血瘀（老舌）」。 實驗應觀察哪組參數能觸發這種更具醫學深度、且符合規則六（氣滯/氣虛 ![][image9] 血瘀）的解釋 9。

### **6.2 舌苔與舌質背離場景**

舌質紅（熱）而舌苔白厚（寒/濕）是臨床常見的複雜表現 37。根據規則七（痰濕 \+ 化熱 \= 濕熱），系統應具備「合成思維」。實驗需監控模型是否會錯誤地將其拆分為獨立的「化熱」與「痰濕」而不進行合成，這通常是 ![][image7] 設置過低導致候選詞概率截斷的表現 4。

## **第七章：實驗尋優的技術工具鏈與實施路徑**

為了高效執行上述實驗設計，開發者應構建一套「提示詞研發流水線」。

### **7.1 使用 Promptfoo 進行自動化評測**

promptfoo 是一款專為 LLM 評測設計的 CLI 工具，非常適合用於參數掃描 20。

**實施步驟**：

1. **定義變量**：在 tests.yaml 中將 {{PREDICTIONS}} 作為變量引入，導入金標準案例。  
2. **配置 Provider**：在 promptfooconfig.yaml 中定義多個 Provider 分支，分別對應不同的 temperature 和 top\_p 組合 40。  
3. **設置斷言 (Assertions)**：使用 llm-rubric 斷言，讓裁判模型自動檢查輸出是否符合診斷規則 21。  
4. **運行與對比**：執行 promptfoo eval 並通過 promptfoo view 在 Web UI 中對比各參數組合的得分矩陣 43。

### **7.2 數據可視化與最優區域鎖定**

在獲得數百組測試結果後，建議繪製「熱力圖（Heatmap）」。橫軸為 ![][image1]，縱軸為 ![][image7]，數值為「裁判模型總分」。

* 通常會發現一個「高原區」，在此區間內模型表現穩定。  
* 對於診斷任務，最優區域通常出現在 ![][image11] 且 ![][image12] 3。

## **第八章：結論與臨床落地建議**

通過對 temperature 與 top\_p 的系統化尋優，以及提示詞邏輯層次的消融實驗，開發者可以將一個傳統的、依賴「體感」調整的 AI 模型，轉變為一個經過量化驗證的臨床輔助工具。

### **8.1 最終參數配置推薦（初步猜測）**

基於現有研究，對於此類嚴格依據對應表的專家系統，最佳配置往往是 3：

* **Temperature**: ![][image13]（確保診斷路徑的唯一性）。  
* **Top-P**: ![][image14]（過濾非專業干擾，保留必要的邏輯組合能力）。  
* **Seed**: 若 API 支持，建議固定隨機種子，以追求極致的結果重現 6。

### **8.2 提示詞維護的長期策略**

AI 診斷模型不是靜態的。隨著基礎模型的更新（如從 GPT-4 升級到 GPT-5），原有的最佳參數可能會失效 46。因此，建議：

1. **版本化提示詞**：使用 Git 跟蹤提示詞的每一次微調 32。  
2. **持續監測**：在線上環境隨機抽取 5% 的數據進行裁判模型審核，監控是否存在「診斷漂移」30。  
3. **神經符號化演進**：考慮將提示詞中的對應表轉化為外部知識庫（RAG），通過向量檢索進一步降低模型的記憶負擔 52。

中醫舌診的客觀化是一個漫長的過程。通過嚴謹的超參數實驗與提示詞工程，我們不僅是在調優一個 AI 模型，更是在嘗試用現代信息論的語言，重構千年醫學的邏輯骨架。這份實驗設計旨在為開發者提供一張導航圖，在隨機性與確定性的天平上，找出那顆最精準的診斷核心 9。

## ---

**附錄：診斷證素與舌象映射統計參考表**

為了方便開發者進行數據檢驗，以下彙總了提示詞中隱含的邏輯優先級，這應作為金標準數據集構建的基準。

| 證素標籤 | 核心強特徵 (High Weight) | 輔助弱特徵 (Low Weight) | 禁忌/排除 |
| :---- | :---- | :---- | :---- |
| **氣虛** | 無根苔、嫩舌 | 齒痕、胖大、舌淡 9 | 若出現厚膩苔需降級 9 |
| **痰濕** | 厚/膩/腐苔、胖大 9 | 齒痕、偏斜、白苔 | 薄苔時不成立 9 |
| **陰虛** | 裂紋、剝苔、瘦薄 9 | 紅/絳舌、津少、少苔 | 濕滑舌不成立 9 |
| **血瘀** | 舌下絡脈異常、瘀斑 9 | 青紫/黯舌、老舌 | 嫩舌通常不優先考慮 9 |
| **濕熱** | 黃厚膩苔 (痰濕+化熱) 9 | 舌邊紅、肉埠腫脹 | 清稀涎水不成立 56 |

開發者應將此表作為檢驗模型「理解力」的標尺。在實驗中，若 AI 能在 ![][image15] 時完美區分這些層次，而在 ![][image16] 時開始混淆，則 ![][image17] 便是更具臨床價值的參數點 9。

#### **引用的著作**

1. TCM-Tongue: A Standardized Tongue Image Dataset with Pathological Annotations for AI-Assisted TCM Diagnosis \- ResearchGate, 檢索日期：5月 14, 2026， [https://www.researchgate.net/publication/393982480\_TCM-Tongue\_A\_Standardized\_Tongue\_Image\_Dataset\_with\_Pathological\_Annotations\_for\_AI-Assisted\_TCM\_Diagnosis](https://www.researchgate.net/publication/393982480_TCM-Tongue_A_Standardized_Tongue_Image_Dataset_with_Pathological_Annotations_for_AI-Assisted_TCM_Diagnosis)  
2. Tongue Diagnosis Chart (Plus How to Read It) \- Huwe Acupuncture, 檢索日期：5月 14, 2026， [https://brianhuwe.com/tongue-diagnosis-chart-plus-how-to-read-it/](https://brianhuwe.com/tongue-diagnosis-chart-plus-how-to-read-it/)  
3. LLM Temperature, Top-P, and Top-K Explained — With Python Simulations, 檢索日期：5月 14, 2026， [https://machinelearningplus.com/gen-ai/llm-temperature-top-p-top-k-explained/](https://machinelearningplus.com/gen-ai/llm-temperature-top-p-top-k-explained/)  
4. Setting Top-K, Top-P and Temperature in LLMs \- Ruman \- Medium, 檢索日期：5月 14, 2026， [https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832](https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832)  
5. Generation configurations: temperature, top-k, top-p, and test time compute \- Chip Huyen, 檢索日期：5月 14, 2026， [https://huyenchip.com/2024/01/16/sampling.html](https://huyenchip.com/2024/01/16/sampling.html)  
6. Understanding Temperature, Top P, and Maximum Length in LLMs \- Learn Prompting, 檢索日期：5月 14, 2026， [https://learnprompting.org/docs/intermediate/configuration\_hyperparameters](https://learnprompting.org/docs/intermediate/configuration_hyperparameters)  
7. Complete Guide to Prompt Engineering with Temperature and Top-p, 檢索日期：5月 14, 2026， [https://promptengineering.org/prompt-engineering-with-temperature-and-top-p/](https://promptengineering.org/prompt-engineering-with-temperature-and-top-p/)  
8. Cheat Sheet: Mastering Temperature and Top\_p in ChatGPT API, 檢索日期：5月 14, 2026， [https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683)  
9. 醫生提示詞參考.txt  
10. Tongue Diagnosis in Chinese Medicine by Giovanni Maciocia, 檢索日期：5月 14, 2026， [https://giovanni-maciocia.com/tongue-gallery/](https://giovanni-maciocia.com/tongue-gallery/)  
11. promptflow-resource-hub/sample\_gallery/golden\_dataset/copilot-golden-dataset-creation-guidance.md at main \- GitHub, 檢索日期：5月 14, 2026， [https://github.com/microsoft/promptflow-resource-hub/blob/main/sample\_gallery/golden\_dataset/copilot-golden-dataset-creation-guidance.md](https://github.com/microsoft/promptflow-resource-hub/blob/main/sample_gallery/golden_dataset/copilot-golden-dataset-creation-guidance.md)  
12. Building a “Golden Dataset” for AI Evaluation: A Step-by-Step Guide \- Maxim AI, 檢索日期：5月 14, 2026， [https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/)  
13. Traditional Chinese medicine-based visual assessment of tongue features associated with current cancer vs. no cancer history: a matched case-control study \- PMC, 檢索日期：5月 14, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12900746/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12900746/)  
14. How to Build Golden Datasets for Content Moderation \- Musubi Labs, 檢索日期：5月 14, 2026， [https://www.musubilabs.ai/post/how-to-build-golden-datasets-for-content-moderation](https://www.musubilabs.ai/post/how-to-build-golden-datasets-for-content-moderation)  
15. TongueNet: a multi-modal fusion and multi-label ... \- Frontiers, 檢索日期：5月 14, 2026， [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1527751/full](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1527751/full)  
16. A Tongue Image Dataset with Pathological Annotations for AI-assisted Diagnosis in Traditional Chinese Medicine \- ResearchGate, 檢索日期：5月 14, 2026， [https://www.researchgate.net/publication/401861514\_A\_Tongue\_Image\_Dataset\_with\_Pathological\_Annotations\_for\_AI-assisted\_Diagnosis\_in\_Traditional\_Chinese\_Medicine](https://www.researchgate.net/publication/401861514_A_Tongue_Image_Dataset_with_Pathological_Annotations_for_AI-assisted_Diagnosis_in_Traditional_Chinese_Medicine)  
17. Harvard and Stanford Physicians Built the Toughest Medical AI Benchmark Yet — Here's How AI Co-Clinician Scored | MindStudio, 檢索日期：5月 14, 2026， [https://www.mindstudio.ai/blog/harvard-stanford-physicians-medical-ai-benchmark-ai-co-clinician-scores](https://www.mindstudio.ai/blog/harvard-stanford-physicians-medical-ai-benchmark-ai-co-clinician-scores)  
18. Ground Truth Data for AI: How to Build the Foundation of Reliable Models \- SuperAnnotate, 檢索日期：5月 14, 2026， [https://www.superannotate.com/blog/ground-truth-data-for-ai](https://www.superannotate.com/blog/ground-truth-data-for-ai)  
19. Grid Search vs Random Search: Which One Should You Use? \- Train in Data's Blog, 檢索日期：5月 14, 2026， [https://www.blog.trainindata.com/grid-search-vs-random-search-which-one-should-you-use/](https://www.blog.trainindata.com/grid-search-vs-random-search-which-one-should-you-use/)  
20. Promptfoo Tutorial: A Hands-On Guide to LLM Evaluation | DataCamp, 檢索日期：5月 14, 2026， [https://www.datacamp.com/tutorial/promptfoo-tutorial](https://www.datacamp.com/tutorial/promptfoo-tutorial)  
21. How to Choose the Right LLM Temperature Setting | Promptfoo, 檢索日期：5月 14, 2026， [https://www.promptfoo.dev/docs/guides/evaluate-llm-temperature/](https://www.promptfoo.dev/docs/guides/evaluate-llm-temperature/)  
22. High Consistency, Limited Accuracy: Evaluating Large Language Models for Binary Medical Diagnosis | medRxiv, 檢索日期：5月 14, 2026， [https://www.medrxiv.org/content/10.64898/2025.12.08.25341823v1.full-text](https://www.medrxiv.org/content/10.64898/2025.12.08.25341823v1.full-text)  
23. Prompt-dependent performance of multimodal AI model in oral diagnosis: a comprehensive analysis of accuracy, narrative quality, calibration, and latency versus human experts \- PMC, 檢索日期：5月 14, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12575769/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12575769/)  
24. Prompt Engineering in Clinical Practice: Tutorial for Clinicians, 檢索日期：5月 14, 2026， [https://www.jmir.org/2025/1/e72644](https://www.jmir.org/2025/1/e72644)  
25. Implementing Automated Rules-Based Evaluations for LLM Applications \- DEV Community, 檢索日期：5月 14, 2026， [https://dev.to/kalio/implementing-automated-rules-based-evaluations-for-llm-applications-468j](https://dev.to/kalio/implementing-automated-rules-based-evaluations-for-llm-applications-468j)  
26. Sprig: Improving Large Language Model Performance by System Prompt Optimization, 檢索日期：5月 14, 2026， [https://arxiv.org/html/2410.14826v3](https://arxiv.org/html/2410.14826v3)  
27. LLM as a Judge \- Primer and Pre-Built Evaluators \- Arize AI, 檢索日期：5月 14, 2026， [https://arize.com/llm-as-a-judge/](https://arize.com/llm-as-a-judge/)  
28. LLM as a Judge: The Complete Guide | Galtea Blog, 檢索日期：5月 14, 2026， [https://galtea.ai/blog/llm-as-a-judge-the-complete-guide](https://galtea.ai/blog/llm-as-a-judge-the-complete-guide)  
29. Rubric-Based Evaluations & LLM-as-a-Judge — Methodologies, Biases, and Empirical Validation in Domain-Specific Contexts. | by Adnan Masood, PhD. \- Medium, 檢索日期：5月 14, 2026， [https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80)  
30. LLM evaluation metrics: Full guide to LLM evals and key metrics \- Articles \- Braintrust, 檢索日期：5月 14, 2026， [https://www.braintrust.dev/articles/llm-evaluation-metrics-guide](https://www.braintrust.dev/articles/llm-evaluation-metrics-guide)  
31. LLM Evaluation: Key Concepts & Best Practices \- Nexla, 檢索日期：5月 14, 2026， [https://nexla.com/ai-readiness/llm-evaluation/](https://nexla.com/ai-readiness/llm-evaluation/)  
32. The Science of Prompt Optimization and Automated Refinement | by Adem Akdogan, 檢索日期：5月 14, 2026， [https://medium.com/data-science-collective/the-science-of-prompt-optimization-and-automated-refinement-044b4b8e5353](https://medium.com/data-science-collective/the-science-of-prompt-optimization-and-automated-refinement-044b4b8e5353)  
33. Benchmarking LLM accuracy for engineering technical QA \- PatSnap, 檢索日期：5月 14, 2026， [https://www.patsnap.com/resources/blog/articles/benchmarking-llm-accuracy-for-engineering-technical-qa/](https://www.patsnap.com/resources/blog/articles/benchmarking-llm-accuracy-for-engineering-technical-qa/)  
34. Enforcing Hierarchical Instruction-Following in LLMs through Logical Consistency \- arXiv, 檢索日期：5月 14, 2026， [https://arxiv.org/html/2604.09075v1](https://arxiv.org/html/2604.09075v1)  
35. In the search for the perfect prompt in medical AI queries \- Frontiers, 檢索日期：5月 14, 2026， [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1689178/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1689178/full)  
36. Investigating the Impact of Prompt Engineering on the Performance of Large Language Models for Standardizing Obstetric Diagnosis Text: Comparative Study \- PMC, 檢索日期：5月 14, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10884897/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884897/)  
37. Tongue Inspection in TCM: Observations in a Study Sample of Patients Living with HIV, 檢索日期：5月 14, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC3929461/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3929461/)  
38. Tongue diagnosis \- Wikipedia, 檢索日期：5月 14, 2026， [https://en.wikipedia.org/wiki/Tongue\_diagnosis](https://en.wikipedia.org/wiki/Tongue_diagnosis)  
39. Promptfoo Cheat Sheet: CLI Commands, Config, and Assertions Reference | PromptFu, 檢索日期：5月 14, 2026， [https://www.promptfu.com/wiki/promptfoo-cheat-sheet/](https://www.promptfu.com/wiki/promptfoo-cheat-sheet/)  
40. LLM Providers | Promptfoo, 檢索日期：5月 14, 2026， [https://www.promptfoo.dev/docs/providers/](https://www.promptfoo.dev/docs/providers/)  
41. OpenAI | Promptfoo, 檢索日期：5月 14, 2026， [https://www.promptfoo.dev/docs/providers/openai/](https://www.promptfoo.dev/docs/providers/openai/)  
42. promptfoo-evaluation | Skills Market... \- LobeHub, 檢索日期：5月 14, 2026， [https://lobehub.com/fr/skills/daymade-claude-code-skills-promptfoo-evaluation](https://lobehub.com/fr/skills/daymade-claude-code-skills-promptfoo-evaluation)  
43. Quickstart guide for using PromptFoo to evaluate LLM prompts via CLI or Colab. \- GitHub, 檢索日期：5月 14, 2026， [https://github.com/ashleysally00/promptfoo-quickstart-guide](https://github.com/ashleysally00/promptfoo-quickstart-guide)  
44. promptfoo — Easy LLM Evaluation with examples | by Sulbha Jain \- Medium, 檢索日期：5月 14, 2026， [https://sulbhajain.medium.com/promptfoo-easy-llm-evaluation-with-examples-e285c2303c9f](https://sulbhajain.medium.com/promptfoo-easy-llm-evaluation-with-examples-e285c2303c9f)  
45. LLM parameters: A complete guide to model evaluation and optimization \- Galileo AI, 檢索日期：5月 14, 2026， [https://galileo.ai/blog/llm-parameters-model-evaluation](https://galileo.ai/blog/llm-parameters-model-evaluation)  
46. LLM Evaluation: The Ultimate Guide to Metrics, Methods & Best Practices \- Comet, 檢索日期：5月 14, 2026， [https://www.comet.com/site/blog/llm-evaluation-guide/](https://www.comet.com/site/blog/llm-evaluation-guide/)  
47. Large Language Model Evaluation in Traditional Chinese Medicine for Stroke: Quantitative Benchmarking Study \- PMC, 檢索日期：5月 14, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12741655/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12741655/)  
48. 7 best prompt management tools in 2026 (tested and compared) \- Articles \- Braintrust, 檢索日期：5月 14, 2026， [https://www.braintrust.dev/articles/best-prompt-management-tools-2026](https://www.braintrust.dev/articles/best-prompt-management-tools-2026)  
49. Prompt Configuration \- Text, Chat, and Dynamic Prompts \- Promptfoo, 檢索日期：5月 14, 2026， [https://www.promptfoo.dev/docs/configuration/prompts/](https://www.promptfoo.dev/docs/configuration/prompts/)  
50. LLM evaluation: Metrics, frameworks, and best practices | genai-research – Weights & Biases \- Wandb, 檢索日期：5月 14, 2026， [https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA)  
51. LLM Temperature Settings: A Complete Guide for Developers \- Tetrate, 檢索日期：5月 14, 2026， [https://tetrate.io/learn/ai/llm-temperature-guide](https://tetrate.io/learn/ai/llm-temperature-guide)  
52. Optimizing LLM Accuracy | OpenAI API, 檢索日期：5月 14, 2026， [https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy](https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy)  
53. TCM-DiffRAG: personalized syndrome differentiation reasoning method for traditional Chinese medicine based on knowledge graph and chain of thought \- Frontiers, 檢索日期：5月 14, 2026， [https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1804478/full](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1804478/full)  
54. TCM-DS: a large language model for intelligent traditional Chinese medicine edible herbal formulas recommendations \- PMC, 檢索日期：5月 14, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12621383/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12621383/)  
55. TCM-3CEval: A Triaxial Benchmark for Assessing Responses From Large Language Models in Traditional Chinese Medicine \- arXiv, 檢索日期：5月 14, 2026， [https://arxiv.org/html/2503.07041v1](https://arxiv.org/html/2503.07041v1)  
56. TCM Tongue Diagnosis: What Your Tongue Reveals \- Thomson Medical, 檢索日期：5月 14, 2026， [https://www.thomsonmedical.com/blog/tcm-tongue-diagnosis](https://www.thomsonmedical.com/blog/tcm-tongue-diagnosis)  
57. How to Calibrate LLM-as-a-Judge with Human Corrections \- LangChain, 檢索日期：5月 14, 2026， [https://www.langchain.com/articles/llm-as-a-judge](https://www.langchain.com/articles/llm-as-a-judge)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAYCAYAAADKx8xXAAAAuUlEQVR4XmNgGDnACYjvAvEjIrELSBMjEE8B4pVArADlg8AcIP4HxB5QPjMQ2wPxAyA2BQmIA/EqIBaDKgABQSA+zQBRJI0kzgPEi4FYBsQBWVuIJAkC+kD8CYjXADELkjjIwElAzAvihAKxGpIkCEQD8X8gLkcTFwbiNAaEdzAAyH+/gdgGXQIfwOU/gsAYiL8yYPqPIMDlP7wA5On5DIPWf6A4PAfE7xggfoPhL0B8nQFi2CgY3AAAzMQr+zx1NKQAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAo0lEQVR4XmNgGAW0ANxA7AzEklA+KxAbADELXAUQ6ALxIyD+D8TvgdgHiGOBOA9ZEUjHVKgkyDQXIN4FxLMYIKbiBFZAPJEB4hScAGRaFxBzokvAACMQhwBxIwMe60CKQoG4GIiZkcRTgdgfic8QDMS/gPg0A8REkMmTgPgEEIvBFKkA8WIgFgJiTSC+zgAJIhAN4sMBBxTDAMh9IgyoThihAAA9GBO5KX/BeAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABMUlEQVR4Xu2TvUoDQRRGr2Bh4Q82EVFMLQSUYCNoIdhoIb6DlUUaGyFVXiAgaiWCWNjZaqEWgo3gMwimUVASGxsR1HOdHZi9s8ZsrAQPHHbZb2Z29t5ZkX/ysoAP+BHYwsfk/gW3cNBP+Il9fMM587wsbtEz7DdZxABe4S2OmEwnX+I7LqajmEls4jH2mmwYbyR7txEr4uqxYQOYxVe8xiGTRWxL9ht14gU+4YzJInwN9K1HuJd4iPd4gON+cDt8fc6xiKOBfcE4RXe4nlwjfH2qNshgCnflm2Og9emote3w5+cOx9JRih5cxVOcNtkXJXzGE4nrEaJ1XBL3+bUwmMeGxP/XWjgooIAT4nZkj0hu9GBqZzM7lgdtyiYuiytJ19RxByviit81Olm7/KtF/gifT5k5dVpTL2EAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAYCAYAAACfpi8JAAAB/klEQVR4Xu2UyyuEURiHX7lESMp1IdeNrCSXwkbkUizsRFlIykKiKLKQFDtJKZcMwkb5A2ymKNY2VhZWVpYsLPD7zTmn73xnvplZmLCYp56Y856Z73zv5YgkjxrY5y46tMEOdzGZlMAtmOcGHNLgPGxxA8liWRJnw1ANd2GOG/gppfAEFrqBGGTAQ9jkBorgDfyy/IDd1p4seOHsWdAx1nxfVNoJ/47DkKgSbMBtWKbjZA5OWZ99DIp6wLEb0KTDc7gEM631Yb1maID98AzuidrLPUfiHZbP4uECYare4KWo9Lm0wlOJru2i1sDGrYXX4qV/FIbFa2YeJNYLR8bvRfxfMPDhIdjorBOm2D4I4QFuRZWdWWA27AzEPQib7gk+w3InNgZXxUutDX/0wFljBlga7q+Ad+J/CcbtcvpgFsLwFdZb61XwSlTKg+BevnG2/mwy8ABnRPVJl44Z1kT1USDsC/bHO2zWa/zRTThkNgXAsvHBLC3hGJv+yBfV5DYFovazAjFhijk5TDfphDvin5IgeuCs/p9NfQ+LvbCPXlGZigubjgfhnPPkvDvqfDuCYeYm4QAcgRNwWqKnrxKuSPTkRcF550HWJcGlEwBLwFLEI1cSZzcCb9NP+CjqznDH+Ncwlxplrf8M3h+8RzheQXfGr8H6tYuqZYoUKf4139CXTvDzpcPmAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAwCAYAAACsRiaAAAAD+0lEQVR4Xu3dW6jlUxwH8KWZkUtuzTAJoYZoZJJrGqVxKQ8yKKY8KC+ejCa3XNKUPCrkUjzIg+JBvIiGssODUFJuKTU88OiZEr+f9d/2fy/7nDn77NmnffL51Lf/f//Wae/z+Gut9V//UgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGBJR0T+ivwS2daMtS6IPFLq3+8fHwIAYJ4+KLUJy+tK7Ip8F9naDgAAMD9/ltq0XdcOLOHMyKttcQavRE5oi2sgZxj3tUUAgEWUzdJHpTZtm5qxWd1Q6vcOHR/5rff56d79ND6MbGiLq5BLvWe0RQCARXRZqY3Vfe3AjIb73vr6n7/v3U8jG79PI7vbgSltLPbkAQDryEulNlOftQMzOFjG98flbN7v3f1RpTZ0fZsjN5a6XHl9OfTDEGlP5N7Ise1Az45SZw/ze/P7+36KXNrUAAAWVjZT2bTd2g6sUn7XgVKbwS8jX5TaNKWLIzu7+7S9jJZk8/94tvx3dm45w1m3VjaGZ5e6FHtlqb//TldPg8hD3T0AwMJ7sIyO+phVNkTZJF0VOTVy4vjwPw1bZui23v3nkXMjd/Zqy3ms1GZw0h68UyInlVGTlsugP5bRk66DyJPdPQDAunBeqY3brK4tdZZsKW3DNpS1O9riBNkEPhe5vB2YIB9+GM7m5bX/fw2KGTYAYB3JRu39trhK70XOb4s9WyJ39z7nXrcLI2+V2ozlsmzOnE3ycls4hEGpy6Lp5zJalk0525bNJQDAwssm5uPI6e3AlPLBgWz6cmn1j8hr48P/yt/rz3Ttj7wdeTTybje23IME0/i11P1teX7c7c3YN5HTmhoAwEL6oS2sgU/awgrlDNykZLM4ycG20Mm/zz12AAALLd9c8FUZXyZcTu4Ze6MtrlL+5gPddR7OijxT6lOq2dC1ciYPAGDh5fLlNA8Z5B6wlTwUsFL3R65oi2vguDL78i8AwNw9XOpToe3SYj95gO3zka9L3ZeW7/6c14wYAAA9eT5ZLhVOm/Y8NQAAAAAAAAAAAFgLu5rP25rPAADM2TGlnlv2beTIrpYH6u6OHF3qy9P7rukCAMAc5SugXizjZ6sNumse33FRdz98UfueyL5SXzmVx3480dUBAJjRpshdkZsjL0ROjlzS1Vv5fs20vYzOXNvRXe+J7Cyjtwc83tUBAJjROd31QKnv0NxQ6iG4/TPWhvJtBrkMenWvlo3b1shTkY1dLWfbsqkDAOAweL3UtxbcEtkb2VJqwzVJzrBNeqAg97j1ZdMHAMBh8mapy5e5Z234oEAuh97UXftyFg4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+j/4G8viDdZPTBFMAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAACmklEQVR4Xu2WTahNURiGP6HIT0QXIddPCoOb/AwkI8SAJKIURVFMbhSFdCQlIkSEkoGBmBgYiIGfAVFGTEQkUYSRAfLzvOdbq73Odu6xNwb31n7r6ey119r7+9b63rX2MatUqVKP0Hq4AcPyHd1Z/eBaQNc9RmPgNezPd3RXDYRRsBK+wWoYAX3TQagXtMMymGeN/YNhMYwN7QGwEGZA7ziooBSnw7IYak+GJZaz8Do4Ay/gC1yE4zAxGaMH78FZWAGH4BGMg/5wEnbCGzgAV2BN+NW784vQSgvgCFyF63AedsMWeAqzs6Gt/T0T3sF289lLg+AunDAPtAumwAc4Zz4ZaRZ8gvmh/Scp9mkYb754enZO6FNsLWpDjl35Wza6BQ+ssUzxvthkvgpL4bs1JjnX3H5bk3ut1AZ7YKj5u9Mk+5hX8Lm5letSsK/hN5XaP+z3CQ2HJ+Yv1yQkjXkJo0Nb0qR+wvLkXhFNgLewI7kXY6rSqnhd8qcG6oFUelCJ5ycUV7IW2tqMN81XRCsjxRWSfWSjMtJG135TnKgYUxaqK/pbgZWANtJBc/uoxJ/NT4coee0ovIdp4V6zFZJPZb9Tlk2mqJpVr2bu+enxRixBtIPK2mmeoBL+CItCn6RjTkmm5Ze/ZYlaaOvZffDY/OQpo7h/NGlNXpoEr2CbZQdE/WIvPINL4ToeX+rbbP6SC+ZH1B2YGvqjNGmtxn24DLfNbTIyHVRQsXoPzd+nuMptlSVJpxoSaCbZSR+puBFTxRWS3XQM6vTJj5NndZ63Que/4qenk+LqpCn7ESukZv7OK36dWxETbObv/y4ltNHc32ut64oVkZLW11oW0Vda12W+uKW0wRrLrf85f6t2OGbZuw6bV6JSpX/RL9mihcIszb9sAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABy0lEQVR4Xu2USygFYRiGP6EouRa5lFiQnXIpJSsbC5dQCgsbsVM2SlnJxtJGSWLlki0pC4qFsrJAWSFlJRsWyOV9zzd//fOf+U+ns9Q89TRnvvlnvnfmzHwiMTHpkw9H4Rpcho3hwykphDOi5y7AyvDhEOwzAYudeoIieAwXYQFshjdwyF7koRZewUmYB3vgHWy31pTCEbgJX+GDeMLOwUtYYtXG4C2ssGouOXAd7ge/DUvwSPTuCYP0w1a4I54gbM4QW069Db7BPqduUw+fRW/EZhC+wxanTtgnMkgTfJHkILwIL8a789ENfyQ5SC/8FX2qLt4gpqEviFu3MQ19Qdw68QYxJ7kN0wnCRlENMwrCtzzTILMS3TCjIL6GvrqNr6GvTrxBzJvvNjRB5p26TSf8kuSGJgi/HhdvEA6wU3ggOpAM/CI+g62Bg68KZgX71fAerpgFAVOiXyK/SBdvEDIOH2FdsM9GnLIXos1JmegE/YAdKdblwj24LeEhZ2CQJ1jjHiA8eRWewAHRi1+LjnoDn9yh6PjmWDcwAOu7oi/+BjyXcKNyeCY6IPmX0W/RQNPWugS8uwY4DLtEw6VLtug7xXO55X5MzP/kD0yBb8dRm/+pAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAiElEQVR4XmNgGAWjgGqAA4jTgJgHXYIcwAjErUBsjC5BLgAZ1AvELOgS5ACQ6wqAOA7KRgECQCxJIpYD4vlAPBmI+RiggBuIq4F4Fhl4BxB/BeJmIGZnoACYAPFqIJZBlyAVCAPxYiCWR5cgB2QBcQS6IDkAlGinArE0ugQ5AJQUeKH0KBhMAABVixNKp22j3QAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIUAAAAYCAYAAADUIj6hAAAFiElEQVR4Xu2ZacimUxjHrwlFdiO7zCDrB2SpkfiC+GDJ0gjDZLLEB1tRtl5JTUTWCFk+yJ58IKJmLDWWUsqWyJAowhc+DFn+v8599ZznvOec977v59W887r/9e95nnOf+2zX/1zXdc5jNmDAgAEDJsNG4g7izuI2ybMNHfHctkieDaiABXtPfExckTzb0IHIV4ovilcmzwZUgCiesOk7aTPxbPEh8TZx3/HHVWwlXm7h3Rst9FEC/Sy3yb1UrZ1DxUvSwhgXiK+LC9MH/1PkRLG1hTW6pSk/WPxUPD2qU8Ie4kfiheKm4oniF+IRUZ3txKXi4+Kv4jdWF04JbdupioJBvtyQ7/MFGI7FiJnu/BJyorhW/EDcNio7R/xM3DEqS7Gx+Ij4fPPdcav4qoXdDDDmKeJh4tNWNuZMaNtOVRS7id9ZGOR8wAHiKvENC64aXiqeIe4f1ashFQVCQBCUxThc/E08OSmPsaf4gwVRxThN/N2CcVLQT8mYXVBrJysK30lnin+KZ1lQ/CZxJWGBuEg8VTzaxp8TJ3GFuze/NxePt9AhWW4X0M9BNuqD3/uIJ1m7sEb95eKzFoQ+CVJRIKafm7IYzBPD1jbUseLfNl0UzOsfC94mRc2YXVBrJyuK8y3soq/FdeKT4j3iXlEdjLJGfNjCTrtd/NBCjMTt3S9eJ35vIaPFRTJJPmk7FVgNx4l3ii+Jr1nI/G8QL7Pp8TcHJvmChdg/KVJRuPFLokjLY7jxS6JIy0HNmF1QaycrClDLJ4hJP4rXWNiFYEvxbfE+C0a83ka7iLjp8RG3SpLDLmkD+n5QXGxBmLx7ZPOMvhFsbowO6nAaWJI+6IlUFG7A1PhtRIHRc8afs6Io5RMsxmrxfRt33V4OL7awe4mnf9m4AI6yEJKuispq4DLlJguxe7WNC4DkDM/zlZUTOsaF19rPwgKUWBJVilQUhMi+omANcsafs6LAkH80n2k5cTAVy/biJxYM5wtGnbXirs1vgGCYMMlUF+SSMu8TD4WnyoGxkEsQcjy5zBHjtkEqipLxS+UxSsYvlYOaMbug1k5RFOQDGAFjxGCgiCIVi3uAqeY3iSVZfnzc8p1NSGmb7Tsw2joL/Ti8T8JKCXgAPEUszEmQisLFmhrfRcE6luDjT43voshtnJoxu6DWTlYUnk9gVIxLUkhcJqTg8pgsLzoWiHeJP4kHNmW5nb3YQkh6wMbP5W2Q8zpTFnKMQ6KyHPBOcDaQisLDZprX5Dwtie4uNsrDmMtamy5qxlraODVjYifCbZskvtZOVhTulj1EoNgrLEyGF34RT2ieAY6aCCBWNvkEap9qfvMuN34fWzihdIEvPIJCWGBv8VvxahstcgkYgxCSerc+SEUBzrUwFh+bz/VdG514yL+4ucTbLanUw6CM9SnLbxz6Zh1yR+s7bHzNa6i1kxUFg71Z/FJ8pvnu6uMZFz40SMMcE9+ycDEUA0Gxi5nwc+KbFkLHTnGllnCvwyUR7dEvY1tqMwvCwW0eC00o4Wjd9a7EkRMFa4P3W2XhzgZDs6m47nZQ/xULR+h4UyAGyllnQuSj4js2bix2P3kTl2EYHZLAY4PYeGxcQjsePh6fo207WVE4+LMk94cJwFWyQLnOY5fKUZRdktZjAdJkL+VKC/3Hpxj6ZXJ9jIqAmPDd4ucWdrdzWVSvhpwoAG1zd8OdTXqRNxOYC+PiXT77zM2Bl7/XQtjvi6oo+iKXT6RgUVngGt34uXxifaEkirkCQhP3R5Ng1kXBYl1kwTWdZ2VP0wYIAldP2FjTfO+yA/8LzGVRMCbCjyf7fTHrolhh4yGA/036YpEFV+9tkUhhlPWJuSwKTivHpIU9MOuimO/wZI08hP9j5hPiuQ2iGDBgQEf8C47GTcB+ZhkSAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGsAAAAYCAYAAAD9CQNjAAAENklEQVR4Xu2YW4hNURjHP6HIvSl3GRIpQgwpUULILWNKeFPyIKFQnhTKZXhwiaQkyaV5xANiSsmlFDWmSCaJUjwIueTy/1t7Net8Z62z1zlnLoz9q3+n/e2z9/7W+u/1rbWXSEZGxv9HZ6g/NAjqq851VNhOtrcC6qTO/dUw6XvQaWitOtdRYTvZ3ktQT3XuD7Oh59DLSM0xl7U6NOuM5CfdHVoFnYT2Q2NyT0cxCZqng0XAZ/LZzIG5MKdi4cjZBK1XcbZ3d/KbAy84Cl2EKpNjcgr6Cc1PjlmSZkFNUFUSa218ZvWBrkO7kvhE6AlU7fwnxDRoG/QA+gVtzz0dDZ/FZ/LZzIG5MCfmVgzM55Pk5xE0a4CYIce5wdJPTIOaoCFOnBefhYY6sTTceceKx4yn4TOLDWNuzNGyGmoU05ZCsHMWQwvF30kxDIOeiXmmxfbXBieWBo29Kf6XJmgWS9pmFZsAfYDqoC5OnEkdhno5sRAsC1uhBjEdbkvWSmhBcj4NbZbtFMZcONI/QktUPMRkKd0smsRreQ8Lq9E5qF48HeyB/98CHRB/HkGzaqDRKsaEfI5zhbJO0lcpA6HLYmp5zAgKoc0aC71LYi628/eoeIhyzOLLqs0izOkNNFLFfXCE10LTxZ9H0CwfnK++QzP0iQg4Eo9J3ByShjbLdnLILB0PUY5ZfEbILF9cw/LHKlMp4TyizQrNV7GME2OWWz5LRZvF+YYjXpvSVmYxj3rxmxJjFisSp5ya5DiUR7RZ9gZ1UlqHs0M3Su6CQqtC0ksp0WZxrmtPs3pAN8RvSoxZU6GDUNfkOJRHtFmh+SoWmnVNzFAPaa/E7Uhos0KmhOIhQp0UQ8iUUNzCNpwQU/4soTyizOLbzq/nUucr4ltdloo2i5M3J3Ftim30DhUPEeqkGLiI8ZnCnF5J+LNmPPRYcjcX3ooZGFzJupsNUWaVO18RfkOdT37LRZtl54wrULckRtjIb8mvhRP5YPGX20Jm2e9C9/4u/Dz4IbnP4n+Zk5sX71PpHPsI5RFllr241PnKUi1mRckaXw7aLLJGzFs4IjmmGdxBuCvNOwgV0CPoq5jlsabQSOTeHN92vnC+PuC970M7ndgoMaOK35CWtPuQKuiz5OcRNIvfWA+h92JubsWh2Sj5wz0GduAiMZuwy8Xz0Eh8ZnFyPg7dgpaJMapBzNaPhf+/Cj2Fhjtx7sG9ltx2st23pbkScM79ImavlM/3MQV6IWbraoWYarRPmhcOxN6H/9Gjm8+6I2aE2jxYEosqgy1NbzFvGFdQbq3mPhrf0DR8ZhE2ni8ZO2qm5HZSS8DSdUQKb1+xanAjmC8Mt6BaknYxq1xCZrU2LLG1Ei5frU1mViRcGNCoufpEG5KZFQmfuVTy55m25J80ixMxJ3/Oc4fUuY4K28n2XpDyV9MZGRmp/AbVuAFKypqBTgAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAXCAYAAABH92JbAAAC2ElEQVR4Xu2XzatNURjGX0kRJRmIyGdkIAoDioEkBiRX+fwLpChFGZ3CgFIyUSYyMDRkQtwyESIKRfKRyIARA0o8v7P2ctZee+19996dwU37V0+3+659znrfZ6/1rnXMOjo6OsYnU6T90mXpnLQsP1wJz/IZPntMmp0frs0maXEcbECYB7VQUxMWSHvjoGe6dEs6JU2TVkkvpJHwoRJ45rq0UtogPZF+SbvCh2pyQtoeB2tCHuRM7tRALdREbVUslw5Jd6Tf0tX88ACSeyjNCGIHpJfSrCAWw9g9abM0IYstkT5Jr6V5WawubU1iHuYjZw+1UNPhIJYCk3ZK66WPVmKS/7J4cK30XdoRxUNWSz+kNzYwE7OuSX+kbVmsLm1NwhzyIB+Pz2PU3MoaC1rEeyv60Acnv1px0BtwJoqHsJTZalcsnwjfhUlNC25r0kUrmgTk8VlaFMVTVJrkzYgHy+Jj4VfmF2vW/KGtSeRYZlIqnqLSJJLirceDbU3aY64BHrdBn4rZam6LfojE9v6WiD+WlvY/WYQVPGppM4ZmEn1jWCbNN9fse9Kk/FAt2qykqdJtS5sxNJPKzCiLl0F/uiEdkSZGY3VpYxKUmVEWT1FpEk2N5hYPepNORvEU3iC2mt9iG83dWZrQ1iQOl5QZ1MSxPjeKp6g0ye9pipwcxLn7cCnkrwcz5li+17CtLljx8nja3DWiCW1N4ppCHwxzpRZqCusiV1pCqhVUmgQHzTXIhdn/mMCN9b4NbqwzpafST2ldFmOyXhYLGy1v753VO3pD2ppEbg/M5eLhUkse4c+M8+b6by+IebxJ3K3CRfAPir0k3TV3+8Sg55bfLqy4m9Irc28D/JZk4ljxDT5knxVPsKrT7ZG5oqtYI701d6ruNjf/WcuvmqPmdsdIEGP1YSYr0edOHs+kFcFzfXCPY5YJ6CepJTne4aTbYu5FN/1J1NHR0dHxv/AXcP+4n1RRAJsAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAXCAYAAABH92JbAAACqElEQVR4Xu2Xz4vNURjGHwtFlKRGRMaPRhZK+RULC01iQWoov/4AyUIWlNWULAgLG8VCFsofMGyIWQopCkUSiRUrFkg8z33ndM99v+d858xtujPl+9Sn233Pvff7vs95z48LNGrUqNH01GxyiFwjF8jqzuFiLSO3SZ8fKNB2stIHJyDlrNxVg2pRTRNRPzngg0HzyD1ylswl68grMhR/qEAzyQ3ygSxyYyU6TXb7YKGUq3JW7qpBtagm1VanNeQYeUD+kJudw20puSdkfhQ7TF6ThVFsPO0nv9F7k5aSt7Ccg1SLajoexVKSSXvJVvIJGZPCj/nBjeQ72ePiOfXD2nwEvTdJ5vwg66PYDHKLjMI6azwpX+XtfWhJTn5FdVAP1IPPuXhKWmaXyCbY7/TapCuomiQply9khYunVGtSMMMP5uIpaT84CZu9qTBJz8yZlIqnVGuSkvqL6mCpSTrNrqO9QZaYtJO8Ix8dWt7fEvFnZKD1zaq0lEaRNmPSTNqF7k3SMrtINkexEpNy6qaT5pD7SJsxaSblzMjFY2lTD8ssqNcmSTkzcvGUak3SpqbNzQ8Gk864eKzLqC6NX7DO/Ay7VGqmS9WtSTpcUmaoJh3rS1w8pVqTwpq+Q2ZF8UFYwXoN0r6zGJ2d4zUVnaSO1kUwzlW1qKa4Lm0P2kP16lVrknQE1gXLx97LBN1YH6G9IS8gz8lPsmUs5qXv6W5SOnte3Zqk3B6T4Si2CpZH/DdD1xR1+XAUCwomKf9kE8jZq+Qh7PYpg17CrvhB6ri75A1sNryOwk4mJSHUhbnldhDVZVp3uj2FFV2nDeQ9OUX2wS7I59HZNSdgeQ1FMXWfzFQnhtyVxwuyNvpcS3JPx6wesA3plpzu0oTsgE20/qo0atSoUaP/Uf8AGAy1Je9T2wIAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAYCAYAAACldpB6AAACdUlEQVR4Xu2WTYhNYRjH/5NRxPiONCZDIZuxEDZqaigs1CRW7ISlsLCa1SSb2ZCIppEsZkbW0ySraRZKKaXUlLqkhFDCRuH/7znvOe957z1n7iXnUu+vfp17nvP1vs99Px4gEolE6hmgL+irJt1vj1XKMnqW3qJDdH3+ciEddA+9Sq/TY3RJ7g7YTdfoJO1NzsUo/UEPJucLaD+t0V1JrCo20qf0FF1ED9E5utu/qQGd9Aq9RDfQLXSGPoO9M2UdvUfXerGV9DGsw91efCm9C3thVagj+kPuJ78d6tg0XezFQrbTD/QBrO3iOP0JGxkpGtrn/ADZQT+j/sNKjh7u8mJ/m830Db0YxI/Qr3RnEPfZRt/CRtHqJKbnlAQlNkVzZKsfQJat8MN60WlkU6YK9CdpWoZtOQxro9paxgpko8BNfb1PyShFWfpO94YX2oDrbFESwngRSsAB+pGO0IX5y3mK1oNmWAWbp+FuUqZW+jLUyUadbSUJGk361jt6k67JX65Hc0xzLVwP2sV5NO5sK0lwaIe7TD9hnm2+aD1oF0WdLYrPh7b3b7Bt0t8RUzRvbuP31wM9r8VThUyzauEqQ+1Qe8LOuiSULXB9sN1MR4e++RIlO8ufrAdCi80+erQFVc2VoXbUEOzr5AysBlAt4FDN47ZCcQeWKB0dbrqHz6b8a+uB0Ogapo/o8iSmZKvAG0fWTlcTvKabkphGz3tkVa84AUvMDXh9VI3wBLZ16KLzC32OgiFTMer8FKy0V8k8RmeRr1z1W0XRQ2TJ0lGJUuwkvQDr14R3z3+FVnb9IZpCOuq8GTSSeulgYk/uaiQSiUQiDfkF/VuPy7vxkQoAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAYCAYAAACldpB6AAAChElEQVR4Xu2WTahNURiGX6HIP4Ub8lPcjAyEklJcxYCBnwk3mWCiK1cxoZRMDGUkkRTdy8REyNBAkVJEok66JYmBGMnP+55vrXPWWfusYyP7UOupp3v2t9fad/1+awGZTCZTZB19RV+XtM+qVcpkepCeo8dpT+vrjvTS07C6/XRC62tgFD1Lh+gC9yzO0290o3seTdfSGl3hYlUxnz6me+k4uom+oCvDQgm20WG6lM6lA/QB7JsNZrlCM4PYNFjBGp0TxCfSy7CPVcUY2IRcd789p+gtOj6IxahvKhO210/6iSBWX9qHwgBZRj+i+I81OGfopCD2t1lE39CjUXwr/UyXR/EQvXtOF0dxfUsD22AHXRIGyC76HcV/PIPuQ3PLVIEmSdsybstmWBvV1hR+AF/CtrKYQu/QLb5QCo3SF7omftEFfGdTgxDHQzRZR2Dl5EV6gx5275Kk8kEZpsP2YHyadFKZvhPqZLvOlhkEoYSu/OEH4h3dgJ8MgvaR9lqcD7rFINp3tswgqKMHYMt/NWwVqM5XujsoVyCVD7pFqrOpeIhm/Amd5561KnbCJvkZ7PQooJHTvvndfKD6Sp66yJR1ar1mGrVD7Yk76wdBp0QK5TadZjG6+3xA4mT5k3wgxtL1dPsvuKpeM43aUUOxM/vpe9glyKOZ1SR4LsHyQYzuDQ/RWrfBv5YPhFbXSXofdrwJDbYueFfRbKeuxm/pCF3oYjoGn8Juwh59bw+9gqCPuiM8gi0Pn0HlJ9i+abtkKkadvwm72uvKfIHeQ+tNUL91tb6L5mApB+iI1Io5Bst31+htOtuV+a9QhzQh2kL6q+eyaIsoh6iuJr3j8ZjJZDKZDPADOhOQR7oarTIAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABrUlEQVR4Xu2UzStFURTFt1AUSb4iEomMkAxMJCEGJIwwV6YGSqlXmPgHlEz8BUZMDIiJUmSAEgMSI0wQysdad5/z3rnn9R4yUd6qX+/ddc/Z+559974iKf1lZYMRsAQWQF34dlKVghnRvZOgInxblQc2wCzIAY3gGAy5ixKoD6yDBlAkGuMZDLqLqCmwB/IdbxScgBLH85UF1sA9aDJeNbgBR6DQeEFgJlixhlELeAD9nu+KSVbBG2g3Xjm4AufiPGA9uJX4JM3gEcx7vi8mKgBp5roHvINlkGEX2WCJkvh+MvEU26KV4f+o+OI+JD7YT5IUgx3RMu2LNoE9WaBe+X0SVx3gCURApjUTBUvkf6Vc0VOxGbqsaVvOD2aTTHu+K85XRLTd3fIwFqvD0QjE4dsS7Xd2ilUneDW/VgxaJrGA9n1eiE49ZePRHzdeoDFwCarMNYNwcndFA1Ns00PwAlqNx9PegTmJ1b8GXIsOcqjDuGARbIIB0QScWH5erPiE/Hycgkrj8WEmRNeyrCzbATiT8N6ouKEWDIM2cTrjG+Jpu0X38nTp4dsp/Ut9AnLuU6pRF8+8AAAAAElFTkSuQmCC>