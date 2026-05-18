import os
import json
import time
import pandas as pd
import itertools
from google import genai
from google.genai import types

# --- 配置區 ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get("GEMINI_API_KEY")
    except ImportError:
        pass

if not API_KEY:
    API_KEY = None # 請在 .env 檔案中設定您的 GEMINI_API_KEY

MODEL_NAME = "gemini-2.5-flash"
IMAGE_PATH = "MyTongue.jpg"
PROMPT_PATH = "醫生提示詞參考.txt"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TCM_KEYWORDS = [
    "舌質老嫩", "舌苔厚薄", "色澤榮枯", "津潤乾燥",
    "病機分析", "證型建議", "養生調理",
    "氣虛", "陽虛", "陰虛", "血虛", "痰濕", "濕熱", "氣滯", "血瘀", "化熱"
]

# 讀取你的黃金提示詞
if not os.path.exists(PROMPT_PATH):
    print(f"❌ 找不到 {PROMPT_PATH}！請確保檔案在同一資料夾中。")
    exit(1)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    BASE_DOCTOR_PROMPT = f.read()

# 定義微調的 Prompt 組合
PROMPTS = [
    {"prefix": "V1", "desc": "醫生原版", "prompt": BASE_DOCTOR_PROMPT},
    {"prefix": "V2", "desc": "思維鏈(CoT)版", "prompt": BASE_DOCTOR_PROMPT + "\n\n【額外指示】：請在輸出報告之前，先在最開頭寫出你的 <思考過程>，一步步解釋你是如何根據對應表得出這個結論的。"},
    {"prefix": "V3", "desc": "極度精簡版", "prompt": BASE_DOCTOR_PROMPT + "\n\n【額外指示】：請嚴格遵守上述格式，去除所有多餘的客套話（如'好的'、'這是一份報告'等），直接輸出條列式結果。"},
]

# 針對嚴格規則特製的收斂網格 (不測 > 0.5 的溫度，避免破壞規則)
CONFIG_GRID = [
    {"temp": 0.0, "top_p": 0.1},
    {"temp": 0.0, "top_p": 0.9},
    {"temp": 0.1, "top_p": 0.5},
    {"temp": 0.3, "top_p": 0.5},
    {"temp": 0.5, "top_p": 0.9},
]

# 動態 N (測試穩定度)
def get_repeat_count(temp: float) -> int:
    return 1 if temp == 0.0 else 3

# --- 評分功能 ---
def rule_based_score(text, keywords):
    found = sum(1 for k in keywords if k in text)
    coverage = found / len(keywords) if keywords else 0
    fmt = 5 if "中醫體質" in text and "警語" in text else 2
    rule = 5 if coverage > 0.2 else 3
    return {"format": fmt, "rule": rule, "coverage": coverage}

def jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    set1, set2 = set(s1), set(s2)
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 1.0

def calculate_stability(responses):
    if len(responses) < 2: return 1.0
    sims = [jaccard(s1, s2) for s1, s2 in itertools.combinations(responses, 2)]
    return sum(sims) / len(sims) if sims else 1.0

def call_judge(client, responses):
    if not responses: return {"causal": 3, "language": 3}
    prompt = "你是一位客觀的AI評分裁判。請對以下中醫診斷模型的多筆輸出進行綜合評分（1-5分，5為最佳）：\n1. 推導因果 (causal)\n2. 語言適切 (language)\n\n"
    for i, res in enumerate(responses): prompt += f"--- 輸出 {i+1} ---\n{res[:500]}...\n"
    prompt += "\n請嚴格回傳 JSON 格式：{\"causal\": 平均分, \"language\": 平均分}"
    try:
        res = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"))
        return json.loads(res.text)
    except: return {"causal": 3, "language": 3}

# --- 執行主程式 ---
def main():
    print("🚀 初始化即時 (Sync) API 測試環境...")
    client = genai.Client(api_key=API_KEY)
    
    # 1. 上傳圖片
    print(f"🖼️ 上傳舌象照片 ({IMAGE_PATH})...")
    with open(IMAGE_PATH, "rb") as f:
        uploaded_image = client.files.upload(file=f, config=types.UploadFileConfig(mime_type="image/jpeg", display_name="sync_tongue_image"))
    print(f"✅ 照片上傳完成: {uploaded_image.uri}")
    
    results = []
    exp_idx = 1
    total_configs = len(PROMPTS) * len(CONFIG_GRID)
    
    print("\n🔥 開始進行組合測試...")
    for p in PROMPTS:
        for c in CONFIG_GRID:
            exp_id = f"EXP{exp_idx:02d}"
            repeats = get_repeat_count(c["temp"])
            print(f"[{exp_idx}/{total_configs}] {exp_id} - {p['desc']} | T={c['temp']}, P={c['top_p']} | 推論次數={repeats}")
            
            responses = []
            for i in range(repeats):
                try:
                    # 即時呼叫 API
                    res = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[uploaded_image, p["prompt"]],
                        config=types.GenerateContentConfig(
                            temperature=c["temp"],
                            top_p=c["top_p"],
                            max_output_tokens=2048
                        )
                    )
                    responses.append(res.text)
                    print(f"  └─ run {i+1}: 成功 (長度 {len(res.text)})")
                except Exception as e:
                    print(f"  └─ run {i+1}: 失敗 ❌ {e}")
                
                # 為了避免觸發 API Rate Limit，每次呼叫完停頓 3 秒
                time.sleep(3)
                
            # 評分計算
            if responses:
                print("  ⚖️ 正在進行 LLM Judge 評分...")
                rule = rule_based_score(responses[0], TCM_KEYWORDS)
                stab = calculate_stability(responses)
                judge = call_judge(client, responses)
                
                accuracy = (rule["rule"] + rule["format"] + judge.get("causal", 3) + judge.get("language", 3)) / 4
                final = (accuracy * 0.6) + (stab * 5.0 * 0.4)
                
                results.append({
                    "實驗編號": exp_id,
                    "T": c["temp"],
                    "P": c["top_p"],
                    "提示詞": p["desc"],
                    "術語覆蓋": f"{rule['coverage']:.1%}",
                    "一致性": f"{stab:.2f}",
                    "綜合評分": f"{final:.2f}"
                })
            
            exp_idx += 1
            print("-" * 40)

    # 產出最終報告
    df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, "doctor_sync_tuning_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print(f"\n🎉 實驗完成！報告已儲存至: {save_path}")
    print(df.to_string())

if __name__ == "__main__":
    main()
