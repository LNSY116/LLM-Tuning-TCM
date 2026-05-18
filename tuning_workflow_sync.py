import os
import json
import time
import pandas as pd
import itertools
from pathlib import Path
from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt

# =============================================================================
# 🔬 LLM Parameter Tuning Framework for TCM Tongue Diagnosis
# 
# 📌 專案定位：參數調優實驗工具（非產品端）
# 🔗 主應用系統：https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis
# 
# 🔄 協作流程：
#   1. 從主應用取得「黃金 Prompt」與測試圖片
#   2. 本框架執行網格搜尋 + 自動化評分
#   3. 將最佳參數回填至主應用的 assets/config/ 或 prompts/
# 
# 📄 授權：本框架僅供內部研發使用，模型權重與主應用程式碼請參閱主 Repo
# =============================================================================

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
    raise ValueError("請在 .env 設定 GEMINI_API_KEY")


GENERATOR_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-2.5-pro"
ROOT = Path(__file__).resolve().parent

# 評分權重配置
WEIGHT_ACCURACY = 0.6
WEIGHT_STABILITY = 0.4

def find_asset(rel_path: str, allow_archive_fallback: bool = True) -> str | None:
    """Return a filesystem path to an asset."""
    candidate = ROOT / "assets" / rel_path
    if candidate.exists():
        return str(candidate)
    if allow_archive_fallback:
        fallback = ROOT / "archive" / "Docs_and_Assets" / "assets" / rel_path
        if fallback.exists():
            return str(fallback)
    return None

def find_test_images() -> list[str]:
    """Return a list of filesystem paths to test images."""
    candidate_dir = ROOT / "assets" / "test_images"
    if candidate_dir.exists():
        images = list(candidate_dir.glob("*.jpg")) + list(candidate_dir.glob("*.jpeg")) + list(candidate_dir.glob("*.png"))
        if images:
            return [str(p) for p in images]
            
    # Fallback
    fallback_img = find_asset("MyTongue.jpg", allow_archive_fallback=False)
    if fallback_img:
        return [fallback_img]
    return []

IMAGE_PATHS = find_test_images()

# PROMPT: prefer consolidated system_prompt.md in assets/prompts/ then fallback to archive
def find_prompt():
    p = find_asset(os.path.join("prompts", "system_prompt.md"))
    if p: return p
    p = find_asset(os.path.join("prompts", "system.current.md"))
    if p: return p
    return find_asset(os.path.join("prompts", "system.default.md"))

PROMPT_PATH = find_prompt()
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TCM_KEYWORDS = [
    "舌質老嫩", "舌苔厚薄", "色澤榮枯", "津潤乾燥",
    "病機分析", "證型建議", "養生調理",
    "氣虛", "陽虛", "陰虛", "血虛", "痰濕", "濕熱", "氣滯", "血瘀", "化熱"
]

if not PROMPT_PATH:
    raise FileNotFoundError(
        "找不到提示詞檔案。請確保 archive/Docs_and_Assets/assets/prompts/ 中存在 system.current.md 或 system.default.md"
    )

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    BASE_DOCTOR_PROMPT = f.read()

PROMPTS = [
    {"prefix": "V1", "desc": "醫生原版", "prompt": BASE_DOCTOR_PROMPT},
    {"prefix": "V2", "desc": "思維鏈(CoT)版", "prompt": BASE_DOCTOR_PROMPT + "\n\n【額外指示】：請在輸出報告之前，先在最開頭寫出你的 <思考過程>，一步步解釋你是如何根據對應表得出這個結論的。"},
    {"prefix": "V3", "desc": "極度精簡版", "prompt": BASE_DOCTOR_PROMPT + "\n\n【額外指示】：請嚴格遵守上述格式，去除所有多餘的客套話（如'好的'、'這是一份報告'等），直接輸出條列式結果。"},
]

CONFIG_GRID = [
    {"temp": 0.0, "top_p": 0.1},
    {"temp": 0.0, "top_p": 0.9},
    {"temp": 0.1, "top_p": 0.5},
    {"temp": 0.3, "top_p": 0.5},
    {"temp": 0.5, "top_p": 0.9},
]

# 動態 N (測試穩定度)
def get_repeat_count(temp: float) -> int:
    if temp == 0.0:
        return 1
    elif temp >= 0.5:
        return 5
    else:
        return 3

# --- API 呼叫 ---
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_inference(client, uploaded_image, prompt_text, temp, top_p):
    return client.models.generate_content(
        model=GENERATOR_MODEL,
        contents=[uploaded_image, prompt_text],
        config=types.GenerateContentConfig(
            temperature=temp,
            top_p=top_p,
            max_output_tokens=2048
        )
    )

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
def _do_call_judge(client, prompt):
    res = client.models.generate_content(
        model=JUDGE_MODEL, 
        contents=prompt, 
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    )
    return json.loads(res.text)

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
        return _do_call_judge(client, prompt)
    except Exception as e:
        print(f"⚠️ LLM Judge 評分失敗: {e}")
        return {"causal": 3, "language": 3}

# --- 執行主程式 ---
def run_tuning_experiment(api_key, image_paths, prompt_path, output_dir="outputs"):
    print("🚀 初始化即時 (Sync) API 測試環境...")
    client = genai.Client(api_key=api_key)
    
    # 1. 上傳圖片
    if not image_paths:
        raise FileNotFoundError(
            "找不到實驗用影像。請在 assets/test_images/ 資料夾放入測試照片，或放置於 assets/MyTongue.jpg"
        )

    print(f"🖼️ 上傳 {len(image_paths)} 張測試照片...")
    uploaded_images = []
    for path in image_paths:
        print(f"  └─ 上傳: {os.path.basename(path)}")
        with open(path, "rb") as f:
            uploaded_img = client.files.upload(file=f, config=types.UploadFileConfig(mime_type="image/jpeg", display_name=os.path.basename(path)))
            uploaded_images.append(uploaded_img)
    print("✅ 所有照片上傳完成")
    
    save_path = os.path.join(output_dir, "doctor_sync_tuning_results.csv")
    results = []
    
    # --- 增量邏輯：載入歷史紀錄 ---
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path)
            results = existing_df.to_dict("records")
            print(f"📦 載入過往實驗紀錄，共 {len(results)} 筆，將略過已完成之組合。")
        except Exception as e:
            print(f"⚠️ 無法讀取舊紀錄: {e}")

    # 讀取提示詞
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"找不到 {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    prompts_to_test = [
        {"prefix": "V1", "desc": "醫生原版", "prompt": base_prompt},
        {"prefix": "V2", "desc": "思維鏈(CoT)版", "prompt": base_prompt + "\n\n【額外指示】：請在輸出報告之前，先在最開頭寫出你的 <思考過程>，一步步解釋你是如何根據對應表得出這個結論的。"},
        {"prefix": "V3", "desc": "極度精簡版", "prompt": base_prompt + "\n\n【額外指示】：請嚴格遵守上述格式，去除所有多餘的客套話（如'好的'、'這是一份報告'等），直接輸出條列式結果。"},
    ]

    def is_completed(prompt_desc, temp, top_p):
        for r in results:
            if r.get("提示詞") == prompt_desc and abs(float(r.get("T", -1)) - temp) < 0.01 and abs(float(r.get("P", -1)) - top_p) < 0.01:
                return True
        return False

    exp_idx = 1
    total_configs = len(prompts_to_test) * len(CONFIG_GRID)
    
    print("\n🔥 開始進行組合測試...")
    for p in prompts_to_test:
        for c in CONFIG_GRID:
            exp_id = f"EXP{exp_idx:02d}"
            repeats = get_repeat_count(c["temp"])
            
            if is_completed(p['desc'], c['temp'], c['top_p']):
                print(f"[{exp_idx}/{total_configs}] {exp_id} - {p['desc']} | T={c['temp']}, P={c['top_p']} | ⏩ 略過")
                exp_idx += 1
                continue
            
            print(f"[{exp_idx}/{total_configs}] {exp_id} - {p['desc']} | T={c['temp']}, P={c['top_p']} | 每圖推論次數={repeats}")
            
            total_accuracy = 0
            total_stab = 0
            total_coverage = 0
            valid_images_count = 0
            
            for img_idx, uploaded_img in enumerate(uploaded_images):
                print(f"  └─ 圖片 {img_idx+1}/{len(uploaded_images)} 處理中...")
                responses = []
                for i in range(repeats):
                    try:
                        res = generate_inference(client, uploaded_img, p["prompt"], c["temp"], c["top_p"])
                        responses.append(res.text)
                        print(f"      └─ run {i+1}: 成功")
                    except Exception as e:
                        print(f"      └─ run {i+1}: 失敗 ❌ {e}")
                    
                    time.sleep(2)
                    
                if responses:
                    print("      ⚖️ 正在進行 LLM Judge 評分...")
                    rule = rule_based_score(responses[0], TCM_KEYWORDS)
                    stab = calculate_stability(responses)
                    judge = call_judge(client, responses)
                    
                    accuracy = (rule["rule"] + rule["format"] + judge.get("causal", 3) + judge.get("language", 3)) / 4
                    total_accuracy += accuracy
                    total_stab += stab
                    total_coverage += rule["coverage"]
                    valid_images_count += 1
            
            if valid_images_count > 0:
                avg_accuracy = total_accuracy / valid_images_count
                avg_stab = total_stab / valid_images_count
                avg_coverage = total_coverage / valid_images_count
                final = (avg_accuracy * WEIGHT_ACCURACY) + (avg_stab * 5.0 * WEIGHT_STABILITY)
                
                results.append({
                    "實驗編號": exp_id,
                    "T": c["temp"],
                    "P": c["top_p"],
                    "提示詞": p["desc"],
                    "術語覆蓋": f"{avg_coverage:.1%}",
                    "一致性": f"{avg_stab:.2f}",
                    "綜合評分": f"{final:.2f}"
                })
            
            exp_idx += 1
            print("-" * 40)

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    # 產出 best_config.yaml
    if not df.empty:
        df["綜合評分_num"] = pd.to_numeric(df["綜合評分"], errors='coerce')
        best_row = df.sort_values("綜合評分_num", ascending=False).iloc[0]
        
        yaml_content = f"""# =========================================================
# 🎯 最佳參數配置 (由 LLM Tuning Framework 自動產出)
# 請將此配置回填至主應用的 assets/config/ 或 prompts/
# =========================================================

best_parameters:
  prompt_version: "{best_row['提示詞']}"
  temperature: {best_row['T']}
  top_p: {best_row['P']}
  score: {best_row['綜合評分']}
"""
        yaml_path = os.path.join(output_dir, "best_config.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
    
    return df, save_path

def main():
    try:
        df, save_path = run_tuning_experiment(API_KEY, IMAGE_PATHS, PROMPT_PATH, OUTPUT_DIR)
        print(f"\n✅ 實驗完成！報告已儲存至: {save_path}")
        print(df.to_string())
    except Exception as e:
        print(f"❌ 執行出錯: {e}")

if __name__ == "__main__":
    main()
