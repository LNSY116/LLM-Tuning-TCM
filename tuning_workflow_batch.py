import os
import json
import time
import httpx
import pandas as pd
import itertools
from google import genai
from google.genai import types
from typing import TypedDict

class Experiment(TypedDict):
    id: str
    temp: float
    top_p: float
    prompt: str
    desc: str

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("請設定 GEMINI_API_KEY 環境變數。")
# 根據資料夾結構，MyTongue.jpg 位於專案根目錄
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "MyTongue.jpg")
MODEL_NAME = "gemini-2.5-flash"
REQUESTS_JSONL = "batch_requests.jsonl"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

TCM_KEYWORDS = [
    "舌質老嫩", "舌苔厚薄", "色澤榮枯", "津潤乾燥",
    "病機分析", "證型建議", "養生調理",
    "氣虛", "陽虛", "陰虛", "血虛", "痰濕", "濕熱", "氣滯", "血瘀", "化熱"
]

# --- Prompt Versions ---
BASE_PROMPT = """你是一個專業的中醫舌診辨證系統。
請觀察圖片中的舌象，並給出診斷報告。
輸出格式：
1. 主要中醫體質
2. 次要中醫體質
3. 體質說明
4. 警語：此為AI自動生成，不具醫療建議。"""

V2_PROMPT = BASE_PROMPT + """
請在分析中加入以下專業術語描述：
- 舌質老嫩
- 舌苔厚薄
- 色澤榮枯
- 津潤乾燥
並要求輸出包含：
- 病機分析（說明致病機轉）
- 證型建議
- 養生調理方向"""

V3_PROMPT = V2_PROMPT + """
請確保報告內容詳盡，語言風格應嚴謹且具臨床參考價值。
針對「病機分析」，請從氣血陰陽的角度深入剖析。
針對「養生調理」，請包含飲食、作息與穴位建議。"""

# 💡 優化策略 1: 精簡網格搜索 (Fractional Factorial)
# 取代 3x3=9，選用 5 組具代表性的組合
CONFIG_GRID = [
    {"temp": 0.0, "top_p": 0.1},
    {"temp": 0.0, "top_p": 0.9},
    {"temp": 0.3, "top_p": 0.5},
    {"temp": 0.7, "top_p": 0.1},
    {"temp": 0.7, "top_p": 0.9},
]

PROMPTS = [
    {"prefix": "V1", "prompt": BASE_PROMPT, "desc": "基礎版"},
    {"prefix": "V2", "prompt": V2_PROMPT, "desc": "專業加強版"},
    {"prefix": "V3", "prompt": V3_PROMPT, "desc": "全面分析版"},
]

EXPERIMENTS: list[Experiment] = []
exp_idx = 1
for p in PROMPTS:
    for c in CONFIG_GRID:
        EXPERIMENTS.append({
            "id": f"EXP{exp_idx:02d}",
            "temp": c["temp"],
            "top_p": c["top_p"],
            "prompt": p["prompt"],
            "desc": f"{p['desc']} (T={c['temp']}, P={c['top_p']})"
        })
        exp_idx += 1

# 💡 優化策略 2: 動態多重推論
def get_repeat_count(temp: float) -> int:
    return 1 if temp == 0.0 else 3

# --- Metrics / Scoring ---
def jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    set1, set2 = set(s1), set(s2)
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 1.0

def calculate_stability(responses: list[str]) -> float:
    if len(responses) < 2:
        return 1.0  # T=0.0 時只有 1 次，預設完全穩定
    sims = []
    for s1, s2 in itertools.combinations(responses, 2):
        sims.append(jaccard(s1, s2))
    return sum(sims) / len(sims) if sims else 1.0

# 💡 優化策略 4: 混合評分機制 (Rule + LLM)
def rule_based_score(text: str, keywords: list[str]) -> dict:
    found = sum(1 for k in keywords if k in text)
    coverage = found / len(keywords) if keywords else 0
    # 簡單的正則/關鍵字判斷格式
    fmt = 5 if "中醫體質" in text and "警語" in text else 2
    rule = 5 if coverage > 0.2 else 3
    return {"format": fmt, "rule": rule, "coverage": coverage}

# 💡 優化策略 3: 裁判模型批次評分
def call_judge(client, responses: list[str]) -> dict:
    """呼叫 Gemini 作為裁判進行語言與因果評分"""
    if not responses:
        return {"causal": 3, "language": 3}
    
    prompt = "你是一位客觀的AI評分裁判。請對以下中醫診斷模型的多筆輸出進行綜合評分（1-5分，5為最佳）：\n"
    prompt += "1. 推導因果 (causal)：內容是否具明確病理關聯？\n"
    prompt += "2. 語言適切 (language)：用詞是否專業、無幻覺矛盾？\n\n"
    for i, res in enumerate(responses):
        prompt += f"--- 輸出 {i+1} ---\n{res[:500]}...\n"  # 取前段防超長
        
    prompt += "\n請嚴格回傳 JSON 格式：{\"causal\": 平均分, \"language\": 平均分}"
    
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        scores = json.loads(res.text)
        return {
            "causal": scores.get("causal", 3), 
            "language": scores.get("language", 3)
        }
    except Exception as e:
        print(f"裁判評分失敗: {e}")
        return {"causal": 3, "language": 3}

# --- Step 1: Upload image ---
def upload_image(client, path):
    print(f"上傳圖片到 File API: {path}")
    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config=types.UploadFileConfig(
                mime_type="image/jpeg",
                display_name="tongue_image",
            ),
        )
    print(f"  圖片上傳完成，URI: {uploaded.uri}")
    return uploaded.uri

# --- Step 2: Build JSONL ---
def build_jsonl(image_uri, output_path):
    rows = []
    for exp in EXPERIMENTS:
        repeats = get_repeat_count(exp["temp"])
        for i in range(repeats):
            request_obj = {
                "custom_id": f"{exp['id']}_run{i+1}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "file_data": {
                                        "mime_type": "image/jpeg",
                                        "file_uri": image_uri,
                                    }
                                },
                                {"text": exp["prompt"]},
                            ],
                        }
                    ],
                    "generation_config": {
                        "temperature": exp["temp"],
                        "top_p": exp["top_p"],
                        "max_output_tokens": 2048,
                    },
                },
            }
            rows.append(json.dumps(request_obj, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    print(f"JSONL 已寫入: {output_path}，共 {len(rows)} 條請求 (動態N優化)")

# --- Step 3: Submit batch job ---
def submit_batch(client, jsonl_path):
    print("上傳 JSONL 到 File API...")
    with open(jsonl_path, "rb") as f:
        uploaded_jsonl = client.files.upload(
            file=f,
            config=types.UploadFileConfig(
                mime_type="application/jsonl",
                display_name="tuning_requests",
            ),
        )

    print(f"建立 Batch Job (模型: {MODEL_NAME})...")
    batch_job = client.batches.create(
        model=MODEL_NAME,
        src=uploaded_jsonl.name,
        config=types.CreateBatchJobConfig(
            display_name="tongue-diagnosis-tuning-optimized",
        ),
    )
    print(f"Batch Job 已建立: {batch_job.name}")
    return batch_job.name

# --- Step 4: Poll until done ---
def wait_for_batch(client, job_name, poll_interval=30):
    print(f"\n開始輪詢 Batch Job 狀態 (每 {poll_interval} 秒)...")
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        print(f"  [{time.strftime('%H:%M:%S')}] 狀態: {state}")
        if state in terminal_states:
            return job
        time.sleep(poll_interval)

# --- Step 5: Download ---
def download_results(client, job):
    if not hasattr(job, "dest") or job.dest is None:
        raise RuntimeError(f"job.dest 不存在，job 資訊：{job}")

    output_file_name = job.dest.file_name
    print(f"\n下載結果檔案: {output_file_name}")

    file_info = client.files.get(name=output_file_name)
    download_uri = file_info.uri
    
    response = httpx.get(
        download_uri,
        params={"key": API_KEY},
        follow_redirects=True,
        timeout=120,
    )
    response.raise_for_status()
    return response.text

# --- Step 6: Parse & Mixed Report ---
def parse_and_report(client, raw_text):
    lines = [l for l in raw_text.strip().split("\n") if l.strip()]

    exp_responses: dict[str, list[str]] = {str(exp["id"]): [] for exp in EXPERIMENTS}
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        custom_id = obj.get("custom_id", "")
        exp_id = custom_id.split("_")[0] if "_" in custom_id else custom_id
        run_tag = custom_id.split("_")[1] if "_" in custom_id else "run1"
        
        try:
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = ""
            
        if exp_id in exp_responses:
            exp_responses[exp_id].append(text)
            
            exp_dir = f"experiment_data/{exp_id}"
            os.makedirs(exp_dir, exist_ok=True)
            with open(f"{exp_dir}/{run_tag}.txt", "w", encoding="utf-8") as f:
                f.write(text)

    # 進行評分與產出
    results = []
    exp_map = {exp["id"]: exp for exp in EXPERIMENTS}
    
    print("\n開始進行混合評分 (Rule + LLM Judge)...")
    for exp_id, responses in exp_responses.items():
        exp = exp_map[exp_id]
        if not responses:
            continue
            
        # 1. 規則型評分 (Rule)
        # 取第一筆代表算 coverage
        rule_scores = rule_based_score(responses[0], TCM_KEYWORDS)
        
        # 2. 穩定性評分 (Stability)
        stability = calculate_stability(responses)
        
        # 3. 裁判模型評分 (LLM Judge)
        llm_scores = call_judge(client, responses)
        
        # 綜合評估
        avg_len = sum(len(r)//2 for r in responses) / len(responses) if responses else 0
        
        # 總分 = (Rule + Format + Causal + Language) / 4 * 0.6 + Stability * 5 * 0.4
        accuracy = (rule_scores["rule"] + rule_scores["format"] + llm_scores["causal"] + llm_scores["language"]) / 4
        final_score = (accuracy * 0.6) + (stability * 5.0 * 0.4)
        
        results.append({
            "實驗編號": exp_id,
            "temperature": exp["temp"],
            "top_p": exp["top_p"],
            "提示詞版本": exp["desc"],
            "平均Token數": int(avg_len),
            "術語覆蓋率": f"{rule_scores['coverage']:.1%}",
            "一致性": f"{stability:.2f}",
            "LLM裁判分": f"{(llm_scores['causal']+llm_scores['language'])/2:.1f}",
            "綜合評分": f"{final_score:.2f}",
        })

    df = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "tuning_results_optimized.csv"), index=False, encoding="utf-8-sig")

    best_exp = max(results, key=lambda x: float(x["綜合評分"]))
    report = f"""# 最佳參數組合報告 (成本優化版)

## 推薦組合
- **實驗編號**: {best_exp['實驗編號']}
- **設定值**: temperature={best_exp['temperature']}, top_p={best_exp['top_p']}
- **提示詞版本**: {best_exp['提示詞版本']}

## 表現摘要
- **術語覆蓋率**: {best_exp['術語覆蓋率']}
- **一致性 (Jaccard)**: {best_exp['一致性']}
- **LLM 裁判評分 (因果/語言)**: {best_exp['LLM裁判分']}
- **綜合評分 (滿分5.0)**: {best_exp['綜合評分']}

## 降本策略總結
1. **精簡網格**: 採用 5 組核心 T/P 組合。
2. **動態N**: T=0.0 時無隨機性，僅發送 1 次請求，節省冗餘 Token。
3. **混合評分**: 透過 Rule 判斷格式與關鍵字，僅將高階邏輯交給裁判模型，大幅降低裁判成本。
"""
    with open(os.path.join(OUTPUT_DIR, "best_parameter_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== 完成！結果已寫入 tuning_results_optimized.csv 與 best_parameter_report.md ===")
    print(df.to_string(index=False))


# --- Main ---
def run_experiment():
    client = genai.Client(api_key=API_KEY)

    image_uri = upload_image(client, IMAGE_PATH)
    build_jsonl(image_uri, REQUESTS_JSONL)
    job_name = submit_batch(client, REQUESTS_JSONL)
    
    job = wait_for_batch(client, job_name, poll_interval=30)
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    
    if state != "JOB_STATE_SUCCEEDED":
        print(f"任務未成功，最終狀態: {state}")
        return

    raw_text = download_results(client, job)
    parse_and_report(client, raw_text)

if __name__ == "__main__":
    os.makedirs("experiment_data", exist_ok=True)
    run_experiment()
