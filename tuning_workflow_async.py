import os
import json
import asyncio
import pandas as pd
import itertools
import re
from pathlib import Path
from typing import List, Set
from evaluator import extract_final_syndromes, calculate_f1, calculate_jaccard_sets

from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# =============================================================================
# 🔬 LLM Parameter Tuning Framework (Async Version)
# =============================================================================

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
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 評分權重配置
WEIGHT_ACCURACY = 0.6
WEIGHT_STABILITY = 0.4

# 評分相關模組已移至 evaluator.py

def find_asset(rel_path: str) -> str | None:
    candidate = ROOT / "assets" / rel_path
    if candidate.exists():
        return str(candidate)
    return None

def find_test_images() -> list[str]:
    candidate_dir = ROOT / "assets" / "test_images"
    if candidate_dir.exists():
        images = list(candidate_dir.glob("*.jpg")) + list(candidate_dir.glob("*.jpeg")) + list(candidate_dir.glob("*.png"))
        if images:
            return [str(p) for p in images]
            
    fallback_img = find_asset("MyTongue.jpg")
    if fallback_img:
        return [fallback_img]
    return []

# --- 評分功能（已移至 evaluator.py） ---

# --- 異步 API 呼叫 ---
def is_retryable_error(e):
    return isinstance(e, APIError) and getattr(e, 'code', None) == 429

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception), reraise=True)
async def async_generate(client, uploaded_image, prompt_text, temp, top_p):
    return await client.aio.models.generate_content(
        model=GENERATOR_MODEL,
        contents=[uploaded_image, prompt_text] if uploaded_image else prompt_text,
        config=types.GenerateContentConfig(
            temperature=temp,
            top_p=top_p,
            max_output_tokens=2048
        )
    )

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception), reraise=True)
async def async_call_judge(client, ai_response, ground_truth, base_prompt, judge_template):
    judge_prompt = judge_template.replace("{base_prompt}", base_prompt)
    judge_prompt = judge_prompt.replace("{characteristics}", str(ground_truth.get('characteristics', [])))
    judge_prompt = judge_prompt.replace("{true_syndromes}", str(ground_truth.get('true_syndromes', [])))
    judge_prompt = judge_prompt.replace("{ai_response}", ai_response)
    
    res = await client.aio.models.generate_content(
        model=JUDGE_MODEL,
        contents=judge_prompt,
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
    )
    return json.loads(res.text)

# --- 異步 Worker ---
async def worker(queue, client, save_path, lock, base_prompt, judge_template, ground_truth_db):
    while True:
        task = await queue.get()
        if task is None:
            break
            
        exp_id, p_desc, temp, top_p, img_idx, uploaded_img, img_filename, prompt_text, repeats = task
        try:
            responses = []
            for _ in range(repeats):
                res = await async_generate(client, uploaded_img, prompt_text, temp, top_p)
                responses.append(res.text)
                await asyncio.sleep(1) # 稍微緩解 rate limit
            
            # 取第一次回答作為代表來算 judge
            rep = responses[0]
            # 計算穩定性
            extracted_list = [extract_final_syndromes(r) for r in responses]
            stab = calculate_jaccard_sets(extracted_list)
            
            gt = ground_truth_db.get(img_filename, {})
            gt_syndromes = gt.get("true_syndromes", [])
            pred_syndromes = extract_final_syndromes(rep)
            metrics = calculate_f1(pred_syndromes, gt_syndromes)
            
            try:
                judge = await async_call_judge(client, rep, gt, base_prompt, judge_template)
                judge_status = "success"
            except Exception as je:
                print(f"⚠️ Judge 失敗: {je}")
                judge = None
                judge_status = "failed"
                
            if judge is not None:
                causal = judge.get("causal", 3)
                rule = judge.get("rule_following", 3)
                accuracy = (metrics["f1"] * 5 + causal + rule) / 3
                final = (accuracy * WEIGHT_ACCURACY) + (stab * 5.0 * WEIGHT_STABILITY)
            else:
                causal = None
                rule = None
                final = None
            
            row = {
                "實驗編號": exp_id,
                "提示詞": p_desc,
                "T": temp,
                "P": top_p,
                "圖片": img_filename,
                "F1": f"{metrics['f1']:.2f}",
                "一致性": f"{stab:.2f}",
                "Causal": causal if causal is not None else "",
                "Rule": rule if rule is not None else "",
                "綜合評分": f"{final:.2f}" if final is not None else "",
                "Judge_Status": judge_status
            }
            
            async with lock:
                df = pd.DataFrame([row])
                df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False, encoding='utf-8-sig')
                print(f"✅ [{exp_id}] 圖片 {img_filename} 測試完成 (綜合評分: {final:.2f})")
                
        except Exception as e:
            print(f"❌ [{exp_id}] 圖片 {img_filename} 失敗: {e}")
        finally:
            queue.task_done()

# --- 主程式 ---
async def main_async():
    client = genai.Client(api_key=API_KEY)
    
    # 載入 Ground Truth
    gt_path = ROOT / "assets" / "ground_truth.json"
    ground_truth_db = {}
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth_db = json.load(f)
            
    # 讀取提示詞
    prompt_path = ROOT / "prompts" / "system_prompt_professional.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到 {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
        
    judge_prompt_path = ROOT / "prompts" / "judge_prompt.md"
    if not judge_prompt_path.exists():
        raise FileNotFoundError(f"找不到 {judge_prompt_path}")
    with open(judge_prompt_path, "r", encoding="utf-8") as f:
        judge_template = f.read()
        
    prompts_to_test = [
        {"desc": "專業版(僅限)", "prompt": base_prompt},
    ]
    
    # 網格搜尋配置：T=0時Top-P無效（因為貪婪解碼下，最高機率的 token 累積機率在第一個就達 100%），故僅取 P=1.0 作為 Baseline。
    # T>0 時，測試不同溫度與 Top-P 的組合。
    CONFIG_GRID = [{"temp": 0.0, "top_p": 1.0}]
    for t in [0.1, 0.3, 0.5]:
        for p in [0.1, 0.5, 0.9]:
            CONFIG_GRID.append({"temp": t, "top_p": p})
    
    def get_repeat_count(temp: float) -> int:
        if temp == 0.0: return 1
        elif temp >= 0.5: return 5
        else: return 3

    image_paths = find_test_images()
    if not image_paths:
        raise FileNotFoundError("找不到測試照片")
        
    # 上傳圖片
    print(f"🖼️ 上傳 {len(image_paths)} 張測試照片...")
    uploaded_images = []
    for path in image_paths:
        with open(path, "rb") as f:
            filename = os.path.basename(path)
            # 使用同步上傳即可，因為只做一次
            uploaded_img = client.files.upload(file=f, config=types.UploadFileConfig(mime_type="image/jpeg", display_name=filename))
            uploaded_images.append((filename, uploaded_img))
            
    save_path = str(OUTPUT_DIR / "doctor_async_tuning_results.csv")
    completed = set()
    if os.path.exists(save_path):
        df_exist = pd.read_csv(save_path)
        for _, r in df_exist.iterrows():
            completed.add(f"{r['提示詞']}_{float(r['T']):.1f}_{float(r['P']):.1f}_{r['圖片']}")
            
    queue = asyncio.Queue()
    lock = asyncio.Lock()
    
    exp_idx = 1
    total = 0
    for p in prompts_to_test:
        for c in CONFIG_GRID:
            exp_id = f"EXP{exp_idx:02d}"
            repeats = get_repeat_count(c["temp"])
            for img_filename, uploaded_img in uploaded_images:
                task_key = f"{p['desc']}_{c['temp']:.1f}_{c['top_p']:.1f}_{img_filename}"
                if task_key not in completed:
                    queue.put_nowait((exp_id, p['desc'], c['temp'], c['top_p'], exp_idx, uploaded_img, img_filename, p['prompt'], repeats))
                    total += 1
            exp_idx += 1
            
    print(f"🔥 開始異步測試，共 {total} 個未完成任務...")
    
    # 啟動 3 個 Worker
    workers = [asyncio.create_task(worker(queue, client, save_path, lock, base_prompt, judge_template, ground_truth_db)) for _ in range(3)]
    
    await queue.join()
    for _ in range(3): queue.put_nowait(None)
    await asyncio.gather(*workers)
    
    print(f"✅ 實驗完成！報告已儲存至: {save_path}")

if __name__ == "__main__":
    asyncio.run(main_async())
