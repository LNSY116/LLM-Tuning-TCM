import os
import json
import pandas as pd
from google import genai

# 1. 配置
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("請設定 GEMINI_API_KEY 環境變數。")
JOB_NAME = "batches/6tn6qopchhkbt87qgpo4nkvcwv2u6jqpq9j8"
OUTPUT_DIR = "outputs"

def main():
    client = genai.Client(api_key=API_KEY)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📡 正在連線至 Job: {JOB_NAME}...")
    job = client.batches.get(name=JOB_NAME)
    
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"❌ 任務狀態未完成: {job.state.name}")
        return

    print("📥 正在下載實驗結果內容...")
    try:
        # 直接傳入檔名 (不寫 name= 避免 keyword error)
        file_name = job.dest.file_name if hasattr(job.dest, "file_name") else str(job.dest)
        raw_bytes = client.files.download(file=file_name) 
        raw_text = raw_bytes.decode("utf-8")
        print("✅ 內容下載成功！")
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        # 如果 file= 還是報錯，嘗試完全不用 keyword argument
        try:
            print("🔄 嘗試備用下載方式...")
            raw_bytes = client.files.download(file_name)
            raw_text = raw_bytes.decode("utf-8")
            print("✅ 備用下載成功！")
        except Exception as e2:
            print(f"❌ 備用下載也失敗: {e2}")
            return

    # 解析 JSONL
    lines = [l for l in raw_text.strip().split("\n") if l.strip()]
    results = []
    
    print(f"🔍 正在解析 {len(lines)} 筆實驗數據...")
    
    for line in lines:
        try:
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "unknown")
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
            
            results.append({
                "實驗編號": custom_id.split("_")[0],
                "推論序號": custom_id.split("_")[1] if "_" in custom_id else "1",
                "回應長度": len(text),
                "診斷摘要": text[:100].replace("\n", " ") + "..."
            })
        except:
            continue

    if results:
        df = pd.DataFrame(results)
        save_path = os.path.join(OUTPUT_DIR, "final_rescued_results.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 數據終於救回來了！")
        print(f"📍 儲存路徑: {save_path}")
        print("\n--- 數據預覽 ---")
        print(df.head())
    else:
        print("😭 解析後清單依然為空。")

if __name__ == "__main__":
    main()
