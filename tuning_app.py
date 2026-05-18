import gradio as gr
import os
import pandas as pd
from tuning_workflow_sync import run_tuning_experiment, API_KEY, IMAGE_PATH, PROMPT_PATH, OUTPUT_DIR

def start_tuning(api_key, image, prompt_file):
    if not api_key:
        return "❌ 請輸入 GEMINI_API_KEY", None, None
    
    if image is None:
        return "❌ 請上傳舌象圖片", None, None

    # 儲存暫存圖片
    temp_image_path = "temp_tongue.jpg"
    image.save(temp_image_path)
    
    # 處理提示詞檔案
    if prompt_file is not None:
        p_path = prompt_file.name
    else:
        p_path = PROMPT_PATH

    try:
        df, save_path = run_tuning_experiment(api_key, temp_image_path, p_path, OUTPUT_DIR)
        
        # 讀取 best_config.yaml
        best_config_path = os.path.join(OUTPUT_DIR, "best_config.yaml")
        best_config_text = ""
        if os.path.exists(best_config_path):
            with open(best_config_path, "r", encoding="utf-8") as f:
                best_config_text = f.read()
        
        return f"🎉 實驗完成！報告已儲存至: {save_path}", df, best_config_text
    except Exception as e:
        return f"❌ 執行出錯: {str(e)}", None, None

# 建立 Gradio 介面
with gr.Blocks(title="TCM Tongue Diagnosis LLM Tuner") as demo:
    gr.Markdown("# 🔬 中醫舌診 LLM 參數調優工具")
    gr.Markdown("本工具可自動測試不同 Prompt 與參數組合，找出最適合主應用的配置。")
    
    with gr.Row():
        with gr.Column():
            api_key_input = gr.Textbox(label="GEMINI_API_KEY", placeholder="輸入你的 API Key", value=API_KEY or "", type="password")
            image_input = gr.Image(label="上傳舌象圖片", type="pil")
            prompt_input = gr.File(label="上傳自定義提示詞 (可選)", file_types=[".txt"])
            run_btn = gr.Button("🚀 開始調優實驗", variant="primary")
        
        with gr.Column():
            status_output = gr.Textbox(label="執行狀態")
            best_config_output = gr.Code(label="最佳配置 (best_config.yaml)", language="yaml")
            results_table = gr.Dataframe(label="實驗結果摘要")

    run_btn.click(
        fn=start_tuning,
        inputs=[api_key_input, image_input, prompt_input],
        outputs=[status_output, results_table, best_config_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("🔗 主應用系統: [FJCU-AI-APPLICATION/Tongue-Diagnosis](https://github.com/FJCU-AI-APPLICATION/Tongue-Diagnosis)")

if __name__ == "__main__":
    demo.launch()
