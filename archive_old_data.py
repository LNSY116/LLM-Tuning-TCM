import shutil
import os

def cleanup_project():
    target_dir = "archive"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 核心保留清單（修正版：加入醫生提示詞參考）
    keep_list = {
        "tuning_workflow_batch.py",
        "MyTongue.jpg",
        "tests.yaml",
        "promptfooconfig.yaml",
        ".env",
        "pyproject.toml",
        "uv.lock",
        ".venv",
        "src",
        "chat-Optimizing Experimental Costs.txt",
        "醫生提示詞參考.txt",  # 👈 保持在根目錄
        "archive",
        "archive_old_data.py",
        ".gitignore",
        ".python-version"
    }

    # 檢查是否需要從 archive 搬回來
    ref_file = "醫生提示詞參考.txt"
    archived_ref = os.path.join(target_dir, ref_file)
    if os.path.exists(archived_ref) and not os.path.exists(ref_file):
        shutil.move(archived_ref, ".")
        print(f"📦 已將 {ref_file} 恢復至根目錄")

    print("🚀 開始清理專案環境...")

    for item in os.listdir("."):
        if item in keep_list:
            continue
        
        try:
            dest = os.path.join(target_dir, item)
            if os.path.exists(dest):
                if os.path.isdir(dest):
                    shutil.rmtree(dest)
                else:
                    os.remove(dest)
            
            shutil.move(item, target_dir)
            print(f"📦 已隔離: {item} -> {target_dir}")
        except Exception as e:
            print(f"❌ 搬移 {item} 失敗: {e}")

    print("\n✅ 清理完成！「醫生提示詞參考.txt」已恢復並保留。")

if __name__ == "__main__":
    cleanup_project()
