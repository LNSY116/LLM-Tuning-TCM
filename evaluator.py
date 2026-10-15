import json
import re
import logging
from typing import List, Dict

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 合法中醫證素
VALID_SYNDROMES = {"氣虛", "陽虛", "陰虛", "血虛", "痰濕", "濕熱", "氣滯", "血瘀", "化熱"}

def clean_markdown(text: str) -> str:
    """清除 markdown 代碼塊標記與多餘空白"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def extract_final_syndromes(response_text: str) -> List[str]:
    """
    Robust 解析 LLM 回傳的 JSON，優先使用 json.loads()，
    若失敗則用 Regex 萃取。自動清除 markdown 代碼塊與多餘空白。
    """
    if not isinstance(response_text, str) or not response_text.strip():
        logger.warning("接收到空字串或非字串輸入")
        return []

    cleaned_text = clean_markdown(response_text)

    # 1. 嘗試標準 JSON 解析
    try:
        data = json.loads(cleaned_text)
        if isinstance(data, dict) and "最終證素" in data:
            syndromes = data["最終證素"]
            if isinstance(syndromes, list):
                return [s for s in syndromes if s in VALID_SYNDROMES]
    except json.JSONDecodeError as e:
        logger.debug(f"JSON 解析失敗，嘗試使用正則表達式萃取。原因: {e}")

    # 2. Fallback: 使用 Regex 萃取
    # 匹配 "最終證素": ["氣虛", "陽虛"] 這種格式，考慮 trailing comma
    match = re.search(r'"最終證素"\s*:\s*\[(.*?)\]', cleaned_text, re.DOTALL)
    if match:
        array_content = match.group(1)
        # 萃取所有帶引號的字串
        items = re.findall(r'"([^"]+)"', array_content)
        if not items:
            items = re.findall(r"'([^']+)'", array_content)
        
        return [s for s in items if s in VALID_SYNDROMES]
        
    logger.warning("無法從文本中萃取最終證素")
    return []

def calculate_f1(pred: List[str], gt: List[str]) -> Dict[str, float]:
    """計算 Precision, Recall, F1"""
    pred_set = set(pred)
    gt_set = set(gt)
    
    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_jaccard_sets(list_of_syndromes: List[List[str]]) -> float:
    """計算多次推論的證素集合 Jaccard 一致性"""
    import itertools
    if not list_of_syndromes or len(list_of_syndromes) < 2:
        return 1.0
        
    extracted_sets = [set(s) for s in list_of_syndromes]
    sims = []
    for s1, s2 in itertools.combinations(extracted_sets, 2):
        if not s1 and not s2:
            sims.append(1.0)
        elif not s1 or not s2:
            sims.append(0.0)
        else:
            sims.append(len(s1 & s2) / len(s1 | s2))
            
    return sum(sims) / len(sims) if sims else 1.0
