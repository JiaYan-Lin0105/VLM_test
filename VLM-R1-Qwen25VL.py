import os
from PIL import Image
import io
import base64
# 從 transformers 庫中僅保留 AutoProcessor
from transformers import AutoProcessor 
from llama_cpp import Llama 

# --- Base64 輔助函數 (新增) ---
def pil_to_base64(image: Image.Image) -> str:
    """將 PIL 圖像轉換為 Base64 編碼的字符串 (JPEG 格式)。"""
    # 這裡通常使用 JPEG 格式進行壓縮和編碼
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
# ------------------------------

# --- 配置 ---

PROCESSOR_ID = "Qwen/Qwen2.5-VL-3B-Instruct" 
GGUF_MODEL_REPO_ID = "mradermacher/VLM-R1-Qwen2.5VL-3B-OVD-0321-i1-GGUF"
# 請使用正確的檔名，我們使用 Q4_K_M
GGUF_MODEL_FILENAME = "VLM-R1-Qwen2.5VL-3B-OVD-0321.i1-Q4_K_M.gguf" 

# 1. 載入 GGUF 模型 (使用 Llama 類)
print(f"正在載入 GGUF 模型: {GGUF_MODEL_REPO_ID}...")
try:
    llm = Llama.from_pretrained(
        repo_id=GGUF_MODEL_REPO_ID,
        filename=GGUF_MODEL_FILENAME,
        n_ctx=32768,          
        n_gpu_layers=-1    # 在 CPU 上運行
    )
    print("模型載入成功。")
except Exception as e:
    print(f"\n❌ Llama 載入 GGUF 模型時發生錯誤。請檢查配置。錯誤: {e}")
    exit()

# 2. 載入處理器 (用於獲取 Tokenizer 和模型配置)
print(f"正在載入處理器: {PROCESSOR_ID}...")
# 雖然我們不再用它來做 Base64 轉換，但仍需要它來確認 token 資訊
processor = AutoProcessor.from_pretrained(PROCESSOR_ID, trust_remote_code=True)
print("處理器載入成功。")

# --- 準備輸入 ---

IMAGE_PATH = "./test.png"  
PROMPT = "圖片中人物的位置在哪裡？"

if not os.path.exists(IMAGE_PATH):
    print(f"\n錯誤：找不到圖片檔案 '{IMAGE_PATH}'。")
    exit()

# 3. 載入圖像
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"成功載入圖片: {IMAGE_PATH}")

# 4. 準備 Qwen-VL 特定的輸入格式
# 💡 關鍵修正：使用我們定義的 pil_to_base64 函數來取代 processor.image_to_base64
encoded_image = pil_to_base64(image) # <-- 修正後的程式碼

# Qwen-VL-GGUF 的 Prompt 格式：
full_prompt = (
    f"系統: 你是一個樂於助人的視覺語言模型。<|im_end|>\n"
    f"<|im_start|>用戶:\n圖片：{encoded_image} 問題：{PROMPT}<|im_end|>\n"
    f"<|im_start|>助理:"
)

# --- 模型生成 ---

print("\n--- 開始生成回答 ---")

# 5. 使用 Llama 進行推理
output = llm(
    prompt=full_prompt,
    max_tokens=512,
    stop=["<|im_end|>"],  
    echo=False,          
    temperature=0.1
)

# 6. 輸出結果
response_text = output["choices"][0]["text"].strip()

# 清理可能的特殊結束標記
final_answer = response_text.replace("<|im_end|>", "").strip()

print("\n✅ **模型輸出結果:**")
print("---------------------------------------")
print(final_answer)
print("---------------------------------------")
