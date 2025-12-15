import requests

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from PIL import Image, ImageDraw
import os
import math
model_id = "IDEA-Research/grounding-dino-base"
# 確保使用 CUDA，如果可用
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"使用的設備: {device}")

# --- 載入模型與處理器 ---
processor = AutoProcessor.from_pretrained(model_id)
# 載入模型並移動到設備
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# --- 輸入設定 ---
image_path = "./test.png"
text = "person in the picture." # 查詢文本必須小寫並以點號結束

if not os.path.exists(image_path):
    print(f"錯誤：找不到圖片檔案 '{image_path}'。")
    exit()
    
# 載入圖像
image = Image.open(image_path).convert("RGB")
# 獲取圖像的尺寸 (W, H)
W, H = image.size

# --- 模型推理 ---
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# --- 後處理 ---
# 注意：target_sizes 需要是 (H, W)，因此使用 image.size[::-1]
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[(H, W)]
)

# 提取第一個結果
result = results[0]

# --- 繪圖邏輯 ---

# 創建一個 ImageDraw 物件來在圖像上繪圖
draw = ImageDraw.Draw(image)

print("\n--- 繪圖結果 ---")
print(f"檢測到 {len(result['boxes'])} 個物體。")

# 遍歷所有檢測到的物體
for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
    # 將張量轉換為 Python 列表或 NumPy 陣列，並取整數
    # box 格式為 (xmin, ymin, xmax, ymax)
    box = box.cpu().numpy().astype(int) 
    
    xmin, ymin, xmax, ymax = box.tolist()

    # 繪製邊界框 (使用紅色，線寬 3)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

    # 準備標籤文本
    # 標籤文本通常放在框的左上角
    score_percent = f"{score.item():.2f}"
    label_text = f"{label}: {score_percent}"

    # 繪製標籤背景 (白色背景)
    # 根據字體大小調整背景框的高度
    text_size = draw.textlength(label_text)
    
    # 繪製一個填充框作為文本背景
    draw.rectangle([xmin, ymin - 20, xmin + text_size + 5, ymin], fill="red")
    
    # 繪製文本 (黑色字體)
    draw.text((xmin + 2, ymin - 18), label_text, fill="white")
    
    print(f" - 標籤: {label_text}, 座標: ({xmin}, {ymin}, {xmax}, {ymax})")

# --- 顯示與儲存結果 ---
output_path = "output_detection2.png"
image.save(output_path)

print(f"\n✅ 繪製完成。結果已儲存到 '{output_path}'")

# 您可以在程式碼結束後手動打開此檔案查看，或者使用 image.show() 顯示
# image.show()
