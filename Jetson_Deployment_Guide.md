# NVIDIA Jetson 邊緣端部署指南 (YOLOv11)

您詢問是否可以直接將 Windows 上生成的模型檔案部署到 NVIDIA Jetson (如 Orin, Xavier, Nano) 上。

**簡短回答：**
*   ❌ **`.engine` (TensorRT)**: **不能直接使用**。TensorRT 引擎是針對特定硬體（GPU 型號）、CUDA 版本和 TensorRT 版本生成的。您無法將 Windows PC 生成的引擎拿到 Jetson 上跑。
*   ✅ **`.pt` (PyTorch)**: **可以直接使用**。將 `.pt` 檔案複製到 Jetson 上即可。
*   ✅ **`.onnx` (ONNX)**: **可以直接使用**。這是最推薦的跨平台格式。

為了在 Jetson 上獲得最佳效能（使用 TensorRT），您需要在 **Jetson 設備上** 重新執行轉換步驟。

---

## 🚀 部署步驟流程

### 步驟 1: 準備模型文件
在您的 Windows 電腦上，您只需要保留原始的 `.pt` 權重檔（例如 `yolov11x.pt`）。

### 步驟 2: 環境設置 (在 Jetson 上)
確保您的 Jetson 已經刷好 JetPack 系統，並安裝了必要的庫。

```bash
# 更新系統
sudo apt-get update
sudo apt-get install python3-pip cmake

# 安裝 PyTorch 和 TorchVision (需使用 NVIDIA 提供的 Jetson 版本)
# 通常 JetPack 預裝了或者需要從 NVIDIA 論壇下載對應的 wheels
# 參考: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# 安裝 Ultralytics (YOLO)
pip3 install ultralytics
```

### 步驟 3: 傳輸模型
將 `yolov11x.pt` 從 Windows 傳輸到 Jetson (使用 USB 隨身碟、SCP 或 SMB)。

### 步驟 4: 在 Jetson 上導出 TensorRT 引擎
**這是最關鍵的一步。** 您必須利用 Jetson 的 GPU 來編譯模型。

在 Jetson 的終端機中執行：

```python
from ultralytics import YOLO

# 載入模型
model = YOLO("yolov11x.pt")

# 導出為 TensorRT 引擎 (這會花費幾分鐘，且消耗大量記憶體)
# half=True 會使用 FP16 半精度，在 Jetson 上速度更快且記憶體更省
model.export(format="engine", device=0, half=True)
```

或者使用 CLI 命令：
```bash
yolo export model=yolov11x.pt format=engine device=0 half=True
```

### 步驟 5: 執行推論
現在您在 Jetson 上擁有了一個生成的 `yolov11x.engine`。您可以使用相同的 Python 代碼來執行它：

```python
from ultralytics import YOLO

# 載入 Jetson 本地生成的引擎
model = YOLO("yolov11x.engine", task="detect")

# 進行推論
results = model("image.jpg")
```

---

## 📋 常見問題

### Q: 為什麼不能用 Windows 的 `.engine`？
**A:** TensorRT 引擎就像是編譯好的二進制程式 (Binary)。它針對特定的 GPU 架構（例如 PC 的 Ampere 架構 vs Jetson 的 Orin 架構）進行了底層優化。如果不匹配，TensorRT 會報錯無法載入。

### Q: Jetson 記憶體不足 (OOM) 怎麼辦？
**A:** 
1.  導出時增加 Swap (交換記憶體) 空間。
2.  使用 `half=True` 開啟 FP16。
3.  如果還是不行，嘗試使用較小的模型版本 (如 `yolo11n.pt` 或 `yolo11s.pt`)，`yolo11x` 對於某些 Jetson (如 Nano) 來說可能太大。

### Q: 是否需要安裝 TensorRT？
**A:** JetPack 系統通常預裝了 TensorRT。但是，Python 的綁定可能需要手動檢查。Ultralytics 會自動嘗試調用系統的 `tensorrt` 庫。
