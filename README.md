# 荧光衰减多周期分析系统 (Fluorescence Decay Multi-cycle Analysis Pipeline)

本套脚本提供了一个从原始视频数据到高精度荧光寿命拟合的完整工作流。主要用于处理周期性发光信号，通过多周期数据合并和非线性最小二乘法（LSM）实现精确的寿命提取。

## 📂 文件结构与功能总览

| 文件名             | 核心功能                                                                 | 主要处理对象    |
| `videodepart.py`   | **视频分帧**：将 `.mp4` 或 `.avi` 视频转换为逐帧的 `.jpg` 图片。         | 原始视频文件    |
| `roi_cycle_get.py` | **圆形 ROI 提取**：在灰度图中提取指定圆心和半径区域的平均灰度值。         | 视频帧序列      |
| `getResult.py`     | **矩形 ROI 提取**：在彩色图中提取红色通道特定矩形区域的均值。             | 视频帧序列      |
| `multicycle.py` | **多周期拟合与合并**：自动识别衰减段，对齐多个周期并合并拟合，输出寿命 。    | 生成的 CSV 数据 |

---

## 🚀 工作流程

### 1. 视频预处理 (`videodepart.py`)

首先运行此脚本将实验录制的视频拆分为单帧图片，以便后续进行图像处理。

* **配置**：修改 `base_dir` 指向视频所在文件夹，设置 `video_name`。
* **输出**：在目标目录下生成以视频命名的文件夹，包含所有分帧图片。

### 2. 特征信号提取 (`roi_cycle_get.py` 或 `getResult.py`)

根据您的实验需求选择其中一个脚本，将图像序列转化为随时间变化的信号数值（CSV 文件）。

* **`roi_cycle_get.py`**：适用于全色/灰度分析，支持自定义圆形感兴趣区域（ROI）。
* **`getResult.py`**：专门针对红色通道信号，采用矩形 ROI。
* **输出**：生成一个包含 `Index`（帧号）和 `MeanGrayValue`（平均值）的 CSV 文件。

### 3. 多周期分析与寿命拟合 (`multicycle.py`)

这是整个项目的核心算法部分，包含以下关键技术：

* **自动段提取**：识别信号中的下降沿。
* **对齐算法**：通过计算“理论起始点”和时间偏移优化（`optimize_time_shift`），将不同周期的衰减曲线在微秒尺度上精确对齐。
* **算法对比**：内置 **RLD** (Rapid Lifetime Determination)、**CMM** (Center of Mass Method) 和 **LSM** (Least Squares Method) 三种算法。
* **数据合并**：将多个低采样率的周期合并为一个高密度的数据集，进行最终拟合。

---

## 🛠 依赖项 (Prerequisites)

运行这些脚本需要以下 Python 环境：

```bash
pip install numpy pandas matplotlib opencv-python scipy

```

---

## ⚙️ 配置说明

在使用前，您通常需要修改脚本顶部的 `PARAMS` 或路径变量：

* **路径设置**：修改 `BASE_DIR` 为您的本地工作路径。
* **ROI 坐标**：
* 在 `roi_cycle_get.py` 中修改 `CIRCLE_CENTER`。
* 在 `getResult.py` 中修改 `ROI_TOP_LEFT` 和 `ROI_BOTTOM_RIGHT`。


* **时间参数**：在 `multicycle.py` 中，`dt`（采样间隔）需根据相机的 FPS 设定（例如 100FPS 对应 ）。

---

## 📊 算法详情：`multicycle.py`

该脚本采用了一种创新的多周期合并策略：

1. **寻找最佳周期**：首先对每个周期进行初步拟合，筛选出  最高、残差最小的周期作为基准。
2. **时间轴平移**：计算其余周期相对于基准周期的微秒级偏移量。
3. **全局拟合**：将所有对齐后的数据点汇聚，利用指数衰减模型  进行统一拟合。

---

## 📝 开发者备注

* **数据精度**：`multicycle` 脚本默认将时间单位统一为微秒 ()。
* **可视化**：拟合过程中会实时弹出 Matplotlib 窗口，展示当前合并的进度和最终结果图。


