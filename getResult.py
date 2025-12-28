import cv2
import numpy as np
import os
import pandas as pd
import time
from glob import glob
import re

# ============== 只需修改这里 ==============
BASE_DIR = "D:/study/work/test_photo"
PARAMS = {
    "μF": "1μF",  # 修改μF值
    "suffix": 4 # 修改尾缀
}
# =========================================

# 固定参数配置
ROI_TOP_LEFT = (473, 486)
ROI_BOTTOM_RIGHT = (491, 603)
GAUSSIAN_SIGMA = 2.0
FILE_PATTERN = "MB7_201HZ_200FPS_12_1"

# 自动生成路径
folder_name = FILE_PATTERN.format(**PARAMS)
INPUT_DIR = os.path.join(BASE_DIR, folder_name)
OUTPUT_FILE = os.path.join(INPUT_DIR, f"{folder_name}_MBL_frame_mean_values.csv")


def extract_index(filename):
    """使用正则表达式提取索引"""
    match = re.search(r'frame_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else None


def main():
    start_time = time.time()

    # 获取已排序的文件列表
    image_paths = sorted(glob(os.path.join(INPUT_DIR, "*.jpg")),
                         key=lambda x: extract_index(x) or 0)

    # 数据处理
    data = []
    for idx, path in enumerate(image_paths, 1):
        try:
            # 读取彩色图像（BGR格式）
            img = cv2.imread(path)
            # 提取红色通道（在BGR格式中是第三个通道）
            red_channel = img[:, :, 2]
            # 对红色通道进行高斯模糊
            blurred = cv2.GaussianBlur(red_channel,(3,3),GAUSSIAN_SIGMA)
            roi = blurred[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1],
                  ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
            data.append({
                "Index": extract_index(os.path.basename(path)),
                "MeanGrayValue": np.mean(roi)  # 改变列名以反映这是红色通道的值
            })
        except Exception as e:
            print(f"跳过 {os.path.basename(path)}: {str(e)}")

    # 保存结果
    pd.DataFrame(data).sort_values('Index').to_csv(OUTPUT_FILE, index=False)

    print(f"处理完成！耗时 {time.time() - start_time:.1f}秒")
    print(f"结果保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    print(f"当前处理参数: μF={PARAMS['μF']}, 尾缀={PARAMS['suffix']}")
    main()
    print("=== 文件名数据已提取完毕 ===")
