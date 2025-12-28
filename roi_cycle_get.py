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
    "suffix": 4  # 修改尾缀
}
# =========================================

# 固定参数配置
CIRCLE_CENTER = (914, 554)  # 圆心坐标 (x, y)
CIRCLE_RADIUS = 20  # 圆的半径
GAUSSIAN_SIGMA = 2.0
FILE_PATTERN = "MBS_100.1HZ_100FPS_2"

# 自动生成路径
folder_name = FILE_PATTERN.format(**PARAMS)
INPUT_DIR = os.path.join(BASE_DIR, folder_name)
OUTPUT_FILE = os.path.join(INPUT_DIR, f"{folder_name}_frame_mean_values.csv")


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
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # blurred = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            blurred = cv2.GaussianBlur(img, (3, 3), GAUSSIAN_SIGMA)
            # 创建圆形掩码
            mask = np.zeros_like(blurred)
            cv2.circle(mask, CIRCLE_CENTER, CIRCLE_RADIUS, 255, -1)

            # 应用掩码，提取圆形区域
            roi = cv2.bitwise_and(blurred, blurred, mask=mask)

            # 计算圆形区域内非零像素的均值
            mean_value = cv2.mean(roi, mask)[0]

            data.append({
                "Index": extract_index(os.path.basename(path)),
                "MeanGrayValue": mean_value
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