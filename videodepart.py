import cv2
import os
import time
import re

def process_video(video_path, output_base):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base, video_name)
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), frame)
        frame_count += 1

    cap.release()
    print(f"处理完成: {video_path} | 耗时: {time.time() - start_time:.2f}s | 生成帧数: {frame_count}")


# 配置参数
base_dir = "D:/study/work"
output_base = "D:/study/work/test_photo/"
# μF_values = [2]  # 根据实际情况修改
# suffixes = [1,2,3,4]

# #批量处理
# for uF in μF_values:
#     for suffix in suffixes:
        # video_name = f"25μs_{uF}μF_200FPS_200Hz_{suffix}.avi"
video_name = f"MB7_201HZ_200FPS_12_1.mp4"
video_path = os.path.join(base_dir, "test_vedio", video_name)

if os.path.exists(video_path):
    process_video(video_path, output_base)
else:
    print(f"文件不存在: {video_path}")
