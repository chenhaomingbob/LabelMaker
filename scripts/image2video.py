import cv2
import os
import re

# --- 用户配置参数 ---

# 包含图片的文件夹路径
# 示例: 'C:/Users/YourUser/Desktop/MyImages' 或 'path/to/your/images'
IMAGE_FOLDER = '/data1/chm/datasets/wheeltec/record_20250625/outputs_dirs/projection_visualization_masked'

# 输出的视频文件名
VIDEO_NAME = '/data1/chm/datasets/wheeltec/record_20250625/outputs_dirs/projection_visualization_masked/output_video.mp4'

# 生成视频的帧率 (Frames Per Second)
FPS = 10

# 要查找的图片文件扩展名
IMAGE_EXTENSION = '.png'


# --- 脚本主逻辑 ---

def create_video_from_images():
    """
    查找指定文件夹中的图片，并将它们合并成一个MP4视频。
    """
    print(f"正在从文件夹 '{IMAGE_FOLDER}' 创建视频...")

    # 1. 获取所有符合条件的图片文件
    images = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(IMAGE_EXTENSION)]

    if not images:
        print(f"错误：在文件夹 '{IMAGE_FOLDER}' 中未找到任何 '{IMAGE_EXTENSION}' 文件。请检查路径和文件扩展名。")
        return

    # 2. 对图片进行自然排序 (例如，正确处理 1, 2, 10 的顺序)
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    images.sort(key=natural_sort_key)

    print(f"找到了 {len(images)} 张图片。")

    # 3. 读取第一张图片以获取视频的宽度和高度
    first_image_path = os.path.join(IMAGE_FOLDER, images[0])
    try:
        frame = cv2.imread(first_image_path)
        if frame is None:
            print(f"错误：无法读取第一张图片 '{first_image_path}'。文件可能已损坏或格式不受支持。")
            return
        height, width, layers = frame.shape
        size = (width, height)
    except Exception as e:
        print(f"读取图片尺寸时出错: {e}")
        return

    # 4. 初始化视频写入器 (VideoWriter)
    # FourCC 是一个4字节的代码，用于指定视频编解码器。'mp4v' 是用于 .mp4 格式的标准编解码器。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, size)

    # 5. 遍历所有图片并写入视频帧
    print("正在逐帧写入视频...")
    for i, image_name in enumerate(images):
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        frame = cv2.imread(image_path)

        # 确保读取的帧尺寸与视频尺寸一致
        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"警告: 图片 '{image_name}' 的尺寸与第一张图片不同，将跳过此帧。")
            continue

        out.write(frame)

        # 打印进度
        print(f"  -> 已写入第 {i + 1}/{len(images)} 帧: {image_name}")

    # 6. 释放资源
    out.release()
    print("-" * 30)
    print(f"视频创建成功！已保存为 '{VIDEO_NAME}'")
    print("-" * 30)


if __name__ == '__main__':
    create_video_from_images()