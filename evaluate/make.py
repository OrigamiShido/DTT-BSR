import os
import argparse
from pathlib import Path


def generate_file_list(src_dir, dst_dir, output_filename):
    src_path_obj = Path(src_dir)

    if not src_path_obj.exists():
        print(f"错误: 源目录不存在 -> {src_dir}")
        return

    print(f"正在扫描目录: {src_dir} ...")
    files = sorted(list(src_path_obj.glob("*.flac")))

    if not files:
        print("提示: 未找到 .flac 文件，正在尝试查找 .wav ...")
        files = sorted(list(src_path_obj.glob("*.wav")))

    if not files:
        print("错误: 未找到任何音频文件 (.flac 或 .wav)。")
        return

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for src_file in files:
                filename = src_file.name

                src_full_path = str(src_file.resolve())

                dst_full_path = os.path.join(dst_dir, filename)

                f.write(f"{src_full_path}|{dst_full_path}\n")

        print("-" * 30)
        print(f"✅ 成功生成清单: {output_filename}")
        print(f"📂 源目录: {src_dir}")
        print(f"📂 目标目录: {dst_dir}")
        print(f"📊 总计行数: {len(files)}")
        print("-" * 30)

    except Exception as e:
        print(f"写入文件时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成用于模型评估或处理的音频文件路径对照列表。")
    parser.add_argument("--src", type=str, required=True, help="源音频文件夹路径 (Input/Reference)")
    parser.add_argument("--dst", type=str, required=True, help="目标音频文件夹路径 (Output/MSG)")
    parser.add_argument("--out", type=str, default="file_list.txt", help="输出的清单文件名 (默认: file_list.txt)")
    args = parser.parse_args()
    generate_file_list(args.src, args.dst, args.out)