"""
batch_process_arkit.py - 批量处理ARKit场景数据的脚本 (带进度条版本)
"""
import argparse
import os
import subprocess
from os.path import join, abspath, dirname, exists
from multiprocessing import Pool
from tqdm import tqdm

# 获取当前脚本所在目录
SCRIPT_DIR = dirname(abspath(__file__))


def should_skip_process(scan_dir, target_dir, args_dict):
    """检查是否应该跳过该扫描的处理"""
    # 检查目标目录是否存在
    if not exists(target_dir):
        return False

    # 检查mesh.ply文件是否存在
    mesh_file = join(target_dir, 'mesh.ply')
    if not exists(mesh_file):
        return False

    # 如果需要更严格的检查，可以添加其他条件
    # 例如检查文件大小、修改时间等
    return True


def process_single_scan(scan_dir, target_base_dir, args_dict):
    """处理单个扫描目录"""
    scan_name = os.path.basename(scan_dir)
    target_dir = join(target_base_dir, scan_name)

    # 检查是否应该跳过处理
    if should_skip_process(scan_dir, target_dir, args_dict):
        return {
            'name': scan_name,
            'status': 'skipped',
            'message': '输出文件已存在，跳过处理',
            'stdout': '',  # 添加空输出
            'stderr': ''  # 添加空错误
        }

    cmd = [
        'python',
        join(SCRIPT_DIR, 'arkitscenes2labelmaker.py'),
        '--scan_dir', scan_dir,
        '--target_dir', target_dir,
        '--sdf_trunc', str(args_dict['sdf_trunc']),
        '--voxel_length', str(args_dict['voxel_length']),
        '--depth_trunc', str(args_dict['depth_trunc'])
    ]

    if args_dict['config']:
        cmd.extend(['--config', args_dict['config']])

    # 使用subprocess.PIPE捕获输出，避免干扰进度条
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("stdout", result.stdout)
    print("stderr", result.stderr)
    return {
        'name': scan_name,
        'status': 'completed',
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def main():
    parser = argparse.ArgumentParser(description='批量处理ARKit场景数据(带进度条)')
    parser.add_argument('--scan_base_dir', required=True,
                        help='包含多个扫描目录的根目录')
    parser.add_argument('--target_base_dir', required=True,
                        help='输出目录的根路径')
    parser.add_argument('--max_scans', type=int, default=0,
                        help='最大处理的扫描数量(0表示全部)')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行工作进程数')
    parser.add_argument('--sdf_trunc', type=float, default=0.04)
    parser.add_argument('--voxel_length', type=float, default=0.008)
    parser.add_argument('--depth_trunc', type=float, default=3.0)
    parser.add_argument('--config', help='gin配置文件路径')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细输出')

    args = parser.parse_args()

    # 获取所有扫描目录
    all_scans = [join(args.scan_base_dir, d)
                 for d in os.listdir(args.scan_base_dir)
                 if os.path.isdir(join(args.scan_base_dir, d))]

    # 限制处理数量
    if args.max_scans > 0:
        all_scans = all_scans[:args.max_scans]

    print(f"准备处理 {len(all_scans)} 个扫描...")
    print("待处理的扫描目录列表：")
    for scan in all_scans:
        print(scan)
    # 准备参数
    args_dict = {
        'sdf_trunc': args.sdf_trunc,
        'voxel_length': args.voxel_length,
        'depth_trunc': args.depth_trunc,
        'config': args.config
    }

    # 创建目标目录
    os.makedirs(args.target_base_dir, exist_ok=True)

    # 进度条设置
    pbar = tqdm(
        total=len(all_scans),
        desc="处理进度",
        unit="scan",
        dynamic_ncols=True
    )

    def update_pbar(result):
        """更新进度条的回调函数"""
        pbar.update(1)
        if args.verbose:
            if result['stdout']:
                tqdm.write(f"[{result['name']}]输出:\n{result['stdout']}")
            if result['stderr']:
                tqdm.write(f"[{result['name']}]错误:\n{result['stderr']}")
        return result['name']

        # 统计信息

    stats = {
        'total': len(all_scans),
        'completed': 0,
        'skipped': 0,
        'failed': 0
    }

    # 并行处理
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = []
            for scan_dir in all_scans:
                res = pool.apply_async(
                    process_single_scan,
                    (scan_dir, args.target_base_dir, args_dict),
                    callback=update_pbar
                )
                results.append(res)

            # 等待所有任务完成
            pool.close()
            pool.join()

            # 收集统计信息
            for res in results:
                try:
                    result = res.get()
                    stats[result['status']] += 1
                except Exception as e:
                    stats['failed'] += 1
                    tqdm.write(f"处理异常: {str(e)}")
    else:
        # 单进程模式
        for scan_dir in all_scans:
            try:
                result = process_single_scan(
                    scan_dir, args.target_base_dir, args_dict
                )
                stats[result['status']] += 1
                update_pbar(result)
            except Exception as e:
                stats['failed'] += 1
                pbar.update(1)
                tqdm.write(f"处理异常: {os.path.basename(scan_dir)} - {str(e)}")
    pbar.close()
    print("\n批量处理完成!")
    print(f"总扫描数: {stats['total']}")
    print(f"成功处理: {stats['completed']}")
    print(f"跳过处理: {stats['skipped']}")
    print(f"处理失败: {stats['failed']}")


if __name__ == "__main__":
    main()

# python scripts/batch_process_arkit.py  --scan_base_dir /data1/chm/datasets/arkitscenes/raw/Training/  --target_base_dir /data1/chm/datasets/arkitscenes/LabelMaker/Training   --max_scans 50  --workers 4 --verbose
