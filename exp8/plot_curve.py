import os
import argparse
import json
from typing import List
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def find_event_files(log_dir: str) -> List[str]:
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, f))
    return sorted(event_files)


def extract_scalars_from_run(run_dir: str):
    """
    从TensorBoard日志目录中提取Loss和Accuracy标量数据
    :param run_dir: TensorBoard日志文件所在的文件夹路径
    :return: 字典，key=标量标签，value=按步数排序的(step, value)列表
    """
    # 核心：使用TensorBoard的EventAccumulator类解析日志标量，**无需安装TensorFlow**
    # run_dir：日志目录；size_guidance={'scalars': 0}：加载所有标量数据（0表示无数量限制）
    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    # 重新加载日志文件，将磁盘中的TensorBoard数据读取到内存中
    acc.Reload()
    # 获取所有标量数据的标签（tags），如果没有标量则返回空列表
    # acc.Tags() 返回包含scalars/images/audio等分类的字典，取scalars分类
    tags = acc.Tags().get('scalars', [])
    # 初始化空字典，用于存储最终筛选、整理后的标量数据
    data = {}
    # 遍历所有标量数据的标签
    for tag in tags:
        # 筛选条件：只保留以 Loss/ 或 Accuracy/ 开头的标量标签（过滤无关数据）
        if tag.startswith('Loss/') or tag.startswith('BLEU-4/'):
            # 根据标签，从解析器中获取该标量对应的所有事件（包含步数、数值、时间戳）
            events = acc.Scalars(tag)
            
            # 遍历事件，提取 训练步数(step) 和 标量数值(value)，并**按训练步数从小到大排序**
            # 排序后保证数据是按训练顺序排列的，避免乱序
            data[tag] = sorted([(e.step, e.value) for e in events], key=lambda x: x[0])
    
    # 返回整理好的标量数据字典
    return data


def plot_metrics(data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Group by prefix (Loss, Accuracy)
    grouped = {}
    # 遍历上一步提取到的所有标量数据（key=标签tag，value=数值序列series）
    for tag, series in data.items():
        # 对标签按 / 分割，取第一部分作为分组前缀（例如 Loss/train → Loss，Accuracy/val → Accuracy）
        prefix = tag.split('/')[0]
        # 分组存储：
        #    - 如果前缀（Loss/Accuracy）不存在，就创建空列表
        #    - 把当前的 (标签, 数值数据) 元组追加到对应前缀的列表中
        grouped.setdefault(prefix, []).append((tag, series))

    saved = []
    for prefix, items in grouped.items():
        plt.figure(figsize=(6,4))
        for tag, series in items:
            steps = [s for s,_ in series]
            values = [v for _,v in series]
            plt.plot(steps, values, label=tag)
        plt.xlabel('Epoch')
        plt.ylabel(prefix)
        plt.title(f'{prefix} curves')
        plt.legend()
        out_path = os.path.join(output_dir, f'{prefix.lower()}_curves.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append(out_path)
    return saved


def main(args):
    # locate latest run matching net
    target_prefix = args.net
    candidate_runs = []
    if os.path.isdir(args.log_dir):
        for d in os.listdir(args.log_dir):
            if d.startswith(target_prefix):
                full = os.path.join(args.log_dir, d)
                if os.path.isdir(full):
                    candidate_runs.append(full)
    if not candidate_runs:
        print('No matching log runs found.')
        return
    # choose most recent by mtime
    candidate_runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    run_dir = candidate_runs[1]
    print('Using log run:', run_dir)

    event_files = find_event_files(run_dir)
    if not event_files:
        print('No event files found.')
        return

    data = extract_scalars_from_run(run_dir)
    saved = plot_metrics(data, args.output_dir)
    print('Saved figures:', json.dumps(saved, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export TensorBoard scalar plots to PNG.')
    parser.add_argument('--log_dir', default='logs', help='Root log directory containing event files.')
    parser.add_argument('--net', default='cnn-transformer', help='Network name used in log folder prefix.')
    parser.add_argument('--output_dir', default='train_results', help='Directory to save plots.')
    args = parser.parse_args()

    main(args)