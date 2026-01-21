import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

# 硬编码的输入和输出文件地址
RESULT_DIR = 'data/exp1_cross_eval/cross_eval_res/res_genimage_2'
JSON_NAME = 'final_results.json'
CSV_NAME = 'results.csv'

json_file = Path(RESULT_DIR) / JSON_NAME
csv_file = Path(RESULT_DIR) / CSV_NAME

def load_results(input_file):
    """加载JSON结果文件"""
    with open(input_file, 'r') as f:
        results = json.load(f)
    return results


def compute_statistics(results):
    """
    计算沿axis=0的mean, std, 以及全局统计
    
    Args:
        results: 嵌套字典 {train_sub: {eval_sub: {metric: {mean, std}}}}
    
    Returns:
        统计结果的字典
    """
    # 获取所有train_subs和eval_subs
    train_subs = list(results.keys())
    eval_subs = list(results[train_subs[0]].keys())
    
    metrics = ['acc', 'ap', 'auc']
    stats = {}
    
    for metric in metrics:
        # 构建mean矩阵和std矩阵
        mean_matrix = np.array([[results[tr][va][metric]['mean'] for va in eval_subs]
                                for tr in train_subs])
        std_matrix = np.array([[results[tr][va][metric]['std'] for va in eval_subs]
                               for tr in train_subs])
        
        # 沿axis=0计算：对每个eval_sub，遍历所有train_sub
        col_means = mean_matrix.mean(axis=0)  # shape: (n_eval_subs,)
        col_stds = std_matrix.std(axis=0)     # shape: (n_eval_subs,)
        
        # 计算全局统计值
        global_mean = col_means.mean()
        global_std = col_stds.mean()
        
        # 保存结果
        stats[metric] = {
            'eval_subs': eval_subs,
            'col_means': col_means,
            'col_stds': col_stds,
            'global_mean': global_mean,
            'global_std': global_std
        }
    
    return stats


def format_value(mean, std):
    """格式化为 mean±std 的形式"""
    return f"{mean*100:.1f}±{std*100:.1f}"


def save_to_csv(stats, output_file):
    """保存统计结果到CSV文件"""
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    metrics = ['acc', 'ap', 'auc']
    rows = []
    
    # 第1行：eval_subs名称 + "global"
    row1 = [''] + stats[metrics[0]]['eval_subs'] + ['avg']
    rows.append(row1)
    
    # 第2-4行：每个指标的mean±std
    for metric in metrics:
        row = [metric]
        col_means = stats[metric]['col_means']
        col_stds = stats[metric]['col_stds']
        global_mean = stats[metric]['global_mean']
        global_std = stats[metric]['global_std']
        
        # 添加每个eval_sub的值
        for mean, std in zip(col_means, col_stds):
            row.append(format_value(mean, std))
        
        # 添加全局统计值
        row.append(format_value(global_mean, global_std))
        rows.append(row)
    
    # 写入CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, header=False)
    print(f"结果已保存到: {output_file}")
    
    # 打印到控制台
    print("\n输出内容：")
    for row in rows:
        print(','.join(str(x) for x in row))


def main():
    """主函数"""
    print(f"加载文件: {json_file.name}")
    results = load_results(json_file)
    
    print("计算统计量...")
    stats = compute_statistics(results)
    
    print(f"保存到CSV: {csv_file.name}")
    save_to_csv(stats, csv_file)
    
    print("\n完成！")


if __name__ == '__main__':
    main()
