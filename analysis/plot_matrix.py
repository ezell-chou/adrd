import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

from configs.config import PlotConfig

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

class PlotConfigExtended(PlotConfig):
    RESULTS_DIR = 'data/exp3_0_ablation/res_rand'
    RESULTS_JSON = 'final_results.json'

    NETWORK = 'BetterCNN'


res_dir = Path(PlotConfigExtended.RESULTS_DIR)
res_json = PlotConfigExtended.RESULTS_JSON
net_name = PlotConfigExtended.NETWORK

sns.set_style("whitegrid")

def load_results(json_file = res_dir / res_json):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_cross_evaluation_stats(results, train_subs, eval_subs, retrain_times):
    """与 verification.py 里完全一致，直接拷过来即可"""
    os.makedirs('result', exist_ok=True)

    metrics = ['acc', 'ap', 'auc']
    titles = {
        'acc': f'Accuracy (Train→Eval)\n({net_name}, {retrain_times} runs)',
        'ap': f'Average Precision (Train→Eval)\n({net_name}, {retrain_times} runs)',
        'auc': f'AUC (Train→Eval)\n({net_name}, {retrain_times} runs)'
    }

    for metric in metrics:
        mean_matrix = np.array([[results[tr][va][metric]['mean'] for va in eval_subs]
                                for tr in train_subs])
        std_matrix = np.array([[results[tr][va][metric]['std'] for va in eval_subs]
                               for tr in train_subs])

        # 计算每一行的平均值和标准差（用于最后一列的 avg）
        row_means = mean_matrix.mean(axis=1)  # shape: (n_train_subs,)
        row_stds = std_matrix.std(axis=1)     # shape: (n_train_subs,)
        
        # 计算每一列的平均值和标准差（用于最下面一行的 avg）
        col_means = mean_matrix.mean(axis=0)  # shape: (n_eval_subs,)
        col_stds = std_matrix.std(axis=0)     # shape: (n_eval_subs,)
        
        # 打印 col_means 和 col_stds
        print(f"\n{'='*80}")
        print(f"Metric: {metric.upper()}")
        print(f"{'='*80}")
        print("Subsets:  " + "  ".join(eval_subs))
        print("Means:    " + "  ".join([f"{col_means[i]*100:.1f}" for i in range(len(eval_subs))]))
        print("Stds:     " + "  ".join([f"{col_stds[i]*100:.1f}" for i in range(len(eval_subs))]))
        
        # 将行平均值列附加到矩阵
        mean_matrix_with_avg = np.hstack([mean_matrix, row_means.reshape(-1, 1)])
        std_matrix_with_avg = np.hstack([std_matrix, row_stds.reshape(-1, 1)])
        
        # 将列平均值行附加到矩阵（追加到最下面）
        # 右下角的单元格为全局平均值
        global_mean = col_means.mean()
        global_std = col_stds.mean()
        col_means_with_global = np.hstack([col_means, global_mean])
        col_stds_with_global = np.hstack([col_stds, global_std])
        
        mean_matrix_with_avg = np.vstack([mean_matrix_with_avg, col_means_with_global.reshape(1, -1)])
        std_matrix_with_avg = np.vstack([std_matrix_with_avg, col_stds_with_global.reshape(1, -1)])

        # 创建扩展后的注释矩阵（包括最后一行）
        annot_matrix = np.empty((len(train_subs) + 1, len(eval_subs) + 1), dtype=object)
        for i in range(len(train_subs)):
            for j in range(len(eval_subs)):
                annot_matrix[i, j] = f'{mean_matrix[i, j]:.1%}\n±{std_matrix[i, j]:.1%}'
            # 添加平均值列的注释（每行的平均值和标准差）
            annot_matrix[i, -1] = f'{row_means[i]:.1%}\n±{row_stds[i]:.1%}'
        
        # 添加最下面一行的注释（每列的平均值和标准差）
        for j in range(len(eval_subs)):
            annot_matrix[-1, j] = f'{col_means[j]:.1%}\n±{col_stds[j]:.1%}'
        # 右下角为全局平均值
        annot_matrix[-1, -1] = f'{global_mean:.1%}\n±{global_std:.1%}'

        # 更新 xticklabels，添加 "avg" 标签
        xticklabels_with_avg = list(eval_subs) + ['avg']
        
        # 更新 yticklabels，添加 "avg" 标签
        yticklabels_with_avg = list(train_subs) + ['avg']

        # 设定每个方块的尺寸为正方形 (0.8 x 0.8 英寸)
        cell_size = 0.8
        fig_width = max(8, (len(eval_subs) + 1) * cell_size)
        fig_height = max(6, (len(train_subs) + 1) * cell_size)
        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(mean_matrix_with_avg, annot=annot_matrix, fmt='', cmap='Blues',
                         xticklabels=xticklabels_with_avg, yticklabels=yticklabels_with_avg,
                         linewidths=.5, cbar_kws={'label': metric.upper()},
                         vmin=0.5, vmax=1.0, annot_kws={'fontsize': 10})
        plt.xlabel('Evaluation subsets', fontsize=12)
        plt.ylabel('Training subsets', fontsize=12)
        # 将横坐标标签放到上方
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=45, ha='left')
        plt.tight_layout()


        filename = f'{res_dir}/{metric}_cross_eval_{net_name}_{retrain_times}runs.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f'已保存 -> {filename}')
        plt.close()

if __name__ == '__main__':
    # 1. 读 JSON
    final_results = load_results()

    # 2. 重建子集顺序（JSON 的 key 顺序就是原始顺序）
    train_subs = list(final_results.keys())
    eval_subs  = list(list(final_results.values())[0].keys())

    # 3. 画图
    RETRAIN_TIMES = len(list(list(final_results.values())[0].values())[0])
    plot_cross_evaluation_stats(final_results, train_subs, eval_subs, RETRAIN_TIMES)
