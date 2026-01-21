import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# ------------------------------------------------------
# 配置
# ------------------------------------------------------
SUMMARY_JSON = "data/ablation_exp2/summary.json"  # 汇总JSON路径
OUTPUT_DIR = "data/ablation_exp2/plots"  # 图表输出目录
PLOT_TITLE = "Model Performance vs Reconstruction Strength"  # 图表标题

# 样式配置
PLOT_STYLE = {
    'figsize': (10, 8),
    'dpi': 150,
    'font_size': 11,
    'line_width': 2,
    'marker_size': 8,
    'grid_alpha': 0.3
}

# ------------------------------------------------------
# 绘图函数
# ------------------------------------------------------
def load_summary(path: Path) -> Dict[str, Any]:
    """加载汇总JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_both_metrics(summary_data: Dict[str, Any], output_dir: Path):
    """
    将acc和ap画在同一张图上，不显示误差棒
    """
    strengths = summary_data['strengths']
    
    # 提取acc和ap数据（仅使用平均值）
    acc_means = [summary_data['summary'][str(s)]['acc']['mean'] for s in strengths]
    ap_means = [summary_data['summary'][str(s)]['ap']['mean'] for s in strengths]
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=PLOT_STYLE['figsize'], dpi=PLOT_STYLE['dpi'])
    
    # 绘制acc (左轴) - 实心圆形
    color_acc = '#1f77b4'
    ax1.plot(strengths, acc_means,
            marker='o', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='ACC', color=color_acc)
    
    ax1.set_xlabel('Strength', fontsize=PLOT_STYLE['font_size'] + 1)
    ax1.set_ylabel('ACC', fontsize=PLOT_STYLE['font_size'] + 1, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0.45, 0.8])
    
    # 绘制ap (右轴) - 实心方形
    ax2 = ax1.twinx()
    color_ap = '#ff7f0e'
    ax2.plot(strengths, ap_means,
            marker='s', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AP', color=color_ap)
    
    ax2.set_ylabel('AP', fontsize=PLOT_STYLE['font_size'] + 1, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([0.45, 0.8])
    
    # 标题和网格
    plt.title('ACC/AP with Different Diffusion Strength', 
              fontsize=PLOT_STYLE['font_size'] + 3)
    ax1.grid(True, alpha=PLOT_STYLE['grid_alpha'], linestyle='--')
    ax1.set_xticks(strengths)
    ax1.set_xticklabels([f"{s:.1f}" for s in strengths])
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='best', fontsize=PLOT_STYLE['font_size'])
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "both_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 组合图表已保存: {output_path}")


def main():
    """主流程: 加载汇总数据并绘图"""
    print("="*70)
    print("绘制性能曲线图")
    print("="*70)
    
    summary_path = Path(SUMMARY_JSON)
    if not summary_path.exists():
        print(f"✗ 汇总文件不存在: {summary_path}")
        print("请运行: python aggregate_results.py")
        return
    
    # 加载数据
    summary_data = load_summary(summary_path)
    print(f"✓ 加载成功: {len(summary_data['strengths'])} 个strength点")
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制组合图表
    print("\n绘制组合图表...")
    plot_both_metrics(summary_data, output_dir)
    
    print("\n" + "="*70)
    print("绘图完成!")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
