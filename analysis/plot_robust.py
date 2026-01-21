import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# ------------------------------------------------------
# 配置
# ------------------------------------------------------
ROBUSTNESS_JSON = "data/exp4_robnest/gaussian_blur.json"  # 鲁棒性数据JSON路径
COMPRESSION_JSON = "data/exp4_robnest/jpeg_compression.json"  # jpeg压缩数据JSON路径
OUTPUT_DIR = "data/exp4_robnest/plots"  # 图表输出目录


# 样式配置
PLOT_STYLE = {
    'figsize': (10, 8),
    'dpi': 300,
    'font_size': 11,
    'line_width': 2,
    'marker_size': 8,
    'grid_alpha': 0.3
}

# ------------------------------------------------------
# 绘图函数
# ------------------------------------------------------
def load_data(path: Path) -> Dict[str, Any]:
    """加载数据JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_robustness_data(robustness_data: Dict[str, Any], output_dir: Path):
    """
    绘制鲁棒性数据：ACC/AP折线图
    展示模型性能随扰动强度变化的趋势
    """
    x_values = robustness_data['x_values']
    acc_means = robustness_data['acc_means']
    ap_means = robustness_data['ap_means']
    auc_means = robustness_data['auc_means']
    
    # 使用等间距的索引作为x坐标
    x_positions = np.arange(len(x_values))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'], dpi=PLOT_STYLE['dpi'])
    
    # 绘制AUC曲线（绿色）
    color_auc = '#388E3C'
    ax.plot(x_positions, auc_means,
            marker='^', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AUC', color=color_auc, zorder=2)
    
    
    # 绘制AP曲线（红色）- 先绘制
    color_ap = '#D32F2F'
    ax.plot(x_positions, ap_means,
            marker='s', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AP', color=color_ap, zorder=3)
    
    # 绘制ACC曲线（蓝色）- 后绘制，叠放在AP上方
    color_acc = '#1f77b4'
    ax.plot(x_positions, acc_means,
            marker='o', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='ACC', color=color_acc, zorder=4)
    
    # 设置标签和标题
    ax.set_xlabel('Gaussian Blur', fontsize=PLOT_STYLE['font_size'] + 2)
    ax.set_ylabel('ACC/AP/AUC', fontsize=PLOT_STYLE['font_size'] + 2)
    ax.set_ylim([0.5, 0.8])
    
    # 设置X轴刻度（等间距，显示原始值）
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(x) for x in x_values])
    
    # 标题和网格
    PLOT_TITLE = "Robustness to Gaussian Blur"  # 图表标题
    plt.title(PLOT_TITLE, fontsize=PLOT_STYLE['font_size'] * 2)
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'], linestyle='--', zorder=0)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=int(PLOT_STYLE['font_size'] * 2))
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "gaussian_blur_robustness.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 图表已保存: {output_path}")

def plot_compression_data(compression_data: Dict[str, Any], output_dir: Path):
    """
    绘制鲁棒性数据：ACC/AP折线图
    展示模型性能随扰动强度变化的趋势
    """
    x_values = compression_data['x_values']
    acc_means = compression_data['acc_means']
    ap_means = compression_data['ap_means']
    auc_means = compression_data['auc_means']
    
    # 使用等间距的索引作为x坐标
    x_positions = np.arange(len(x_values))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'], dpi=PLOT_STYLE['dpi'])
    
    # 绘制AUC曲线（绿色）
    color_auc = '#388E3C'
    ax.plot(x_positions, auc_means,
            marker='^', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AUC', color=color_auc, zorder=2)
    
    
    # 绘制AP曲线（红色）- 先绘制
    color_ap = '#D32F2F'
    ax.plot(x_positions, ap_means,
            marker='s', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AP', color=color_ap, zorder=3)
    
    # 绘制ACC曲线（蓝色）- 后绘制，叠放在AP上方
    color_acc = '#1f77b4'
    ax.plot(x_positions, acc_means,
            marker='o', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='ACC', color=color_acc, zorder=4)
    
    # 设置标签和标题
    ax.set_xlabel('Jpeg Image Quality', fontsize=PLOT_STYLE['font_size'] + 2)
    ax.set_ylabel('ACC/AP/AUC', fontsize=PLOT_STYLE['font_size'] + 2)
    ax.set_ylim([0.5, 0.8])
    
    # 设置X轴刻度（等间距，显示原始值）
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(x) for x in x_values])
    
    # 标题和网格
    PLOT_TITLE = "Robustness to jpeg compression quality"  # 图表标题
    plt.title(PLOT_TITLE, fontsize=PLOT_STYLE['font_size'] * 2)
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'], linestyle='--', zorder=0)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=int(PLOT_STYLE['font_size'] * 2))
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "jpeg_compression_robustness.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 图表已保存: {output_path}")

def main():
    """主流程: 加载鲁棒性数据并绘图"""

    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("绘制鲁棒性曲线图")
    print("="*70)
    
    robustness_data_path = Path(ROBUSTNESS_JSON)
    if not robustness_data_path.exists():
        print(f"✗ gaussian blur 数据文件不存在: {robustness_data_path}")
        return
    
    robustness_data = load_data(robustness_data_path)
    plot_robustness_data(robustness_data, output_dir)

    compression_data_path = Path(COMPRESSION_JSON)
    if not compression_data_path.exists():
        print(f"✗ jpeg compression 数据文件不存在: {compression_data_path}")
        return

    compression_data = load_data(compression_data_path)
    plot_compression_data(compression_data, output_dir)

    print("="*70)
    print("绘制完成")


if __name__ == "__main__":
    main()
