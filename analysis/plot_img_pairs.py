import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# ------------------------------------------------------
# 配置
# ------------------------------------------------------
IMG_PAIRS_JSON = "data/exp3_1_img_pairs/img_pairs_1_16.json"  # 图片对数据JSON路径
OUTPUT_DIR = "data/exp3_1_img_pairs/plots"  # 图表输出目录
PLOT_TITLE = "Model Performance vs Image Pairs"  # 图表标题

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
def load_img_pairs_data(path: Path) -> Dict[str, Any]:
    """加载图片对数据JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_img_pairs(img_pairs_data: Dict[str, Any], output_dir: Path):
    """
    绘制图片对数据：ACC/AP曲线和耗时柱状图
    要求：
    1. X轴使用对数坐标
    2. ACC叠放在AP上方（先绘AP，后绘ACC）
    3. AP使用红色
    4. 添加耗时柱状图（浅橙色，50%透明度）
    5. 对数坐标系下柱子宽度视觉一致
    """
    x_values = img_pairs_data['x_values']
    acc_means = img_pairs_data['acc_means']
    ap_means = img_pairs_data['ap_means']
    auc_means = img_pairs_data['auc_means']
    
    # 耗时数据（分钟）
    time_costs = [12, 23, 44, 87, 174]
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=PLOT_STYLE['figsize'], dpi=PLOT_STYLE['dpi'])
    
    # 设置对数坐标
    ax1.set_xscale('log', base=2)
    

    # 绘制AUC曲线（低调绿色）- 先绘制
    color_auc = '#388E3C'
    ax1.plot(x_values, auc_means,
            marker='^', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AUC', color=color_auc, zorder=2)


    # 绘制AP曲线（低调红色）- 先绘制
    color_ap = '#D32F2F'
    ax1.plot(x_values, ap_means,
            marker='s', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='AP', color=color_ap, zorder=3)
    
    # 绘制ACC曲线（蓝色）- 后绘制，叠放在AP上方
    color_acc = '#1f77b4'
    ax1.plot(x_values, acc_means,
            marker='o', markersize=PLOT_STYLE['marker_size'],
            linewidth=PLOT_STYLE['line_width'],
            label='ACC', color=color_acc, zorder=4)
    
    ax1.set_xlabel('Image Pairs', fontsize=PLOT_STYLE['font_size'] + 2)
    ax1.set_ylabel('ACC/AP/AUC', fontsize=PLOT_STYLE['font_size'] + 2, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0.55, 0.80])
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([str(x) for x in x_values])
    

    
    # 计算对数空间中的柱子宽度（确保视觉一致）
    log_x_values = np.log2(x_values)
    log_distances = np.diff(log_x_values)
    min_log_distance = np.min(log_distances)
    
    # 设置柱子宽度为最小对数距离的40%
    bar_width_factor = 0.4
    widths_linear = []
    
    for x in x_values:
        # 对数空间宽度
        log_width = min_log_distance * bar_width_factor
        # 转换为线性空间: width = x * (2^log_width - 1)
        width_linear = x * (2**log_width - 1)
        widths_linear.append(width_linear)
    
    # 创建右轴用于耗时柱状图
    ax2 = ax1.twinx()
    # 绘制耗时柱状图（橙色，30%透明度）
    color_time = '#FFB366'
    ax2.bar(x_values, time_costs,
           color=color_time, alpha=0.5,
           label='Running Time', width=widths_linear, zorder=1)
    
    ax2.set_ylabel('Running Time (minutes)', fontsize=PLOT_STYLE['font_size'] + 1, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim([0, 200])
    
    # 标题和网格
    plt.title(PLOT_TITLE, font='Timeserif', fontsize=PLOT_STYLE['font_size']*2)
    ax1.grid(True, alpha=PLOT_STYLE['grid_alpha'], linestyle='--', zorder=0)
    
    # 合并图例，放大50%
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper left', fontsize=int(PLOT_STYLE['font_size'] * 1.5))
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "img_pairs_performance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 图表已保存: {output_path}")


def main():
    """主流程: 加载图片对数据并绘图"""
    print("="*70)
    print("绘制图片对性能曲线图")
    print("="*70)
    
    data_path = Path(IMG_PAIRS_JSON)
    if not data_path.exists():
        print(f"✗ 数据文件不存在: {data_path}")
        return
    
    # 加载数据
    img_pairs_data = load_img_pairs_data(data_path)
    print(f"✓ 加载成功: {len(img_pairs_data['x_values'])} 个数据点")
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制图表
    print("\n绘制图表...")
    plot_img_pairs(img_pairs_data, output_dir)
    
    print("\n" + "="*70)
    print("绘图完成!")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
