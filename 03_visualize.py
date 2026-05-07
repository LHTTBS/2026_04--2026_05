"""
真实气象+PM2.5  20方法评估结果可视化
====================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

out_dir = r'C:\Users\Administrator\Desktop\data\气象PM2.5'
results_df = pd.read_csv(os.path.join(out_dir, '20methods_results.csv'), encoding='utf-8-sig')
summary = pd.read_csv(os.path.join(out_dir, '20methods_summary.csv'), encoding='utf-8-sig')

missing_modes = results_df['missing_mode'].unique()
missing_rates = sorted(results_df['missing_rate'].unique())

# ============================================================
# 图1: 各缺失模式下分组柱状图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, mode in enumerate(missing_modes):
    ax = axes[idx]
    sub = summary[summary['missing_mode'] == mode]
    pivot = sub.pivot(index='method', columns='missing_rate', values='rmse_mean')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{mode} 缺失模式', fontsize=14)
    ax.set_xlabel('方法', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.legend(title='缺失率')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '图1_RMSE分组柱状图.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, '图1_RMSE分组柱状图.pdf'), bbox_inches='tight')
print("[OK] 图1_RMSE分组柱状图")
plt.close()

# ============================================================
# 图2: 热力图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for idx, mode in enumerate(missing_modes):
    ax = axes[idx]
    sub = summary[summary['missing_mode'] == mode]
    pivot = sub.pivot(index='method', columns='missing_rate', values='rmse_mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{c*100:.0f}%' for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f'{mode}', fontsize=14)
    ax.set_xlabel('缺失率', fontsize=12)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i,j]:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='RMSE')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '图2_RMSE热力图.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, '图2_RMSE热力图.pdf'), bbox_inches='tight')
print("[OK] 图2_RMSE热力图")
plt.close()

# ============================================================
# 图3: 综合平均排名
# ============================================================
rank_df = results_df.copy()
rank_df['rank'] = rank_df.groupby(['missing_mode', 'missing_rate', 'repeat'])['rmse'].rank()
avg_rank = rank_df.groupby('method')['rank'].mean().sort_values()

plt.figure(figsize=(10, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(avg_rank)))
bars = plt.barh(range(len(avg_rank)), avg_rank.values, color=colors)
plt.yticks(range(len(avg_rank)), avg_rank.index, fontsize=10)
plt.xlabel('平均排名（越小越好）', fontsize=12)
plt.title('20种方法综合平均排名', fontsize=14)
plt.gca().invert_yaxis()
for i, (m, v) in enumerate(avg_rank.items()):
    plt.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=9)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '图3_综合平均排名.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, '图3_综合平均排名.pdf'), bbox_inches='tight')
print("[OK] 图3_综合平均排名")
plt.close()

# ============================================================
# 图4: Top5方法敏感度分析（随缺失率变化）
# ============================================================
top5 = avg_rank.head(5).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, mode in enumerate(missing_modes):
    ax = axes[idx]
    for method in top5:
        sub = summary[(summary['missing_mode']==mode) & (summary['method']==method)]
        ax.plot(sub['missing_rate']*100, sub['rmse_mean'], marker='o', label=method)
    ax.set_title(f'{mode}', fontsize=14)
    ax.set_xlabel('缺失率 (%)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '图4_Top5敏感度分析.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, '图4_Top5敏感度分析.pdf'), bbox_inches='tight')
print("[OK] 图4_Top5敏感度分析")
plt.close()

print("\n[OK] 所有图表生成完毕！")
