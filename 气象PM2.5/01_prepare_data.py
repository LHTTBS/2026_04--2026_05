"""
真实气象+PM2.5 数据准备
=======================
输入：
  - xlsx月数据（广州/深圳/东莞/佛山 1980-2022.3）
  - 气象数据_完整_2000_2022.csv（温度+风向）
输出：
  - meteo_pm25_real.csv（合并后的真实数据集）
"""
import os
import pandas as pd
import numpy as np

# 路径
xlsx_path = r'C:\Users\Administrator\Desktop\data\基于气象与社会经济因子的PM2.5缺失值重建\国自然数据\PM2.5浓度重构\01_原始数据\1980年1月~2022年3月各城市地表PM2.5质量浓度(微克每立方米).xlsx'
meteo_path = r'C:\Users\Administrator\Desktop\data\基于气象与社会经济因子的PM2.5缺失值重建\国自然数据\气象数据_完整_2000_2022.csv'
out_path = r'C:\Users\Administrator\Desktop\data\气象PM2.5\meteo_pm25_real.csv'

# ============================================================
# 1. 读取PM2.5月数据
# ============================================================
print("读取xlsx...")
df_pm = pd.read_excel(xlsx_path, sheet_name='Sheet1')
print(f"原始形状: {df_pm.shape}")
print(f"列名: {df_pm.columns.tolist()}")

# 提取4市
df_pm['month'] = pd.to_datetime(df_pm['month'], errors='coerce')
df_pm['year'] = df_pm['month'].dt.year

target_cities = ['广州市', '深圳市', '东莞市', '佛山市']
short_names = {'广州市':'广州', '深圳市':'深圳', '东莞市':'东莞', '佛山市':'佛山'}

records = []
for city in target_cities:
    city_df = df_pm[df_pm['市'] == city].copy()
    if city_df.empty:
        print(f"  [WARN] {city} 无数据")
        continue
    annual = city_df.groupby('year')['地表PM2.5质量浓度'].mean().reset_index()
    annual.columns = ['年份', 'PM2.5浓度']
    annual['城市'] = short_names[city]
    records.append(annual)
    print(f"  {short_names[city]}: {len(annual)} 年, PM2.5范围={annual['PM2.5浓度'].min():.2f}~{annual['PM2.5浓度'].max():.2f}")

df_pm25 = pd.concat(records, ignore_index=True)
print(f"\nPM2.5年数据形状: {df_pm25.shape}")
print(df_pm25.head())

# ============================================================
# 2. 读取并合并气象数据
# ============================================================
print("\n读取气象数据...")
df_meteo = pd.read_csv(meteo_path, encoding='utf-8-sig')
print(f"气象数据形状: {df_meteo.shape}")

# 合并（取交集年份）
df = pd.merge(df_pm25, df_meteo, on=['年份', '城市'], how='inner')
print(f"\n合并后形状: {df.shape}")
print(df.head())
print(f"\n各城市样本数:\n{df['城市'].value_counts().sort_index()}")
print(f"年份范围: {df['年份'].min()} - {df['年份'].max()}")

# 保存
df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] 数据已保存: {out_path}")
