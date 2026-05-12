"""
真实气象+PM2.5  20种缺失值重构方法评估
======================================
数据：meteo_pm25_real.csv（温度+风向+PM2.5，92条）
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. 加载数据
# ============================================================
data_path = r'C:\Users\Administrator\Desktop\data\气象PM2.5\meteo_pm25_real.csv'
df = pd.read_csv(data_path, encoding='utf-8-sig')
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(df.head())

target_col = 'PM2.5浓度'
feature_cols = [c for c in df.columns if c not in ['城市', '年份', target_col]]
print(f"\n特征: {feature_cols}")

# 确保数值型
for c in feature_cols + [target_col]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ============================================================
# 2. 缺失模拟
# ============================================================
def simulate_missing(df_full, target_col, missing_rate=0.2, mode='MCAR'):
    df_missing = df_full.copy()
    cities = df_missing['城市'].unique()
    for city in cities:
        idx = df_missing[df_missing['城市'] == city].index
        n = len(idx)
        n_miss = max(1, int(n * missing_rate))
        if mode == 'MCAR':
            miss_idx = np.random.choice(idx, size=n_miss, replace=False)
        elif mode == 'MAR':
            # 基于温度排序，高温更容易缺失
            temp_vals = df_missing.loc[idx, '温度'].values
            probs = np.argsort(np.argsort(-temp_vals)) + 1
            probs = probs / probs.sum()
            miss_idx = np.random.choice(idx, size=n_miss, replace=False, p=probs)
        elif mode == 'MNAR':
            # 高PM2.5更容易缺失
            pm_vals = df_missing.loc[idx, target_col].values
            threshold = np.nanpercentile(pm_vals, 60)
            high_idx = idx[pm_vals > threshold]
            if len(high_idx) >= n_miss:
                miss_idx = np.random.choice(high_idx, size=n_miss, replace=False)
            else:
                extra = n_miss - len(high_idx)
                other_idx = np.setdiff1d(idx, high_idx)
                miss_idx = np.concatenate([
                    high_idx,
                    np.random.choice(other_idx, size=extra, replace=False)
                ])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        df_missing.loc[miss_idx, target_col] = np.nan
    return df_missing

# ============================================================
# 3. 20种方法
# ============================================================
def get_methods():
    return {
        '01_均值填充': 'mean',
        '02_中位数填充': 'median',
        '03_KNN填充_k3': KNNImputer(n_neighbors=3),
        '04_KNN填充_k5': KNNImputer(n_neighbors=5),
        '05_线性回归': LinearRegression(),
        '06_岭回归': Ridge(alpha=1.0),
        '07_Lasso': Lasso(alpha=0.1, max_iter=5000),
        '08_ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        '09_贝叶斯岭回归': BayesianRidge(),
        '10_决策树': DecisionTreeRegressor(random_state=42, max_depth=5),
        '11_随机森林': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        '12_极端随机树': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        '13_GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        '14_AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        '15_SVR_rbf': SVR(kernel='rbf', C=1.0),
        '16_KNN回归_k3': KNeighborsRegressor(n_neighbors=3),
        '17_KNN回归_k5': KNeighborsRegressor(n_neighbors=5),
        '18_MLP神经网络': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
        '19_迭代插补_贝叶斯': IterativeImputer(random_state=42, max_iter=10),
        '20_迭代插补_随机树': IterativeImputer(random_state=42, estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), max_iter=10),
    }

# ============================================================
# 4. 评估函数
# ============================================================
def evaluate_method(df_missing, df_true, target_col, feature_cols, method_name, method, scaler):
    cities = df_missing['城市'].unique()
    y_true_list = []
    y_pred_list = []

    for city in cities:
        city_idx = df_missing[df_missing['城市'] == city].index
        missing_mask = df_missing.loc[city_idx, target_col].isna()
        if missing_mask.sum() == 0:
            continue

        X_test_raw = df_missing.loc[city_idx, feature_cols].values
        train_idx = df_missing[
            (df_missing['城市'] != city) | (~df_missing[target_col].isna())
        ].index
        X_train_raw = df_missing.loc[train_idx, feature_cols].values
        y_train = df_missing.loc[train_idx, target_col].values
        y_train_filled = pd.Series(y_train).fillna(pd.Series(y_train).median()).values

        # 特征缺失填充（中位数）+ 标准化
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test_raw, columns=feature_cols)
        for c in feature_cols:
            med = X_train_df[c].median()
            X_train_df[c] = X_train_df[c].fillna(med)
            X_test_df[c] = X_test_df[c].fillna(med)

        if len(feature_cols) == 1:
            # 单特征时fit_transform需要reshape
            X_train = scaler.fit_transform(X_train_df.values.reshape(-1, 1))
            X_test = scaler.transform(X_test_df.values.reshape(-1, 1))
        else:
            X_train = scaler.fit_transform(X_train_df.values)
            X_test = scaler.transform(X_test_df.values)

        try:
            if method_name.startswith(('01_', '02_')):
                pred_val = np.nanmean(y_train_filled) if method == 'mean' else np.nanmedian(y_train_filled)
                y_pred = np.full(missing_mask.sum(), pred_val)
            elif method_name.startswith(('03_', '04_', '19_', '20_')):
                full_train = df_missing.loc[train_idx, feature_cols + [target_col]].copy()
                full_test_city = df_missing.loc[city_idx, feature_cols + [target_col]].copy()
                combined = pd.concat([full_train, full_test_city])
                combined_imputed = pd.DataFrame(
                    method.fit_transform(combined),
                    columns=combined.columns,
                    index=combined.index
                )
                y_pred = combined_imputed.loc[city_idx[missing_mask], target_col].values
            else:
                valid_train = ~np.isnan(y_train)
                if valid_train.sum() < 5:
                    y_pred = np.full(missing_mask.sum(), np.nanmedian(y_train_filled))
                else:
                    m = method.fit(X_train[valid_train], y_train[valid_train])
                    y_pred = m.predict(X_test[missing_mask])
        except Exception as e:
            print(f"    [ERR] {method_name} @ {city}: {e}")
            y_pred = np.full(missing_mask.sum(), np.nanmedian(y_train_filled))

        y_true_list.extend(df_true.loc[city_idx[missing_mask], target_col].values)
        y_pred_list.extend(y_pred)

    if len(y_true_list) == 0:
        return np.nan
    rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
    return rmse

# ============================================================
# 5. 主实验
# ============================================================
def main():
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    missing_modes = ['MCAR', 'MAR', 'MNAR']
    n_repeats = 5
    methods = get_methods()
    scaler = StandardScaler()

    results = []

    for mode in missing_modes:
        for mr in missing_rates:
            print(f"\n{'='*50}")
            print(f"模式: {mode}, 缺失率: {mr}")
            print(f"{'='*50}")
            for rep in range(n_repeats):
                np.random.seed(42 + rep)
                df_missing = simulate_missing(df, target_col, missing_rate=mr, mode=mode)
                for method_name, method in methods.items():
                    rmse = evaluate_method(df_missing, df, target_col, feature_cols, method_name, method, scaler)
                    results.append({
                        'method': method_name,
                        'missing_mode': mode,
                        'missing_rate': mr,
                        'repeat': rep + 1,
                        'rmse': rmse,
                    })
                    print(f"  {method_name} (rep{rep+1}): RMSE={rmse:.4f}")

    results_df = pd.DataFrame(results)
    out_dir = r'C:\Users\Administrator\Desktop\data\气象PM2.5'
    results_df.to_csv(os.path.join(out_dir, '20methods_results.csv'), index=False, encoding='utf-8-sig')
    print(f"\n[OK] 详细结果已保存: 20methods_results.csv")

    # 汇总统计
    summary = results_df.groupby(['method', 'missing_mode', 'missing_rate'])['rmse'].agg(['mean', 'std']).reset_index()
    summary.columns = ['method', 'missing_mode', 'missing_rate', 'rmse_mean', 'rmse_std']
    summary.to_csv(os.path.join(out_dir, '20methods_summary.csv'), index=False, encoding='utf-8-sig')
    print(f"[OK] 汇总结果已保存: 20methods_summary.csv")

    # 快速结论
    print("\n" + "="*60)
    print("各场景最优方法（RMSE最低）")
    print("="*60)
    for mode in missing_modes:
        for mr in missing_rates:
            sub = summary[(summary['missing_mode']==mode) & (summary['missing_rate']==mr)]
            best = sub.loc[sub['rmse_mean'].idxmin()]
            print(f"{mode} {mr*100:.0f}%: {best['method']} (RMSE={best['rmse_mean']:.4f}±{best['rmse_std']:.4f})")

    # 综合排名
    rank_df = results_df.copy()
    rank_df['rank'] = rank_df.groupby(['missing_mode', 'missing_rate', 'repeat'])['rmse'].rank()
    avg_rank = rank_df.groupby('method')['rank'].mean().sort_values()
    print("\n" + "="*60)
    print("综合平均排名（越小越好）")
    print("="*60)
    for m, r in avg_rank.items():
        print(f"{m}: {r:.2f}")

if __name__ == '__main__':
    main()
