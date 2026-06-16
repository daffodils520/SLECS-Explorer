import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import os

class NMFEvaluator:
    def __init__(self, data_file, output_dir):
        self.data_file = data_file
        self.output_dir = output_dir
        self.matrix_df = None
        self.data_matrix = None

    def load_data(self):
        """加载数据矩阵"""
        self.matrix_df = pd.read_csv(self.data_file, index_col=0)
        self.data_matrix = self.matrix_df.values  # 转换为 NumPy 数组

    def evaluate(self, max_components=20, min_components=2, step=1):
        """评估不同的 n_components 值，生成两个图：重构误差图和解释方差图"""
        if self.data_matrix is None:
            self.load_data()

        # 确定 n_components 的范围
        n_samples, n_features = self.data_matrix.shape
        max_components = min(max_components, min(n_samples, n_features))
        components_range = range(min_components, max_components + 1, step)

        # 计算重构误差
        reconstruction_errors = []
        for n in components_range:
            model = NMF(n_components=n, init='random', random_state=42, max_iter=1000)
            W = model.fit_transform(self.data_matrix)
            H = model.components_
            reconstruction_error = model.reconstruction_err_  # sklearn 内置误差计算
            reconstruction_errors.append(reconstruction_error)

        # 绘制重构误差图
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, reconstruction_errors, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Components (n_components)")
        plt.ylabel("Reconstruction Error")
        plt.title("Choosing Optimal n_components for CNMF")
        plt.grid()
        error_plot_path = os.path.join(self.output_dir, "reconstruction_error.png")
        plt.savefig(error_plot_path)
        plt.close()

        # 计算解释方差
        explained_variances = []
        total_variance = np.var(self.data_matrix)  # 计算原始数据矩阵的总方差
        for n in components_range:
            nmf_model = NMF(n_components=n, init='random', random_state=42, max_iter=1000)
            W_matrix = nmf_model.fit_transform(self.data_matrix)
            component_variance = np.var(W_matrix, axis=0)
            explained_variance_ratio = np.sum(component_variance) / total_variance
            explained_variances.append(explained_variance_ratio)

        # 绘制解释方差图
        plt.figure(figsize=(8, 5))
        plt.plot(components_range, explained_variances, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Components (n_components)")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Choosing Optimal n_components using Explained Variance")
        plt.grid()
        variance_plot_path = os.path.join(self.output_dir, "explained_variance.png")
        plt.savefig(variance_plot_path)
        plt.close()