# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
import networkx as nx
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 解决中文显示问题 (Windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NMFProcessor:
    def __init__(self, w_matrix_file, metadata_file, gnps_graphml_path, output_dir, n_components=11):
        self.w_matrix_file = w_matrix_file
        self.metadata_file = metadata_file
        self.gnps_graphml_path = gnps_graphml_path
        self.output_dir = output_dir
        self.n_components = int(n_components)

        self.W_df = None
        self.H_df = None
        self.edges_df = pd.DataFrame(columns=['Node 1', 'Node 2', 'Cosine Score'])
        self.color_map = {}
        self.cnmf_w_path = None  # ✅ 记录真正用于 UMAP 的 W

        self.custom_palette = [
            "#E46A6A", "#64B5F6", "#81C784", "#FFD54F",
            "#C37BCF", "#4DB6AC", "#F27AA2", "#FCBB74",
            "#A1887F", "#7D949F", "#7986CB", "#DCE775",
            "#B3E5FC", "#FF8A80", "#E0F7FA"
        ]

    # ---------------- 工具 ----------------
    def normalize_id(self, x):
        x = str(x).strip()
        if x.endswith(".0"):
            x = x[:-2]
        return x

    def _safe_format(self, value, fmt):
        if pd.isna(value):
            return "N/A"
        try:
            val = float(value)
            return f"{val:{fmt}}"
        except Exception:
            return str(value)

    # ---------------- 数据/分解 ----------------
    def load_data(self):
        """
        - 如果传入的是 W（包含 Component_* 列），直接使用并记录 cnmf_w_path；
        - 否则把传入看作原始矩阵，执行 NMF 得到 W/H，并保存到 output_dir，同步记录 cnmf_w_path。
        """
        os.makedirs(self.output_dir, exist_ok=True)

        df = pd.read_csv(self.w_matrix_file, index_col=0)
        comp_cols = [c for c in df.columns if str(c).startswith("Component_")]

        # ✅ Case 1: 传入即 W
        if len(comp_cols) >= 1:
            self.W_df = df.copy()
            self.H_df = pd.DataFrame()
            self.cnmf_w_path = self.w_matrix_file
            return

        # ✅ Case 2: 原始矩阵 -> CNMF
        data_matrix_df = df
        data_matrix = data_matrix_df.values

        # 固定随机
        np.random.seed(42)
        W_init = np.abs(np.random.rand(data_matrix.shape[0], self.n_components))
        H_init = np.abs(np.random.rand(self.n_components, data_matrix.shape[1]))

        nmf_model = NMF(n_components=self.n_components, init='custom', random_state=42, max_iter=1000)
        W_matrix = nmf_model.fit_transform(data_matrix, W=W_init, H=H_init)
        H_matrix = nmf_model.components_

        component_names = [f'Component_{i+1}' for i in range(self.n_components)]
        self.W_df = pd.DataFrame(W_matrix, index=data_matrix_df.index, columns=component_names)
        self.H_df = pd.DataFrame(H_matrix, index=component_names, columns=data_matrix_df.columns)

        # 保存
        w_out = os.path.join(self.output_dir, f"CNMF_W_matrix_n{self.n_components}.csv")
        h_out = os.path.join(self.output_dir, f"CNMF_H_matrix_n{self.n_components}.csv")
        self.W_df.to_csv(w_out)
        self.H_df.to_csv(h_out)
        self.cnmf_w_path = w_out  # ✅ UMAP 用它

        # 生成附加产物：top features & 可视化
        self._save_top_features()
        self._plot_h_matrix()
        self._plot_w_matrix()

    def _save_top_features(self):
        component_names = [f'Component_{i+1}' for i in range(self.n_components)]
        top_features = []
        for comp in component_names:
            top_mz = self.H_df.loc[comp].nlargest(15).index.tolist()
            for mz in top_mz:
                top_features.append([comp, mz])
        out = os.path.join(self.output_dir, f"top_features_per_component-{self.n_components}.csv")
        pd.DataFrame(top_features, columns=["Component", "Top_m/z"]).to_csv(out, index=False)

    def _plot_h_matrix(self):
        component_names = [f'Component_{i+1}' for i in range(self.n_components)]
        color_dict = {comp: self.custom_palette[i % len(self.custom_palette)] for i, comp in enumerate(component_names)}
        plt.figure(figsize=(12, 6), dpi=300)
        for comp in component_names:
            plt.plot(self.H_df.columns, self.H_df.loc[comp, :],
                     label=comp, linewidth=1.2, alpha=0.8, color=color_dict[comp])
        plt.xlabel("m/z（原始顺序）", fontsize=12)
        plt.ylabel("Component Weight", fontsize=12)
        plt.title("CNMF 成分与 m/z 关系", fontsize=14)
        xticks_itv = max(len(self.H_df.columns) // 15, 1)
        plt.xticks(range(0, len(self.H_df.columns), xticks_itv), self.H_df.columns[::xticks_itv], rotation=45, fontsize=10)
        plt.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out = os.path.join(self.output_dir, f"NMF_H_Matrix_Visualization_n{self.n_components}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_w_matrix(self):
        # 层次聚类并过滤低贡献样本（与 notebook 一致）
        linkage_matrix = linkage(self.W_df, method='ward', metric='euclidean')
        ordered_indices = leaves_list(linkage_matrix)
        W_sorted = self.W_df.iloc[ordered_indices, :]
        low_contrib = W_sorted.sum(axis=1) < 0.01
        W_filtered = W_sorted[~low_contrib]

        color_dict = {c: self.custom_palette[i % len(self.custom_palette)]
                      for i, c in enumerate(self.W_df.columns)}

        plt.figure(figsize=(14, 6), dpi=300)
        W_norm = W_filtered.div(W_filtered.sum(axis=1), axis=0)
        ax = W_norm.plot(kind="bar", stacked=True, width=0.8, figsize=(14, 6),
                         color=[color_dict[comp] for comp in self.W_df.columns])
        ax.set_xlabel("样本（已聚类）", fontsize=12)
        ax.set_ylabel("成分贡献比例", fontsize=12)
        ax.set_title("样本与成分的关系（聚类后）", fontsize=14)
        xticks_itv = max(len(W_filtered) // 20, 1)
        ax.set_xticks(range(0, len(W_filtered), xticks_itv))
        ax.set_xticklabels(W_filtered.index[::xticks_itv], rotation=45, fontsize=10)
        ax.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out = os.path.join(self.output_dir, f"NMF_W_Matrix_Visualization_n{self.n_components}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------- UMAP ----------------
    def process_umap(self, n_neighbors=10, min_dist=1.0, random_state=42):
        # ✅ 统一入口：优先使用记录的 W
        w_path = self.cnmf_w_path if self.cnmf_w_path and os.path.exists(self.cnmf_w_path) else self.w_matrix_file
        self.W_df = pd.read_csv(w_path, index_col=0)

        # 严格只用 Component_* 列
        component_cols = [c for c in self.W_df.columns if str(c).startswith("Component_")]
        component_cols = sorted(component_cols, key=lambda x: int(str(x).split("_")[-1]))
        X = self.W_df[component_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Top components / Node type
        top_n = 3
        self.W_df["Top_Components"] = self.W_df[component_cols].apply(lambda r: r.nlargest(top_n).index.tolist(), axis=1)
        self.W_df["Top_Contributions"] = self.W_df[component_cols].apply(lambda r: r.nlargest(top_n).values.tolist(), axis=1)

        def _node_type(row, thr=0.05):
            c1 = row["Top_Contributions"][0]
            c2 = row["Top_Contributions"][1] if len(row["Top_Contributions"]) > 1 else 0
            diff_ratio = abs(c1 - c2) / (c1 + c2) if (c1 + c2) > 0 else 1
            return "multi" if diff_ratio < thr else "single"
        self.W_df["Node_Type"] = self.W_df.apply(_node_type, axis=1)
        # >>> 调试：主成分分布 & 见到的成分集合
        print("[DEBUG] Primary component counts:\n",
              self.W_df["Top_Components"].str[0].value_counts().sort_index())
        print("[DEBUG] Components seen (Top_1/2/3 union):",
              sorted(self.W_df["Top_Components"].explode().unique(),
                     key=lambda x: int(x.split("_")[-1])))
        print("[DEBUG] Component columns in W:", component_cols)

        # UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', random_state=random_state)
        embedding = reducer.fit_transform(X)
        self.W_df["UMAP_1"] = embedding[:, 0]
        self.W_df["UMAP_2"] = embedding[:, 1]

        # 元数据
        if self.metadata_file and os.path.exists(self.metadata_file):
            meta = pd.read_csv(self.metadata_file)
            meta["row ID"] = meta["row ID"].apply(self.normalize_id)
            self.W_df.index = self.W_df.index.map(self.normalize_id)
            meta_dict = meta.set_index("row ID")[["row m/z", "row retention time"]].to_dict(orient="index")
            self.W_df["m/z"] = self.W_df.index.map(lambda idx: float(meta_dict[idx]["row m/z"])
                                                   if idx in meta_dict else np.nan)
            self.W_df["Retention Time"] = self.W_df.index.map(lambda idx: float(meta_dict[idx]["row retention time"])
                                                               if idx in meta_dict else np.nan)
        else:
            self.W_df["m/z"] = np.nan
            self.W_df["Retention Time"] = np.nan

        # 颜色映射
        all_components = sorted(self.W_df["Top_Components"].explode().unique(),
                                key=lambda x: int(str(x).split("_")[-1]))
        self.color_map = {comp: self.custom_palette[i % len(self.custom_palette)]
                          for i, comp in enumerate(all_components)}
        self.W_df["Primary_Component"] = self.W_df["Top_Components"].apply(lambda x: x[0])
        self.W_df["Color"] = self.W_df["Primary_Component"].map(self.color_map)

        # GNPS 边
        if self.gnps_graphml_path and os.path.exists(self.gnps_graphml_path):
            gnps_network = nx.read_graphml(self.gnps_graphml_path)
            edges_data = []
            for n1, n2, edata in gnps_network.edges(data=True):
                cosine_score = float(edata.get('cosine_score', 1.0))
                node1 = self.normalize_id(n1)
                node2 = self.normalize_id(n2)
                edges_data.append({'Node 1': node1, 'Node 2': node2, 'Cosine Score': cosine_score})
            self.edges_df = pd.DataFrame(edges_data)
            all_nodes = set(self.W_df.index)
            self.edges_df = self.edges_df[
                self.edges_df["Node 1"].isin(all_nodes) &
                self.edges_df["Node 2"].isin(all_nodes)
            ]
        else:
            self.edges_df = pd.DataFrame(columns=['Node 1', 'Node 2', 'Cosine Score'])

        # 保存全部节点信息
        all_nodes_info = os.path.join(self.output_dir, "all_nodes_info.csv")
        self.W_df.to_csv(all_nodes_info)

    # ---------------- 绘图 ----------------
    def plot_graph(self, highlight_samples=None, display_option="None", cosine_range=None):
        if cosine_range is None:
            cosine_range = [0.6, 1.0]
        if highlight_samples is None:
            highlight_samples = []

        highlight_samples = [self.normalize_id(idx) for idx in highlight_samples]
        fig = go.Figure()

        # 成分集合
        all_components = sorted(self.W_df["Top_Components"].explode().unique(),
                                key=lambda x: int(str(x).split("_")[-1]))

        # A) UMAP 节点
        for comp in all_components:
            comp_df = self.W_df[self.W_df["Primary_Component"] == comp]
            color = self.color_map.get(comp, "#888888")

            # 文本
            if display_option == "Node ID":
                text_info = comp_df.index.astype(str)
            elif display_option == "PEPMASS":
                text_info = comp_df["m/z"].apply(lambda x: self._safe_format(x, ".4f"))
            elif display_option == "Retention Time":
                text_info = comp_df["Retention Time"].apply(lambda x: self._safe_format(x, ".2f"))
            elif display_option == "All":
                text_info = comp_df.apply(
                    lambda row: f"{row.name}, {self._safe_format(row['m/z'], '.4f')}, {self._safe_format(row['Retention Time'], '.2f')}",
                    axis=1
                )
            else:
                text_info = ""

            # hover
            hover_info = comp_df.apply(
                lambda row: (
                    f"Sample ID: {row.name}<br>"
                    f"Component: {' & '.join(row['Top_Components'][:2]) if row['Node_Type']=='multi' else row['Primary_Component']}<br>"
                    f"m/z: {self._safe_format(row['m/z'], '.4f')}<br>"
                    f"Retention Time: {self._safe_format(row['Retention Time'], '.2f')}"
                ),
                axis=1
            )

            # multi 节点用第二成分颜色作为边框
            line_colors, line_widths = [], []
            for idx in comp_df.index:
                row = self.W_df.loc[idx]
                if row["Node_Type"] == "multi" and len(row["Top_Components"]) > 1:
                    second_comp = row["Top_Components"][1]
                    edge_c = self.color_map.get(second_comp, "black")
                    line_colors.append(edge_c); line_widths.append(4)
                else:
                    line_colors.append("rgba(0,0,0,0)"); line_widths.append(0)

            fig.add_trace(go.Scatter(
                x=comp_df["UMAP_1"], y=comp_df["UMAP_2"],
                mode=("markers+text" if display_option != "None" else "markers"),
                marker=dict(
                    size=12,
                    color=color,
                    opacity=0.85,
                    line=dict(width=line_widths, color=line_colors)
                ),
                text=text_info,
                textposition="top center",
                textfont=dict(size=7),
                hoverinfo="text",
                hovertext=hover_info,
                name=f"Component {comp.split('_')[-1]}"
            ))

        # B) 高亮
        highlight_samples = [idx for idx in highlight_samples if idx in self.W_df.index]
        if highlight_samples:
            highlight_df = self.W_df.loc[highlight_samples]
            fig.add_trace(go.Scatter(
                x=highlight_df["UMAP_1"], y=highlight_df["UMAP_2"],
                mode="markers",
                marker=dict(
                    size=14,
                    color=highlight_df["Primary_Component"].map(self.color_map),
                    opacity=1.0,
                    line=dict(width=4, color="red")
                ),
                hoverinfo="text",
                hovertext=highlight_df.apply(
                    lambda row: (
                        f"🔍 FOUND!<br>"
                        f"Sample ID: {row.name}<br>"
                        f"Component: {' & '.join(row['Top_Components'][:2]) if row['Node_Type']=='multi' else row['Primary_Component']}<br>"
                        f"m/z: {self._safe_format(row['m/z'], '.4f')}<br>"
                        f"Retention Time: {self._safe_format(row['Retention Time'], '.2f')}"
                    ),
                    axis=1
                ),
                showlegend=False
            ))

            if not highlight_df.empty:
                min_x, max_x = highlight_df["UMAP_1"].min(), highlight_df["UMAP_1"].max()
                min_y, max_y = highlight_df["UMAP_2"].min(), highlight_df["UMAP_2"].max()
                fig.update_layout(xaxis=dict(range=[min_x - 2, max_x + 2]),
                                  yaxis=dict(range=[min_y - 2, max_y + 2]))

        # C) GNPS 边
        cmin, cmax = cosine_range
        drawn = 0
        if not self.edges_df.empty:
            edges = self.edges_df[(self.edges_df["Cosine Score"] >= cmin) & (self.edges_df["Cosine Score"] <= cmax)]
            for _, ed in edges.iterrows():
                n1, n2, score = ed["Node 1"], ed["Node 2"], ed["Cosine Score"]
                if n1 in self.W_df.index and n2 in self.W_df.index:
                    x1, y1 = self.W_df.loc[n1, "UMAP_1"], self.W_df.loc[n1, "UMAP_2"]
                    x2, y2 = self.W_df.loc[n2, "UMAP_1"], self.W_df.loc[n2, "UMAP_2"]
                    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
                        continue
                    fig.add_trace(go.Scatter(
                        x=[x1, x2], y=[y1, y2], mode="lines",
                        line=dict(width=score * 2, color='gray'),
                        opacity=score, hoverinfo="text",
                        hovertext=f"Cosine Score: {score:.2f}", showlegend=False
                    ))
                    drawn += 1

        fig.update_layout(
            width=1000, height=650, template="plotly_white",
            title="UMAP Visualization with Interactive Search",
            xaxis_title="UMAP Component 1", yaxis_title="UMAP Component 2",
            hovermode="closest", legend_title="Components",
            xaxis=dict(showgrid=False, zeroline=False, showline=False),
            yaxis=dict(showgrid=False, zeroline=False, showline=False),
            paper_bgcolor='white', plot_bgcolor='white'
        )
        return fig

    # Dash 布局占位
    def build_dash_layout(self, dash_app):
        pass
