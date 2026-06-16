import os
import re
import gzip
import numpy as np
import pandas as pd
from itertools import product

from pyteomics.mass import calculate_mass
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------- 工具 ----------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _write_empty_csv(path, columns):
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    return path


# ---------------- Step①：构建完整碎片矩阵 ----------------

def extract_spectra_from_mgf(mgf_file_path):
    """
    逐行解析 .mgf/.mgf.gz，提取 (FEATURE_ID, PEPMASS, [(mz, intensity), ...])
    """
    spectra_data = []
    cur_id, cur_pm, cur_ints = None, None, []

    opener = gzip.open if str(mgf_file_path).lower().endswith('.gz') else open
    mode = 'rt' if opener is gzip.open else 'r'

    with opener(mgf_file_path, mode, encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("FEATURE_ID="):
                cur_id = line.split('=', 1)[1].strip()
            elif line.startswith("PEPMASS="):
                try:
                    cur_pm = float(line.split('=', 1)[1].split()[0].strip())
                except Exception:
                    cur_pm = None
            elif line.startswith("END IONS"):
                if cur_id and cur_ints:
                    spectra_data.append((cur_id, cur_pm, cur_ints))
                cur_id, cur_pm, cur_ints = None, None, []
            elif cur_pm is not None and ' ' in line:
                try:
                    mz, inten = map(float, line.split()[:2])
                    cur_ints.append((mz, inten))
                except Exception:
                    continue
    return spectra_data


def build_fragment_matrix(spectra_data, output_csv):
    """
    列 = m/z（字符串），行 = FEATURE_ID；值 = intensity
    与 jupyter 版本保持一致：to_csv() 默认 header+index，供 Step② 以 header=None 读取。
    """
    all_mz = sorted({mz for _, _, ints in spectra_data for mz, _ in ints})
    df = pd.DataFrame(0, index=[fid for fid, _, _ in spectra_data],
                      columns=[str(m) for m in all_mz])
    for fid, _, ints in spectra_data:
        for mz, inten in ints:
            col = str(mz)
            if col in df.columns:
                df.at[fid, col] = inten
    df.to_csv(output_csv)
    return output_csv


# ---------------- Step②：分箱 ----------------

def bin_mz_values(input_csv, output_csv, bin_size=0.01):
    data = pd.read_csv(input_csv, header=None)
    mz_values = data.iloc[0, 1:].astype(float).values
    intensity = data.iloc[1:, 1:].values

    bins = np.floor(mz_values / bin_size) * bin_size
    uniq = np.unique(bins)

    result_mz = []
    result_intensity = []
    for bv in uniq:
        idx = np.where(bins == bv)[0]
        bin_ints = intensity[:, idx]
        if len(idx) == 1:
            selected_mz = mz_values[idx[0]]
            max_per_row = bin_ints[:, 0]
        else:
            bin_mz_vals = mz_values[idx]
            max_idx = np.unravel_index(np.argmax(bin_ints), bin_ints.shape)
            selected_mz = bin_mz_vals[max_idx[1]]
            max_per_row = bin_ints.max(axis=1)
        result_mz.append(selected_mz)
        result_intensity.append(max_per_row)

    result_intensity = np.array(result_intensity).T
    result_df = pd.DataFrame(result_intensity, columns=[f"{mz:.4f}" for mz in result_mz])
    result_df.insert(0, "FEATURE_ID", data.iloc[1:, 0].values)
    mz_row = pd.DataFrame([[np.nan] + result_mz], columns=result_df.columns)
    out = pd.concat([mz_row, result_df], ignore_index=True)
    out.to_csv(output_csv, index=False, header=False)
    return output_csv


# ---------------- Step③：CHO/CHON 化学式注释 ----------------

def annotate_formulas(binned_csv, output_csv, tolerance=0.02):
    data = pd.read_csv(binned_csv, header=None)
    mz_values = data.iloc[0, 1:].astype(float).values

    elements_cho = {"C": range(1, 40), "H": range(1, 50), "O": range(0, 15)}
    elements_chon = {"C": range(1, 40), "H": range(1, 50), "O": range(0, 15), "N": range(0, 3)}
    mass_formula_cache = {}

    def is_valid_formula(C, H, O, N=0):
        if H + O + N > 4 * C: return False
        if N > 0 and H > 3 * N: return False
        return True

    def generate(elements):
        names = list(elements.keys())
        ranges = [elements[k] for k in names]
        possible = []
        for counts in product(*ranges):
            f = {n: c for n, c in zip(names, counts)}
            C, H, O, N = f.get("C", 0), f.get("H", 0), f.get("O", 0), f.get("N", 0)
            if O > 0 and not (5 <= C / O <= 15):
                continue
            if not is_valid_formula(C, H, O, N):
                continue
            mass = calculate_mass(f)
            possible.append((mass, f))
        return possible

    cho = generate(elements_cho)
    chon = generate(elements_chon)

    def best(mz):
        mass = mz - 1.0078
        if mass in mass_formula_cache:
            return mass_formula_cache[mass]
        best_formula, best_diff = None, float("inf")
        for pool in (cho, chon):
            for mass_val, f in pool:
                d = abs(mass_val - mass)
                if d <= tolerance and d < best_diff:
                    best_diff = d
                    best_formula = "".join(f"{k}{v}" for k, v in f.items() if v > 0)
            if best_formula:
                break
        res = best_formula if best_formula else "No match"
        mass_formula_cache[mass] = res
        return res

    formulas = [best(mz) for mz in tqdm(mz_values, desc="Processing m/z")]
    pd.DataFrame({"m/z": mz_values, "Formula": formulas}).to_csv(output_csv, index=False)
    return output_csv


# ---------------- Step④：筛选 CHO 并保留原始矩阵列 ----------------

def filter_CHO(binned_csv, formula_csv, output_csv):
    def is_CHO(s):
        return bool(re.fullmatch(r"C\d*H\d*O?\d*", str(s)))

    formulas = pd.read_csv(formula_csv)
    keep_mz = formulas[formulas["Formula"].apply(is_CHO)]["m/z"].astype(str).tolist()

    orig = pd.read_csv(binned_csv)
    orig.columns = orig.columns.astype(str).str.strip()

    match_cols = [c for c in orig.columns if c in keep_mz]
    cols = [orig.columns[0]] + match_cols
    filtered = orig[cols]

    # 首列列名置空，index 不输出
    filtered.columns.values[0] = ""
    filtered.to_csv(output_csv, index=False)
    return output_csv


# ---------------- Step⑤：结构性质量差匹配与转化关系 ----------------

def match_structural_transformations(cho_matrix_csv, output_csv):
    data = pd.read_csv(cho_matrix_csv, index_col=0)
    if data.shape[1] == 0:
        return _write_empty_csv(output_csv,
                                ["Feature_1_mz", "Feature_2_mz", "Observed_Difference", "Matched_Transformation",
                                 "Estimated_M"])

    mz_cols = sorted(data.columns.astype(float), reverse=True)

    adduct_masses = {
        "[M+H]+": 1.007276,
        "[M+Na]+": 22.989218,
        "[M+NH4]+": 18.033823
    }
    neutral_losses = {
        ("[M+H]+", "[M+Na]+"): 21.981942,
        ("[M+H]+", "[M+NH4]+"): 17.026547,
        ("[M+H]+", "[M+H-H2O]+"): 18.010565,
        ("[M+H-H2O]+", "[M+H-2H2O]+"): 18.010565
    }

    def within(v, t, tol=0.01):  # 按脚本 0.01
        return abs(v - t) <= tol

    results, used = [], set()

    for i, mz1 in enumerate(mz_cols):
        if mz1 in used:
            continue

        # 估计中性母体质量（取最大）
        est_M = None
        for _, off in adduct_masses.items():
            cand = round(mz1 - off, 5)
            if est_M is None or cand > est_M:
                est_M = cand

        matched = []
        for mz2 in mz_cols[i + 1:]:
            diff = mz1 - mz2
            for (a1, a2), theo in neutral_losses.items():
                if within(diff, theo):
                    matched.append({
                        "Feature_1_mz": round(mz1, 5),
                        "Feature_2_mz": round(mz2, 5),
                        "Observed_Difference": round(diff, 5),
                        "Matched_Transformation": f"{a1} -> {a2}",
                        "Estimated_M": est_M
                    })
                    used.add(mz2)
        if matched:
            results.extend(matched)
            used.add(mz1)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    return output_csv


# ---------------- Step⑥：根据结构转化匹配过滤矩阵 ----------------

def filter_by_transformations(cho_matrix_csv, transform_csv, output_csv):
    data = pd.read_csv(cho_matrix_csv, index_col=0)
    if not os.path.exists(transform_csv) or os.path.getsize(transform_csv) == 0:
        data.iloc[:, :0].to_csv(output_csv)  # 只保留行索引，无列
        return output_csv

    trans = pd.read_csv(transform_csv)
    if trans.empty:
        data.iloc[:, :0].to_csv(output_csv)
        return output_csv

    matched = set(trans["Feature_1_mz"]).union(set(trans["Feature_2_mz"]))
    keep_cols = [c for c in data.columns if float(c) in matched]

    filtered = data[keep_cols]
    filtered.to_csv(output_csv)
    return output_csv


# ---------------- Step⑦a：频率统计 + 20Da 分箱高分位筛选 ----------------

def frequency_correct_and_bin_filter(step6_matrix_csv,
                                     freq_out_csv,
                                     freq_filtered_out_csv,
                                     bin_size_da=20,
                                     top_quantile=0.999):
    df = pd.read_csv(step6_matrix_csv, index_col=0)
    if df.shape[1] == 0:
        _write_empty_csv(freq_out_csv, ["mz", "Frequency_Count", "Frequency_Rate"])
        _write_empty_csv(freq_filtered_out_csv, ["mz", "Frequency_Count", "Frequency_Rate", "mz_bin"])
        return freq_out_csv, freq_filtered_out_csv

    df.columns = df.columns.astype(float)

    counts = (df > 0).sum(axis=0)
    rates = counts / df.shape[0]
    freq_df = pd.DataFrame({
        "mz": counts.index,
        "Frequency_Count": counts.values,
        "Frequency_Rate": rates.values
    }).sort_values("mz")
    freq_df.to_csv(freq_out_csv, index=False)

    freq_df["mz_bin"] = (freq_df["mz"] // bin_size_da).astype(int)
    kept = []
    for _, bin_df in freq_df.groupby("mz_bin"):
        threshold = bin_df["Frequency_Rate"].quantile(top_quantile)
        kept.append(bin_df[bin_df["Frequency_Rate"] >= threshold])
    kept_df = pd.concat(kept).sort_values("mz")
    kept_df.to_csv(freq_filtered_out_csv, index=False)
    return freq_out_csv, freq_filtered_out_csv


# ---------------- Step⑦b：样条平滑并找谷点 ----------------

def smooth_and_find_valleys(freq_filtered_csv, valleys_csv, plot_png, spline_s=0.05):
    kept = pd.read_csv(freq_filtered_csv)
    if kept.empty:
        _write_empty_csv(valleys_csv, ["mz", "Frequency_Rate"])
        plt.figure(figsize=(8, 3))
        plt.title("Smoothed Frequency Curve with Valleys (empty)")
        plt.savefig(plot_png, dpi=300)
        plt.close()
        return valleys_csv, plot_png

    x = kept["mz"].values
    y = kept["Frequency_Rate"].values

    if len(x) < 4:
        _write_empty_csv(valleys_csv, ["mz", "Frequency_Rate"])
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label="Retained Points", markersize=3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_png, dpi=300)
        plt.close()
        return valleys_csv, plot_png

    spline = UnivariateSpline(x, y, s=spline_s)
    xs = np.linspace(x.min(), x.max(), 2000)
    ys = spline(xs)
    valley_idx = argrelextrema(ys, np.less)[0]
    valleys = pd.DataFrame({"mz": xs[valley_idx], "Frequency_Rate": ys[valley_idx]})
    valleys.to_csv(valleys_csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label="Retained Points", markersize=3)
    plt.plot(xs, ys, '-', label="Smoothed Fit")
    if len(valley_idx) > 0:
        plt.plot(xs[valley_idx], ys[valley_idx], 'ro', label="Valleys")
    plt.xlabel("m/z")
    plt.ylabel("Frequency Rate")
    plt.title("Smoothed Frequency Curve with Valleys")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_png, dpi=300)
    plt.close()
    return valleys_csv, plot_png


# ---------------- Step⑦c：固定阈值按 m/z 分段做最终筛选 ----------------

def final_threshold_filter(step6_matrix_csv, freq_corrected_csv,
                           output_csv, mz_split_threshold=300,
                           high_freq_rate=0.01, low_freq_rate=0.1):
    freq_df = pd.read_csv(freq_corrected_csv)
    inten_df = pd.read_csv(step6_matrix_csv, index_col=0)
    if inten_df.shape[1] == 0 or freq_df.empty:
        inten_df.iloc[:, :0].to_csv(output_csv)
        return output_csv

    inten_df.columns = inten_df.columns.astype(float)

    sel = freq_df[
        ((freq_df["mz"] < mz_split_threshold) & (freq_df["Frequency_Rate"] > low_freq_rate)) |
        ((freq_df["mz"] >= mz_split_threshold) & (freq_df["Frequency_Rate"] > high_freq_rate))
        ]["mz"].tolist()

    kept = inten_df.loc[:, inten_df.columns.isin(sel)]
    kept = kept.sort_index(axis=1)
    kept.to_csv(output_csv)
    return output_csv


# ---------------- 便捷封装：给 app 的两个独立小功能 ----------------

def run_freq_valley_analysis(intensity_matrix_csv, output_dir,
                             bin_size_da=20, top_quantile=0.999, spline_s=0.05):
    """
    功能①：给定 Step⑥ 强度矩阵 -> 频率表/分箱高分位/谷点CSV+PNG
    """
    _ensure_dir(output_dir)
    freq_csv = os.path.join(output_dir, "fv_frequency_corrected.csv")
    kept_csv = os.path.join(output_dir, "fv_frequency_kept.csv")
    valleys_csv = os.path.join(output_dir, "fv_mz_valleys.csv")
    valleys_png = os.path.join(output_dir, "fv_valleys_plot.png")

    frequency_correct_and_bin_filter(intensity_matrix_csv, freq_csv, kept_csv,
                                     bin_size_da=bin_size_da, top_quantile=top_quantile)
    smooth_and_find_valleys(kept_csv, valleys_csv, valleys_png, spline_s=spline_s)

    return {
        "frequency_csv": freq_csv,
        "filtered_csv": kept_csv,
        "valleys_csv": valleys_csv,
        "plot_png": valleys_png
    }


def run_segmented_frequency_filter(intensity_matrix_csv, output_dir,
                                   mz_split_threshold=300, high_freq_rate=0.01, low_freq_rate=0.1):
    """
    功能②：给定 Step⑥ 强度矩阵 + 三阈值 -> 频率表 + 最终过滤矩阵
    """
    _ensure_dir(output_dir)
    freq_csv = os.path.join(output_dir, "seg_frequency_corrected.csv")
    _ = os.path.join(output_dir, "seg_tmp_kept.csv")  # 产物不强制使用，但函数需要
    # 先得到频率表（同时会生成一个 kept 表，不用也没关系）
    frequency_correct_and_bin_filter(intensity_matrix_csv, freq_csv, _,
                                     bin_size_da=20, top_quantile=0.999)

    final_csv = os.path.join(output_dir, "seg_final_filtered_matrix.csv")
    final_threshold_filter(intensity_matrix_csv, freq_csv, final_csv,
                           mz_split_threshold=mz_split_threshold,
                           high_freq_rate=high_freq_rate,
                           low_freq_rate=low_freq_rate)

    return {
        "frequency_csv": freq_csv,
        "final_filtered_matrix_csv": final_csv
    }


# ---------------- 总流程（供 app 调用） ----------------

def process_matrix_builder(mgf_path, output_dir,
                           bin_size=0.01, tolerance=0.02,
                           mz_split_threshold=300,
                           high_freq_rate=0.01, low_freq_rate=0.1,
                           bin20_quantile=0.999, spline_s=0.05):
    """
    将“修改后的 jupyter 流程”完整封装；返回所有产物的绝对路径字典。
    """
    _ensure_dir(output_dir)

    # Step①
    spectra = extract_spectra_from_mgf(mgf_path)
    step1_csv = os.path.join(output_dir, "step1_fragment_matrix.csv")
    build_fragment_matrix(spectra, step1_csv)

    # Step②
    step2_csv = os.path.join(output_dir, "step2_binned_fragment_matrix.csv")
    bin_mz_values(step1_csv, step2_csv, bin_size=bin_size)

    # Step③
    step3_csv = os.path.join(output_dir, "step3_formula_annotation.csv")
    annotate_formulas(step2_csv, step3_csv, tolerance=tolerance)

    # Step④
    step4_csv = os.path.join(output_dir, "step4_CHO_filtered_matrix.csv")
    filter_CHO(step2_csv, step3_csv, step4_csv)

    # Step⑤
    step5_csv = os.path.join(output_dir, "step5_matched_transformations.csv")
    match_structural_transformations(step4_csv, step5_csv)

    # Step⑥
    step6_csv = os.path.join(output_dir, "step6_transformation_filtered_matrix.csv")
    filter_by_transformations(step4_csv, step5_csv, step6_csv)

    # Step⑦a
    step7_freq_csv = os.path.join(output_dir, "step6_transformation_frag_frequency_corrected.csv")
    step7_freq_kept_csv = os.path.join(output_dir, "step6_transformation_frag_frequency_filtered.csv")
    frequency_correct_and_bin_filter(step6_csv, step7_freq_csv, step7_freq_kept_csv,
                                     bin_size_da=20, top_quantile=bin20_quantile)

    # Step⑦b
    valleys_csv = os.path.join(output_dir, "step6_transformation_mz_valleys.csv")
    valleys_png = os.path.join(output_dir, "frag_frequency_valleys_plot.png")
    smooth_and_find_valleys(step7_freq_kept_csv, valleys_csv, valleys_png, spline_s=spline_s)

    # Step⑦c
    final_csv = os.path.join(output_dir, "final_filtered_matrix.csv")
    final_threshold_filter(step6_csv, step7_freq_csv, final_csv,
                           mz_split_threshold=mz_split_threshold,
                           high_freq_rate=high_freq_rate,
                           low_freq_rate=low_freq_rate)

    return {
        "step1_fragment_matrix_csv": step1_csv,
        "step2_binned_fragment_matrix_csv": step2_csv,
        "step3_formula_annotation_csv": step3_csv,
        "step4_CHO_filtered_matrix_csv": step4_csv,
        "step5_matched_transformations_csv": step5_csv,
        "step6_transformation_filtered_matrix_csv": step6_csv,
        "step6_transformation_frag_frequency_corrected_csv": step7_freq_csv,
        "step6_transformation_frag_frequency_filtered_csv": step7_freq_kept_csv,
        "step6_transformation_mz_valleys_csv": valleys_csv,
    }
