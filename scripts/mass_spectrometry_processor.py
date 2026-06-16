import pandas as pd
import numpy as np
import os
from datetime import datetime
def match_features_from_mgf(mgf_file_path, features, tolerance=0.05, intensity_threshold=1e6):
    """从 MGF 文件中匹配特征峰并计算得分"""
    with open(mgf_file_path, 'r') as file:
        mgf_content = file.readlines()

    score_data = []
    current_spectrum = None

    for line in mgf_content:
        line = line.strip()
        if line == "BEGIN IONS":
            current_spectrum = {'params': {}, 'm/z array': [], 'intensity array': []}
        elif line == "END IONS":
            if current_spectrum:
                peaks = np.array(current_spectrum['m/z array'])
                intensities = np.array(current_spectrum['intensity array'])
                score = 0
                found_features = []
                for feature in features:
                    matching_peaks = abs(peaks - feature) / feature <= tolerance
                    if any(matching_peaks & (intensities > intensity_threshold)):
                        score += 1
                        found_features.append(feature)
                scan_id = (current_spectrum['params'].get('FEATURE_ID') or
                           current_spectrum['params'].get('SCANS', 'unknown'))
                score_data.append({
                    'scan': scan_id,
                    'score': score,
                    'found_features': ', '.join(map(str, found_features))
                })
            current_spectrum = None
        elif current_spectrum is not None:
            if '=' in line:
                key, value = line.split('=', 1)
                current_spectrum['params'][key] = value
            else:
                try:
                    mz, intensity = map(float, line.split())
                    current_spectrum['m/z array'].append(mz)
                    current_spectrum['intensity array'].append(intensity)
                except ValueError:
                    pass

    return pd.DataFrame(score_data)

def filter_quant_csv(score_df, input_csv, output_csv):
    """根据得分筛选量化 CSV 文件"""
    found_scans = score_df[score_df['score'] > 0]['scan'].astype(str).tolist()
    df_input = pd.read_csv(input_csv)
    filtered_df = df_input[df_input['row ID'].astype(str).isin(found_scans)]
    if 'Unnamed: 14' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['Unnamed: 14'])
    filtered_df.to_csv(output_csv, index=False)
    return filtered_df

def detect_dehydration_peaks(mgf_file_path, water_mass, water_count, tolerance_dehydration, intensity_threshold=70000, mz_threshold=150):
    """检测脱水峰（支持单次或连续脱水）"""
    with open(mgf_file_path, 'r') as file:
        mgf_content = file.readlines()

    score_data = []
    current_spectrum = None

    for line in mgf_content:
        line = line.strip()
        if line == "BEGIN IONS":
            current_spectrum = {'params': {}, 'm/z array': [], 'intensity array': [], 'dehydration_peaks': []}
        elif line == "END IONS":
            if current_spectrum:
                peaks = np.array(current_spectrum['m/z array'])
                intensities = np.array(current_spectrum['intensity array'])
                high_intensity_indices = (intensities > intensity_threshold) & (peaks > mz_threshold)
                high_intensity_peaks = peaks[high_intensity_indices]
                dehydration_peaks = []

                if water_count == '1':  # 单次脱水
                    for i in range(len(high_intensity_peaks)):
                        for j in range(len(high_intensity_peaks)):
                            if i != j and abs(high_intensity_peaks[i] - high_intensity_peaks[j] - water_mass) < tolerance_dehydration:
                                dehydration_peaks.append(f"{high_intensity_peaks[i]:.5f}-{high_intensity_peaks[j]:.5f}")
                elif water_count == '2':  # 连续两次脱水
                    for i in range(len(high_intensity_peaks)):
                        for j in range(len(high_intensity_peaks)):
                            for k in range(len(high_intensity_peaks)):
                                if (i != j and j != k and i != k and
                                        abs(high_intensity_peaks[i] - high_intensity_peaks[j] - water_mass) < tolerance_dehydration and
                                        abs(high_intensity_peaks[j] - high_intensity_peaks[k] - water_mass) < tolerance_dehydration):
                                    dehydration_peaks.append(
                                        f"{high_intensity_peaks[i]:.5f}-{high_intensity_peaks[j]:.5f}-{high_intensity_peaks[k]:.5f}")

                scan_id = current_spectrum['params'].get('FEATURE_ID', 'unknown')
                score_data.append({
                    'scan': scan_id,
                    'dehydration_peaks': ', '.join(dehydration_peaks),
                    'score': len(dehydration_peaks)
                })
            current_spectrum = None
        elif current_spectrum is not None:
            if '=' in line:
                key, value = line.split('=', 1)
                current_spectrum['params'][key] = value
            else:
                try:
                    mz, intensity = map(float, line.split())
                    current_spectrum['m/z array'].append(mz)
                    current_spectrum['intensity array'].append(intensity)
                except ValueError:
                    pass

    return pd.DataFrame(score_data)

def filter_dehydration_quant(score_df, quant_csv, output_csv):
    """根据脱水峰筛选量化 CSV 文件"""
    matched_ids = score_df[score_df['score'] > 0]['scan'].astype(str).str.strip()
    quant_df = pd.read_csv(quant_csv)
    quant_df['row ID'] = quant_df['row ID'].astype(str).str.strip()
    filtered_quant_df = quant_df[quant_df['row ID'].isin(matched_ids)]
    filtered_quant_df.to_csv(output_csv, index=False)
    return filtered_quant_df

def generate_filtered_mgf(mgf_file_path, quant_csv, output_mgf_path):
    """生成筛选后的 MGF 文件"""
    csv_data = pd.read_csv(quant_csv)
    csv_data['row ID'] = csv_data['row ID'].astype(str).str.strip()
    row_ids = csv_data['row ID'].tolist()

    with open(mgf_file_path, 'r') as file:
        mgf_content = file.readlines()

    filtered_mgf_content = []
    inside_ions_block = False
    current_spectrum_id = None
    current_block = []

    for line in mgf_content:
        if line.startswith("BEGIN IONS"):
            inside_ions_block = True
            current_spectrum_id = None
            current_block = [line]
        elif line.startswith("END IONS"):
            if current_spectrum_id in row_ids:
                filtered_mgf_content.extend(current_block)
                filtered_mgf_content.append(line)
                filtered_mgf_content.append('\n')
            inside_ions_block = False
        elif inside_ions_block:
            if line.startswith("FEATURE_ID="):
                current_spectrum_id = line.split('=')[1].strip()
            current_block.append(line)

    with open(output_mgf_path, 'w') as output_file:
        output_file.writelines(filtered_mgf_content)
    return output_mgf_path

def process_mass_spectrometry_data(mgf_file_path, input_csv, output_path, tolerance=0.05, intensity_threshold=1e6,
                                  features=[], water_mass=18.01528, water_count='1', tolerance_dehydration = 0.05,
                                  intensity_threshold_dehydration=70000, mz_threshold=150):
    """主函数：执行所有处理步骤，支持用户指定的输出路径和参数"""
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    suffix = "-H2O" if water_count == '1' else "-2H2O"
    base_name = os.path.splitext(os.path.basename(mgf_file_path))[0]
    base_name = f"{base_name}_{today}"
    score_csv = os.path.join(output_dir, f"{base_name}_score_data.csv")
    filtered_quant_csv = os.path.join(output_dir, f"{base_name}_quant_filtered.csv")
    h2o_score_csv = os.path.join(output_dir, f"{base_name}_quant_score_data{suffix}.csv")
    h2o_quant_csv = os.path.join(output_dir, f"{base_name}_quant_filtered{suffix}.csv")
    filtered_mgf = os.path.join(output_dir, f"{base_name}_filtered{suffix}.mgf")
    print(features)

    # 1. 特征峰匹配与打分
    print("[INFO] 开始特征峰匹配与打分...")
    score_df = match_features_from_mgf(mgf_file_path, features, tolerance, intensity_threshold)
    score_df.to_csv(score_csv, index=False)
    print(f"[INFO] 特征峰匹配结果已保存到: {score_csv}")

    # 2. 筛选量化数据
    print("[INFO] 开始筛选量化数据...")
    filtered_df = filter_quant_csv(score_df, input_csv, filtered_quant_csv)
    print(f"[INFO] 筛选后的量化数据已保存到: {filtered_quant_csv}")

    # 3. 单次或连续脱水峰检测
    print(f"[INFO] 开始{'单次' if water_count == '1' else '连续两次'}脱水峰检测...")
    h2o_score_df = detect_dehydration_peaks(mgf_file_path, water_mass, water_count,
                                           tolerance_dehydration, intensity_threshold_dehydration, mz_threshold)
    h2o_score_df.to_csv(h2o_score_csv, index=False)
    print(f"[INFO] 脱水峰检测结果已保存到: {h2o_score_csv}")

    # 4. 筛选脱水数据
    print("[INFO] 开始筛选脱水数据...")
    filter_dehydration_quant(h2o_score_df, filtered_quant_csv, h2o_quant_csv)
    print(f"[INFO] 脱水筛选后的量化数据已保存到: {h2o_quant_csv}")

    # 5. 生成筛选后的 MGF 文件
    print("[INFO] 开始生成筛选后的 MGF 文件...")
    generate_filtered_mgf(mgf_file_path, h2o_quant_csv, filtered_mgf)
    print(f"[INFO] 筛选后的 MGF 文件已保存到: {filtered_mgf}")

    return {
        "score_csv": os.path.abspath(score_csv),
        "filtered_quant_csv": os.path.abspath(filtered_quant_csv),
        "h2o_score_csv": os.path.abspath(h2o_score_csv),
        "h2o_quant_csv": os.path.abspath(h2o_quant_csv),
        "filtered_mgf": os.path.abspath(filtered_mgf),
        "filtered_spectra_count": len(score_df),
        "matched_count": len(filtered_df)
    }