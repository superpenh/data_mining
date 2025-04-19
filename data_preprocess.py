import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pandas.api.types import is_numeric_dtype


def load_parquet_files(directory_path):
    """
    加载目录中的所有parquet文件并合并为一个DataFrame
    """
    dataframes = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(directory_path, filename)
            try:
                # 读取parquet文件
                df = pd.read_parquet(file_path)
                dataframes.append(df)
                print(f"成功加载文件: {filename}, 行数: {len(df)}")
            except Exception as e:
                print(f"无法加载文件 {filename}: {str(e)}")

    if not dataframes:
        raise ValueError("没有找到有效的parquet文件")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"合并完成，总行数: {len(combined_df)}")

    return combined_df


def check_missing_values(df):
    """
    检查并统计缺失值
    """
    # 计算每列的缺失值数量和比例
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    # 创建一个包含缺失值统计的DataFrame
    missing_stats = pd.DataFrame({
        '缺失值数量': missing_count,
        '缺失值比例(%)': missing_percentage
    })

    # 仅返回有缺失值的列
    return missing_stats[missing_stats['缺失值数量'] > 0].sort_values('缺失值数量', ascending=False)


def check_duplicates(df):
    """
    检查重复行
    """
    # 检查完全重复的行
    duplicates = df.duplicated(subset=['user_name', 'fullname', 'email'])
    duplicate_rows = df[duplicates]

    # 计算重复行数量和比例
    dup_count = len(duplicate_rows)
    dup_percentage = (dup_count / len(df)) * 100

    return dup_count, dup_percentage, duplicate_rows


def check_outliers(df, numeric_columns=None):
    """
    使用IQR方法检测数值列中的异常值
    """
    outlier_stats = {}

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    for column in numeric_columns:
        if not is_numeric_dtype(df[column]):
            continue

        # 计算Q1、Q3和IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # 定义异常值界限
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 检测异常值
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100

        outlier_stats[column] = {
            '异常值数量': outlier_count,
            '异常值比例(%)': outlier_percentage,
            '下界': lower_bound,
            '上界': upper_bound
        }

    return outlier_stats


def check_inconsistent_data(df):
    """
    检查数据一致性问题
    """
    consistency_issues = {}

    # 检查年龄合理性
    if 'age' in df.columns:
        invalid_ages = len(df[(df['age'] < 0) | (df['age'] > 120)])
        consistency_issues['invalid_ages'] = {
            '无效年龄数量': invalid_ages,
            '无效年龄比例(%)': (invalid_ages / len(df)) * 100
        }
    # 检查性别值一致性
    if 'gender' in df.columns:
        gender_values = df['gender'].dropna().unique()
        consistency_issues['gender_values'] = {
            '不同性别值': list(gender_values),
            '唯一值数量': len(gender_values)
        }
    # 检查电子邮件格式
    if 'email' in df.columns:
        # 简单的电子邮件格式检查 - 包含@符号
        invalid_emails = len(df[~df['email'].str.contains('@', na=False)])
        consistency_issues['invalid_emails'] = {
            '无效邮箱数量': invalid_emails,
            '无效邮箱比例(%)': (invalid_emails / len(df)) * 100
        }
    # 检查JSON字段的格式 - login_history字段的检查
    if 'login_history' in df.columns:
        invalid_json = 0

        for json_str in df['login_history']:
            if pd.isna(json_str):
                continue

            try:
                if isinstance(json_str, str):
                    data = json.loads(json_str)
                elif isinstance(json_str, dict):
                    data = json_str
                else:
                    invalid_json += 1
                    continue

            except:
                invalid_json += 1

        consistency_issues['invalid_login_history'] = {
            '无效JSON数量': invalid_json,
            '无效JSON比例(%)': (invalid_json / len(df)) * 100
        }
    # 检查purchase_history字段的格式
    if 'purchase_history' in df.columns:
        invalid_json = 0
        for json_str in df['purchase_history']:
            if pd.isna(json_str):
                continue
            try:
                if isinstance(json_str, str):
                    json.loads(json_str)
            except:
                invalid_json += 1

        consistency_issues['invalid_purchase_history'] = {
            '无效JSON数量': invalid_json,
            '无效JSON比例(%)': (invalid_json / len(df)) * 100
        }

    return consistency_issues


def preprocess_data(df, missing_threshold=0.1):
    """
    数据预处理
    """
    original_shape = df.shape
    processing_log = []
    processing_log.append(f"原始数据形状: {original_shape}")

    # 创建副本以避免修改原始数据
    processed_df = df.copy()

    # 记录是否有数据被删除的标志
    data_deleted = False

    # 1. 处理重复行
    print("正在处理重复行...")
    before_dedup = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=['user_name', 'fullname', 'email'])
    after_dedup = len(processed_df)
    if before_dedup > after_dedup:
        data_deleted = True
        processing_log.append(
            f"删除了 {before_dedup - after_dedup} 行重复数据 ({(before_dedup - after_dedup) / before_dedup:.2%})，原因：这些行基于user_name、fullname和email字段完全重复")

    # 2. 删除缺失值过多的列
    print("正在处理缺失值...")
    missing_stats = processed_df.isnull().sum()
    missing_percent = missing_stats / len(processed_df)
    cols_to_drop = missing_percent[missing_percent > missing_threshold].index.tolist()
    if cols_to_drop:
        data_deleted = True
        processed_df = processed_df.drop(columns=cols_to_drop)
        processing_log.append(
            f"删除了 {len(cols_to_drop)} 列，原因：它们的缺失值比例超过 {missing_threshold:.0%}。被删除的列: {', '.join(cols_to_drop)}")

    # 3. 删除income、login_history和purchase_history缺失值的行
    target_columns = []

    if 'income' in processed_df.columns:
        target_columns.append('income')

    if 'login_history' in processed_df.columns:
        target_columns.append('login_history')

    if 'purchase_history' in processed_df.columns:
        target_columns.append('purchase_history')

    if target_columns:
        print(f"正在删除 {', '.join(target_columns)} 列中有缺失值的行...")
        rows_before = len(processed_df)

        for col in target_columns:
            missing_counts = processed_df[col].isnull().sum()
            if missing_counts > 0:
                processed_df = processed_df.dropna(subset=[col])
                processing_log.append(f"删除了 {missing_counts} 行，原因：'{col}' 列中存在缺失值")
                data_deleted = True

        rows_after = len(processed_df)
        if rows_before > rows_after:
            processing_log.append(f"总共删除了 {rows_before - rows_after} 行，原因：指定列中存在缺失值")

    # 4. 处理日期列
    print("正在处理日期列...")
    date_columns = ['last_login', 'registration_date']
    for col in date_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            null_dates_after = processed_df[col].isnull().sum()
            if null_dates_after > 0:
                processing_log.append(f"列 '{col}' 中有 {null_dates_after} 个无效日期值被转换为NaT")

    # 5. 处理数值列中的异常值
    print("正在处理数值列中的异常值...")
    numeric_cols = ['age', 'income']
    for col in numeric_cols:
        if col not in processed_df.columns:
            continue

        # 计算IQR
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 统计并将异常值替换为NaN
        outliers_mask = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        if outliers_count > 0:
            processed_df.loc[outliers_mask, col] = np.nan
            processing_log.append(
                f"列 '{col}' 中有 {outliers_count} 个异常值 ({outliers_count / len(processed_df):.2%}) 被替换为NaN，异常值范围：小于 {lower_bound:.2f} 或大于 {upper_bound:.2f}")

            # 如果是income列的异常值，需要删除这些行
            if col == 'income':
                rows_before = len(processed_df)
                processed_df = processed_df.dropna(subset=['income'])
                rows_deleted = rows_before - len(processed_df)
                if rows_deleted > 0:
                    processing_log.append(f"删除了 {rows_deleted} 行，原因：'income' 列中检测到异常值后转为NaN")
                    data_deleted = True

    final_shape = processed_df.shape
    processing_log.append(f"最终数据形状: {final_shape}")
    processing_log.append(
        f"行数变化: {original_shape[0]} -> {final_shape[0]} ({(final_shape[0] - original_shape[0]) / original_shape[0]:.2%})")
    processing_log.append(
        f"列数变化: {original_shape[1]} -> {final_shape[1]} ({(final_shape[1] - original_shape[1]) / original_shape[1]:.2%})")

    return processed_df, processing_log, data_deleted


def analyze_data_quality(df):
    """
    全面分析数据质量并生成报告
    """
    results = {
        "basic_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    }
    # 检查缺失值
    results["missing_values"] = check_missing_values(df).to_dict()
    # 检查重复行
    dup_count, dup_percentage, _ = check_duplicates(df)
    results["duplicates"] = {
        "duplicate_rows": dup_count,
        "duplicate_percentage": dup_percentage
    }
    # 检查数值列中的异常值
    numeric_columns = ['age', 'income']
    results["outliers"] = check_outliers(df, numeric_columns)
    # 检查数据一致性问题
    results["consistency_issues"] = check_inconsistent_data(df)

    return results


def generate_data_quality_report(analysis_results):
    """
    生成数据质量报告
    """
    report = []
    report.append("=" * 80)
    report.append("数据质量评估报告")
    report.append("=" * 80)

    # 基本信息
    basic_info = analysis_results["basic_info"]
    report.append("\n1. 基本信息")
    report.append(f"- 行数: {basic_info['rows']}")
    report.append(f"- 列数: {basic_info['columns']}")
    report.append(f"- 列名: {', '.join(basic_info['column_names'])}")

    report.append("\n2. 缺失值问题")
    # Check if any columns have missing values by seeing if any inner dictionaries have content
    has_missing_values = False
    for col, stats in analysis_results["missing_values"].items():
        if stats and ('缺失值数量' in stats) and stats['缺失值数量'] > 0:
            has_missing_values = True
            missing_count = stats['缺失值数量']
            missing_percent = stats['缺失值比例(%)']
            report.append(f"- 列 '{col}': {missing_count} 行缺失 ({missing_percent:.2f}%)")

    if not has_missing_values:
        report.append("- 未发现缺失值问题")

    # 重复值问题
    report.append("\n3. 重复行问题")
    dup_count = analysis_results["duplicates"]["duplicate_rows"]
    dup_percent = analysis_results["duplicates"]["duplicate_percentage"]
    if dup_count > 0:
        report.append(f"- 发现 {dup_count} 行重复数据 ({dup_percent:.2f}%)")
    else:
        report.append("- 未发现重复行问题")

    # 异常值问题
    report.append("\n4. 异常值问题")
    if analysis_results["outliers"]:
        for col, stats in analysis_results["outliers"].items():
            outlier_count = stats['异常值数量']
            outlier_percent = stats['异常值比例(%)']
            if outlier_count > 0:
                report.append(f"- 列 '{col}': {outlier_count} 个异常值 ({outlier_percent:.2f}%)")
                report.append(f"  异常值界限: 小于 {stats['下界']:.2f} 或大于 {stats['上界']:.2f}")
    else:
        report.append("- 未发现异常值问题")

    # 数据一致性问题
    report.append("\n5. 数据一致性问题")
    consistency_issues = analysis_results.get("consistency_issues", {})
    if consistency_issues:
        # 日期格式问题
        for key, value in consistency_issues.items():
            if key.endswith('_invalid_format') and '无效日期数量' in value:
                col = key.replace('_invalid_format', '')
                count = value['无效日期数量']
                percent = value['无效日期比例(%)']
                if count > 0:
                    report.append(f"- 列 '{col}': {count} 个无效日期格式 ({percent:.2f}%)")

            # 年龄问题
            if key == 'invalid_ages':
                count = value['无效年龄数量']
                percent = value['无效年龄比例(%)']
                if count > 0:
                    report.append(f"- 年龄列: {count} 个无效值 ({percent:.2f}%), 有效范围应为0-120岁")

            # 性别值问题
            if key == 'gender_values':
                values = value['不同性别值']
                report.append(f"- 性别列: 包含 {len(values)} 种不同的值: {', '.join(map(str, values))}")

            # 邮箱格式问题
            if key == 'invalid_emails':
                count = value['无效邮箱数量']
                percent = value['无效邮箱比例(%)']
                if count > 0:
                    report.append(f"- 邮箱列: {count} 个无效格式 ({percent:.2f}%)")

            # JSON格式问题
            if key == 'invalid_login_history':
                count = value['无效JSON数量']
                percent = value['无效JSON比例(%)']
                if count > 0:
                    report.append(f"- login_history列: {count} 个无效JSON格式 ({percent:.2f}%)")
                    if value['无效时间戳数量'] > 0:
                        report.append(f"  包含 {value['无效时间戳数量']} 个无效时间戳")
                    if value['无效首次登录日期数量'] > 0:
                        report.append(f"  包含 {value['无效首次登录日期数量']} 个无效首次登录日期")

            if key == 'invalid_purchase_history':
                count = value['无效JSON数量']
                percent = value['无效JSON比例(%)']
                if count > 0:
                    report.append(f"- purchase_history列: {count} 个无效JSON格式 ({percent:.2f}%)")
    else:
        report.append("- 未发现数据一致性问题")

    return "\n".join(report)


def save_results(df, analysis_results, processing_log, quality_report, data_deleted,
                 output_dir="./data_quality_results"):
    """
    保存分析结果和处理后的数据
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 仅当数据有删除时保存处理后的数据
    if data_deleted:
        df.to_parquet(os.path.join(output_dir, "processed_data.parquet"))
        print(f"处理后的数据已保存到 {os.path.join(output_dir, 'processed_data.parquet')}")
    else:
        print("未删除任何数据，不保存处理后的parquet文件")

    # 保存分析结果
    with open(os.path.join(output_dir, "analysis_results.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)

    # 保存处理日志
    with open(os.path.join(output_dir, "processing_log.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(processing_log))

    # 保存数据质量报告
    with open(os.path.join(output_dir, "data_quality_report.txt"), 'w', encoding='utf-8') as f:
        f.write(quality_report)

    print(f"结果已保存到 {output_dir} 目录")
    print("数据质量报告已保存到 data_quality_report.txt 文件")


def main():
    # 设置输出目录
    output_base_dir = "./data_quality_assessment"
    os.makedirs(output_base_dir, exist_ok=True)

    try:
        # 输入parquet文件目录
        directory_path = '/home/pengxiao/virtualenvs/shujuwajue/30G_new_data'

        # 加载parquet文件
        print("正在加载parquet文件...")
        df = load_parquet_files(directory_path)

        # 分析原始数据质量
        print("正在分析原始数据质量...")
        raw_analysis = analyze_data_quality(df)

        # 生成原始数据质量报告
        print("正在生成原始数据质量报告...")
        raw_quality_report = generate_data_quality_report(raw_analysis)
        print("\n" + "=" * 50)
        print("原始数据质量评估结果:")
        print("=" * 50)
        print(raw_quality_report)

        # 进行数据预处理
        print("\n开始数据预处理...")
        processed_df, processing_log, data_deleted = preprocess_data(df)

        # 打印处理日志
        print("\n" + "=" * 50)
        print("数据处理日志:")
        print("=" * 50)
        for log in processing_log:
            print(log)

        # 分析处理后的数据质量
        print("\n正在分析处理后的数据质量...")
        processed_analysis = analyze_data_quality(processed_df)

        # 生成处理后数据质量报告
        print("正在生成处理后数据质量报告...")
        processed_quality_report = generate_data_quality_report(processed_analysis)
        print("\n" + "=" * 50)
        print("处理后数据质量评估结果:")
        print("=" * 50)
        print(processed_quality_report)

        # 保存结果
        print("\n正在保存分析结果和处理后的数据...")
        results_dir = os.path.join(output_base_dir, "results")
        save_results(
            processed_df,
            {
                "raw_data_analysis": raw_analysis,
                "processed_data_analysis": processed_analysis
            },
            processing_log,
            raw_quality_report + "\n\n\n" + processed_quality_report,
            data_deleted,
            results_dir
        )

        # 显示是否保存了处理后的数据
        if data_deleted:
            print("\n数据发生了变化，已保存处理后的parquet文件")
        else:
            print("\n数据未发生变化，未保存处理后的parquet文件")

        print(f"\n结果已保存到: {output_base_dir}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time

    start_time = time.time()

    main()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n总执行时间: {execution_time:.2f} 秒 ({execution_time / 60:.2f} 分钟)")