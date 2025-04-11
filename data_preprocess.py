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
    duplicates =df.duplicated(subset=['user_name', 'chinese_name', 'email'])
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

    # 检查日期格式一致性（针对timestamp和registration_date列）
    date_columns = ['timestamp', 'registration_date']
    for col in date_columns:
        if col in df.columns:
            invalid_dates = 0
            for date_str in df[col]:
                if pd.isna(date_str):
                    continue
                try:
                    # 尝试解析日期字符串
                    pd.to_datetime(date_str)
                except:
                    invalid_dates += 1

            consistency_issues[f'{col}_invalid_format'] = {
                '无效日期数量': invalid_dates,
                '无效日期比例(%)': (invalid_dates / len(df)) * 100
            }

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

    # 检查JSON字段的格式
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



def preprocess_data(df, missing_threshold=0.5):
    """
    数据预处理
    """
    original_shape = df.shape
    processing_log = []
    processing_log.append(f"原始数据形状: {original_shape}")

    # 创建副本以避免修改原始数据
    processed_df = df.copy()

    # 1. 处理重复行
    print("正在处理重复行...")
    before_dedup = len(processed_df)
    processed_df = processed_df.drop_duplicates()
    after_dedup = len(processed_df)
    if before_dedup > after_dedup:
        processing_log.append(
            f"删除了 {before_dedup - after_dedup} 行重复数据 ({(before_dedup - after_dedup) / before_dedup:.2%})")

    # 2. 删除缺失值过多的列
    print("正在处理缺失值...")
    missing_stats = processed_df.isnull().mean()
    cols_to_drop = missing_stats[missing_stats > missing_threshold].index.tolist()
    if cols_to_drop:
        processed_df = processed_df.drop(columns=cols_to_drop)
        processing_log.append(
            f"删除了 {len(cols_to_drop)} 列，因为它们的缺失值比例超过 {missing_threshold:.0%}: {', '.join(cols_to_drop)}")

    # 3. 处理日期列
    print("正在处理日期列...")
    date_columns = ['timestamp', 'registration_date']
    for col in date_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            null_dates_after = processed_df[col].isnull().sum()
            if null_dates_after > 0:
                processing_log.append(f"列 '{col}' 中有 {null_dates_after} 个无效日期值被转换为NaT")

    # 4. 处理数值列中的异常值
    print("正在处理数值列中的异常值...")
    numeric_cols = ['age', 'income', 'credit_score']
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
        outliers_count = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
        if outliers_count > 0:
            processed_df.loc[(processed_df[col] < lower_bound) | (processed_df[col] > upper_bound), col] = np.nan
            processing_log.append(
                f"列 '{col}' 中有 {outliers_count} 个异常值 ({outliers_count / len(processed_df):.2%}) 被替换为NaN")


    # 5. 根据需要填充缺失值
    # 对数值列使用中位数填充
    print("正在填充缺失值...")
    for col in processed_df.select_dtypes(include=np.number).columns:
        null_count = processed_df[col].isnull().sum()
        if null_count > 0:
            median_value = processed_df[col].median()
            processed_df[col].fillna(median_value, inplace=True)
            processing_log.append(f"列 '{col}' 中的 {null_count} 个缺失值使用中位数 {median_value} 填充")

    # 对分类列使用众数填充
    for col in ['gender', 'country']:
        if col in processed_df.columns:
            null_count = processed_df[col].isnull().sum()
            if null_count > 0:
                mode_value = processed_df[col].mode()[0]
                processed_df[col].fillna(mode_value, inplace=True)
                processing_log.append(f"列 '{col}' 中的 {null_count} 个缺失值使用众数 '{mode_value}' 填充")


    final_shape = processed_df.shape
    processing_log.append(f"最终数据形状: {final_shape}")
    processing_log.append(
        f"行数变化: {original_shape[0]} -> {final_shape[0]} ({(final_shape[0] - original_shape[0]) / original_shape[0]:.2%})")
    processing_log.append(
        f"列数变化: {original_shape[1]} -> {final_shape[1]} ({(final_shape[1] - original_shape[1]) / original_shape[1]:.2%})")

    return processed_df, processing_log


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
    numeric_columns = ['age', 'income', 'credit_score']
    results["outliers"] = check_outliers(df, numeric_columns)

    return results


def generate_statistics_report(df):
    """
    生成详细的数据统计报告
    """
    report = []
    report.append("=" * 80)
    report.append("数据统计报告")
    report.append("=" * 80)

    # 基本信息
    report.append("\n1. 基本信息")
    report.append(f"- 行数: {len(df)}")
    report.append(f"- 列数: {len(df.columns)}")
    report.append(f"- 列名: {', '.join(df.columns)}")

    # 数据类型
    report.append("\n2. 数据类型")
    for col, dtype in df.dtypes.items():
        report.append(f"- {col}: {dtype}")


    # 缺失值统计
    report.append("\n3. 缺失值统计")
    missing_stats = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_stats / len(df) * 100).round(2)
    for col, count in missing_stats.items():
        if count > 0:
            report.append(f"- {col}: {count} 行 ({missing_percent[col]}%)")


    # 重复行统计
    dup_count = df.duplicated().sum()
    dup_percent = (dup_count / len(df) * 100).round(2)
    report.append(f"\n4. 重复行: {dup_count} 行 ({dup_percent}%)")


    # 数值列统计
    report.append("\n5. 数值列统计")
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        stats = df[col].describe()
        report.append(f"\n  {col}:")
        report.append(f"  - 计数: {stats['count']}")
        report.append(f"  - 均值: {stats['mean']:.2f}")
        report.append(f"  - 标准差: {stats['std']:.2f}")
        report.append(f"  - 最小值: {stats['min']:.2f}")
        report.append(f"  - 25%分位: {stats['25%']:.2f}")
        report.append(f"  - 中位数: {stats['50%']:.2f}")
        report.append(f"  - 75%分位: {stats['75%']:.2f}")
        report.append(f"  - 最大值: {stats['max']:.2f}")

        # 计算异常值
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = round((outlier_count / len(df) * 100), 2)
        report.append(f"  - 异常值: {outlier_count} 行 ({outlier_percent}%)")
        report.append(f"  - 异常值界限: [{lower_bound:.2f}, {upper_bound:.2f}]")


    # 分类列统计
    report.append("\n6. 分类列统计")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        unique_count = len(value_counts)
        report.append(f"\n  {col}:")
        report.append(f"  - 唯一值数量: {unique_count}")
        report.append(f"  - 出现频率最高的值 (top 5):")
        for value, count in value_counts.head(5).items():
            percent=round((count / len(df) * 100), 2)
            report.append(f"    - {value}: {count} 行 ({percent}%)")

    # 日期列统计
    date_columns = ['timestamp', 'registration_date']
    report.append("\n7. 日期列统计")
    for col in date_columns:
        if col in df.columns:
            dates = pd.to_datetime(df[col], errors='coerce')
            valid_dates = dates.dropna()
            invalid_dates = len(df) - len(valid_dates)
            invalid_percent = round((invalid_dates / len(df) * 100),2)

            report.append(f"\n  {col}:")
            report.append(f"  - 有效日期: {len(valid_dates)} 行")
            report.append(f"  - 无效日期: {invalid_dates} 行 ({invalid_percent}%)")
            if len(valid_dates) > 0:
                report.append(f"  - 最早日期: {valid_dates.min()}")
                report.append(f"  - 最晚日期: {valid_dates.max()}")


    # JSON数据统计
    if 'purchase_history' in df.columns:
        report.append("\n9. purchase_history JSON分析")
        valid_json = 0
        invalid_json = 0
        categories = {}
        avg_prices = []

        for json_str in df['purchase_history'].dropna():
            try:
                if isinstance(json_str, str):
                    data = json.loads(json_str)
                    valid_json += 1

                    # 统计类别
                    if 'category' in data:
                        category = data['category']
                        categories[category] = categories.get(category, 0) + 1

                    # 统计平均价格
                    if 'average_price' in data:
                        avg_prices.append(data['average_price'])
            except:
                invalid_json += 1

        report.append(f"  - 有效JSON: {valid_json} 行")
        report.append(f"  - 无效JSON: {invalid_json} 行")

        if categories:
            report.append("  - 类别统计:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percent = round((count / valid_json * 100),2)
                report.append(f"    - {category}: {count} 行 ({percent}%)")

        if avg_prices:
            report.append(f"  - 平均价格统计:")
            report.append(f"    - 均值: {np.mean(avg_prices):.2f}")
            report.append(f"    - 中位数: {np.median(avg_prices):.2f}")
            report.append(f"    - 最小值: {min(avg_prices):.2f}")
            report.append(f"    - 最大值: {max(avg_prices):.2f}")

    return "\n".join(report)


def save_results(df, analysis_results, processing_log, statistics_report, output_dir="./data_quality_results"):
    """
    保存分析结果和处理后的数据
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存处理后的数据
    df.to_parquet(os.path.join(output_dir, "processed_data.parquet"))

    # 保存分析结果
    with open(os.path.join(output_dir, "analysis_results.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)

    # 保存处理日志
    with open(os.path.join(output_dir, "processing_log.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(processing_log))

    # 保存统计报告
    with open(os.path.join(output_dir, "statistics_report.txt"), 'w', encoding='utf-8') as f:
        f.write(statistics_report)

    print(f"结果已保存到 {output_dir} 目录")
    print("统计报告已保存到 statistics_report.txt 文件")


def main():
    # 设置输出目录
    output_base_dir = "./data_quality_assessment"
    os.makedirs(output_base_dir, exist_ok=True)

    try:
        # 输入parquet文件目录
        directory_path = '/home/pengxiao/virtualenvs/shujuwajue/30G_data'

        # 加载parquet文件
        print("正在加载parquet文件...")
        df = load_parquet_files(directory_path)

        # 分析原始数据质量
        print("正在分析原始数据质量...")
        raw_analysis = analyze_data_quality(df)

        # 生成原始数据统计报告
        print("正在生成原始数据统计报告...")
        raw_stats_report = generate_statistics_report(df)
        print("\n" + "=" * 50)
        print("原始数据统计结果:")
        print("=" * 50)
        print(raw_stats_report)

        # 进行数据预处理
        print("\n开始数据预处理...")
        processed_df, processing_log = preprocess_data(df)

        # 打印处理日志
        print("\n" + "=" * 50)
        print("数据处理日志:")
        print("=" * 50)
        for log in processing_log:
            print(log)

        # 分析处理后的数据质量
        print("\n正在分析处理后的数据质量...")
        processed_analysis = analyze_data_quality(processed_df)

        # 生成处理后数据统计报告
        print("正在生成处理后数据统计报告...")
        processed_stats_report = generate_statistics_report(processed_df)
        print("\n" + "=" * 50)
        print("处理后数据统计结果:")
        print("=" * 50)
        print(processed_stats_report)

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
            raw_stats_report + "\n\n" + processed_stats_report,
            results_dir
        )

        # 显示处理前后对比
        print("\n" + "=" * 50)
        print("数据处理前后对比:")
        print("=" * 50)
        print(f"- 原始数据: {raw_analysis['basic_info']['rows']} 行, {raw_analysis['basic_info']['columns']} 列")
        print(
            f"- 处理后数据: {processed_analysis['basic_info']['rows']} 行, {processed_analysis['basic_info']['columns']} 列")

        # 显示处理前后缺失值变化
        print("\n缺失值变化:")
        raw_missing = sum(raw_analysis['missing_values'].get(col, {}).get('缺失值数量', 0) for col in df.columns)
        processed_missing = sum(
            processed_analysis['missing_values'].get(col, {}).get('缺失值数量', 0) for col in processed_df.columns)
        print(f"- 原始数据缺失值: {raw_missing} 个")
        print(f"- 处理后数据缺失值: {processed_missing} 个")
        print(
            f"- 缺失值减少: {raw_missing - processed_missing} 个 ({100 * (raw_missing - processed_missing) / raw_missing if raw_missing > 0 else 0:.2f}%)")

        # 显示处理前后异常值变化
        print("\n异常值变化:")
        raw_outliers = sum(stats['异常值数量'] for col, stats in raw_analysis['outliers'].items())
        processed_outliers = sum(stats['异常值数量'] for col, stats in processed_analysis['outliers'].items())
        print(f"- 原始数据异常值: {raw_outliers} 个")
        print(f"- 处理后数据异常值: {processed_outliers} 个")
        print(
            f"- 异常值减少: {raw_outliers - processed_outliers} 个 ({100 * (raw_outliers - processed_outliers) / raw_outliers if raw_outliers > 0 else 0:.2f}%)")

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