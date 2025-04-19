import pandas as pd
import numpy as np
import json
from pathlib import Path
import gc


def parse_purchase_history(df):
    """Efficiently parse purchase_history JSON column"""
    if 'purchase_history' not in df.columns:
        return df

    print("解析purchase_history JSON字段...")

    # Create a function to parse a single JSON string
    def parse_json(json_str):
        if pd.isna(json_str):
            return None, None, None
        try:
            data = json.loads(json_str)
            return (
                data.get('categories'),
                len(data.get('items', [])),
                data.get('avg_price')
            )
        except:
            return None, None, None

    # Apply function to the entire column at once (vectorized)
    parsed_data = df['purchase_history'].apply(parse_json)

    # Create new columns from the parsed results
    df['purchase_category'], df['purchase_items_count'], df['avg_price'] = zip(*parsed_data)

    # Drop the original JSON column
    df = df.drop(columns=['purchase_history'])

    return df


def parse_login_history(df):
    """Efficiently parse login_history JSON column"""
    if 'login_history' not in df.columns:
        return df

    print("解析login_history JSON字段...")

    # Create a function to parse a single JSON string
    def parse_json(json_str):
        if pd.isna(json_str):
            return None, None
        try:
            data = json.loads(json_str)
            return (
                data.get('avg_session_duration'),
                data.get('login_count')
            )
        except:
            return None, None

    # Apply function to the entire column at once (vectorized)
    parsed_data = df['login_history'].apply(parse_json)

    # Create new columns from the parsed results
    df['avg_session_duration'], df['login_count'] = zip(*parsed_data)

    # Drop the original JSON column
    df = df.drop(columns=['login_history'])

    return df


def load_data(data_dir, dtype=None):
    """
    加载所有parquet文件到一个DataFrame中
    """
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(f"目录不存在或不是有效目录: {data_dir}")

    parquet_files = list(data_path.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"在目录 {data_dir} 中未找到parquet文件")

    print(f"发现 {len(parquet_files)} 个parquet文件")

    if dtype is None:
        dtype = {
            'id': 'int32',
            'user_name': 'category',
            'fullname': 'category',
            'email': 'category',
            'age': 'int32',  # 不再使用int8来避免潜在的溢出问题
            'income': 'int64',  # 使用更大的整数类型
            'gender': 'category',
            'country': 'category',
            'address': 'category',
            'is_active': 'bool',
            'credit_score': 'int32',  # 不再使用int16
            'phone_number': 'category'
        }

    all_dfs = []
    metadata = {'total_rows': 0, 'files': []}

    for file_path in parquet_files:
        try:
            print(f"处理文件: {file_path}")

            df = pd.read_parquet(file_path)
            file_rows = len(df)
            print(f"文件包含 {file_rows} 行数据")

            metadata['total_rows'] += file_rows
            metadata['files'].append({
                'path': str(file_path),
                'rows': file_rows
            })

            print('开始数据类型转换...')
            chinese_columns = ['fullname', 'address']
            for col in chinese_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            # 应用数据类型转换
            for col, type_name in dtype.items():
                if col in df.columns:
                    if type_name == 'category':
                        df[col] = df[col].astype('category')
                    else:
                        try:
                            df[col] = df[col].astype(type_name)
                        except:
                            pass  # 如果转换失败，保留原始类型

            # 解析JSON字段
            df = parse_purchase_history(df)
            df = parse_login_history(df)  # 添加对login_history的解析
            all_dfs.append(df)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 合并所有数据帧
    print("合并所有数据...")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"成功加载 {len(combined_df)} 行数据")

        del all_dfs
        gc.collect()

        return combined_df, metadata
    else:
        raise ValueError("没有成功加载任何数据")


def identify_high_value_users(df, output_file='high_value_users.csv'):
    """
    识别高价值用户 - 使用login_history中的数据代替credit_score
    """
    print(f"开始高价值用户分析，数据集有 {len(df)} 行")

    # 计算阈值
    print("计算高价值用户阈值...")
    thresholds = {
        'income': df['income'].mean(),
        'avg_session_duration': df['avg_session_duration'].mean() if 'avg_session_duration' in df.columns else 0,
        'login_count': df['login_count'].mean() if 'login_count' in df.columns else 0,
        'purchase_items_count': df['purchase_items_count'].mean() if 'purchase_items_count' in df.columns else 0,
        'avg_price': df['avg_price'].mean() if 'avg_price' in df.columns else 0
    }

    print(f"计算得到的阈值: {thresholds}")

    # 筛选高价值用户 - 不再使用credit_score
    print("开始筛选高价值用户...")
    conditions = [
        df['income'] > thresholds['income']
    ]

    # 添加avg_session_duration和login_count条件
    if 'avg_session_duration' in df.columns:
        conditions.append(df['avg_session_duration'] > thresholds['avg_session_duration'])

    if 'login_count' in df.columns:
        conditions.append(df['login_count'] > thresholds['login_count'])

    # 只有当purchase_items_count列存在时才添加该条件
    if 'purchase_items_count' in df.columns:
        conditions.append(df['purchase_items_count'] > thresholds['purchase_items_count'])

    # 添加average_purchase_price条件
    if 'avg_price' in df.columns:
        conditions.append(df['avg_price'] > thresholds['avg_price'])

    high_value_users = df[np.logical_and.reduce(conditions)]

    high_value_count = len(high_value_users)
    hvu_percentage = (high_value_count / len(df)) * 100 if len(df) > 0 else 0

    print(f"找到 {high_value_count} 个高价值用户 ({hvu_percentage:.2f}%)")

    # 保存高价值用户数据
    if high_value_count > 0:
        # 准备输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        # 删除敏感信息列
        columns_to_exclude = ['fullname', 'email', 'address']
        filtered_df = high_value_users.drop(
            columns=[col for col in columns_to_exclude if col in high_value_users.columns])

        # 保存到CSV
        filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"高价值用户数据已保存到 {output_file}")

    # 计算高价值用户特征统计
    if high_value_count > 0:
        numeric_cols = ['age', 'income']
        # 添加新的评估指标
        if 'avg_session_duration' in df.columns:
            numeric_cols.append('avg_session_duration')
        if 'login_count' in df.columns:
            numeric_cols.append('login_count')
        if 'purchase_items_count' in df.columns:
            numeric_cols.append('purchase_items_count')
        if 'avg_price' in df.columns:
            numeric_cols.append('avg_price')

        hvu_stats = high_value_users[numeric_cols].describe()
    else:
        hvu_stats = pd.DataFrame()

    return {
        'high_value_count': high_value_count,
        'high_value_percentage': hvu_percentage,
        'high_value_stats': hvu_stats,
        'thresholds': thresholds,
        'high_value_users': high_value_users
    }


def generate_high_value_report(result, output_dir):
    """
    生成高价值用户报告
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 保存高价值用户报告
    with open(output_path / 'high_value_users_report.txt', 'w', encoding='utf-8') as f:
        f.write("高价值用户分析报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"高价值用户数量: {result['high_value_count']:,}\n")
        f.write(f"高价值用户占比: {result['high_value_percentage']:.2f}%\n\n")

        f.write("筛选阈值:\n")
        for key, value in result['thresholds'].items():
            f.write(f"  {key}: {value:.2f}\n")

        if not result['high_value_stats'].empty:
            f.write("\n高价值用户统计信息:\n")
            f.write(result['high_value_stats'].to_string())


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='大数据用户分析程序')
    parser.add_argument('--data-dir', type=str,
                        default='/home/pengxiao/virtualenvs/shujuwajue/30G_new_data',
                        help='包含parquet文件的目录路径')
    parser.add_argument('--output-dir', type=str, default='select_results', help='结果输出目录')

    args = parser.parse_args()

    print(f"启动用户分析程序...")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")

    try:
        print("开始加载数据...")
        df, metadata = load_data(data_dir=args.data_dir)

        print("开始识别高价值用户...")
        output_file = Path(args.output_dir) / 'high_value_users.csv'
        hvu_result = identify_high_value_users(df, output_file=output_file)

        print("生成高价值用户报告...")
        generate_high_value_report(hvu_result, args.output_dir)

        print("分析成功完成!")

    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import time

    start_time = time.time()

    main()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n高质量用户筛选总执行时间: {execution_time:.2f} 秒 ({execution_time / 60:.2f} 分钟)")