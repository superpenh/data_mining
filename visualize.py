import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def configure_droid_sans_fallback():
    plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif font family
    plt.rcParams['font.sans-serif'] = [
        'Nimbus Sans L',
        # Secondary options
        *plt.rcParams['font.sans-serif']  # Keep original backup fonts
    ]
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue


def optimize_visualization(df, column, bins=20, figsize=(10, 6), title=None, xlabel=None, ylabel=None,
                           output_path=None, kind='hist'):
    plt.figure(figsize=figsize)

    if kind == 'hist':
        counts, bin_edges = np.histogram(df[column].dropna(), bins=bins)
        plt.bar(bin_edges[:-1], counts, width=bin_edges[1] - bin_edges[0], alpha=0.7)
    elif kind == 'pie':
        counts = df[column].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    elif kind == 'bar':
        counts = df[column].value_counts()
        plt.bar(counts.index, counts)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.close()
        return None


def load_data_in_parts(data_dir, dtype=None, chunksize=5000):
    """
    Load parquet files in chunks instead of using Dask, strictly controlling each batch size
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    import json

    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(f"Directory does not exist or is not a valid directory: {data_dir}")

    parquet_files = list(data_path.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in directory {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    if dtype is None:
        dtype = {
            'id': 'int32',
            'user_name': 'category',
            'fullname': 'category',
            'email': 'category',
            'age': 'int8',
            'income': 'int32',
            'gender': 'category',
            'country': 'category',
            'address': 'category',
            'is_active': 'bool',
            'phone_number': 'category'
            # login_history will be handled separately
        }

    metadata = {'total_rows': 0, 'files': []}

    for file_path in parquet_files:
        try:
            parquet_file = pq.ParquetFile(file_path)
            num_rows = parquet_file.metadata.num_rows

            metadata['total_rows'] += num_rows
            metadata['files'].append({
                'path': str(file_path),
                'rows': num_rows
            })

            if len(metadata.get('columns', [])) == 0:
                metadata['columns'] = parquet_file.schema.names
        except Exception as e:
            print(f"Error reading metadata for file {file_path}: {str(e)}")

    print(f"All parquet files have a total of {metadata['total_rows']} rows")

    def data_generator():
        processed_rows = 0

        for file_info in metadata['files']:
            file_path = file_info['path']
            print(f"Processing file: {file_path}")

            try:
                table = pq.read_table(file_path)
                df = table.to_pandas()
                file_rows = len(df)
                print(f"File contains {file_rows} rows of data")

                chinese_columns = ['fullname', 'address']
                for col in chinese_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

                for col, type_name in dtype.items():
                    if col in df.columns:
                        if type_name == 'category':
                            df[col] = df[col].astype('category')
                        else:
                            try:
                                df[col] = df[col].astype(type_name)
                            except:
                                pass

                # Process login_history JSON data if present
                if 'login_history' in df.columns:
                    # Extract login_count and avg_session_duration from JSON
                    login_count_list = []
                    avg_session_duration_list = []

                    for idx, json_data in enumerate(df['login_history']):
                        if pd.isna(json_data):
                            login_count_list.append(np.nan)
                            avg_session_duration_list.append(np.nan)
                            continue

                        try:
                            if isinstance(json_data, str):
                                data = json.loads(json_data)
                            elif isinstance(json_data, dict):
                                data = json_data
                            else:
                                login_count_list.append(np.nan)
                                avg_session_duration_list.append(np.nan)
                                continue

                            # Extract login_count
                            if 'login_count' in data:
                                login_count_list.append(data['login_count'])
                            else:
                                login_count_list.append(np.nan)

                            # Extract avg_session_duration
                            if 'avg_session_duration' in data:
                                avg_session_duration_list.append(data['avg_session_duration'])
                            else:
                                avg_session_duration_list.append(np.nan)

                        except:
                            login_count_list.append(np.nan)
                            avg_session_duration_list.append(np.nan)

                    # Add extracted data as new columns
                    df['login_count'] = login_count_list
                    df['avg_session_duration'] = avg_session_duration_list

                for start_idx in range(0, file_rows, chunksize):
                    end_idx = min(start_idx + chunksize, file_rows)
                    chunk = df.iloc[start_idx:end_idx].copy()

                    processed_rows += len(chunk)
                    print(f"Total processed: {processed_rows}/{metadata['total_rows']} rows")

                    yield chunk

                    del chunk
                    gc.collect()

                del df
                gc.collect()

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    return data_generator, metadata


def generate_specific_visuals(gender_data, income_data, login_data, output_dir='visuals'):
    """
    Generate the specified three types of visualizations:
    gender ratio, income distribution, and login history data (replacing credit score)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Gender ratio pie chart (unchanged)
    if gender_data is not None and not gender_data.empty:
        plt.figure(figsize=(10, 10))
        plt.pie(gender_data['count'], labels=gender_data['gender'], autopct='%1.1f%%',
                colors=['#5DA5DA', '#FAA43A', '#60BD68'],
                textprops={'fontsize': 14})
        plt.title('Gender Distribution', fontsize=18)
        plt.savefig(output_path / 'gender_ratio.png', dpi=300)
        plt.close()
        print(f"Generated gender ratio chart: {output_path / 'gender_ratio.png'}")

    # 2. Income distribution histogram (unchanged)
    if income_data is not None and len(income_data) > 0:
        plt.figure(figsize=(12, 8))
        sns.histplot(income_data, bins=30, kde=True, color='#4D85BD')
        plt.title('Income Distribution', fontsize=18)
        plt.xlabel('Income Amount', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add statistical information
        income_mean = np.mean(income_data)
        income_median = np.median(income_data)
        plt.axvline(income_mean, color='red', linestyle='--', label=f'Mean: {income_mean:.2f}')
        plt.axvline(income_median, color='green', linestyle='--', label=f'Median: {income_median:.2f}')
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path / 'income_distribution.png', dpi=300)
        plt.close()
        print(f"Generated income distribution chart: {output_path / 'income_distribution.png'}")

    # 3. Login Data Visualizations (replacing Credit score distribution chart)
    if login_data is not None and not login_data.empty:
        # 3.1. Login Count Distribution
        if 'login_count' in login_data.columns and login_data['login_count'].notna().any():
            plt.figure(figsize=(12, 8))

            login_count_data = login_data['login_count'].dropna()

            # Boxplot for login count
            plt.subplot(2, 1, 1)
            sns.boxplot(x=login_count_data, color='#5DA5DA')
            plt.title('Login Count Distribution (Boxplot)', fontsize=16)
            plt.xlabel('Number of Logins', fontsize=14)

            # Histogram for login count
            plt.subplot(2, 1, 2)
            sns.histplot(login_count_data, bins=25, kde=True, color='#F17CB0')
            plt.title('Login Count Distribution (Histogram)', fontsize=16)
            plt.xlabel('Number of Logins', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)

            # Add segment descriptions
            login_count_array = np.array(login_count_data)
            low_activity = np.sum(login_count_array < 30)
            medium_activity = np.sum((login_count_array >= 30) & (login_count_array < 60))
            high_activity = np.sum(login_count_array >= 60)
            total = len(login_count_array)

            login_stats = (
                f"Low Activity (<30 logins): {low_activity / total * 100:.1f}%, "
                f"Medium Activity (30-60 logins): {medium_activity / total * 100:.1f}%, "
                f"High Activity (>60 logins): {high_activity / total * 100:.1f}%"
            )
            plt.figtext(0.5, 0.01, login_stats, ha='center', fontsize=12,
                        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(output_path / 'login_count_distribution.png', dpi=300)
            plt.close()
            print(f"Generated login count distribution chart: {output_path / 'login_count_distribution.png'}")

        # 3.2. Average Session Duration Distribution
        if 'avg_session_duration' in login_data.columns and login_data['avg_session_duration'].notna().any():
            plt.figure(figsize=(12, 8))

            session_duration_data = login_data['avg_session_duration'].dropna()

            # Boxplot for session duration
            plt.subplot(2, 1, 1)
            sns.boxplot(x=session_duration_data, color='#B2912F')
            plt.title('Average Session Duration Distribution (Boxplot)', fontsize=16)
            plt.xlabel('Duration (seconds)', fontsize=14)

            # Histogram for session duration
            plt.subplot(2, 1, 2)
            sns.histplot(session_duration_data, bins=25, kde=True, color='#60BD68')
            plt.title('Average Session Duration Distribution (Histogram)', fontsize=16)
            plt.xlabel('Duration (seconds)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)

            # Add segment descriptions
            duration_array = np.array(session_duration_data)
            short_session = np.sum(duration_array < 50)
            medium_session = np.sum((duration_array >= 50) & (duration_array < 70))
            long_session = np.sum(duration_array >= 70)
            total = len(duration_array)

            duration_stats = (
                f"Short Sessions (<50s): {short_session / total * 100:.1f}%, "
                f"Medium Sessions (50-70s): {medium_session / total * 100:.1f}%, "
                f"Long Sessions (>70s): {long_session / total * 100:.1f}%"
            )
            plt.figtext(0.5, 0.01, duration_stats, ha='center', fontsize=12,
                        bbox={"facecolor": "lightblue", "alpha": 0.2, "pad": 5})

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(output_path / 'session_duration_distribution.png', dpi=300)
            plt.close()
            print(f"Generated session duration distribution chart: {output_path / 'session_duration_distribution.png'}")

    return output_path


def process_data(data_generator, output_dir='results'):
    """
    Process data and generate visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize results
    processed_batches = 0

    gender_summary = {}
    income_data = []
    login_data_list = []  # Changed from credit_score_data to login_data_list

    # Process in batches
    for batch in data_generator():
        processed_batches += 1

        # Process gender data
        if 'gender' in batch.columns:
            batch_gender_counts = batch['gender'].value_counts().to_dict()
            for gender, count in batch_gender_counts.items():
                gender_summary[gender] = gender_summary.get(gender, 0) + count

        # For income, sample data for visualization
        # To keep memory manageable, sample at most 1000 rows per batch
        sample_size = min(len(batch), 1000)
        if 'income' in batch.columns:
            income_data.append(batch['income'].sample(sample_size).values)

        # For login data (replacing credit_score)
        login_columns = ['login_count', 'avg_session_duration']
        has_login_data = any(col in batch.columns for col in login_columns)

        if has_login_data:
            login_sample = batch.sample(min(len(batch), 1000))
            login_data_cols = [col for col in login_columns if col in login_sample.columns]
            login_data_list.append(login_sample[login_data_cols])

        # Free memory
        del batch
        gc.collect()

    # Create separate dataframes for each metric
    gender_data = None
    income_data_array = None
    login_data_df = None  # Changed from credit_score_data_array

    # Prepare gender data
    if gender_summary:
        gender_data = pd.DataFrame({
            'gender': list(gender_summary.keys()),
            'count': list(gender_summary.values())
        })

    # Prepare income data
    if income_data:
        income_data_array = np.concatenate(income_data)

    # Prepare login data (replacing credit_score data)
    if login_data_list:
        login_data_df = pd.concat(login_data_list, ignore_index=True)

    # Generate visualizations
    print("Generating the specified data visualizations...")
    vis_path = generate_specific_visuals(gender_data, income_data_array, login_data_df,
                                         output_dir=str(output_path / 'visuals'))

    # Print statistical summary
    print("\nData Statistical Summary:")
    if gender_data is not None and not gender_data.empty:
        total = gender_data['count'].sum()
        print("Gender Distribution:")
        for i, row in gender_data.iterrows():
            gender = row['gender']
            count = row['count']
            percentage = (count / total) * 100
            print(f"  - {gender}: {count} ({percentage:.2f}%)")

    if income_data_array is not None and len(income_data_array) > 0:
        print("\nIncome Statistics:")
        print(f"  - Average Income: {np.mean(income_data_array):.2f}")
        print(f"  - Median Income: {np.median(income_data_array):.2f}")
        print(f"  - Minimum Income: {np.min(income_data_array)}")
        print(f"  - Maximum Income: {np.max(income_data_array)}")

    # Login data statistics (replacing credit score statistics)
    if login_data_df is not None and not login_data_df.empty:
        print("\nLogin Data Statistics:")

        if 'login_count' in login_data_df.columns and login_data_df['login_count'].notna().any():
            login_count_data = login_data_df['login_count'].dropna()
            print("\nLogin Count Statistics:")
            print(f"  - Average Login Count: {np.mean(login_count_data):.2f}")
            print(f"  - Median Login Count: {np.median(login_count_data):.2f}")
            print(f"  - Minimum Login Count: {np.min(login_count_data)}")
            print(f"  - Maximum Login Count: {np.max(login_count_data)}")

            # Activity level segmentation
            low_activity = np.sum(login_count_data < 30)
            medium_activity = np.sum((login_count_data >= 30) & (login_count_data < 60))
            high_activity = np.sum(login_count_data >= 60)
            total = len(login_count_data)

            print(f"  - Low Activity Users (<30 logins): {low_activity} ({low_activity / total * 100:.2f}%)")
            print(f"  - Medium Activity Users (30-60 logins): {medium_activity} ({medium_activity / total * 100:.2f}%)")
            print(f"  - High Activity Users (>60 logins): {high_activity} ({high_activity / total * 100:.2f}%)")

        if 'avg_session_duration' in login_data_df.columns and login_data_df['avg_session_duration'].notna().any():
            session_duration_data = login_data_df['avg_session_duration'].dropna()
            print("\nSession Duration Statistics:")
            print(f"  - Average Session Duration: {np.mean(session_duration_data):.2f} seconds")
            print(f"  - Median Session Duration: {np.median(session_duration_data):.2f} seconds")
            print(f"  - Minimum Session Duration: {np.min(session_duration_data):.2f} seconds")
            print(f"  - Maximum Session Duration: {np.max(session_duration_data):.2f} seconds")

            # Session duration segmentation
            short_session = np.sum(session_duration_data < 60)
            medium_session = np.sum((session_duration_data >= 60) & (session_duration_data < 180))
            long_session = np.sum(session_duration_data >= 180)
            total = len(session_duration_data)

            print(f"  - Short Sessions (<60s): {short_session} ({short_session / total * 100:.2f}%)")
            print(f"  - Medium Sessions (60-180s): {medium_session} ({medium_session / total * 100:.2f}%)")
            print(f"  - Long Sessions (>180s): {long_session} ({long_session / total * 100:.2f}%)")

    return output_path


def run_analysis(data_dir, output_dir='visual_results', chunksize=10000):
    """
    Main function to run the specified analysis
    :param data_dir: Data directory
    :param output_dir: Output directory
    :param chunksize: Batch size
    """
    # Configure font
    configure_droid_sans_fallback()

    # Load data
    data_gen, metadata = load_data_in_parts(data_dir, chunksize=chunksize)

    # Process data
    result_path = process_data(data_gen, output_dir=output_dir)

    print(f"Data analysis complete, results saved to: {result_path}")
    return result_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Analysis - Gender Ratio, Income Distribution, and Credit Score Distribution')
    parser.add_argument('--data-dir', type=str, default='/home/pengxiao/virtualenvs/shujuwajue/30G_new_data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='visual_results', help='Output directory')
    parser.add_argument('--chunksize', type=int, default=10000, help='Batch size')

    args = parser.parse_args()
    import time
    start_time = time.time()

    run_analysis(args.data_dir, args.output_dir, args.chunksize)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n可视化总执行时间: {execution_time:.2f} 秒 ({execution_time / 60:.2f} 分钟)")