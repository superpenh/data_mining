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
            'chinese_name': 'category',
            'email': 'category',
            'age': 'int8',
            'income': 'int32',
            'gender': 'category',
            'country': 'category',
            'chinese_address': 'category',
            'is_active': 'bool',
            'credit_score': 'int16',
            'phone_number': 'category'
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

                chinese_columns = ['chinese_name', 'chinese_address']
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


def generate_specific_visuals(gender_data, income_data, credit_score_data, output_dir='visuals'):
    """
    Generate the specified three types of visualizations: gender ratio, income distribution, and credit score distribution
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Gender ratio pie chart
    if gender_data is not None and not gender_data.empty:
        plt.figure(figsize=(10, 10))
        plt.pie(gender_data['count'], labels=gender_data['gender'], autopct='%1.1f%%',
                colors=['#5DA5DA', '#FAA43A', '#60BD68'],
                textprops={'fontsize': 14})
        plt.title('Gender Distribution', fontsize=18)
        plt.savefig(output_path / 'gender_ratio.png', dpi=300)
        plt.close()
        print(f"Generated gender ratio chart: {output_path / 'gender_ratio.png'}")

    # 2. Income distribution histogram
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

    # 3. Credit score distribution chart
    if credit_score_data is not None and len(credit_score_data) > 0:
        plt.figure(figsize=(12, 8))

        # Use boxplot and histogram to show distribution
        plt.subplot(2, 1, 1)
        sns.boxplot(x=credit_score_data, color='#5DA5DA')
        plt.title('Credit Score Distribution (Boxplot)', fontsize=16)
        plt.xlabel('Credit Score', fontsize=14)

        plt.subplot(2, 1, 2)
        sns.histplot(credit_score_data, bins=25, kde=True, color='#F17CB0')
        plt.title('Credit Score Distribution (Histogram)', fontsize=16)
        plt.xlabel('Credit Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        # Add segment descriptions
        credit_score_array = np.array(credit_score_data)
        low_credit = np.sum(credit_score_array < 600)
        medium_credit = np.sum((credit_score_array >= 600) & (credit_score_array < 750))
        high_credit = np.sum(credit_score_array >= 750)
        total = len(credit_score_array)

        credit_stats = (
            f"Low Credit (<600): {low_credit / total * 100:.1f}%, "
            f"Medium Credit (600-750): {medium_credit / total * 100:.1f}%, "
            f"High Credit (>750): {high_credit / total * 100:.1f}%"
        )
        plt.figtext(0.5, 0.01, credit_stats, ha='center', fontsize=12,
                    bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(output_path / 'credit_score_distribution.png', dpi=300)
        plt.close()
        print(f"Generated credit score distribution chart: {output_path / 'credit_score_distribution.png'}")

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
    credit_score_data = []

    # Process in batches
    for batch in data_generator():
        processed_batches += 1

        # Process gender data
        if 'gender' in batch.columns:
            batch_gender_counts = batch['gender'].value_counts().to_dict()
            for gender, count in batch_gender_counts.items():
                gender_summary[gender] = gender_summary.get(gender, 0) + count

        # For income and credit score, sample data for visualization
        # To keep memory manageable, sample at most 1000 rows per batch
        sample_size = min(len(batch), 1000)
        if 'income' in batch.columns:
            income_data.append(batch['income'].sample(sample_size).values)

        if 'credit_score' in batch.columns:
            credit_score_data.append(batch['credit_score'].sample(sample_size).values)

        # Free memory
        del batch
        gc.collect()

    # Create separate dataframes for each metric instead of trying to merge arrays of different lengths
    gender_data = None
    income_data_array = None
    credit_score_data_array = None

    # Prepare gender data
    if gender_summary:
        gender_data = pd.DataFrame({
            'gender': list(gender_summary.keys()),
            'count': list(gender_summary.values())
        })

    # Prepare income data
    if income_data:
        income_data_array = np.concatenate(income_data)

    # Prepare credit score data
    if credit_score_data:
        credit_score_data_array = np.concatenate(credit_score_data)

    # Generate visualizations
    print("Generating the three specified data visualizations...")
    vis_path = generate_specific_visuals(gender_data, income_data_array, credit_score_data_array,
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

    if credit_score_data_array is not None and len(credit_score_data_array) > 0:
        print("\nCredit Score Statistics:")
        print(f"  - Average Score: {np.mean(credit_score_data_array):.2f}")
        print(f"  - Median Score: {np.median(credit_score_data_array):.2f}")
        print(f"  - Minimum Score: {np.min(credit_score_data_array)}")
        print(f"  - Maximum Score: {np.max(credit_score_data_array)}")

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
    parser.add_argument('--data-dir', type=str, default='/home/pengxiao/virtualenvs/shujuwajue/data_quality_assessment/results', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='visual_results', help='Output directory')
    parser.add_argument('--chunksize', type=int, default=10000, help='Batch size')

    args = parser.parse_args()
    import time
    start_time = time.time()

    run_analysis(args.data_dir, args.output_dir, args.chunksize)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n可视化总执行时间: {execution_time:.2f} 秒 ({execution_time / 60:.2f} 分钟)")