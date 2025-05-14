# main.py
import os
import logging
import argparse
import glob
from data_processing import load_product_catalog, expand_parquet_paths, stream_process_parquet
from analysis import (
    analyze_category_associations,
    analyze_payment_associations,
    analyze_time_series_patterns,
    analyze_refund_patterns
)
from util import (
    visualize_association_rules,
    visualize_high_value_payments,
    visualize_time_series_patterns,
    visualize_refund_rates,
    print_and_save_frequent_itemsets,
    print_and_save_association_rules
)


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("analysis.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(parquet_files, product_catalog_file, output_dir='.', batch_size=10000):
    """运行整个分析的主函数"""
    logger = setup_logging()

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 处理parquet文件路径，支持文件夹
        files_to_process = expand_parquet_paths(parquet_files)

        if not files_to_process:
            logger.error("未找到任何 parquet 文件！")
            return

        logger.info(f"将处理以下 parquet 文件: {files_to_process}")
        logger.info(f"产品目录: {product_catalog_file}")
        logger.info(f"输出目录: {output_dir}")

        # 加载产品目录
        product_catalog = load_product_catalog(product_catalog_file)

        # 流式处理parquet文件，收集数据
        collected_data = stream_process_parquet(files_to_process, product_catalog, batch_size)

        # 分析任务1：商品类别关联规则挖掘
        logger.info("开始任务1：商品类别关联规则挖掘")
        category_results = analyze_category_associations(
            collected_data['user_day_categories'],
            min_support=0.02,
            min_confidence=0.5
        )

        # 分析任务2：支付方式与商品类别关联分析
        logger.info("开始任务2：支付方式与商品类别关联分析")
        payment_category_results = analyze_payment_associations(
            collected_data['user_payment_categories'],
            min_support=0.01,
            min_confidence=0.6
        )

        # 分析任务3：时间序列模式挖掘
        logger.info("开始任务3：时间序列模式挖掘")
        time_series_results = analyze_time_series_patterns(
            collected_data['quarterly_categories'],
            collected_data['monthly_categories'],
            collected_data['dow_categories'],
            collected_data['user_category_sequence']
        )

        # 分析任务4：退款模式分析
        logger.info("开始任务4：退款模式分析")
        refund_results = analyze_refund_patterns(
            collected_data['refund_user_day_categories'],
            collected_data['category_refund_count'],
            collected_data['category_total_count'],
            min_support=0.005,
            min_confidence=0.4
        )

        # 可视化结果
        logger.info("开始生成可视化结果")

        # 可视化任务1结果
        visualize_association_rules(category_results['rules'], '类别关联规则', output_dir)
        if 'electronics_rules' in category_results and not category_results['electronics_rules'].empty:
            visualize_association_rules(category_results['electronics_rules'], '电子产品相关规则', output_dir)

        # 可视化任务2结果
        visualize_association_rules(payment_category_results['rules'], '支付-类别关联规则', output_dir)
        visualize_high_value_payments(collected_data['high_value_payments'], output_dir)

        # 可视化任务3结果
        visualize_time_series_patterns(time_series_results, output_dir)

        # 可视化任务4结果
        visualize_association_rules(refund_results['rules'], '退款关联规则', output_dir)
        visualize_refund_rates(refund_results['refund_rates'], output_dir)

        logger.info(f"分析完成，结果保存到 {output_dir}")
        # 在main.py中的main函数中添加，位置在可视化结果之后

        # 打印频繁项集和关联规则
        logger.info("打印挖掘结果")
        print("\n\n" + "=" * 80)
        print("购物数据挖掘结果摘要")
        print("=" * 80)

        # 任务1：类别关联规则结果
        print_and_save_frequent_itemsets(category_results['frequent_itemsets'], '类别关联', output_dir)
        print_and_save_association_rules(category_results['rules'], '类别关联规则', output_dir)

        # 任务2：支付方式与类别关联结果
        print_and_save_frequent_itemsets(payment_category_results['frequent_itemsets'], '支付-类别关联', output_dir)
        print_and_save_association_rules(payment_category_results['rules'], '支付-类别关联规则', output_dir)

        # 任务4：退款模式分析结果
        if not refund_results['frequent_itemsets'].empty:
            print_and_save_frequent_itemsets(refund_results['frequent_itemsets'], '退款模式', output_dir)
        print_and_save_association_rules(refund_results['rules'], '退款关联规则', output_dir)

        # 打印关于电子产品的特殊规则（如果有）
        if 'electronics_rules' in category_results and not category_results['electronics_rules'].empty:
            print_and_save_association_rules(category_results['electronics_rules'], '电子产品相关规则', output_dir)

        # 打印时间序列模式摘要
        print("\n\n" + "=" * 80)
        print("时间序列模式摘要")
        print("=" * 80)

        seq_patterns = time_series_results['sequential_patterns']
        if not seq_patterns.empty and len(seq_patterns) > 0:
            print("\n最常见的顺序购买模式（前10个）：")
            print("-" * 60)
            for idx, row in seq_patterns.head(10).iterrows():
                print(f"{row['antecedent']} => {row['consequent']}   出现次数: {row['count']}")
    except Exception as e:
        logger.error(f"主函数中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析购物数据并挖掘关联规则。')
    parser.add_argument('--parquet-files', default='/home/pengxiao/virtualenvs/shujuwajue/30G_new_data', help='Parquet文件路径')
    parser.add_argument('--product-catalog', default='/home/pengxiao/virtualenvs/shujuwajue/homework2/product_catalog.json', help='产品目录JSON文件路径')
    parser.add_argument('--output-dir', default='2_analysis_results', help='保存结果的目录')
    parser.add_argument('--batch-size', type=int, default=10000, help='处理大文件的批处理大小')

    args = parser.parse_args()

    main(args.parquet_files, args.product_catalog, args.output_dir, args.batch_size)