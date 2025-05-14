# visualization.py
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def visualize_association_rules(rules, title, output_dir='.', top_n=10):
    """保存关联规则到CSV（已删除图表生成）"""
    if rules.empty:
        logger.warning(f"没有找到{title}的规则")
        return

    try:
        # 保存规则到CSV
        csv_file = os.path.join(output_dir, f"{title.replace(' ', '_')}_rules.csv")
        rules.to_csv(csv_file, index=False, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码

        logger.info(f"规则数据已保存到{csv_file}")
    except Exception as e:
        logger.error(f"保存关联规则数据时出错: {str(e)}")


def visualize_high_value_payments(high_value_payments, output_dir):
    """保存高价值商品的支付方式统计（已删除图表生成）"""
    try:
        # 转换为DataFrame
        df = pd.DataFrame({
            'payment_method': list(high_value_payments.keys()),
            'count': list(high_value_payments.values())
        })

        if not df.empty:
            df = df.sort_values('count', ascending=False)

            # 保存数据
            csv_file = os.path.join(output_dir, "high_value_payment_methods.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码

            logger.info(f"高价值商品支付方式数据已保存到{csv_file}")
    except Exception as e:
        logger.error(f"保存高价值商品支付方式数据时出错: {str(e)}")


def visualize_time_series_patterns(time_series_results, output_dir):
    """保存时间序列模式数据（已删除图表生成）"""
    try:
        # 保存季度模式
        quarterly_df = time_series_results['quarterly_patterns']
        if not quarterly_df.empty:
            csv_file = os.path.join(output_dir, "quarterly_patterns.csv")
            quarterly_df.to_csv(csv_file, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码
            logger.info(f"季度模式数据已保存到{csv_file}")

        # 保存月度模式
        monthly_df = time_series_results['monthly_patterns']
        if not monthly_df.empty:
            csv_file = os.path.join(output_dir, "monthly_patterns.csv")
            monthly_df.to_csv(csv_file, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码
            logger.info(f"月度模式数据已保存到{csv_file}")

        # 保存星期几模式
        dow_df = time_series_results['dow_patterns']
        if not dow_df.empty:
            csv_file = os.path.join(output_dir, "dow_patterns.csv")
            dow_df.to_csv(csv_file, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码
            logger.info(f"星期几模式数据已保存到{csv_file}")

        # 保存顺序模式
        seq_df = time_series_results['sequential_patterns']
        if not seq_df.empty and len(seq_df) > 0:
            csv_file = os.path.join(output_dir, "sequential_patterns.csv")
            seq_df.to_csv(csv_file, index=False, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码
            logger.info(f"顺序模式数据已保存到{csv_file}")
    except Exception as e:
        logger.error(f"保存时间序列模式数据时出错: {str(e)}")


def visualize_refund_rates(refund_rates_df, output_dir):
    """保存退款率数据（已删除图表生成）"""
    try:
        if not refund_rates_df.empty and len(refund_rates_df) > 0:
            # 保存数据
            csv_file = os.path.join(output_dir, "refund_rates.csv")
            refund_rates_df.to_csv(csv_file, index=False, encoding='utf-8-sig')  # 使用utf-8-sig解决中文乱码

            logger.info(f"退款率数据已保存到{csv_file}")
    except Exception as e:
        logger.error(f"保存退款率数据时出错: {str(e)}")


def print_and_save_frequent_itemsets(frequent_itemsets, title, output_dir, top_n=20):
    """打印并保存频繁项集"""
    if frequent_itemsets.empty:
        logger.warning(f"没有找到{title}的频繁项集")
        return

    try:
        # 按支持度排序
        sorted_itemsets = frequent_itemsets.sort_values('support', ascending=False)

        # 转换frozenset为可读字符串
        sorted_itemsets['itemset_str'] = sorted_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

        # 打印到控制台
        print(f"\n{'=' * 50}")
        print(f"{title}的频繁项集（按支持度排序，前{min(top_n, len(sorted_itemsets))}个）：")
        print(f"{'=' * 50}")

        for idx, row in sorted_itemsets.head(top_n).iterrows():
            print(f"项集: {row['itemset_str']}")
            print(f"支持度: {row['support']:.4f}")
            print('-' * 50)

        # 保存到文本文件
        output_file = os.path.join(output_dir, f"{title.replace(' ', '_')}_frequent_itemsets.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{title}的频繁项集（按支持度排序）：\n")
            f.write("=" * 50 + "\n")

            for idx, row in sorted_itemsets.iterrows():
                f.write(f"项集: {row['itemset_str']}\n")
                f.write(f"支持度: {row['support']:.4f}\n")
                f.write('-' * 50 + "\n")

        logger.info(f"将频繁项集信息保存到{output_file}")
    except Exception as e:
        logger.error(f"打印和保存频繁项集时出错: {str(e)}")


def print_and_save_association_rules(rules, title, output_dir, top_n=20):
    """打印并保存关联规则"""
    if rules.empty:
        logger.warning(f"没有找到{title}的关联规则")
        return

    try:
        # 按提升度排序
        sorted_rules = rules.sort_values('lift', ascending=False)

        # 转换frozensets为可读字符串
        sorted_rules['antecedents_str'] = sorted_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        sorted_rules['consequents_str'] = sorted_rules['consequents'].apply(lambda x: ', '.join(list(x)))

        # 打印到控制台
        print(f"\n{'=' * 60}")
        print(f"{title}的关联规则（按提升度排序，前{min(top_n, len(sorted_rules))}个）：")
        print(f"{'=' * 60}")

        for idx, row in sorted_rules.head(top_n).iterrows():
            print(f"规则: {row['antecedents_str']} => {row['consequents_str']}")
            print(f"支持度: {row['support']:.4f}, 置信度: {row['confidence']:.4f}, 提升度: {row['lift']:.4f}")
            print('-' * 60)

        # 保存到文本文件
        output_file = os.path.join(output_dir, f"{title.replace(' ', '_')}_association_rules.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{title}的关联规则（按提升度排序）：\n")
            f.write("=" * 60 + "\n")

            for idx, row in sorted_rules.iterrows():
                f.write(f"规则: {row['antecedents_str']} => {row['consequents_str']}\n")
                f.write(f"支持度: {row['support']:.4f}, 置信度: {row['confidence']:.4f}, 提升度: {row['lift']:.4f}\n")
                f.write('-' * 60 + "\n")

        logger.info(f"将关联规则信息保存到{output_file}")
    except Exception as e:
        logger.error(f"打印和保存关联规则时出错: {str(e)}")