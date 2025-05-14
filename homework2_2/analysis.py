# analysis.py
import pandas as pd
import numpy as np
import logging
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

logger = logging.getLogger(__name__)


def analyze_category_associations(user_day_categories, min_support=0.02, min_confidence=0.5):
    """分析商品类别关联规则"""
    try:
        # 转换数据格式以适合关联规则挖掘
        transactions = [list(categories) for categories in user_day_categories.values()]

        if not transactions:
            logger.warning("没有找到任何交易数据")
            return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}

        # 创建独热编码
        unique_categories = set()
        for transaction in transactions:
            unique_categories.update(transaction)

        logger.info(f"找到{len(unique_categories)}个唯一类别")

        # 创建独热编码的DataFrame
        encoded_df = pd.DataFrame(0, index=range(len(transactions)), columns=list(unique_categories))

        for i, items in enumerate(transactions):
            for item in items:
                encoded_df.loc[i, item] = 1

        # 使用FP-Growth算法挖掘频繁项集
        logger.info(f"使用FP-Growth算法挖掘频繁项集，最小支持度={min_support}")
        frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            logger.warning("没有找到频繁项集")
            return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}

        logger.info(f"找到{len(frequent_itemsets)}个频繁项集")

        # 生成关联规则
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            logger.warning("没有找到符合条件的关联规则")
        else:
            logger.info(f"生成了{len(rules)}条关联规则")

            # 筛选与电子产品相关的规则
            electronics_rules = rules[
                (rules['antecedents'].apply(lambda x: '电子产品' in x)) |
                (rules['consequents'].apply(lambda x: '电子产品' in x))
                ]

            if not electronics_rules.empty:
                logger.info(f"找到{len(electronics_rules)}条与电子产品相关的规则")

        return {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'electronics_rules': electronics_rules if 'electronics_rules' in locals() else pd.DataFrame()
        }
    except Exception as e:
        logger.error(f"分析类别关联时出错: {str(e)}")
        return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame(), 'electronics_rules': pd.DataFrame()}


def analyze_payment_associations(user_payment_categories, min_support=0.01, min_confidence=0.6):
    """分析支付方式与商品类别的关联规则"""
    try:
        # 转换数据格式以适合关联规则挖掘
        transactions = [list(payment_categories) for payment_categories in user_payment_categories.values()]

        if not transactions:
            logger.warning("没有找到任何支付-类别交易数据")
            return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}

        # 创建独热编码
        unique_payment_categories = set()
        for transaction in transactions:
            unique_payment_categories.update(transaction)

        logger.info(f"找到{len(unique_payment_categories)}个唯一支付-类别组合")

        # 创建独热编码的DataFrame
        encoded_df = pd.DataFrame(0, index=range(len(transactions)), columns=list(unique_payment_categories))

        for i, items in enumerate(transactions):
            for item in items:
                encoded_df.loc[i, item] = 1

        # 使用FP-Growth算法挖掘频繁项集
        logger.info(f"使用FP-Growth算法挖掘支付-类别关联，最小支持度={min_support}")
        frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            logger.warning("没有找到支付-类别频繁项集")
            return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}

        logger.info(f"找到{len(frequent_itemsets)}个支付-类别频繁项集")

        # 生成关联规则
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            logger.warning("没有找到符合条件的支付-类别关联规则")
        else:
            logger.info(f"生成了{len(rules)}条支付-类别关联规则")

        return {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules
        }
    except Exception as e:
        logger.error(f"分析支付-类别关联时出错: {str(e)}")
        return {'frequent_itemsets': pd.DataFrame(), 'rules': pd.DataFrame()}


def analyze_time_series_patterns(quarterly_categories, monthly_categories, dow_categories, user_category_sequence):
    """分析时间序列模式"""
    try:
        # 转换季度数据为DataFrame
        quarter_df = pd.DataFrame({
            quarter: pd.Series(categories) for quarter, categories in quarterly_categories.items()
        })

        # 转换月度数据为DataFrame
        month_df = pd.DataFrame({
            month: pd.Series(categories) for month, categories in monthly_categories.items()
        })

        # 转换星期几数据为DataFrame
        dow_df = pd.DataFrame({
            dow: pd.Series(categories) for dow, categories in dow_categories.items()
        })

        # 分析顺序模式 (先A后B)
        sequential_patterns = {}

        # 对每个用户的购买序列按日期排序
        for user_id, purchases in user_category_sequence.items():
            # 排序购买记录
            sorted_purchases = sorted(purchases, key=lambda x: x[0])
            categories = [p[1] for p in sorted_purchases]

            # 分析序列中的模式 (在30天内，先买A后买B)
            for i in range(len(categories) - 1):
                date_i = sorted_purchases[i][0]
                for j in range(i + 1, len(categories)):
                    date_j = sorted_purchases[j][0]
                    days_diff = (date_j - date_i).days

                    # 只考虑30天内的购买
                    if 0 <= days_diff <= 30:
                        pattern = (categories[i], categories[j])
                        sequential_patterns[pattern] = sequential_patterns.get(pattern, 0) + 1

        # 转换为DataFrame
        seq_patterns_df = pd.DataFrame([
            {'antecedent': a, 'consequent': c, 'count': count}
            for (a, c), count in sequential_patterns.items()
        ])

        if not seq_patterns_df.empty:
            seq_patterns_df = seq_patterns_df.sort_values('count', ascending=False)

        return {
            'quarterly_patterns': quarter_df,
            'monthly_patterns': month_df,
            'dow_patterns': dow_df,
            'sequential_patterns': seq_patterns_df
        }
    except Exception as e:
        logger.error(f"分析时间序列模式时出错: {str(e)}")
        return {
            'quarterly_patterns': pd.DataFrame(),
            'monthly_patterns': pd.DataFrame(),
            'dow_patterns': pd.DataFrame(),
            'sequential_patterns': pd.DataFrame()
        }


def analyze_refund_patterns(refund_user_day_categories, category_refund_count, category_total_count, min_support=0.005,
                            min_confidence=0.4):
    """分析退款模式"""
    try:
        # 分析退款关联规则
        transactions = [list(categories) for categories in refund_user_day_categories.values()]

        rules = pd.DataFrame()
        frequent_itemsets = pd.DataFrame()

        if transactions:
            # 创建独热编码
            unique_categories = set()
            for transaction in transactions:
                unique_categories.update(transaction)

            if unique_categories:
                # 创建独热编码的DataFrame
                encoded_df = pd.DataFrame(0, index=range(len(transactions)), columns=list(unique_categories))

                for i, items in enumerate(transactions):
                    for item in items:
                        encoded_df.loc[i, item] = 1

                # 使用FP-Growth算法挖掘频繁项集
                logger.info(f"使用FP-Growth算法挖掘退款模式，最小支持度={min_support}")
                frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)

                if not frequent_itemsets.empty:
                    # 生成关联规则
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # 计算退款率
        refund_rates = []
        for category, refund_count in category_refund_count.items():
            total_count = category_total_count.get(category, 0)
            if total_count > 0:
                refund_rate = refund_count / total_count
                refund_rates.append({
                    'category': category,
                    'refund_rate': refund_rate,
                    'refund_count': refund_count,
                    'total_count': total_count
                })

        refund_rates_df = pd.DataFrame(refund_rates)
        if not refund_rates_df.empty:
            refund_rates_df = refund_rates_df.sort_values('refund_rate', ascending=False)

        return {
            'rules': rules,
            'frequent_itemsets': frequent_itemsets,
            'refund_rates': refund_rates_df
        }
    except Exception as e:
        logger.error(f"分析退款模式时出错: {str(e)}")
        return {
            'rules': pd.DataFrame(),
            'frequent_itemsets': pd.DataFrame(),
            'refund_rates': pd.DataFrame()
        }