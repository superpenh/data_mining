# data_processing.py
import json
import pyarrow.parquet as pq
import pandas as pd
import logging
import glob
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_product_catalog(file_path):
    """加载产品目录，创建ID到类别和价格的映射"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)

        # 创建产品ID到类别和价格的映射
        product_map = {}
        if 'products' in catalog and isinstance(catalog['products'], list):
            for item in catalog['products']:
                if 'id' in item:
                    product_map[item['id']] = {
                        'category': item.get('category', '未知'),
                        'price': item.get('price', 0)
                    }

        logger.info(f"加载了产品目录，共{len(product_map)}个产品")
        return product_map
    except Exception as e:
        logger.error(f"加载产品目录时出错: {str(e)}")
        return {}


def expand_parquet_paths(parquet_files):
    """扩展路径，支持文件夹处理"""
    files_to_process = []
    if os.path.isdir(parquet_files):
        # 如果是文件夹，找出所有 parquet 文件
        folder_files = glob.glob(os.path.join(parquet_files, "*.parquet"))
        logger.info(f"在文件夹 {parquet_files} 中找到 {len(folder_files)} 个parquet文件")
        files_to_process.extend(folder_files)
    else:
        # 如果是单个文件，直接添加
        files_to_process.append(parquet_files)

    return files_to_process


def stream_process_parquet(parquet_files, product_catalog, batch_size=10000):
    """
    流式处理parquet文件，收集用于分析的数据

    Args:
        parquet_files: 要处理的parquet文件列表
        product_catalog: 产品ID到类别和价格的映射
        batch_size: 批处理大小

    Returns:
        dict: 包含用于各种分析的收集数据
    """
    # 用于任务1：类别关联规则
    user_day_categories = defaultdict(set)  # {user_id-date: set(categories)}

    # 用于任务2：支付方式与商品类别的关联分析
    user_payment_categories = defaultdict(set)  # {user_id: set(payment_method_category)}
    high_value_payments = defaultdict(int)  # {payment_method: count}

    # 用于任务3：时间序列模式
    quarterly_categories = defaultdict(lambda: defaultdict(int))  # {quarter: {category: count}}
    monthly_categories = defaultdict(lambda: defaultdict(int))  # {month: {category: count}}
    dow_categories = defaultdict(lambda: defaultdict(int))  # {day_of_week: {category: count}}
    user_category_sequence = defaultdict(list)  # {user_id: [(date, category)]}

    # 用于任务4：退款模式分析
    refund_user_day_categories = defaultdict(set)  # {user_id-date: set(categories)} for refunds
    category_refund_count = defaultdict(int)  # {category: refund_count}
    category_total_count = defaultdict(int)  # {category: total_count}

    # 处理每个parquet文件
    total_items_processed = 0

    for file_path in parquet_files:
        try:
            logger.info(f"处理parquet文件: {file_path}")
            pf = pq.ParquetFile(file_path)

            # 分批处理文件
            for batch in pf.iter_batches(batch_size=batch_size):
                batch_df = batch.to_pandas()
                batch_items_processed = 0

                for _, row in batch_df.iterrows():
                    try:
                        user_id = row['id']

                        # 解析purchase_history JSON字段
                        if isinstance(row['purchase_history'], str):
                            try:
                                purchase_history = json.loads(row['purchase_history'])
                            except json.JSONDecodeError:
                                continue
                        else:
                            # 如果已经是字典，则直接使用
                            purchase_history = row['purchase_history']

                        # 获取purchase_history中的信息
                        payment_method = purchase_history.get('payment_method', '未知')
                        payment_status = purchase_history.get('payment_status', '未知')
                        purchase_date_str = purchase_history.get('purchase_date', None)

                        if purchase_date_str:
                            try:
                                purchase_date = pd.to_datetime(purchase_date_str)
                                user_day_key = f"{user_id}-{purchase_date.date()}"
                                quarter_key = f"{purchase_date.year}-Q{purchase_date.quarter}"
                                month_key = f"{purchase_date.year}-{purchase_date.month}"
                                dow_key = purchase_date.dayofweek
                            except:
                                purchase_date = None
                                continue
                        else:
                            continue

                        # 处理商品列表
                        items = purchase_history.get('items', [])
                        categories_in_transaction = set()

                        for item in items:
                            try:
                                item_id = item.get('id')
                                if item_id is None:
                                    continue

                                batch_items_processed += 1

                                # 从产品目录获取类别和价格
                                category = '未知'
                                price = 0

                                if item_id in product_catalog:
                                    category = product_catalog[item_id]['category']
                                    price = product_catalog[item_id]['price']

                                # 任务1：收集类别关联数据
                                categories_in_transaction.add(category)

                                # 任务2：支付方式与类别关联
                                payment_category = f"{payment_method}_{category}"
                                user_payment_categories[user_id].add(payment_category)

                                # 处理高价值商品
                                if price > 5000:
                                    high_value_payments[payment_method] += 1

                                # 任务3：时间序列模式
                                quarterly_categories[quarter_key][category] += 1
                                monthly_categories[month_key][category] += 1
                                dow_categories[dow_key][category] += 1

                                # 记录用户购买序列，按日期排序
                                user_category_sequence[user_id].append((purchase_date, category))

                                # 任务4：退款模式分析
                                category_total_count[category] += 1

                                if payment_status in ['已退款', '部分退款']:
                                    category_refund_count[category] += 1
                                    refund_user_day_categories[user_day_key].add(category)
                            except Exception as e:
                                logger.debug(f"处理商品时出错: {str(e)}")

                        # 更新任务1的交易数据
                        if categories_in_transaction:
                            user_day_categories[user_day_key].update(categories_in_transaction)

                    except Exception as e:
                        logger.debug(f"处理用户行时出错: {str(e)}")

                total_items_processed += batch_items_processed
                # logger.info(f"已处理 {batch_items_processed} 个商品，累计 {total_items_processed} 个")

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    logger.info(f"所有数据处理完成，共处理 {total_items_processed} 个商品")

    # 返回收集的数据
    return {
        'user_day_categories': user_day_categories,
        'user_payment_categories': user_payment_categories,
        'high_value_payments': high_value_payments,
        'quarterly_categories': quarterly_categories,
        'monthly_categories': monthly_categories,
        'dow_categories': dow_categories,
        'user_category_sequence': user_category_sequence,
        'refund_user_day_categories': refund_user_day_categories,
        'category_refund_count': category_refund_count,
        'category_total_count': category_total_count
    }