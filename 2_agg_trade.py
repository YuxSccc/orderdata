import pandas as pd
import json
import os
import time
import csv
import decimal

BTC_pre_agg_duration = 100

def get_decimal_precision(number):
    decimal_value = decimal.Decimal(str(number))
    return -decimal_value.as_tuple().exponent

def has_header(file_path):
    # 尝试读取文件的第一行并检查列名
    first_row = pd.read_csv(file_path, nrows=0)
    return "id" in first_row.columns  # 检查第一列是否是 "id"

path_prefix = '/mnt/e/orderdata/binance'
for file in os.listdir(f'{path_prefix}/rawdata'):
    filename = file.split('.')[0]
    if os.path.exists(f'{path_prefix}/agg_trade/{filename}.csv'):
        continue

    if has_header(f'{path_prefix}/rawdata/{file}'):
        df = pd.read_csv(f'{path_prefix}/rawdata/{file}')
    else:
        df = pd.read_csv(f'{path_prefix}/rawdata/{file}', header=None, names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'])

    total_rows = len(df)
    start_time = time.time()
    pre_agg_duration = BTC_pre_agg_duration
    last_print_time = start_time
    aggregated = []

    buySideAggregated = None
    sellSideAggregated = None
    max_qty_precision = 0
    max_quote_qty_precision = 0
    for row in df.itertuples():
        if row.Index % 10000 == 0:
            current_time = time.time()

            if current_time - last_print_time >= 10:
                progress = (row.Index + 1) / total_rows * 100
                print(f"已处理 {row.Index + 1}/{total_rows} 行，进度: {progress:.2f}%")
                last_print_time = current_time

        ts = int(row.time // pre_agg_duration * pre_agg_duration)

        op_agg = buySideAggregated if row.is_buyer_maker else sellSideAggregated

        qty_precision = get_decimal_precision(row.qty)
        quote_qty_precision = get_decimal_precision(row.quote_qty)
        if qty_precision > max_qty_precision:
            max_qty_precision = qty_precision
        if quote_qty_precision > max_quote_qty_precision:
            max_quote_qty_precision = quote_qty_precision

        if op_agg and op_agg['time'] != ts:
            aggregated.append(op_agg)
            if op_agg['is_buyer_maker'] == True:
                buySideAggregated = None
            else:
                sellSideAggregated = None
            op_agg = None
        if op_agg is None:
            op_agg = {
                'id': row.id,
                'price': row.price,
                'qty': row.qty,
                'quote_qty': row.quote_qty,
                'time': ts,
                'is_buyer_maker': row.is_buyer_maker,
                'count': 1
            }
            if op_agg['is_buyer_maker'] == True:
                buySideAggregated = op_agg
            else:
                sellSideAggregated = op_agg
        else:
            op_agg['qty'] += row.qty
            op_agg['quote_qty'] += row.quote_qty
            op_agg['count'] += 1

    if buySideAggregated:
        aggregated.append(buySideAggregated)
    if sellSideAggregated:
        aggregated.append(sellSideAggregated)

    aggregated.sort(key=lambda x: (x['time'], x['id']))

    with open(f'{path_prefix}/agg_trade/{filename}.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in aggregated:
            data['qty'] = round(data['qty'], max_qty_precision)
            data['quote_qty'] = round(data['quote_qty'], max_quote_qty_precision)
            writer.writerow(data)
