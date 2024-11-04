import pandas as pd
import json
import os
import time
from common import FootprintBar, Tick
from concurrent.futures import ProcessPoolExecutor
import logging
import concurrent.futures
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler('footprint_process.log'),
        logging.StreamHandler()
    ]
)

BTC_duration = 60 * 5
BTC_scale = 100
BTC_volumePrecision = 2
BTC_pricePrecision = 1

path_prefix = '/mnt/e/orderdata/binance'

def has_header(file_path):
    # 尝试读取文件的第一行并检查列名
    first_row = pd.read_csv(file_path, nrows=0)
    return "id" in first_row.columns  # 检查第一列是否是 "id"

def process_file(file: str):
    try:
        filename = file.split('.')[0]
        output_path = f'./footprint/{filename}.json'
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            logging.info(f"Skip existing file: {filename}")
            return True
            
        logging.info(f"Processing: {filename}")
        start_time = time.time()
        
        # 读取CSV文件
        file_path = f'{path_prefix}/rawdata/{file}'
        if has_header(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, header=None, 
                           names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'])

        total_rows = len(df)
        last_print_time = time.time()

        footprintList = {}
        currentFootprint = FootprintBar(BTC_duration, BTC_scale, BTC_volumePrecision, BTC_pricePrecision)
        
        for row in df.itertuples():
            if row.Index % 50000 == 0:
                current_time = time.time()
                if current_time - last_print_time >= 30:
                    progress = (row.Index + 1) / total_rows * 100
                    logging.info(f"{filename}: Processed {row.Index + 1}/{total_rows} rows, {progress:.2f}%")
                    last_print_time = current_time

            tick = Tick()
            tick.timestamp = int(row.time // 1000)
            tick.price = float(row.price)
            tick.size = float(row.qty)
            tick.isBuy = row.is_buyer_maker == False
            
            if currentFootprint.handle_tick(tick):
                continue
                
            footprintList[currentFootprint.timestamp] = currentFootprint
            currentFootprint = FootprintBar(BTC_duration, BTC_scale, BTC_volumePrecision, BTC_pricePrecision)
            if currentFootprint.handle_tick(tick) == False:
                logging.error(f"Handle tick failed for {filename}")
                logging.error(f"Tick: {tick}")
                logging.error(f"Footprint: {currentFootprint}")
                return False

        if currentFootprint.timestamp != 0:
            footprintList[currentFootprint.timestamp] = currentFootprint

        for footprint in footprintList.values():
            footprint.end_handle_tick()

        footprintList = dict(sorted(footprintList.items()))
        
        # 确保输出目录存在
        os.makedirs('./footprint', exist_ok=True)
        
        with open(output_path, 'w') as f:
            data = {k: v.to_dict() for k, v in footprintList.items()}
            json.dump(data, f, indent=4)

        elapsed_time = time.time() - start_time
        logging.info(f"Completed {filename} in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {file}: {str(e)}", exc_info=True)
        return False

def main():
    # 创建输出目录
    Path('./footprint').mkdir(exist_ok=True)
    
    # 获取所有需要处理的文件
    files = [f for f in os.listdir(f'{path_prefix}/rawdata') if f.endswith('.csv')]
    total_files = len(files)
    logging.info(f"Found {total_files} files to process")
    
    # 获取CPU核心数，留一个核心给系统
    max_workers = max(1, os.cpu_count() - 1)
    logging.info(f"Using {max_workers} processes")
    
    # 使用进程池并行处理
    start_time = time.time()
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并获取结果
        future_to_file = {executor.submit(process_file, file): file for file in files}
        
        # 处理完成的任务
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
            file = future_to_file[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Exception occurred while processing {file}: {str(e)}")
                failed += 1
                
            logging.info(f"Progress: {i}/{total_files} files completed")
    
    # 输出最终统计
    elapsed_time = time.time() - start_time
    logging.info(f"""
    Processing completed:
    Total time: {elapsed_time:.2f} seconds
    Total files: {total_files}
    Successful: {successful}
    Failed: {failed}
    Average time per file: {elapsed_time/total_files:.2f} seconds
    """)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    exit(main())
