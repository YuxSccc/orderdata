import os
import zipfile
import requests
import concurrent.futures
import threading
from datetime import datetime

def download_data(year, month, path_prefix):
    """下载单个月份的数据"""
    month_str = f"{month:02d}"
    year_str = str(year)
    
    # 检查文件是否已存在
    expected_csv = f"BTCUSDT-trades-{year_str}-{month_str}.csv"
    if os.path.exists(f"{path_prefix}/unsorted_rawdata/{expected_csv}"):
        print(f"Data for {year_str}-{month_str} already exists, skipping...")
        return True

    print(f"Downloading data for {year_str}-{month_str}...")
    try:
        url = f"https://data.binance.vision/data/futures/um/monthly/trades/BTCUSDT/BTCUSDT-trades-{year_str}-{month_str}.zip"
        response = requests.get(url)
        zip_path = f"{path_prefix}/unsorted_rawdata/temp_{year_str}-{month_str}.zip"
        
        # 保存zip文件
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(f'{path_prefix}/unsorted_rawdata')
        
        # 删除zip文件
        os.remove(zip_path)
        print(f"Successfully downloaded and extracted data for {year_str}-{month_str}")
        return True
    except Exception as e:
        print(f"Error processing data for {year_str}-{month_str}: {str(e)}")
        return False

def generate_date_range(start_year, start_month, end_year, end_month):
    """生成日期范围"""
    dates = []
    current_year = start_year
    current_month = start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        dates.append((current_year, current_month))
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    return dates

def parallel_download(start_year, start_month, end_year, end_month, max_workers=4):
    """并行下载和解压数据"""
    path_prefix = '/mnt/e/orderdata/binance'
    if not os.path.exists(f'{path_prefix}/unsorted_rawdata'):
        os.makedirs(f'{path_prefix}/unsorted_rawdata')

    # 生成所有需要下载的日期
    dates = generate_date_range(start_year, start_month, end_year, end_month)
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        future_to_date = {
            executor.submit(download_data, year, month, path_prefix): (year, month)
            for year, month in dates
        }
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(future_to_date):
            year, month = future_to_date[future]
            try:
                success = future.result()
                if not success:
                    print(f"Failed to process data for {year}-{month:02d}")
            except Exception as e:
                print(f"Exception occurred while processing {year}-{month:02d}: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    parallel_download(2022, 1, 2024, 10, max_workers=4)
    end_time = datetime.now()
    print(f"Total execution time: {end_time - start_time}")