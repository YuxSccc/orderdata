import json
from pathlib import Path
import logging
from datetime import datetime
import concurrent.futures
from collections import defaultdict

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('timestamp_check.log'),
            logging.StreamHandler()
        ]
    )

def check_file_timestamps(file_path):
    """检查单个文件中的时间戳"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 获取所有时间戳并转换为整数
        timestamps = sorted([int(ts) for ts in data.keys()])
        
        if not timestamps:
            return file_path.name, [], "Empty file"
        
        # 检查时间戳间隔
        gaps = []
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i-1]
            if diff != 300:
                gaps.append((
                    timestamps[i-1],
                    timestamps[i],
                    diff,
                    datetime.fromtimestamp(timestamps[i-1]).strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M:%S')
                ))
        
        return file_path.name, timestamps, gaps
        
    except Exception as e:
        return file_path.name, [], f"Error processing file: {str(e)}"

def main():
    setup_logging()
    footprint_dir = Path('./footprint')
    
    if not footprint_dir.exists():
        logging.error(f"Directory {footprint_dir} does not exist!")
        return
    
    # 获取所有json文件
    json_files = sorted(footprint_dir.glob('*.json'))
    total_files = len(json_files)
    logging.info(f"Found {total_files} JSON files")
    
    # 使用线程池并行处理文件
    results = defaultdict(list)
    gaps_found = False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(check_file_timestamps, f): f for f in json_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
            filename, timestamps, gaps = future.result()
            
            if isinstance(gaps, str):
                # 处理错误情况
                logging.error(f"{filename}: {gaps}")
                continue
                
            if gaps:
                gaps_found = True
                logging.warning(f"\nFound gaps in {filename}:")
                for gap in gaps:
                    logging.warning(
                        f"Gap between {gap[3]} ({gap[0]}) and {gap[4]} ({gap[1]}) - "
                        f"Missing {gap[2]-1} timestamps"
                    )
            
            if timestamps:
                results['min_timestamp'].append(timestamps[0])
                results['max_timestamp'].append(timestamps[-1])
            
            if i % 10 == 0:
                logging.info(f"Processed {i}/{total_files} files")
    
    # 输出总体统计
    if results['min_timestamp'] and results['max_timestamp']:
        overall_min = min(results['min_timestamp'])
        overall_max = max(results['max_timestamp'])
        
        logging.info("\nOverall Statistics:")
        logging.info(f"Earliest timestamp: {datetime.fromtimestamp(overall_min).strftime('%Y-%m-%d %H:%M:%S')} ({overall_min})")
        logging.info(f"Latest timestamp: {datetime.fromtimestamp(overall_max).strftime('%Y-%m-%d %H:%M:%S')} ({overall_max})")
        logging.info(f"Total time span: {(overall_max - overall_min) / 3600:.2f} hours")
        
        if not gaps_found:
            logging.info("No timestamp gaps found in any file!")
    
    return 0 if not gaps_found else 1

if __name__ == "__main__":
    exit(main()) 