import multiprocessing as mp
from pathlib import Path
import os
import logging
import time
from model_feature_process import get_file_list, main as process_file

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(message)s',
        handlers=[
            logging.FileHandler('parallel_process.log'),
            logging.StreamHandler()
        ]
    )

def process_wrapper(filename):
    """
    包装处理函数，添加错误处理和日志
    """
    try:
        logging.info(f"Starting processing {filename}")
        start_time = time.time()
        
        # 检查输出文件是否已存在
        output_files = [
            f'./model_feature/{filename}_number.parquet',
            f'./model_feature/{filename}_event.parquet',
            f'./model_feature/{filename}_target.parquet',
            f'./model_feature/{filename}_flow.parquet'
        ]
        
        if all(Path(f).exists() for f in output_files):
            logging.info(f"Skipping {filename} - output files already exist")
            return True
            
        process_file([filename])
        
        elapsed_time = time.time() - start_time
        logging.info(f"Completed processing {filename} in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return False

def main():
    # 设置日志
    setup_logging()
    
    # 确保输出目录存在
    Path('./model_feature').mkdir(exist_ok=True)
    
    # 获取文件列表
    files = get_file_list()
    total_files = len(files)
    logging.info(f"Found {total_files} files to process")
    
    # 获取CPU核心数，留一个核心给系统
    num_processes = 2
    logging.info(f"Using {num_processes} processes")
    
    # 创建进程池
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        # 使用imap处理文件，这样可以实时获取结果
        results = []
        for i, result in enumerate(pool.imap_unordered(process_wrapper, files), 1):
            results.append(result)
            logging.info(f"Progress: {i}/{total_files} files completed")
    
    # 统计结果
    successful = sum(1 for r in results if r)
    failed = total_files - successful
    elapsed_time = time.time() - start_time
    
    # 输出最终统计
    logging.info(f"""
    Processing completed:
    Total time: {elapsed_time:.2f} seconds
    Total files: {total_files}
    Successful: {successful}
    Failed: {failed}
    Average time per file: {elapsed_time/total_files:.2f} seconds
    """)
    
    # 如果有失败的文件，返回非零状态码
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main()) 