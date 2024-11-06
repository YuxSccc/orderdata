import pandas as pd
import glob

def count_ones_zeros_in_targets(file_pattern):
    """
    统计目标文件中每列的 1 和 0 的数量和比例。

    Args:
        file_pattern: 文件模式，例如 "*.parquet"
    """
    files = glob.glob(file_pattern)
    total_ones = None
    total_zeros = None

    for file in files:
        df = pd.read_parquet(file)
        # 确保目标列为整数类型
        df = df.astype(int)
        ones = (df == 1).sum()
        zeros = (df == 0).sum()

        if total_ones is None:
            total_ones = ones
            total_zeros = zeros
        else:
            total_ones += ones
            total_zeros += zeros

    total_counts = total_ones + total_zeros
    for column in df.columns:
        ones_count = total_ones[column]
        zeros_count = total_zeros[column]
        total = total_counts[column]
        ones_ratio = ones_count / total if total > 0 else 0
        zeros_ratio = zeros_count / total if total > 0 else 0
        print(f'Column: {column}')
        print(f'  Total samples: {total}')
        print(f'  Ones (1): {ones_count} ({ones_ratio:.4%})')
        print(f'  Zeros (0): {zeros_count} ({zeros_ratio:.4%})')
        print('')

if __name__ == "__main__":
    # 假设目标文件位于当前目录，文件名以 "_target.parquet" 结尾
    file_pattern = "./model_feature/*_target.parquet"
    count_ones_zeros_in_targets(file_pattern)
