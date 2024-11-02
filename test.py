import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import json
from visualization.flask_eng import start_server, set_data
from common import *
import pandas as pd
from indicator import *

def get_bars(filename: str) -> list[FootprintBar]:
    with open(filename, 'r') as f:
        data = json.load(f)
        return [FootprintBar().from_dict(bar) for bar in data.values()]


# 假设 high_array 和 low_array 分别是 5 分钟 bar 的最高价和最低价
filename = 'BTCUSDT-trades-2024-09'
bars = get_bars(f'./footprint/{filename}.json')
bars = bars[-5000:]
high_array = np.array([bar.high for bar in bars])
low_array = np.array([bar.low for bar in bars])

avg_price = np.array([(bar.high + bar.low) / 2 for bar in bars])

window=30
# 计算标准差
volatility = np.std(avg_price)  # 基于最近window个数据计算标准差
print(120 / (volatility / len(bars)))
# 293
exit(0)

# 设置动态显著性阈值
high_std_dev = np.std(high_array)
low_std_dev = np.std(low_array)

high_prominence = 2 * high_std_dev  # 高点显著性
low_prominence = 2 * low_std_dev    # 低点显著性
high_prominence = 100
low_prominence = 100

print(high_std_dev, low_std_dev)

# 检测高点
high_peaks, high_properties = find_peaks(
    high_array,
    prominence=high_prominence,
    distance=6  # 调整距离参数以适应市场特性
)

# 检测低点（反转 low_array 寻找谷值）
low_peaks, low_properties = find_peaks(
    -low_array,
    prominence=low_prominence,
    distance=6
)

print(high_peaks, high_properties, low_peaks, low_properties)
# 可视化结果
# plt.plot(high_array, label="High Price", color="blue")
# plt.plot(low_array, label="Low Price", color="orange")
# plt.plot(high_peaks, high_array[high_peaks], "x", label="Detected Highs", color="green")
# plt.plot(low_peaks, low_array[low_peaks], "o", label="Detected Lows", color="red")
# plt.legend()
# plt.savefig("/mnt/d/output.png")
# plt.show()


data = {
    'open': np.array([bar.open for bar in bars]),   # 开盘价数组
    'high': high_array,   # 最高价数组
    'low': low_array,    # 最低价数组
    'close': np.array([bar.close for bar in bars])   # 收盘价数组
}

high_points = np.full_like(high_array, np.nan)
low_points = np.full_like(low_array, np.nan)
high_points[high_peaks] = high_array[high_peaks]
low_points[low_peaks] = low_array[low_peaks]

import mplfinance as mpf 
dates = pd.date_range(start="2023-01-01", periods=len(high_array), freq='5min')  # 示例日期
ohlc = pd.DataFrame(data, index=dates)
apds = [
    mpf.make_addplot(high_points, type='scatter', markersize=20, color='green', marker='^', panel=0),   # 标注高点
    mpf.make_addplot(low_points, type='scatter', markersize=20, color='red', marker='v', panel=0)     # 标注低点
]

mpf.plot(ohlc, type='candle', figsize=(200, 10), style='charles', figscale=4, addplot=apds, title="K-line with Peaks and Troughs", ylabel="Price", savefig=dict(fname="/mnt/d/output.png", dpi=300))