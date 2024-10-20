import numpy as np
from common import *

class TrendSignal(BarsSignal):
    def __init__(self, signal: str):
        super().__init__(signal)

    def get_additional_info_str(self):
        if len(self.additionalInfo) != 3:
            return ""
        max_error, slope, intercept = self.additionalInfo
        return f"{self.signalName}: max_error:{max_error:.2f}, slope:{slope:.2f}, intercept:{intercept:.2f}"

    def has_bar_additional_info(self):
        return False

    def has_color_tensor(self):
        return False

# Trend_PLR
# indicators: max_error, slope, intercept
# signal.additional_info: [max_error, slope, intercept]
# signal.bar_additional_info: []
# signal.color_tensor: []

class TrendSignalCalculator(BarCalculator):
    SignalType = "Trend_PLR"
    DefaultErrorThreshold = 300
    DefaultMinLength = 2

    def __init__(self, bars: list[FootprintBar], error_threshold: float = DefaultErrorThreshold, min_length: int = DefaultMinLength):
        super().__init__()
        self.bars = bars
        self.error_threshold = error_threshold
        self.min_length = min_length

    def calc_error(self, bars: list[FootprintBar], start_idx: int, end_idx: int):
        n = end_idx - start_idx + 1
        if n < 2:
            return 0, 0, 0
        
        x = np.arange(start_idx, end_idx + 1)
        y = np.array([bar.close for bar in bars[start_idx:end_idx + 1]])

        # 线性拟合
        slope, intercept = np.polyfit(x, y, 1)

        # 计算最大误差
        fitted_y = slope * x + intercept
        max_error = np.max(np.abs(fitted_y - y))

        return max_error, slope, intercept

    def top_down_plr(self, bars: list[FootprintBar], start_idx: int, end_idx: int, error_threshold: float, segments: list[int]):
        max_error, slope, intercept = self.calc_error(bars, start_idx, end_idx)
        
        if max_error > error_threshold and (end_idx - start_idx) > 1:
            max_dev = 0
            max_idx = start_idx
            x = np.arange(start_idx, end_idx + 1)
            
            for i in range(start_idx + 1, end_idx):
                y = bars[i].close
                y_fit = slope * i + intercept
                deviation = np.abs(y - y_fit)
                if deviation > max_dev:
                    max_dev = deviation
                    max_idx = i
            
            self.top_down_plr(bars, start_idx, max_idx, error_threshold, segments)
            segments.append(max_idx)
            self.top_down_plr(bars, max_idx + 1, end_idx, error_threshold, segments)
    
    def calc_signal(self) -> list[Signal]:
        segments = []

        segments.append(0)
        self.top_down_plr(self.bars, 0, len(self.bars) - 1, self.error_threshold, segments)
        segments.append(len(self.bars) - 1)
        segments.sort()
        signals = []
        for i in range(len(segments) - 1):
            start_idx = segments[i]
            end_idx = segments[i + 1]
            max_error, slope, intercept = self.calc_error(self.bars, start_idx, end_idx)
            if end_idx - start_idx >= self.min_length:
                signal = TrendSignal(self.SignalType)
                signal.add_additional_info([max_error, slope, intercept])
                signal.set_significance(1)
                for j in range(start_idx, end_idx + 1):
                    signal.add_bar(self.bars[j], [], [])
                signals.append(signal)
        self.cal_finished = True
        self.signals = signals
        return signals
    
    def get_bar_indicators(self, bar: FootprintBar) -> list[float]:
        if not self.cal_finished:
            self.calc_signal()
        for signal in self.signals:
            for s_bar in signal.get_bars():
                if s_bar == bar:
                    return signal.additionalInfo
        return []