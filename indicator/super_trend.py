import numpy as np
from typing import List
from common import FootprintBar, SignalBar, Signal

class SuperTrendRangeCalculator:
    def __init__(self, atr_period=5, multiplier=1, stdev_multiplier=1):
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.stdev_multiplier = stdev_multiplier

    def GenSignal(self, bars: List[FootprintBar]) -> List[Signal]:
        if len(bars) < self.atr_period:
            raise ValueError("bars 数量必须大于 ATR 周期")

        atr = self._calculate_atr(bars)
        stdev = self._calculate_stdev(bars)

        supertrend = []
        lower_band = []
        upper_band = []
        for i in range(len(bars)):
            bar = bars[i]
            atr_value = atr[i] if i < len(atr) else None
            stdev_value = stdev[i] if i < len(stdev) else None

            if atr_value is None or stdev_value is None:
                supertrend.append(None)
                lower_band.append(None)
                upper_band.append(None)
                continue

            # 计算支撑线和阻力线
            upper_band.append((bar.high + bar.low) / 2 + self.multiplier * atr_value + self.stdev_multiplier * stdev_value)
            lower_band.append((bar.high + bar.low) / 2 - self.multiplier * atr_value - self.stdev_multiplier * stdev_value)
            
            if bar.close > upper_band[-1]:
                supertrend.append({'support': lower_band, 'resistance': None, 'direction': 1})
            elif bar.close < lower_band[-1]:
                supertrend.append({'support': None, 'resistance': upper_band, 'direction': -1})
            else:
                supertrend.append({'support': lower_band, 'resistance': upper_band, 'direction': 0})


        # Step 3: 修正空缺的部分
        supertrend = self._handle_missing_segments(supertrend)

        return supertrend

    def _calculate_atr(self, bars: List[FootprintBar]):
        """
        计算 ATR（平均真实波幅）值，基于传入的 FootprintBar 列表。
        """
        tr_values = []
        for i in range(1, len(bars)):
            previous_bar = bars[i-1]
            current_bar = bars[i]
            high_low = current_bar.high - current_bar.low
            high_close_prev = abs(current_bar.high - previous_bar.close)
            low_close_prev = abs(current_bar.low - previous_bar.close)
            tr = max(high_low, high_close_prev, low_close_prev)
            tr_values.append(tr)

        atr = []
        for i in range(self.atr_period):
            atr.append(None)
        for i in range(self.atr_period, len(tr_values)):
            atr.append(np.mean(tr_values[i-self.atr_period:i]))

        return atr

    def _calculate_stdev(self, bars: List[FootprintBar]):
        closes = [bar.close for bar in bars]
        stdev = []
        for i in range(self.atr_period, len(closes)):
            stdev.append(np.std(closes[i-self.atr_period:i]))
        return stdev

    def _handle_missing_segments(self, supertrend: List[dict], lower_band: List[float], upper_band: List[float], missing_threshold: int = 2):
        valid = lambda x: x < len(supertrend) and supertrend[x] != None
        for i in range(len(supertrend)):
            for j in range(missing_threshold+2):
                if not valid(i+j):
                    continue
            if supertrend[i]['support'] == None and supertrend[i+missing_threshold+1]['support'] == None:
                for j in range(1, missing_threshold+1):
                    supertrend[i+j]['support'] = None
            
            if supertrend[i]['resistance'] == None and supertrend[i+missing_threshold+1]['resistance'] == None:
                for j in range(1, missing_threshold+1):
                    supertrend[i+j]['resistance'] = None

        return supertrend
