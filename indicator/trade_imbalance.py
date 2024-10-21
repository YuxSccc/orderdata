from common import *

class TradeImbalanceSignal(BarsSignal):
    SIGNAL_NAME = "TradeImbalance"
    def __init__(self):
        super().__init__(TradeImbalanceSignal.SIGNAL_NAME)

# Trade imbalance:
# 1. find the consecutive price levels that have a diagonal multiple greater than X
# 2. if the sum of the delta of these price levels is greater than Y, then we have a trade imbalance
# significance: 1
# additionalInfo: [start_idx, end_idx + 1, sum_delta, avg_ratio]
# colorTensor: [0] or [1] in the current bar (0 = buy imbalance, 1 = sell imbalance)

class TradeImbalanceSignalCalculator(BarCalculator):
    def __init__(self, bars: list[FootprintBar], diagonal_multiple_threshold: float, 
                 imbalance_price_level_threshold: int, sum_delta_threshold: float, ignore_volume: float):
        super().__init__()
        self.bars = bars
        self.diagonal_multiple_threshold = diagonal_multiple_threshold
        self.imbalance_price_level_threshold = imbalance_price_level_threshold
        self.sum_delta_threshold = sum_delta_threshold
        self.ignore_volume = ignore_volume

    def _find_consecutive_intervals(self, indices):
        if not indices or len(indices) == 0:
            return []
        intervals = []
        start = indices[0]
        end = indices[0]
        for idx in indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                intervals.append([start, end])
                start = idx
                end = idx
        intervals.append([start, end])
        return intervals

    def calc_signal(self) -> list[Signal]:
        signals = []
        for i in range(len(self.bars)):
            priceLevels = list(self.bars[i].priceLevels.values())
            assert sorted(priceLevels, key=lambda x: x.price) == priceLevels
            buy_imbalance_price_idx = []
            sell_imbalance_price_idx = []
            for idx in range(len(priceLevels) - 1):
                if (priceLevels[idx + 1].askSize + 1) / (priceLevels[idx].bidSize + 1) > self.diagonal_multiple_threshold or priceLevels[idx].bidSize < self.ignore_volume:
                    buy_imbalance_price_idx.append(idx)
                if (priceLevels[idx + 1].bidSize + 1) / (priceLevels[idx].askSize + 1) > self.diagonal_multiple_threshold or priceLevels[idx].askSize < self.ignore_volume:
                    sell_imbalance_price_idx.append(idx)

            buy_intervals = self._find_consecutive_intervals(buy_imbalance_price_idx)
            sell_intervals = self._find_consecutive_intervals(sell_imbalance_price_idx)
            for interval in buy_intervals:
                if interval[1] - interval[0] + 1 > self.imbalance_price_level_threshold:
                    sumDelta = 0
                    sumRatio = 0
                    colorTensor = []
                    for idx in range(interval[0], interval[1] + 1):
                        sumDelta += priceLevels[idx+1].askSize - priceLevels[idx].bidSize
                        sumRatio += (priceLevels[idx+1].askSize + 1) / (priceLevels[idx].bidSize + 1)
                    
                    for idx in range(len(priceLevels)):
                        if idx in range(interval[0], interval[1] + 1):
                            colorTensor.append(set([0]))
                        else:
                            colorTensor.append(set())

                    if sumDelta > self.sum_delta_threshold:
                        signal = TradeImbalanceSignal()
                        signal.add_bar(self.bars[i], [interval[0], interval[1] + 1, sumDelta, sumRatio / (interval[1] - interval[0] + 1)], colorTensor)
                        signal.set_significance(1)
                        signal.set_additional_info([priceLevels[interval[0]].price, priceLevels[interval[1] + 1].price, sumDelta, sumRatio / (interval[1] - interval[0] + 1)])
                        signals.append(signal)

            for interval in sell_intervals:
                if interval[1] - interval[0] + 1 > self.imbalance_price_level_threshold:
                    sumDelta = 0
                    sumRatio = 0
                    colorTensor = []
                    for idx in range(len(priceLevels)):
                        if idx in range(interval[0], interval[1] + 1):
                            colorTensor.append(set([1]))
                        else:
                            colorTensor.append(set())
                    for idx in range(interval[0], interval[1] + 1):
                        sumDelta += priceLevels[idx+1].bidSize - priceLevels[idx].askSize
                        sumRatio += (priceLevels[idx+1].bidSize + 1) / (priceLevels[idx].askSize + 1)
                    if sumDelta > self.sum_delta_threshold:
                        signal = TradeImbalanceSignal()
                        signal.add_bar(self.bars[i], [interval[0], interval[1] + 1, sumDelta, sumRatio / (interval[1] - interval[0] + 1)], colorTensor)
                        signal.set_significance(1)
                        signal.set_additional_info([priceLevels[interval[0]].price, priceLevels[interval[1] + 1].price, sumDelta, sumRatio / (interval[1] - interval[0] + 1)])
                        signals.append(signal)
        self.cal_finished = True
        self.signals = signals
        return signals
