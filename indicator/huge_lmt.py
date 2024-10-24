from common import *

class HugeLmtSignal(BarsSignal):
    SIGNAL_NAME = "HugeLmt"
    def __init__(self):
        super().__init__(HugeLmtSignal.SIGNAL_NAME)

class HugeLmtSignalCalculator(BarCalculator):
    def __init__(self, bars: list[FootprintBar], imbalance_threshold: float, volume_threshold: float, price_level_count_threshold: int):
        super().__init__()
        self.bars = bars
        self.imbalance_threshold = imbalance_threshold
        self.volume_threshold = volume_threshold
        self.price_level_count_threshold = price_level_count_threshold

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
        bid_signal_list = []
        ask_signal_list = []
        for bar in self.bars:
            bid_idx_list = []
            ask_idx_list = []
            for idx in range(len(bar.priceLevels)):
                if (bar.priceLevels[idx].bidSize + 1) / (bar.priceLevels[idx].askSize + 1) > self.imbalance_threshold and \
                    bar.priceLevels[idx].volume > self.volume_threshold:
                    bid_idx_list.append(idx)
                elif (bar.priceLevels[idx].askSize + 1) / (bar.priceLevels[idx].bidSize + 1) > self.imbalance_threshold and \
                    bar.priceLevels[idx].volume > self.volume_threshold:
                    ask_idx_list.append(idx)
            bid_intervals = self._find_consecutive_intervals(bid_idx_list)
            ask_intervals = self._find_consecutive_intervals(ask_idx_list)
            for interval in bid_intervals:
                if interval[1] - interval[0] + 1 > self.price_level_count_threshold:
                    signal = HugeLmtSignal()
                    colorTensor = []
                    avg_delta = 0
                    avg_ratio = 0
                    for i in range(len(bar.priceLevels)):
                        if i in interval:
                            colorTensor.append(set([1]))
                            avg_delta += bar.priceLevels[i].delta
                            avg_ratio += (bar.priceLevels[i].bidSize + 1) / (bar.priceLevels[i].askSize + 1)
                        else:
                            colorTensor.append(set())
                    signal.add_bar(bar, [1, interval[0], interval[1]], colorTensor)
                    signal.set_significance(1)
                    signal.set_additional_info([1, interval[0], interval[1], avg_delta / (interval[1] - interval[0] + 1), avg_ratio / (interval[1] - interval[0] + 1)])
                    bid_signal_list.append(signal)
            for interval in ask_intervals:
                if interval[1] - interval[0] + 1 > self.price_level_count_threshold:
                    signal = HugeLmtSignal()
                    colorTensor = []
                    avg_delta = 0
                    avg_ratio = 0
                    for i in range(len(bar.priceLevels)):
                        if i in interval:
                            colorTensor.append(set([2]))
                            avg_delta += bar.priceLevels[i].delta
                            avg_ratio += (bar.priceLevels[i].askSize + 1) / (bar.priceLevels[i].bidSize + 1)
                        else:
                            colorTensor.append(set())
                    signal.add_bar(bar, [2, interval[0], interval[1]], colorTensor)
                    signal.set_significance(1)
                    signal.set_additional_info([2, interval[0], interval[1], avg_delta / (interval[1] - interval[0] + 1), avg_ratio / (interval[1] - interval[0] + 1)])
                    ask_signal_list.append(signal)
        self.is_calculated = True
        self.signals = bid_signal_list + ask_signal_list
        return bid_signal_list + ask_signal_list