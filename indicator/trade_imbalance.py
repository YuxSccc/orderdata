from common import *

class TradeImbalanceSignal(BarsSignal):
    SIGNAL_NAME = "TradeImbalance"
    def __init__(self):
        super().__init__(TradeImbalanceSignal.SIGNAL_NAME)

class TradeImbalanceSignalCalculator(BarCalculator):
    def __init__(self, bars: list[FootprintBar], bar_count_threshold: int, total_volume_threshold: float):
        super().__init__()
        self.bars = bars
        self.bar_count_threshold = bar_count_threshold
        self.total_volume_threshold = total_volume_threshold

    