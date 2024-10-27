from common import *

class PbPatternSignal(BarStatusSignal):
    SIGNAL_TYPE = "PbPattern"
    def __init__(self):
        super().__init__(self.SIGNAL_TYPE)

class PbPatternSignalCalculator(BarStatusCalculator):
    def __init__(self, bars: list[FootprintBar]):
        super().__init__()
        self.bars = bars

    def calc_signal(self) -> list[Signal]:
        signals = []
        

        return signals
