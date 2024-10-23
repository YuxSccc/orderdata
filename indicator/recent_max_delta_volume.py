from common import *

class RecentMaxDeltaVolumeSignal(BarStatusSignal):
    SIGNAL_NAME = "RecentMaxDeltaVolume"
    def __init__(self):
        super().__init__(RecentMaxDeltaVolumeSignal.SIGNAL_NAME)
    
    def get_signal_dict(self) -> dict:
        return {"RecentMaxDelta": self.additionalInfo[0], "RecentMaxVolume": self.additionalInfo[1]}

    def get_signal_str(self) -> str:
        return self.signalName + " : RecentMaxDelta: " + self.additionalInfo[0] + " RecentMaxVolume: " + self.additionalInfo[1]

class RecentMaxDeltaVolumeCalculator(BarStatusCalculator):
    def __init__(self, bars: list[FootprintBar], window_size: int = 30):
        super().__init__()
        self.bars = bars
        self.window_size = window_size

    def calc_signal(self) -> list[Signal]:
        signals = []
        max_delta_sliding_window = []
        max_volume_sliding_window = []
        for i in range(len(self.bars)):
            current_bar = self.bars[i]
            max_volume = max(price_level.volume for price_level in current_bar.priceLevels.values())
            max_delta = max(price_level.delta for price_level in current_bar.priceLevels.values())
            
            max_delta_sliding_window.append(max_delta)
            max_volume_sliding_window.append(max_volume)
            if len(max_delta_sliding_window) > self.window_size:
                max_delta_sliding_window.pop(0)
                max_volume_sliding_window.pop(0)
            
            signal = RecentMaxDeltaVolumeSignal()
            signal.set_additional_info([max(max_delta_sliding_window), max(max_volume_sliding_window)])
            signals.append(signal)
        self.cal_finished = True
        self.signals = signals
        return signals