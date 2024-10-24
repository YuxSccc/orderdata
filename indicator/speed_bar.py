from common import *

class SpeedBarSignal(BarsSignal):
    SIGNAL_TYPE = "SpeedBar"
    def __init__(self):
        super().__init__(self.SIGNAL_TYPE)

class SpeedBarSignalCalculator(BarCalculator):
    def __init__(self, bars: list[FootprintBar], ticks: list[AggTick], check_interval_in_sec: int, trade_count_threshold: int):
        super().__init__()
        self.bars = bars
        self.ticks = ticks
        self.check_interval_in_sec = check_interval_in_sec
        self.trade_count_threshold = trade_count_threshold
        assert bars[0].duration % check_interval_in_sec == 0, f"bars duration {bars[0].duration} must be a multiple of check_interval_in_sec {check_interval_in_sec}"

    def calc_signal(self) -> list[Signal]:
        signal_list = []

        aggregated_ticks = {}
        for tick in self.ticks:
            interval_start = (tick.timestamp // (self.check_interval_in_sec * 1000)) * (self.check_interval_in_sec * 1000)
            key = (interval_start, tick.isBuy)
            if key not in aggregated_ticks:
                aggregated_ticks[key] = {
                    "timestamp": interval_start,
                    "isBuy": tick.isBuy,
                    "price": tick.price,
                    "size": 0,
                    "count": 0
                }
            aggregated_ticks[key]["size"] += tick.size
            aggregated_ticks[key]["count"] += tick.count

        bars_speed_info = {}

        for key, agg_tick in aggregated_ticks.items():
            if agg_tick["count"] >= self.trade_count_threshold:
                new_key = key[0] // self.bars[0].duration * self.bars[0].duration
                if new_key not in bars_speed_info:
                    bars_speed_info[new_key] = {
                        "timestamp": new_key,
                        "buyCount": 0,
                        "sellCount": 0
                    }
                if agg_tick["isBuy"]:
                    bars_speed_info[new_key]["buyCount"] = max(bars_speed_info[new_key]["buyCount"], agg_tick["count"])
                else:
                    bars_speed_info[new_key]["sellCount"] = max(bars_speed_info[new_key]["sellCount"], agg_tick["count"])

        for bar in self.bars:
            if bar.timestamp in bars_speed_info:
                if bars_speed_info[bar.timestamp]["buyCount"] > 0 or bars_speed_info[bar.timestamp]["sellCount"] > 0:
                    signal = SpeedBarSignal()
                    signal.add_bar(bar, [bars_speed_info[bar.timestamp]["buyCount"], bars_speed_info[bar.timestamp]["sellCount"]], [])
                    signal.set_significance(1)
                    signal.set_additional_info([bars_speed_info[bar.timestamp]["buyCount"], bars_speed_info[bar.timestamp]["sellCount"]])
                    signal_list.append(signal)

        self.is_calculated = True
        self.signals = signal_list
        return signal_list