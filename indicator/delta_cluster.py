from common import *
from indicator.recent_max_delta_volume import RecentMaxDeltaVolumeCalculator

class DeltaClusterSignal(BarsSignal):
    SIGNAL_TYPE = "DeltaCluster"
    def __init__(self):
        super().__init__(DeltaClusterSignal.SIGNAL_TYPE)

class DeltaClusterSignalCalculator(BarCalculator):
    def __init__(self, bars: list[FootprintBar], check_bar_interval: int, max_price_level_range: int, min_delta_percentage: float, min_price_level_count: int):
        super().__init__()
        self.bars = bars
        self.check_bar_interval = check_bar_interval
        self.max_price_level_range = max_price_level_range
        self.min_delta_percentage = min_delta_percentage
        self.min_price_level_count = min_price_level_count

    def calc_signal(self) -> list[Signal]:
        signals = []
        recent_max_delta_volume_calculator = RecentMaxDeltaVolumeCalculator(self.bars, self.check_bar_interval)
        recent_max_delta_volume_calculator.calc_signal()
        assert "RecentMaxDelta" in recent_max_delta_volume_calculator.signals[0].get_signal_dict()
        valid_price_level_list = []
        price_level_dict = {}
        price_level_height = self.bars[0].get_price_level_height()
        last_rhs_idx_bar = -self.check_bar_interval
        for idx, bar in enumerate(self.bars):
            # wrong RecentMaxDelta, skip
            if idx < self.check_bar_interval:
                continue
            recent_max_delta = recent_max_delta_volume_calculator.signals[idx].get_signal_dict()["RecentMaxDelta"]
            for price_level in bar.priceLevels:
                if price_level.delta / recent_max_delta >= self.min_delta_percentage:
                    valid_price_level_list.append({
                        "price_level_idx": price_level.price / price_level_height,
                        "bar_idx": idx,
                        "delta_percentage": price_level.delta / recent_max_delta,
                        "recent_max_delta": recent_max_delta
                    })
                    if price_level.price / price_level_height not in price_level_dict:
                        price_level_dict[price_level.price / price_level_height] = {
                            "price_level_idx": price_level.price / price_level_height,
                            "count": 0,
                            "delta_percentage_sum": 0,
                            "delta_sum": 0
                        }
                    price_level_dict[price_level.price / price_level_height]["count"] += 1
                    price_level_dict[price_level.price / price_level_height]["delta_percentage_sum"] += price_level.delta / recent_max_delta
                    price_level_dict[price_level.price / price_level_height]["delta_sum"] += price_level.delta

            while len(valid_price_level_list) > 0 and (valid_price_level_list[0]["bar_idx"] + self.check_bar_interval <= idx or last_rhs_idx_bar + self.check_bar_interval / 3 > valid_price_level_list[0]["bar_idx"]):
                price_level_dict[valid_price_level_list[0]["price_level_idx"]]["count"] -= 1
                price_level_dict[valid_price_level_list[0]["price_level_idx"]]["delta_percentage_sum"] -= valid_price_level_list[0]["delta_percentage"]
                price_level_dict[valid_price_level_list[0]["price_level_idx"]]["delta_sum"] -= valid_price_level_list[0]["delta_percentage"] * valid_price_level_list[0]["recent_max_delta"]
                if price_level_dict[valid_price_level_list[0]["price_level_idx"]]["count"] == 0:
                    price_level_dict.pop(valid_price_level_list[0]["price_level_idx"])
                valid_price_level_list.pop(0)

            if len(valid_price_level_list) >= self.min_price_level_count:
                max_sum = 0
                price_level_dict_sorted = sorted(price_level_dict.items(), key=lambda x: x["price_level_idx"])
                for i in range(len(price_level_dict_sorted)):
                    current_sum = 0
                    current_count = 0
                    current_delta_percentage_sum = 0
                    for j in range(i, len(price_level_dict_sorted)):
                        if price_level_dict_sorted[j]["price_level_idx"] - price_level_dict_sorted[i]["price_level_idx"] + 1 > self.max_price_level_range:
                            break
                        current_sum += price_level_dict_sorted[j]["delta_sum"]
                        current_count += price_level_dict_sorted[j]["count"]
                        current_delta_percentage_sum += price_level_dict_sorted[j]["delta_percentage_sum"]
                        if current_sum > max_sum:
                            max_sum = current_sum
                            best_range = (price_level_dict_sorted[i]["price_level_idx"], price_level_dict_sorted[j]["price_level_idx"])
                            best_delta_percentage_sum = current_delta_percentage_sum
                            
                if max_sum > 0:
                    signal = DeltaClusterSignal()
                    signal_price_level_list = [price_level for price_level in valid_price_level_list if best_range[0] <= price_level["price_level_idx"] <= best_range[1]]
                    bar_price_level_idx_dict = {}
                    for price_level in signal_price_level_list:
                        if price_level["bar_idx"] not in bar_price_level_idx_dict:
                            bar_price_level_idx_dict[price_level["bar_idx"]] = []
                        bar_price_level_idx_dict[price_level["bar_idx"]].append(price_level["price_level_idx"])
                    min_bar_idx = min([price_level["bar_idx"] for price_level in signal_price_level_list])
                    max_bar_idx = max([price_level["bar_idx"] for price_level in signal_price_level_list])
                    for i in range(min_bar_idx, max_bar_idx + 1):
                        color_tensor = [set() for _ in range(len(self.bars[i].priceLevels))]
                        for price_level_idx in bar_price_level_idx_dict[i]:
                            color_tensor[price_level_idx] = set([1])
                        signal.add_bar(self.bars[i], [best_range[0], best_range[1]], color_tensor)

                    signal.set_significance(max_sum)
                    signal.set_additional_info([best_range[0], best_range[1], max_sum, best_delta_percentage_sum / len(signal_price_level_list)])
                    signals.append(signal)
                    last_rhs_idx_bar = max_bar_idx
        self.signals = signals
        self.is_calculated = True
        return self.signals