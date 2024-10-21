from common import *

class SLOrdersSignal(BarsSignal):
    SIGNAL_NAME = "StopLostOrders"
    def __init__(self):
        super().__init__(SLOrdersSignal.SIGNAL_NAME)

# SL orders:
# 1. short SL order:
#   - find the highest price in the past N bars
#   - if the current bar's high is greater than the highest price * (1 + X) and has some price_level.delta > abs(self.delta_threshold)
#     then we have a short SL order
# 2. long SL order:
#   - find the lowest price in the past N bars
#   - if the current bar's low is less than the lowest price * (1 - X) and has some price_level.delta < -abs(self.delta_threshold)
#     then we have a long SL order
# significance: [SL_orders_size] (long SL order) or [-SL_orders_size] (short SL order)
# additionalInfo: [SL_orders_size] (long SL order) or [-SL_orders_size] (short SL order)
# colorTensor: [1] (short SL order) or [0] (long SL order) in the current bar

class SLOrdersSignalCalculator(BarCalculator):

    def __init__(self, bars: list[FootprintBar], delta_threshold: float, bar_count_threshold: int, price_change_ratio_threshold: float, total_volume_threshold: float):
        super().__init__()
        self.bars = bars
        self.delta_threshold = delta_threshold
        self.price_change_ratio_threshold = price_change_ratio_threshold
        self.bar_count_threshold = bar_count_threshold
        self.total_volume_threshold = total_volume_threshold

    def calc_signal(self) -> list[Signal]:
        for idx in range(len(self.bars)):
            if idx < self.bar_count_threshold or idx >= len(self.bars) - self.bar_count_threshold:
                continue

            start_idx = idx - self.bar_count_threshold
            end_idx = idx + self.bar_count_threshold
            
            priceList = []
            for j in range(start_idx, end_idx + 1):
                priceList.append({"high": self.bars[j].high, "low": self.bars[j].low, "close": self.bars[j].close, "idx": j})
            confirm_signal = 0
            # short SL order
            priceList.sort(key=lambda x: x["high"], reverse=True)
            if priceList[0]["idx"] == idx and priceList[0]["high"] > priceList[0]["close"] * (1 + self.price_change_ratio_threshold) \
                and priceList[0]["high"] > priceList[1]["high"] * (1 + self.price_change_ratio_threshold / 2):
                signal = SLOrdersSignal()
                colorTensor = []
                SL_orders_price = []
                SL_orders_size = 0
                if priceList[0]["idx"] == idx and priceList[0]["high"] > priceList[1]["high"] * (1 + self.price_change_ratio_threshold):
                    confirm_signal += 1
                for price_level in self.bars[idx].priceLevels.values():
                    if price_level.delta > abs(self.delta_threshold) and price_level.price > priceList[1]["high"]:
                        SL_orders_price.append(price_level.price)
                        SL_orders_size += abs(price_level.delta)
                        colorTensor.append(set([1]))
                    else:
                        colorTensor.append(set())

                if len(SL_orders_price) > 0 and SL_orders_size > self.total_volume_threshold:
                    for j in range(start_idx, end_idx + 1):
                        if j == idx:
                            signal.add_bar(self.bars[j], [SL_orders_size], colorTensor)
                        else:
                            signal.add_bar(self.bars[j], [], [])
                    signal.set_significance(SL_orders_size)
                    signal.set_additional_info([SL_orders_size, confirm_signal])
                    self.signals.append(signal)
            
            # long SL order
            priceList.sort(key=lambda x: x["low"], reverse=False)
            if priceList[0]["idx"] == idx and priceList[0]["low"] < priceList[0]["close"] * (1 - self.price_change_ratio_threshold) \
                and priceList[0]["low"] < priceList[1]["low"] * (1 - self.price_change_ratio_threshold / 2):
                signal = SLOrdersSignal()
                colorTensor = []
                SL_orders_price = []
                SL_orders_size = 0
                if priceList[0]["idx"] == idx and priceList[0]["low"] < priceList[1]["low"] * (1 - self.price_change_ratio_threshold):
                    confirm_signal += 1
                for price_level in self.bars[idx].priceLevels.values():
                    if price_level.delta < -abs(self.delta_threshold) and price_level.price < priceList[1]["low"]:
                        SL_orders_price.append(price_level.price)
                        SL_orders_size += abs(price_level.delta)
                        colorTensor.append(set([0]))
                    else:
                        colorTensor.append(set())

                if len(SL_orders_price) > 0 and SL_orders_size > self.total_volume_threshold:
                    for j in range(start_idx, end_idx + 1):
                        if j == idx:
                            signal.add_bar(self.bars[j], [SL_orders_size], colorTensor)
                        else:
                            signal.add_bar(self.bars[j], [], [])
                    signal.set_significance(SL_orders_size)
                    signal.set_additional_info([-SL_orders_size, confirm_signal])
                    self.signals.append(signal)
        return self.signals
