from common import *

class BigTradeSignal(TickSignal):
    SignalType = "BigTrade"

    def __init__(self):
        super().__init__(self.SignalType)

    def get_additional_info_str(self):
        return f"{self.signalName}: {self.tick.price}, {self.tick.size}"

class BigTradeSignalCalculator(TickCalculator):
    def __init__(self, ticks: list[Tick], size_threshold: int = 100, aggregate_interval_in_ms: int = 100):
        super().__init__()
        self.ticks = ticks
        self.size_threshold = size_threshold
        self.aggregate_interval_in_ms = aggregate_interval_in_ms

    def calc_signal(self) -> list[Signal]:
        signals = []

        self.ticks.sort(key=lambda x: x.timestamp)
        aggregated_buy_trades = {}
        aggregated_sell_trades = {}
        
        for tick in self.ticks:
            interval_start = tick.timestamp - (tick.timestamp % self.aggregate_interval_in_ms)
            trades_dict = aggregated_buy_trades if tick.isBuy else aggregated_sell_trades
            if tick.isBuy:
                if interval_start not in aggregated_buy_trades:
                    aggregated_buy_trades[interval_start] = {'max_price': float('-inf'), 'min_price': float('inf'), 'size': 0, 'trade_count': 0}
            else:
                if interval_start not in aggregated_sell_trades:
                    aggregated_sell_trades[interval_start] = {'max_price': float('-inf'), 'min_price': float('inf'), 'size': 0, 'trade_count': 0}
            trades_dict[interval_start]['max_price'] = max(trades_dict[interval_start]['max_price'], tick.price)
            trades_dict[interval_start]['min_price'] = min(trades_dict[interval_start]['min_price'], tick.price)
            trades_dict[interval_start]['size'] += tick.size
            trades_dict[interval_start]['trade_count'] += 1

        for interval_start, trades in aggregated_buy_trades.items():
            if trades['size'] >= self.size_threshold:
                signal = BigTradeSignal()
                signal.set_tick(Tick(interval_start, trades['min_price'], trades['size'], True))
                signal.set_additional_info([trades['trade_count'], trades['size'], trades['max_price'] - trades['min_price']])
                signal.set_significance(1)
                signals.append(signal)

        for interval_start, trades in aggregated_sell_trades.items():
            if trades['size'] >= self.size_threshold:
                signal = BigTradeSignal()
                signal.set_tick(Tick(interval_start, trades['max_price'], trades['size'], False))
                signal.set_additional_info([trades['trade_count'], trades['size'], trades['max_price'] - trades['min_price']])
                signal.set_significance(1)
                signals.append(signal)

        self.cal_finished = True
        self.signals = signals
        return signals
