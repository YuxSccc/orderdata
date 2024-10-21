import json
from visualization.flask_eng import start_server, set_data
from common import *
import pandas as pd
from indicator.SL_orders import SLOrdersSignalCalculator
from indicator.big_trade import BigTradeSignalCalculator
from indicator.trade_imbalance import TradeImbalanceSignalCalculator

filename = 'BTCUSDT-trades-2024-09'

def get_bars(filename: str) -> list[FootprintBar]:
    with open(filename, 'r') as f:
        data = json.load(f)
        return [FootprintBar().from_dict(bar) for bar in data.values()]

def get_ticks(filename: str) -> list[Tick]:
    with open(filename, 'r') as f:
        data = pd.read_csv(f)
        res = []
        for row in data.itertuples():
            newTick = Tick()
            newTick.timestamp = int(row.time)
            newTick.price = float(row.price)
            newTick.size = float(row.qty)
            newTick.isBuy = row.is_buyer_maker == False
            res.append(newTick)
        return res

def encode_bars(bars: list[FootprintBar]) -> list[dict]:
    res = []
    for bar in bars:
        price_levels = []
        for price_level in bar.priceLevels.values():
            price_levels.append({
                "price": bars[0].normalize_price(price_level.price),
                "bidSize": price_level.bidSize,
                "askSize": price_level.askSize,
                "volume": price_level.volume,
                "delta": price_level.delta
            })
        assert len(price_levels) == len(set(level['price'] for level in price_levels)), f"Duplicate prices found in price levels: {price_levels}"
        res.append({
            "timestamp": bar.timestamp,
            "duration": bar.duration,
            "open": bars[0].normalize_price(bar.open),
            "close": bars[0].normalize_price(bar.close),
            "high": bars[0].normalize_price(bar.high),
            "low": bars[0].normalize_price(bar.low),
            "priceLevels": price_levels
        })
    return res

def encode_signals(bars: list[FootprintBar], signals: list[Signal]):
    ts_to_idx = {}
    for idx in range(len(bars)):
        ts_to_idx[bars[idx].timestamp] = idx
    res = []
    bar_duration = bars[0].duration
    for signal in signals:
        if isinstance(signal, BarsSignal):
            assert sorted(signal.get_bars(), key=lambda x: x.timestamp) == signal.get_bars()
            res.append({
                "name": signal.get_signal_name(),
                "startTs": signal.get_bars()[0].timestamp,
                "endTs": signal.get_bars()[-1].timestamp,
                "params": signal.get_additional_info(),
                "type": "bars",
                "color": "purple"
            })
        elif isinstance(signal, TickSignal):
            res.append({
                "name": signal.get_signal_name(),
                "timestamp": signal.tick.timestamp / 1000 // bar_duration * bar_duration,
                "price": bars[0].normalize_price(signal.tick.price),
                "params": signal.get_additional_info(),
                "type": "tick",
                "color": signal.get_color()
            })
        else:
            raise ValueError(f"Unknown signal type: {type(signal)}")
    return res

def draw_indicator(bars: list[FootprintBar]) -> list[Signal]:
    SL_orders_calculator = SLOrdersSignalCalculator(bars, 30, 3, 50/60000, 70)
    SL_orders_calculator.calc_signal()
    # ticks = get_ticks(f'./agg_trade/{filename}.csv')
    # big_trade_calculator = BigTradeSignalCalculator(ticks, 200, 200)
    # big_trade_calculator.calc_signal()
    return SL_orders_calculator.signals

if __name__ == "__main__":
    bars = get_bars(f'./footprint/{filename}.json')
    signals = draw_indicator(bars)
    set_data(encode_bars(bars), encode_signals(bars, signals), {"show_price_level_text": False, "price_level_height": (10 ** -bars[0].pricePrecision) * bars[0].scale})
    start_server()
