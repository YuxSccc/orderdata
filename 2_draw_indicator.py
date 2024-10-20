import json
from visualization.flask_eng import start_server, set_data
from common import *
from indicator.SL_orders import SLOrdersSignalCalculator

def get_bars(filename: str) -> list[FootprintBar]:
    with open(filename, 'r') as f:
        data = json.load(f)
        return [FootprintBar().from_dict(bar) for bar in data.values()]
        

def encode_bars(bars: list[FootprintBar]) -> list[dict]:
    res = []
    for bar in bars:
        price_levels = []
        for price_level in bar.priceLevels.values():
            price_levels.append({
                "price": price_level.price,
                "bidSize": price_level.bidSize,
                "askSize": price_level.askSize,
                "volume": price_level.volume,
                "delta": price_level.delta
            })
        assert len(price_levels) == len(set(level['price'] for level in price_levels)), f"Duplicate prices found in price levels: {price_levels}"
        res.append({
            "timestamp": bar.timestamp,
            "duration": bar.duration,
            "open": bar.open,
            "close": bar.close,
            "high": bar.high,
            "low": bar.low,
            "priceLevels": price_levels
        })
    return res

def encode_signals(bars: list[FootprintBar], signals: list[Signal]):
    ts_to_idx = {}
    for idx in range(len(bars)):
        ts_to_idx[bars[idx].timestamp] = idx
    res = []
    for signal in signals:
        assert sorted(signal.get_bars(), key=lambda x: x.timestamp) == signal.get_bars()
        res.append({
            "name": signal.get_signal_name(),
            "startBarIndex": ts_to_idx[signal.get_bars()[0].timestamp],
            "endBarIndex": ts_to_idx[signal.get_bars()[-1].timestamp],
            "params": signal.get_additional_info(),
            "color": "green"
        })
    return res

def draw_indicator(bars: list[FootprintBar]) -> list[Signal]:
    calculator = SLOrdersSignalCalculator(bars, 30, 3, 50/60000)
    calculator.calc_signal()
    return calculator.signals

if __name__ == "__main__":
    bars = get_bars('./footprint/BTCUSDT-trades-2024-10-01.json')
    signals = draw_indicator(bars)
    set_data(encode_bars(bars), encode_signals(bars, signals), {"price_level_height": (10 ** -bars[0].pricePrecision) * bars[0].scale})
    start_server()
