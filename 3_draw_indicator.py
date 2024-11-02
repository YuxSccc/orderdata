import json
from visualization.flask_eng import start_server, set_data
from common import *
import pandas as pd
from indicator import *

filename = 'BTCUSDT-trades-2024-10-01'

def get_bars(filename: str) -> list[FootprintBar]:
    with open(filename, 'r') as f:
        data = json.load(f)
        return [FootprintBar().from_dict(bar) for bar in data.values()]

def get_ticks(filename: str) -> list[Tick]:
    with open(filename, 'r') as f:
        data = pd.read_csv(f)
        res = []
        if 'count' in data.columns:
            isAggTick = True
        else:
            isAggTick = False

        for row in data.itertuples():
            newTick = AggTick() if isAggTick else Tick()
            newTick.timestamp = int(row.time)
            newTick.price = float(row.price)
            newTick.size = float(row.qty)
            newTick.isBuy = row.is_buyer_maker == False
            if isAggTick:
                newTick.count = int(row.count)
            res.append(newTick)
        return res

def encode_bars(bars: list[FootprintBar], status_list: list[dict]) -> list[dict]:
    res = []
    for i in range(len(bars)):
        price_levels = []
        for price_level in bars[i].priceLevels.values():
            price_levels.append({
                "price": bars[0].normalize_price(price_level.price),
                "bidSize": price_level.bidSize,
                "askSize": price_level.askSize,
                "volume": price_level.volume,
                "delta": price_level.delta
            })
        assert len(price_levels) == len(set(level['price'] for level in price_levels)), f"Duplicate prices found in price levels: {price_levels}"
        status_str = ""
        for status_name, status_value in status_list[i].items():
            status_str += f"{status_name}: {status_value}, "
        res.append({
            "timestamp": bars[i].timestamp,
            "duration": bars[i].duration,
            "open": bars[0].normalize_price(bars[i].open),
            "close": bars[0].normalize_price(bars[i].close),
            "high": bars[0].normalize_price(bars[i].high),
            "low": bars[0].normalize_price(bars[i].low),
            "priceLevels": price_levels,
            "status": status_list[i]
        })
    return res

def encode_signals(bars: list[FootprintBar], signals: list[Signal]):
    ts_to_idx = {}
    for idx in range(len(bars)):
        ts_to_idx[bars[idx].timestamp] = idx
    res = []
    bar_duration = bars[0].duration
    for signal in signals:
        if isinstance(signal, MultiBarSignal):
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

def barstatus_encode(bars: list[FootprintBar]) -> list[dict]:
    return [{
        "timestamp": bar.timestamp,
        "status": bar.status
    } for bar in bars]

def generate_signals(bars: list[FootprintBar]) -> list[Signal]:
    SL_orders_calculator = StopLostOrdersSignalCalculator(bars, 30, 3, 50/60000, 70)
    SL_orders_calculator.calc_signal()
    ticks = get_ticks(f'./agg_trade/{filename}.csv')
    big_trade_calculator = BigTradeCalculator(ticks, 200, 200)
    big_trade_calculator.calc_signal()
    trade_imbalance_calculator = TradeImbalanceSignalCalculator(bars, 5, 4, 50, 0.5)
    trade_imbalance_calculator.calc_signal()

    return SL_orders_calculator.signals + big_trade_calculator.signals + trade_imbalance_calculator.signals

def generate_bar_status(bars: list[FootprintBar]):
    recent_max_delta_volume_calculator = RecentMaxDeltaVolumeCalculator(bars, 30)
    recent_max_delta_volume_calculator.calc_signal()
    intraday_session_calculator = IntradaySessionCalculator(bars)
    intraday_session_calculator.calc_signal()
    weekday_holiday_marker_calculator = WeekdayHolidayMarkerCalculator(bars)
    weekday_holiday_marker_calculator.calc_signal()

    calculators = [recent_max_delta_volume_calculator, intraday_session_calculator, weekday_holiday_marker_calculator]

    status_list = []
    for i in range(len(bars)):
        status_dict = {}
        for calculator in calculators:
            assert isinstance(calculator, SingleBarSignalCalculator)
            status_dict.update(calculator.signals[i].get_signal_dict())
        status_list.append(status_dict)
    return status_list

if __name__ == "__main__":
    bars = get_bars(f'./footprint/{filename}.json')
    signals = generate_signals(bars)
    status_list = generate_bar_status(bars)
    set_data(encode_bars(bars, status_list), encode_signals(bars, signals), 
             {"show_price_level_text": False, "price_level_height": (10 ** -bars[0].pricePrecision) * bars[0].scale, "display_bar_status": True})
    start_server()
