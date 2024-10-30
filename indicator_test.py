import json
from visualization.flask_eng import start_server, set_data
from common import *
import pandas as pd
from indicator import *

def get_bars(filename: str) -> list[FootprintBar]:
    with open(filename, 'r') as f:
        data = json.load(f)
        return [FootprintBar().from_dict(bar) for bar in data.values()]

filename = 'BTCUSDT-trades-2024-10-01'

if __name__ == "__main__":
    bars = get_bars(f'./footprint/{filename}.json')
    trend_calculator = TrendSignalCalculator(bars, 0.3, 4)
    trend_calculator.calc_signal()
    for signal in trend_calculator.signals:
        print(signal.get_additional_info_str())