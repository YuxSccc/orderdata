from flask import Flask, jsonify
from flask_cors import CORS
from typing import List, Dict
import time
import random

app = Flask(__name__)
CORS(app)

# 示例数据生成函数，生成一组 k-bar 数据对象
def generate_sample_data():
    bars = []
    num_bars = 30  # 生成 30 个 k-bar
    prev_close = 100  # 设置初始价格
    duration = 60
    price_level_height = 1

    for bar_index in range(num_bars):
        num_price_levels = int(random.random() * 21) + 10  # 限制 num_price_levels 最大不超过 30
        low = (prev_close - price_level_height * (int(random.random() * 6 + 5))) // price_level_height * price_level_height
        high = low + (num_price_levels - 1) * price_level_height
        open_price = prev_close
        close_price = (int(random.random() * ((high - low) // price_level_height + 1))) * price_level_height + low
        price_levels = [
            {
                "priceLevel": low + i * price_level_height,
                "bidSize": int(random.random() * 40) + 10,
                "askSize": int(random.random() * 40) + 10,
                "volume": int(random.random() * 150) + 50,
                "delta": int(random.random() * 60) - 30
            } for i in range(num_price_levels)
        ]
        timestamp = int(time.time()) - (num_bars - bar_index) * duration
        bars.append({
            "timestamp": timestamp,
            "duration": duration,
            "openPrice": open_price,
            "closePrice": close_price,
            "high": high,
            "low": low,
            "priceLevels": price_levels
        })
        prev_close = close_price

    return bars

def set_data(data: list[dict], signals: list[dict], config: dict):
    global flask_data, flask_signals, flask_config
    flask_data = data
    flask_signals = signals
    flask_config = config

@app.route('/data', methods=['GET'])
def get_data():
    global flask_data, flask_signals, flask_config
    config = {
        "signal_font_size": 14 if "signal_font_size" not in flask_config else flask_config["signal_font_size"],
        "price_level_font_size": 10 if "price_level_font_size" not in flask_config else flask_config["price_level_font_size"],
        "price_level_height": 1 if "price_level_height" not in flask_config else flask_config["price_level_height"],
        "show_price_level_text": False if "show_price_level_text" not in flask_config else flask_config["show_price_level_text"]
    }
    return jsonify({"bars": flask_data, "signals": flask_signals, "config": config})


def start_server():
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    start_server()
