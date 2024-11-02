import indicator
from common import *
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch.nn as nn
import torch
import importlib
from model_common import *
from config import *

def get_bars(filename: str) -> list[FootprintBar]:
    with open(f'./footprint/{filename}.json', 'r') as f:
        data = json.load(f)
        res = [FootprintBar().from_dict(bar) for bar in data.values()]
        for bar in res:
            bar.high = bar.normalize_price(bar.high)
            bar.low = bar.normalize_price(bar.low)
            bar.open = bar.normalize_price(bar.open)
            bar.close = bar.normalize_price(bar.close)
        return res

def get_ticks(filename: str) -> list[Tick]:
    with open(f'./agg_trade/{filename}.csv', 'r') as f:
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

class FeatureModule(nn.Module):
    def __init__(self, param_dims, embedding_dim):
        super(FeatureModule, self).__init__()
        self.param_layers = nn.ModuleList()
        for dim in param_dims:
            self.param_layers.append(nn.Linear(dim, embedding_dim))
        # 如果需要，可以添加特征的嵌入层
        # self.feature_embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, params):
        # params: [num_params]
        param_embeds = []
        for i, layer in enumerate(self.param_layers):
            param = params[i].unsqueeze(0)  # [1]
            param_embed = layer(param)  # [embedding_dim]
            param_embeds.append(param_embed)
        # 合并参数嵌入
        feature_vector = sum(param_embeds)  # [embedding_dim]
        return feature_vector

filename = ['BTCUSDT-trades-2024-10-01']
output_bar_parquet_path = './model_feature/output_bar.parquet'
output_onehot_parquet_path = './model_feature/output_onehot.parquet'
output_price_level_parquet_path = './model_feature/output_price_level.parquet'
output_bar_feature_parquet_path = './model_feature/output_bar_feature.parquet'
output_target_parquet_path = './model_feature/output_target.parquet'
output_flow_parquet_path = './model_feature/output_flow.parquet'

first_chunk_dict = {}

def write_to_parquet(df: pd.DataFrame, filename: str):
    if df.empty:
        return
    table = pa.Table.from_pandas(df)
    if filename not in first_chunk_dict:
        first_chunk_dict[filename] = True
        pq.write_table(table, filename)
    else:
        pq.write_table(table, filename, append=True)

def generate_bar_feature(bars: list[FootprintBar]):
    bar_feature_df = pd.DataFrame()
    bar_column_name = ["timestamp", "open", "high", "low", "close", "volume", "delta", "trade_count"]
    bar_feature_list = [[] for _ in range(len(bar_column_name))]
    for i in range(len(bars)):
        bar_feature_list[0].append(bars[i].timestamp)
        bar_feature_list[1].append(bars[i].open)
        bar_feature_list[2].append(bars[i].high)
        bar_feature_list[3].append(bars[i].low)
        bar_feature_list[4].append(bars[i].close)
        bar_feature_list[5].append(bars[i].volume)
        bar_feature_list[6].append(bars[i].delta)
        bar_feature_list[7].append(bars[i].tradesCount)
    for i in range(len(bar_column_name)):
        bar_feature_df[bar_column_name[i]] = bar_feature_list[i]
    return bar_feature_df

def generate_price_level_feature(bars: list[FootprintBar]):
    price_level_column_name = ["timestamp", "price", "bid_size", "bid_count", "ask_size", "ask_count", "delta", "trades_count", "volume", "idx"]
    price_level_features = []
    price_to_price_level_idx = []
    for i in range(len(bars)):
        price_levels = []
        inner_price_to_price_level_idx_dict = {}
        raw_price_levels_list = [price_level for key, price_level in bars[i].priceLevels.items()]
        for j, price_level in enumerate(raw_price_levels_list):
            price_level_data = [
                bars[i].timestamp, # 0
                price_level.price, # 1
                price_level.bidSize, # 2
                price_level.bidCount, # 3
                price_level.askSize, # 4
                price_level.askCount, # 5
                price_level.delta, # 6
                price_level.tradesCount, # 7
                price_level.volume, # 8
                j, # 9
            ]
            price_levels.append(price_level_data)
            inner_price_to_price_level_idx_dict[price_level.price] = j
        price_to_price_level_idx.append(inner_price_to_price_level_idx_dict)
        # while len(price_levels) < max_price_level_size:
        #     padding_data = [0] * len(price_level_column_name)
        #     padding_data[0] = bars[i].timestamp
        #     price_levels.append(padding_data)
        # if len(price_levels) > max_price_level_size:
        #     price_levels.sort(key=lambda x: x[8], reverse=True)
        #     price_levels = price_levels[:max_price_level_size]
        #     price_levels.sort(key=lambda x: x[9])
        price_level_features.append(pd.DataFrame(price_levels, columns=price_level_column_name))
    price_level_features_df = pd.concat(price_level_features, ignore_index=True)
    return price_level_features_df, price_to_price_level_idx

def generate_feature_df(indicator: Calculator, bars: list[FootprintBar], ticks: list[Tick]=None, price_to_price_level_idx: list[dict]=None):
    duration = bars[0].duration
    price_interval = bars[0].get_price_level_height()
    ts_to_idx = {}
    for i in range(len(bars)):
        ts_to_idx[bars[i].timestamp] = i
    indicator_instance = indicator["indicator"]
    feature_prefix = indicator["name"]
    feature_name = indicator_instance.get_feature_name()
    signals_group = indicator_instance.signals
    if isinstance(indicator_instance, SingleBarSignalCalculator):
        data_list = []
        for i in range(len(bars)):
            signal = signals_group[i]
            data_list.append([signal.get_additional_info()])
        return pd.DataFrame({f"{feature_prefix}_{feature_name}": data_list}), None

    elif isinstance(indicator_instance, TickSignalCalculator):
        tick_signal_dict = {}
        for signal in signals_group:
            bar_idx = ts_to_idx[signal.tick.timestamp // 1000 // duration * duration]
            if bar_idx not in tick_signal_dict:
                tick_signal_dict[bar_idx] = []
            tick_signal_dict[bar_idx].append(signal)
        for i in range(len(bars)):
            if i not in tick_signal_dict:
                tick_signal_dict[i] = []
        tick_signal_dict = dict(sorted(tick_signal_dict.items()))
        # onehot feature
        bar_onehot_feature_list = []
        for bar_idx, signals in tick_signal_dict.items():
            bar_onehot_feature = [0] * len(bars[bar_idx].priceLevels)
            for signal in signals:
                price_level_idx = price_to_price_level_idx[bar_idx][signal.tick.price // price_interval * price_interval]
                bar_onehot_feature[price_level_idx] = 1
            bar_onehot_feature_list.append(bar_onehot_feature)
        # signal feature
        bars_feature_list = []
        for bar_idx, signals in tick_signal_dict.items():
            signal_feature_list = []
            for signal in signals:
                signal_feature_list.append(signal.get_additional_info())
            bars_feature_list.append(signal_feature_list)
        return pd.DataFrame({f"{feature_prefix}_{feature_name}": bars_feature_list}), \
            pd.DataFrame({f"{feature_prefix}_{feature_name}": bar_onehot_feature_list})
    elif isinstance(indicator_instance, MultiBarSignalCalculator):
        bars_signal_list = [[] for _ in range(len(bars))]
        for signal in signals_group:
            for i in range(len(signal.get_bars())):
                bars_signal_list[ts_to_idx[signal.get_bars()[i].timestamp]].append((i, signal))
        # onehot feature
        bar_onehot_feature_list = []
        for bar_idx, signals in enumerate(bars_signal_list):
            bar_onehot_feature = [[0] * indicator_instance.get_max_color_size()] * len(bars[bar_idx].priceLevels)
            for signal_bar_idx, signal in signals:
                color_tensor = signal.get_color_tensor()[signal_bar_idx]
                for color_set in color_tensor:
                    assert len(color_set) <= 1
                    for color in color_set:
                        bar_onehot_feature[color] = 1
            bar_onehot_feature_list.append(bar_onehot_feature)
        # signal feature
        bars_feature_list = []
        for bar_idx, signals in enumerate(bars_signal_list):
            signal_feature_list = []
            for signal_bar_idx, signal in signals:
                signal_feature_list.append(signal.get_additional_info())
            bars_feature_list.append(signal_feature_list)
        return pd.DataFrame({f"{feature_prefix}_{feature_name}": bars_feature_list}), \
            pd.DataFrame({f"{feature_prefix}_{feature_name}": bar_onehot_feature_list})

    else:
        raise ValueError(f"Indicator {indicator_instance.__name__} is not supported")

def generate_flow_feature(indicator: Calculator, bars: list[FootprintBar], ticks: list[Tick]=None, price_to_price_level_idx: list[dict]=None):
    ts_to_idx = {}
    for i in range(len(bars)):
        ts_to_idx[bars[i].timestamp] = i
    assert indicator["indicator"].get_feature_name() in flow_feature_list
    indicator_instance = indicator["indicator"]
    feature_prefix = indicator["name"]
    feature_name = indicator_instance.get_feature_name()
    bars_feature_list = []
    if isinstance(indicator_instance, MultiBarSignalCalculator):
        for start_bar_idx in range(len(bars) - seq_len + 1):
            end_bar_idx = start_bar_idx + seq_len - 1
            signals_group = indicator_instance.fast_calc_signal(start_bar_idx, end_bar_idx)
            bars_signal_list = [[] for _ in range(len(bars))]
            for signal in signals_group:
                for i in range(len(signal.get_bars())):
                    bars_signal_list[ts_to_idx[signal.get_bars()[i].timestamp]].append((i, signal))
            # signal feature
            for i in range(start_bar_idx, end_bar_idx + 1):
                signals = bars_signal_list[i]
                signal_feature_list = []
                for signal_bar_idx, signal in signals:
                    signal_feature_list.append(signal.get_additional_info())
                bars_feature_list.append(signal_feature_list)
                
    return pd.DataFrame({f"{feature_prefix}_{feature_name}_flow": bars_feature_list})

def generate_target_feature(bars: list[FootprintBar], step1_size=5, step2_size=10, step_5_target=0.003, step_10_target=0.005):
    df = pd.DataFrame(index=range(len(bars)), columns=["reach_5_steps_high", "reach_10_steps_high", "reach_5_steps_low", "reach_10_steps_low"])
    df.infer_objects(copy=False).fillna(0, inplace=True)
    n = len(bars)
    for i in range(n):
        current_price = bars[i].close
        for j in range(1, step1_size + 1):
            if i + j >= n:
                break
            future_price = bars[i + j].high
            price_change = (future_price - current_price) / current_price
            if price_change >= step_5_target:
                df.loc[i, 'reach_5_steps_high'] = 1
                break
            
        for j in range(1, step2_size + 1):
            if i + j >= n:
                break
            future_price = bars[i + j].high
            price_change = (future_price - current_price) / current_price
            if price_change >= step_10_target:
                df.loc[i, 'reach_10_steps_high'] = 1
                break

    for i in range(n):
        current_price = bars[i].close
        for j in range(1, step1_size + 1):
            if i + j >= n:
                break
            future_price = bars[i + j].low
            price_change = (future_price - current_price) / current_price
            if price_change <= -step_5_target:
                df.loc[i, 'reach_5_steps_low'] = 1
                break
            
        for j in range(1, step2_size + 1):
            if i + j >= n:
                break
            future_price = bars[i + j].low
            price_change = (future_price - current_price) / current_price
            if price_change <= -step_10_target:
                df.loc[i, 'reach_10_steps_low'] = 1
                break
    return df

# def print_plt(data_list: list[list[float]]):
#     data_list = [[],[],[],[]]
#     for i in range(global_flow_feature_df.shape[0]):
#         ta = global_flow_feature_df.iloc[i, 0]
#         if not (isinstance(ta, list) and len(ta) > 0 and len(ta[0]) > 2):
#             print(i, ta)
#         else:
#             val = ta[0][4]
#             if val < 0.008:
#                 data_list[0].append(val)
#             if val <= 0.025 and val >= 0.008:
#                 data_list[1].append(val)
#             if val > 0.025:
#                 data_list[2].append(val)
#             data_list[3].append(val)
#     print(len(data_list[0]), len(data_list[1]), len(data_list[2]))
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 5))
#     plt.hist(data_list[3], bins=150, density=True, alpha=0.6, color='b', label='Histogram')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#     plt.title('Data Distribution (Histogram)')
#     plt.legend()
#     plt.savefig("output.png")

if __name__ == "__main__":
    for file in filename:
        bars = get_bars(file)
        target_feature_df = generate_target_feature(bars)
        pd.set_option('display.max_rows', None)
        ticks = get_ticks(file)
        global_feature_df = pd.DataFrame()

        bar_feature_df = generate_bar_feature(bars)
        price_level_features_df, price_to_price_level_idx = generate_price_level_feature(bars)

        global_onehot_feature_df = pd.DataFrame()
        global_flow_feature_df = pd.DataFrame()
        normalizer = GlobalNormalizer()
        min_value, max_value = normalizer.get_normalize_min_max_value(bar_feature_df.loc[:, "close"])
        normalizer.set_max_price(max_value)
        normalizer.set_min_price(min_value)

        indicator_instance_list = []
        for indicator_name, config_list in config.items():
            for config_name, config_dict in config_list.items():
                indicator_cls = getattr(indicator, indicator_name)
                if issubclass(indicator_cls, indicator.SpeedBarCalculator):
                    indicator_instance = indicator_cls(bars, ticks, **config_dict)
                elif issubclass(indicator_cls, SingleBarSignalCalculator) or issubclass(indicator_cls, MultiBarSignalCalculator):
                    indicator_instance = indicator_cls(bars, **config_dict)
                elif issubclass(indicator_cls, TickSignalCalculator):
                    indicator_instance = indicator_cls(ticks, **config_dict)
                else:
                    raise ValueError(f"Indicator {indicator_cls} is not supported")
                indicator_instance.calc_signal()
                indicator_instance_list.append({"name": config_name, "indicator": indicator_instance})
        sum_feature_size = 0
        for indicator in indicator_instance_list:
            feature_name = indicator["indicator"].get_feature_name()
            sum_feature_size += indicator["indicator"].get_feature_dimension()
            if feature_name == "Trend":
                feature_df = generate_flow_feature(indicator, bars, ticks, price_to_price_level_idx)
                feature_df = indicator["indicator"].normalize_feature(feature_df.iloc[:, 0], indicator['name'] + "_" + feature_name, normalizer)
                global_flow_feature_df = pd.concat([global_flow_feature_df, feature_df], axis=1)
            else:
                feature_df, onehot_feature_df = generate_feature_df(indicator, bars, ticks, price_to_price_level_idx)
                feature_df = indicator["indicator"].normalize_feature(feature_df.iloc[:, 0], indicator['name'] + "_" + feature_name, normalizer)
                global_feature_df = pd.concat([global_feature_df, feature_df], axis=1)
                if onehot_feature_df is not None:
                    global_onehot_feature_df = pd.concat([global_onehot_feature_df, onehot_feature_df], axis=1)
        # print(global_feature_df)
        # print(global_onehot_feature_df)
        # print(sum_feature_size)
        # print(bar_feature_df.shape)
        # print(price_level_features_df.shape)
        write_to_parquet(global_feature_df, output_bar_feature_parquet_path)
        write_to_parquet(global_onehot_feature_df, output_onehot_parquet_path)
        write_to_parquet(global_flow_feature_df, output_flow_parquet_path)
        print(global_flow_feature_df)
        global_flow_feature_df.to_html("output.html")
        write_to_parquet(price_level_features_df, output_price_level_parquet_path)
        write_to_parquet(bar_feature_df, output_bar_parquet_path)
        write_to_parquet(target_feature_df, output_target_parquet_path)