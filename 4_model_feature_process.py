from indicator import *
from common import *
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch.nn as nn
import torch
from model_common import *

def get_bars(filename: str) -> list[FootprintBar]:
    with open(f'./footprint/{filename}.json', 'r') as f:
        data = json.load(f)
        return [FootprintBar().from_dict(bar) for bar in data.values()]

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

indicator_list = [
    RecentMaxDeltaVolumeCalculator,
    IntradaySessionCalculator,
    WeekdayHolidayMarkerCalculator
]

# config = {
#     'RecentMaxDeltaVolumeCalculator': {
#         'main': {
#             'window_size': 30
#         }
#     },
#     'IntradaySessionCalculator': {
#         'main': {
#             'window_size': 30
#         }
#     },
#     'WeekdayHolidayMarkerCalculator': {
#         'main': {}
#     }
# }

config = {}

embedding_config = {
    'BigTradeSignalCalculator': {
        'max_feature_length': 5,
        'embedding_dim': 16,
        'param_dims': 5,
        'num_categories': 2,
        'use_attention': True
    }
}

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

if __name__ == "__main__":
    for file in filename:
        bars = get_bars(file)
        ticks = get_ticks(file)
        global_feature_df = pd.DataFrame()

        bar_feature_df = generate_bar_feature(bars)
        price_level_features_df, price_to_price_level_idx = generate_price_level_feature(bars)

        ts_to_idx = {}
        for i in range(len(bars)):
            ts_to_idx[bars[i].timestamp] = i

        onehot_feature_df = pd.DataFrame()

        indicator_instance_list = []

        for indicator, config_list in config.items():
            for config_name, config_dict in config_list.items():
                if isinstance(indicator, BarStatusCalculator) or isinstance(indicator, BarCalculator):
                    indicator_instance = indicator(bars, **config_dict)
                elif isinstance(indicator, TickCalculator):
                    indicator_instance = indicator(ticks, **config[indicator.__name__])
                else:
                    raise ValueError(f"Indicator {indicator.__name__} is not supported")
                indicator_instance.calc_signal()
                indicator_instance_list.append({"name": config_name, "indicator": indicator_instance})

        for indicator in indicator_instance_list:
            indicator_instance = indicator["indicator"]
            feature_prefix = indicator["name"]
            feature_name = indicator_instance.get_feature_name()
            signals = indicator_instance.signals
            if isinstance(indicator_instance, BarStatusCalculator):
                data_list = [[] for _ in range(indicator_instance.get_feature_dimension())]
                for i in range(len(bars)):
                    signal = signals[i]
                    for j in range(indicator_instance.get_feature_dimension()):
                        data_list[j].append(signal.get_additional_info()[i])
                global_feature_df[feature_prefix + '_' + feature_name] = data_list

            elif isinstance(indicator_instance, TickCalculator):
                tick_signal_dict = {}
                for signal in signals:
                    bar_idx = ts_to_idx[signal.tick.timestamp]
                    if bar_idx not in tick_signal_dict:
                        tick_signal_dict[bar_idx] = []
                    tick_signal_dict[bar_idx].append(signal)
                for i in range(len(bars)):
                    if i not in tick_signal_dict:
                        tick_signal_dict[i] = []
                # onehot feature
                bar_onehot_feature_list = []
                for bar_idx, signals in tick_signal_dict.items():
                    bar_onehot_feature = [0] * len(bars[bar_idx].priceLevels)
                    for signal in signals:
                        price_level_idx = price_to_price_level_idx[bar_idx][signal.tick.price]
                        bar_onehot_feature[price_level_idx] = 1
                    bar_onehot_feature_list.append(bar_onehot_feature)
                onehot_feature_df[feature_prefix + '_' + feature_name] = bar_onehot_feature_list
                # signal feature
                bars_feature_list = []
                for bar_idx, signals in tick_signal_dict.items():
                    signal_feature_list = []
                    for signal in signals:
                        signal_feature_list.append(signal.get_additional_info())
                    bars_feature_list.append(signal_feature_list)
                global_feature_df[feature_prefix + '_' + feature_name] = bars_feature_list
            elif isinstance(indicator_instance, BarCalculator):
                bars_signal_dict = {}
                for signal in signals:
                    for i in range(len(signal.get_bars())):
                        bars_signal_dict[ts_to_idx[signal.get_bars()[i].timestamp]].append([i, signal])
                for i in range(len(bars)):
                    if i not in bars_signal_dict:
                        bars_signal_dict[i] = []    
                # onehot feature
                bar_onehot_feature_list = []
                for bar_idx, [signal_bar_idx, signals] in bars_signal_dict.items():
                    bar_onehot_feature = [0 * indicator_instance.get_max_color_size()] * len(bars[bar_idx].priceLevels)
                    for signal in signals:
                        color_tensor = signal.get_color_tensor()[signal_bar_idx]
                        for color_set in color_tensor:
                            assert len(color_set) == 1
                            for color in color_set:
                                bar_onehot_feature[color] = 1
                    bar_onehot_feature_list.append(bar_onehot_feature)
                onehot_feature_df[feature_prefix + '_' + feature_name] = bar_onehot_feature_list
                # signal feature
                bars_feature_list = []
                for bar_idx, [signal_bar_idx, signals] in bars_signal_dict.items():
                    signal_feature_list = []
                    for signal in signals:
                        signal_feature_list.append(signal.get_additional_info())
                    bars_feature_list.append(signal_feature_list)
                global_feature_df[feature_prefix + '_' + feature_name] = [signal_bar_idx] + bars_feature_list

            else:
                raise ValueError(f"Indicator {indicator_instance.__name__} is not supported")

        write_to_parquet(global_feature_df, output_bar_feature_parquet_path)
        write_to_parquet(onehot_feature_df, output_onehot_parquet_path)
        write_to_parquet(price_level_features_df, output_price_level_parquet_path)
        write_to_parquet(bar_feature_df, output_bar_feature_parquet_path)
