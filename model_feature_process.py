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

path_prefix = '/mnt/e/orderdata/binance'

def get_bars(filename: str) -> list[FootprintBar]:
    with open(f'./footprint/{filename}.json', 'r') as f:
        data = json.load(f)
        res = [FootprintBar().from_dict(bar) for bar in data.values()]
        for bar in res:
            bar.high = bar.normalize_price(bar.high)
            bar.low = bar.normalize_price(bar.low)
            bar.open = bar.normalize_price(bar.open)
            bar.close = bar.normalize_price(bar.close)
        res.sort(key=lambda x: x.timestamp)
        return res

def get_ticks(filename: str) -> list[Tick]:
    with open(f'{path_prefix}/agg_trade/{filename}.csv', 'r') as f:
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

first_chunk_dict = {}

def write_to_parquet(df: pd.DataFrame, filename: str):
    if df.empty:
        return
    df.columns = df.columns.map(str)
    table = pa.Table.from_pandas(df)
    if filename not in first_chunk_dict:
        first_chunk_dict[filename] = True
        pq.write_table(table, filename)
    else:
        pq.write_table(table, filename, append=True)

def generate_bar_feature(bars: list[FootprintBar]):
    bar_feature_df = pd.DataFrame()
    bar_column_name = ["timestamp", "open", "high", "low", "close", "volume", "delta", "trades_count"]
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
        if indicator_instance.get_max_color_size() != 0:
            bar_onehot_feature_list = []
            for bar_idx, signals in enumerate(bars_signal_list):
                bar_onehot_feature = [0 * indicator_instance.get_max_color_size()] * len(bars[bar_idx].priceLevels)
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
            pd.DataFrame({f"{feature_prefix}_{feature_name}": bar_onehot_feature_list}) if indicator_instance.get_max_color_size() != 0 else None

    else:
        raise ValueError(f"Indicator {indicator_instance.__name__} is not supported")

# flow[seqlen * i: seqlen * (i + 1)] for bar[i]
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

# 获取所有列名前缀
def get_column_prefixes(df):
    prefixes = {}
    for col in df.columns:
        name, feature_name, _ = col.split('_', 2)
        prefix = name + "_" + feature_name
        if prefix in prefixes:
            prefixes[prefix].append(col)
        else:
            prefixes[prefix] = [col]
    return prefixes

# 聚合函数
def aggregate_by_prefix(df):
    prefixes = get_column_prefixes(df)
    for prefix, cols in prefixes.items():
        def merge_row(row):
            if row[cols[0]] is None:
                assert all(row[col] is None for col in cols), \
                    f"All columns in {prefix} must be None if the first column is None"
                return None
            max_length = max(len(row[col]) for col in cols)
            combined_row = []
            for i in range(max_length):
                combined_row.append([row[col][i] if i < len(row[col]) else None for col in cols])
            return combined_row
        
        # 创建聚合列并删除原始列
        df[prefix] = df.apply(merge_row, axis=1)
        df.drop(columns=cols, inplace=True)
    return df

def generate_target_feature(bars: list[FootprintBar], step1_size=5, step2_size=10, step_5_target=0.003, step_10_target=0.005):
    df = pd.DataFrame(index=range(len(bars)), columns=["reach_5_steps_high", "reach_10_steps_high", "reach_5_steps_low", "reach_10_steps_low"])
    n = len(bars)
    for i in range(n):
        current_price = bars[i].close
        df.loc[i, 'reach_5_steps_high'] = 0
        df.loc[i, 'reach_10_steps_high'] = 0
        df.loc[i, 'reach_5_steps_low'] = 0
        df.loc[i, 'reach_10_steps_low'] = 0
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

def do_normalize_for_bar_features(bar_feature_df: pd.DataFrame, normalizer: GlobalNormalizer):
    bar_feature_df = bar_feature_df.astype({"trades_count": float})
    bar_feature_df.loc[:, "open"] = normalizer.min_max_normalize_for_price(bar_feature_df.loc[:, "open"])
    bar_feature_df.loc[:, "high"] = normalizer.min_max_normalize_for_price(bar_feature_df.loc[:, "high"])
    bar_feature_df.loc[:, "low"] = normalizer.min_max_normalize_for_price(bar_feature_df.loc[:, "low"])
    bar_feature_df.loc[:, "close"] = normalizer.min_max_normalize_for_price(bar_feature_df.loc[:, "close"])
    bar_feature_df.loc[:, "volume"] = normalizer.log_normalize(bar_feature_df.loc[:, "volume"])
    bar_feature_df.loc[:, "delta"] = normalizer.min_max_normalize_with_negative(bar_feature_df.loc[:, "delta"])
    bar_feature_df.loc[:, "trades_count"] = normalizer.min_max_normalize(bar_feature_df.loc[:, "trades_count"])
    return bar_feature_df

def do_normalize_for_price_level_features(price_level_feature_df: pd.DataFrame, normalizer: GlobalNormalizer):
    price_level_feature_df = price_level_feature_df.astype({"bid_count": float, 'ask_count': float, "trades_count": float, "idx": float})
    price_level_feature_df.loc[:, "price"] = normalizer.min_max_normalize_for_price(price_level_feature_df.loc[:, "price"])
    price_level_feature_df.loc[:, "bid_size"] = normalizer.min_max_normalize(price_level_feature_df.loc[:, "bid_size"])
    price_level_feature_df.loc[:, "bid_count"] = normalizer.min_max_normalize(price_level_feature_df.loc[:, "bid_count"])
    price_level_feature_df.loc[:, "ask_size"] = normalizer.min_max_normalize(price_level_feature_df.loc[:, "ask_size"])
    price_level_feature_df.loc[:, "ask_count"] = normalizer.min_max_normalize(price_level_feature_df.loc[:, "ask_count"])
    price_level_feature_df.loc[:, "delta"] = normalizer.min_max_normalize_with_negative(price_level_feature_df.loc[:, "delta"])
    price_level_feature_df.loc[:, "trades_count"] = normalizer.min_max_normalize(price_level_feature_df.loc[:, "trades_count"])
    price_level_feature_df.loc[:, "volume"] = normalizer.log_normalize(price_level_feature_df.loc[:, "volume"])
    price_level_feature_df.loc[:, "idx"] = normalizer.min_max_normalize(price_level_feature_df.loc[:, "idx"])
    return price_level_feature_df

def pack_price_level_feature(price_level_feature_df: pd.DataFrame, onehot_feature_df: pd.DataFrame):
    grouped = price_level_feature_df.groupby('timestamp')
    
    # 存储每个时间步的特征参数
    feature_params = []
    
    # 获取特征列名（除timestamp外的所有列）
    feature_columns = [col for col in price_level_feature_df.columns if col != 'timestamp']
    onehot_columns = onehot_feature_df.columns.tolist()
    # 按时间顺序处理每个组
    idx = -1
    for timestamp, group in sorted(grouped, key=lambda x: (x[0], x[1]['idx'])):
        idx += 1
        step_features = group[feature_columns].values
        num_price_levels = len(step_features)
        # 获取当前时间步的onehot特征

        # 将字符串形式的列表转换为实际的数组
        onehot_arrays = [onehot_feature_df.loc[idx][col] for col in onehot_columns]
        for arr in onehot_arrays:
            assert len(arr) == num_price_levels, \
                f"Onehot array length ({len(arr)}) doesn't match price level count ({num_price_levels})"
        
        
        # 构建完整的特征矩阵
        full_features = np.zeros((num_price_levels, len(feature_columns) + len(onehot_columns)))
        
        # 填充原始特征
        full_features[:, :len(feature_columns)] = step_features
        
        # 填充onehot特征
        for i, onehot_array in enumerate(onehot_arrays):
            full_features[:, len(feature_columns) + i] = onehot_array
        
        feature_params.append(full_features)
    
    # 返回打包后的特征字典
    return {
        'main_PriceLevels': (
            feature_params,  # list[time_steps][num_features, param_dim]
            None
        )
    }

def combine_feature_df(feature_df: pd.DataFrame, bar_feature_df: pd.DataFrame, price_level_feature_df: pd.DataFrame, onehot_feature_df: pd.DataFrame, embedding_config: dict):
    """
    将特征DataFrame转换为训练所需的格式

    Args:
        feature_df: DataFrame，列名格式为 {name}_{feature_name}_{param_id}
        price_level_feature_df: DataFrame，price level特征
        onehot_feature_df: DataFrame，onehot特征
        embedding_config: dict，特征嵌入配置

    Returns:
        dict: {
            'other_features': list[time_steps][num_samples, other_feature_dim],
            'feature_data': list[time_steps][dict] 每个dict的格式为:
                {
                    'feature_name': (
                        feature_params: list[num_samples][num_features, param_dim],
                        feature_categories: list[num_samples][num_features] or None
                    )
                }
        }
    """
    num_time_steps = len(feature_df)
    feature_data = []
    other_features = []

    # 处理price level特征
    price_level_features = pack_price_level_feature(price_level_feature_df, onehot_feature_df)

    # 按时间步处理特征
    for time_step in range(num_time_steps):
        step_features = {}
        step_other_features = []
        
        # 按feature_name分组处理列
        feature_groups = {}
        for col in feature_df.columns:
            name, feature_name, param_id = col.split('_', 2)
            key = f"{name}_{feature_name}"
            if key not in feature_groups:
                feature_groups[key] = []
            feature_groups[key].append((param_id, col))

        # non_embedding_features = []
        # for key, columns in feature_groups.items():
        #     name, feature_name = key.split('_', 1)
        #     if feature_name not in embedding_config:
        #         non_embedding_features.append(key)
        # print(non_embedding_features)

        # 处理每个特征组
        for key, columns in feature_groups.items():
            name, feature_name = key.split('_', 1)
            
            # 检查是否在embedding_config中
            if feature_name in embedding_config:
                config = embedding_config[feature_name]
                param_values = []
                categories = None
                
                # 获取该时间步的所有参数值
                for param_id, col in sorted(columns, key=lambda x: int(x[0].split('_')[-1])):
                    value = feature_df.iloc[time_step][col]
                    if value is not None and not (isinstance(value, list) and value[0] is None):
                        param_values.extend(value if isinstance(value, list) else [value])
                
                if param_values:
                    # 如果有类别特征，提取第一列作为类别
                    if config['num_categories'] > 0:
                        categories = param_values[::len(columns)]  # 每组参数的第一个值作为类别
                        param_values = [v for i, v in enumerate(param_values) if i % len(columns) != 0]  # 剩余值作为参数
                    
                    # 重塑参数数组
                    num_features = len(param_values) // (len(columns) - (1 if categories else 0))
                    param_dim = len(columns) - (1 if categories else 0)
                    params = np.array(param_values).reshape(num_features, param_dim)
                    
                    if name not in step_features:
                        step_features[name + '_' + feature_name] = (params, categories)
            else:
                # 不在embedding_config中的特征添加到other_features
                for param_id, col in columns:
                    value = feature_df.iloc[time_step][col]
                    if value is not None and not (isinstance(value, list) and value[0] is None):
                        step_other_features.extend(value if isinstance(value, list) else [value])
        
        # 添加price level特征
        if price_level_features:
            step_features['main_PriceLevels'] = [price_level_features['main_PriceLevels'][0][time_step], None]
        
        feature_data.append(step_features)
        other_features.append(np.array(step_other_features))
    
    bar_copy = bar_feature_df.copy()
    bar_copy.drop(columns=["timestamp"], inplace=True)
    number_feature_df = pd.DataFrame([arr.tolist() for arr in other_features])
    number_feature_df = pd.concat([bar_copy, number_feature_df], axis=1)

    flat_data = []
    for entry in feature_data:
        flat_entry = {}
        for key, [arr1, arr2] in entry.items():
            flat_entry[f"{key}_params"] = list(arr1)
            flat_entry[f"{key}_categories"] = list(arr2) if arr2 is not None else None
        flat_data.append(flat_entry)

    event_data_df = pd.DataFrame(flat_data, dtype="object")
    event_data_df = event_data_df.where(pd.notnull(event_data_df), None)
    return number_feature_df, event_data_df

def get_file_list():
    import os
    all_files = set()
    for filename in os.listdir('./footprint/'):
        if filename.endswith('.json'):
            all_files.add(filename.split('.')[0])
    return sorted(list(all_files))

def main(filename_list: list[str]):
    for file in filename_list:
        print(f"Processing {file}")
        bars = get_bars(file)
        target_feature_df = generate_target_feature(bars)
        # pd.set_option('display.max_rows', None)
        ticks = get_ticks(file)
        event_feature_df = pd.DataFrame()

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
                if issubclass(indicator_cls, SpeedBarCalculator):
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
        for indicator_instance_tuple in indicator_instance_list:
            feature_name = indicator_instance_tuple["indicator"].get_feature_name()
            sum_feature_size += indicator_instance_tuple["indicator"].get_feature_dimension()
            if feature_name == "Trend":
                feature_df = generate_flow_feature(indicator_instance_tuple, bars, ticks, price_to_price_level_idx)
                feature_df = indicator_instance_tuple["indicator"].normalize_feature(feature_df.iloc[:, 0], indicator_instance_tuple['name'] + "_" + feature_name, normalizer)
                global_flow_feature_df = pd.concat([global_flow_feature_df, feature_df], axis=1)
            else:
                feature_df, onehot_feature_df = generate_feature_df(indicator_instance_tuple, bars, ticks, price_to_price_level_idx)
                feature_df = indicator_instance_tuple["indicator"].normalize_feature(feature_df.iloc[:, 0], indicator_instance_tuple['name'] + "_" + feature_name, normalizer)
                event_feature_df = pd.concat([event_feature_df, feature_df], axis=1)
                if onehot_feature_df is not None:
                    global_onehot_feature_df = pd.concat([global_onehot_feature_df, onehot_feature_df], axis=1)

        aggregate_by_prefix(global_flow_feature_df)

        bar_feature_df = do_normalize_for_bar_features(bar_feature_df, normalizer)
        price_level_features_df = do_normalize_for_price_level_features(price_level_features_df, normalizer)

        number_feature_df, event_data_df = combine_feature_df(event_feature_df, bar_feature_df, price_level_features_df, global_onehot_feature_df, embedding_config)
        write_to_parquet(number_feature_df, f'./model_feature/{file}_number.parquet')
        write_to_parquet(event_data_df, f'./model_feature/{file}_event.parquet')
        write_to_parquet(target_feature_df, f'./model_feature/{file}_target.parquet')
        write_to_parquet(global_flow_feature_df, f'./model_feature/{file}_flow.parquet')
        # number_feature_df.to_html("output_number.html")
        # event_data_df.to_html("output_event.html")
        # target_feature_df.to_html("output_target.html")
        # global_flow_feature_df.to_html("output_flow.html")
        print(f"Completed processing {file}")
if __name__ == "__main__":
    # filename_list = get_file_list()
    main(['BTCUSDT-trades-2023-11'])
