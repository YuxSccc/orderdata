from indicator import *
import math

config = {
    'BigTradeCalculator': {
        'main': {
            'size_threshold': 100,
            'aggregate_interval_in_ms': 100
        }
    },
    'DeltaClusterCalculator': {
        'buy': {
            'check_bar_interval': 10,
            'max_price_level_range': 5,
            'min_delta_percentage': 0.4,
            'min_price_level_count': 5,
            'positive_side': True,
            'overlap_threshold': 0.3333
        },
        'sell': {
            'check_bar_interval': 10,
            'max_price_level_range': 5,
            'min_delta_percentage': 0.4,
            'min_price_level_count': 5,
            'positive_side': False,
            'overlap_threshold': 0.3333
        }
    },
    'EMACalculator': {
        'main': {
            'period': 20
        }
    },
    'ExtremePriceCalculator': {
        'main': {
            'min_distance': 6,
            'max_prominence': 120
        }
    },
    'HugeLmtCalculator': {
        'buy': {
            'imbalance_threshold': 0.1,
            'volume_threshold': 150,
            'price_level_count_threshold': 1,
            'buy_side': True
        },
        'sell': {
            'imbalance_threshold': 0.1,
            'volume_threshold': 150,
            'price_level_count_threshold': 1,
            'buy_side': False
        }
    }, 
    'IntradaySessionCalculator': {
        'main': {
        }
    },
    'LastImbalanceCalculator': {
        'buyh': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'is_buy_imbalance': True,
            'higher_price_side': True
        },
        'buyl': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'is_buy_imbalance': True,
            'higher_price_side': False
        },
        'sellh': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'is_buy_imbalance': False,
            'higher_price_side': True
        },
        'selll': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'is_buy_imbalance': False,
            'higher_price_side': False
        }
    },
    'LastLowHighPriceCalculator': {
        'main': {
            'reserve_price_count': 1
        }
    },
    'MACDCalculator': {
        'main': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    },
    'NoLiquidityLevelsCalculator': {
        'main': {
            'price_level_threshold': 2,
            'ignore_volume': 3
        }
    },
    'PBPatternCalculator': {
        'main': {
            'price_level_count_threshold': 4,
            'check_ratio': 0.4,
            'max_delta_ratio': 0.5,
            'recent_max_delta_volume_count': 10
        }
    },
    'PriceChangePercentageCalculator': {
        'main': {
            'percentage_threshold': 0.5
        }
    },
    'RecentMaxDeltaVolumeCalculator': {
        'main': {
            'window_size': 30
        }
    },
    'RSICalculator': {
        'main': {
            'period': 14
        }
    },
    'SMACalculator': {
        'main': {
            'period': 20
        }
    },
    'SpeedBarCalculator': {
        'main': {
            'check_interval_in_sec': 15,
            'trade_count_threshold': 800
        }
    },
    'StopLostOrdersCalculator': {
        'buy': {
            'delta_threshold': 20,
            'pre_bar_count_threshold': 3,
            'suf_bar_count_threshold': 0,
            'price_change_ratio_threshold': 50/60000,
            'total_volume_threshold': 70,
            'buy_side': True
        },
        'sell': {
            'delta_threshold': 20,
            'pre_bar_count_threshold': 3,
            'suf_bar_count_threshold': 0,
            'price_change_ratio_threshold': 50/60000,
            'total_volume_threshold': 70,
            'buy_side': False
        }
    },
    'TradeImbalanceCalculator': {
        'buy': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'buy_side': True
        },
        'sell': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'buy_side': False
        }
    },
    'TrendCalculator': {
        'main': {
            'error_threshold': 300,
            'min_length': 2,
            'max_length': 100
        }
    },
    'WeekdayHolidayMarkerCalculator': {
        'main': {
        }
    }
}

# config = {
#     'IntradaySessionCalculator': {
#         'main': {
#         }
#     },
# }

def next_power_of_2(n):
    if n < 1:
        return 1
    return 2 ** math.ceil(math.log2(n))

embedding_config = {
    BigTradeCalculator.get_feature_name(): {
        'max_feature_length': 4,
        'embedding_dim': next_power_of_2(BigTradeCalculator.get_feature_column_name().__len__() * 4),
        'param_dim': BigTradeCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': False
    },
    DeltaClusterCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(DeltaClusterCalculator.get_feature_column_name().__len__() * 1),
        'param_dim': DeltaClusterCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    },
    ExtremePriceCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(ExtremePriceCalculator.get_feature_column_name().__len__() * 1),
        'param_dim': ExtremePriceCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    },
    HugeLmtCalculator.get_feature_name(): {
        'max_feature_length': 3,
        'embedding_dim': next_power_of_2(HugeLmtCalculator.get_feature_column_name().__len__() * 3),
        'param_dim': HugeLmtCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    },
    IntradaySessionCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(IntradaySessionCalculator.get_feature_column_name().__len__() * 1),
        'param_dim': IntradaySessionCalculator.get_feature_column_name().__len__() - 1,
        'category_dim': 2,
        'num_categories': 4,
        'use_attention': True
    },
    LastImbalanceCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(LastImbalanceCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': LastImbalanceCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    },
    LastLowHighPriceCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(LastLowHighPriceCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': LastLowHighPriceCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': False
    },
    NoLiquidityLevelsCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(NoLiquidityLevelsCalculator.get_feature_column_name().__len__() * 1),
        'param_dim': NoLiquidityLevelsCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    },
    PBPatternCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(PBPatternCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': PBPatternCalculator.get_feature_column_name().__len__() - 1,
        'category_dim': 2,
        'num_categories': 4,
        'use_attention': True
    },
    RecentMaxDeltaVolumeCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(RecentMaxDeltaVolumeCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': RecentMaxDeltaVolumeCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': False
    },
    SpeedBarCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(SpeedBarCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': SpeedBarCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': False
    },
    StopLostOrdersCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(StopLostOrdersCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': StopLostOrdersCalculator.get_feature_column_name().__len__() - 1,
        'category_dim': 2,
        'num_categories': 2,
        'use_attention': True
    },
    TradeImbalanceCalculator.get_feature_name(): {
        'max_feature_length': 2,
        'embedding_dim': next_power_of_2(TradeImbalanceCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': TradeImbalanceCalculator.get_feature_column_name().__len__(),
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    },
    TrendCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(TrendCalculator.get_feature_column_name().__len__() * 2),
        'param_dim': TrendCalculator.get_feature_column_name().__len__() - 1,
        'category_dim': 2,
        'num_categories': 4,
        'use_attention': True
    },
    WeekdayHolidayMarkerCalculator.get_feature_name(): {
        'max_feature_length': 1,
        'embedding_dim': next_power_of_2(WeekdayHolidayMarkerCalculator.get_feature_column_name().__len__()),
        'param_dim': WeekdayHolidayMarkerCalculator.get_feature_column_name().__len__() - 1,
        'category_dim': 4,
        'num_categories': 7,
        'use_attention': True
    },
    "PriceLevels": {
        'max_feature_length': 15,
        'embedding_dim': 128,
        'param_dim': 16,
        'category_dim': 0,
        'num_categories': 0,
        'use_attention': True
    }
}

def get_feature_embedding_config(features_name: list[str]):
    result = {}
    for name in features_name:
        _, feature_name = name.split("_")
        if feature_name not in embedding_config:
            raise ValueError(f"Feature {feature_name} not found in embedding_config")
        result[name] = embedding_config[feature_name]

    sorted_result = dict(sorted(result.items()))
    return sorted_result


flow_feature_list = ['Trend']

seq_len = 60
batch_size = 32
begin_skip = 100
end_skip = 50

output_flow_parquet_path = './model_feature/output_flow.parquet'
output_target_parquet_path = './model_feature/output_target.parquet'
output_event_parquet_path = './model_feature/output_event.parquet'
output_number_parquet_path = './model_feature/output_number.parquet'