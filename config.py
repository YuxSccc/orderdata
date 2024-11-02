
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
        'buy': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'is_buy_imbalance': True
        },
        'sell': {
            'diagonal_multiple_threshold': 4,
            'imbalance_price_level_threshold': 3,
            'sum_delta_threshold': 100,
            'ignore_volume': 1,
            'is_buy_imbalance': False
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
            'price_level_threshold': 3,
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
#     'TrendCalculator': {
#         'main': {
#             'error_threshold': 300,
#             'min_length': 2,
#             'max_length': 100
#         }
#     },
# }

embedding_config = {
    'BigTradeSignalCalculator': {
        'max_feature_length': 5,
        'embedding_dim': 16,
        'param_dims': 5,
        'num_categories': 2,
        'use_attention': True
    }
}

flow_feature_list = ['Trend']

seq_len = 100