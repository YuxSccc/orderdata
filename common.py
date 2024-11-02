import json
from abc import ABC, abstractmethod
import pandas as pd
import random
from typing import Optional, Callable
import numpy as np

class Tick:
    def __init__(self):
        self.timestamp = 0
        self.price = 0
        self.size : float = 0
        self.isBuy : bool = False

    def __str__(self):
        return f'timestamp: {self.timestamp}, price: {self.price}, size: {self.size}, isBuy: {self.isBuy}'

class AggTick(Tick):
    def __init__(self):
        super().__init__()
        self.count : int = 0

    def __str__(self):
        return f'timestamp: {self.timestamp}, price: {self.price}, size: {self.size}, isBuy: {self.isBuy}, count: {self.count}'

class FootprintBar:
    class PriceLevel:
        def __init__(self, volumePrecision: int = None, pricePrecision: int = None):
            self.price : float = 0
            self.volume : float = 0
            self.bidSize : float = 0
            self.askSize : float = 0
            self.bidCount : int = 0
            self.askCount : int = 0
            self.delta : float = 0
            self.tradesCount : int = 0
            self.volumePrecision : int = volumePrecision
            self.pricePrecision : int = pricePrecision
        
        def to_dict(self):
            return {
                'price': round(self.price, self.pricePrecision),
                'volume': round(self.volume, self.volumePrecision),
                'bidSize': round(self.bidSize, self.pricePrecision),
                'askSize': round(self.askSize, self.pricePrecision),
                'bidCount': self.bidCount,
                'askCount': self.askCount,
                'delta': round(self.delta, self.volumePrecision),
                'tradesCount': self.tradesCount,
            }
        
        def from_dict(self, dict: dict, volumePrecision: int, pricePrecision: int):
            self.price = dict['price']
            self.volume = dict['volume']
            self.bidSize = dict['bidSize']
            self.askSize = dict['askSize']
            self.bidCount = dict['bidCount']
            self.askCount = dict['askCount']
            self.delta = dict['delta']
            self.tradesCount = dict['tradesCount']
            self.volumePrecision = volumePrecision
            self.pricePrecision = pricePrecision
            return self

    def __init__(self, duration: int = None, scale: int = None, volumePrecision: int = None, pricePrecision: int = None):
        self.timestamp = 0
        self.duration : int = duration
        self.scale : int = scale

        self.priceLevels : dict[int, FootprintBar.PriceLevel] = {}
        
        self.openTime : int = 0
        self.closeTime : int = 0
        self.open : float = 0
        self.high : float = 0
        self.low : float = 0
        self.close : float = 0
        self.volume : float = 0
        self.delta : float = 0
        self.tradesCount : int = 0
        self.volumePrecision : int = volumePrecision
        self.pricePrecision : int = pricePrecision

    def __eq__(self, other):
        return self.timestamp == other.timestamp and self.duration == other.duration and self.scale == other.scale and self.volumePrecision == other.volumePrecision and self.pricePrecision == other.pricePrecision

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'duration': self.duration,
            'scale': self.scale,
            'openTime': self.openTime,
            'closeTime': self.closeTime,
            'open': round(self.open, self.pricePrecision),
            'high': round(self.high, self.pricePrecision),
            'low': round(self.low, self.pricePrecision),
            'close': round(self.close, self.pricePrecision),
            'volume': round(self.volume, self.volumePrecision),
            'delta': round(self.delta, self.volumePrecision),
            'tradesCount': self.tradesCount,
            'volumePrecision': self.volumePrecision,
            'pricePrecision': self.pricePrecision,
            'priceLevels': {k: v.to_dict() for k, v in self.priceLevels.items()},
        }

    def from_dict(self, dict: dict):
        self.timestamp = dict['timestamp']
        self.duration = dict['duration']
        self.scale = dict['scale']
        self.volumePrecision = dict['volumePrecision']
        self.pricePrecision = dict['pricePrecision']
        self.openTime = dict['openTime']
        self.closeTime = dict['closeTime']
        self.open = dict['open']
        self.high = dict['high']
        self.low = dict['low']
        self.close = dict['close']
        self.volume = dict['volume']
        self.delta = dict['delta']
        self.tradesCount = dict['tradesCount']
        self.priceLevels = {k: FootprintBar.PriceLevel().from_dict(v, self.volumePrecision, self.pricePrecision) for k, v in dict['priceLevels'].items()}
        return self

    def __str__(self):
        return f'timestamp: {self.timestamp}, open: {self.open}, high: {self.high}, low: {self.low}, close: {self.close}, volume: {self.volume}, delta: {self.delta}, tradesCount: {self.tradesCount}'

    def get_price_level_height(self) -> float:
        return self.scale * (10 ** -self.pricePrecision)

    def normalize_price(self, price: float) -> int:
        return round(price // self.get_price_level_height() * self.get_price_level_height(), self.pricePrecision)

    def handle_tick(self, tick: Tick) -> bool:
        if self.timestamp == 0:
            self.timestamp = tick.timestamp // self.duration * self.duration
            self.openTime = tick.timestamp
            self.closeTime = tick.timestamp
            self.open = self.normalize_price(tick.price)
            self.close = self.normalize_price(tick.price)
            self.high = self.normalize_price(tick.price)
            self.low = self.normalize_price(tick.price)

        if (tick.timestamp < self.timestamp) or (tick.timestamp >= self.timestamp + self.duration):
            return False
        
        tickScalePrice = self.normalize_price(tick.price)

        if tickScalePrice not in self.priceLevels:
            self.priceLevels[tickScalePrice] = self.PriceLevel(self.volumePrecision, self.pricePrecision)
            self.priceLevels[tickScalePrice].price = tickScalePrice

        priceLevel = self.priceLevels[tickScalePrice]

        if tick.timestamp > self.closeTime:
            self.closeTime = tick.timestamp
            self.close = tick.price
        
        if tick.timestamp < self.openTime:
            self.openTime = tick.timestamp
            self.open = tick.price
        
        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        
        if tick.isBuy:
            priceLevel.bidSize += tick.size
            priceLevel.bidCount += 1
            priceLevel.delta += tick.size
        else:
            priceLevel.askSize += tick.size
            priceLevel.askCount += 1
            priceLevel.delta -= tick.size

        priceLevel.volume += tick.size
        priceLevel.tradesCount += 1

        self.volume += tick.size
        self.tradesCount += 1
        self.delta += tick.size if tick.isBuy else -tick.size
        return True
    
    def end_handle_tick(self):
        self._fill_no_trades_price_levels()
        self._sort_price_levels()

    def _sort_price_levels(self):
        self.priceLevels = dict(sorted(self.priceLevels.items()))

    def _fill_no_trades_price_levels(self):
        price_interval = (10 ** -self.pricePrecision) * self.scale
        price = self.normalize_price(self.open)
        price_end = self.normalize_price(self.close)
        while price <= price_end:
            if price not in self.priceLevels:
                self.priceLevels[price] = self.PriceLevel(self.volumePrecision, self.pricePrecision)
                self.priceLevels[price].price = price
            price += price_interval

class Signal(ABC):
    def __init__(self, signalName: str):
        self.signalName : str = signalName
        self.additionalInfo : list[float] = []

    def get_signal_name(self) -> str:
        return self.signalName
    
    def set_significance(self, significance: float):
        self.significance = significance

    def get_additional_info(self) -> list[float]:
        return self.additionalInfo

    def set_additional_info(self, additionalInfo: list[float]):
        self.additionalInfo = additionalInfo

    def add_additional_info(self, additionalInfo: list[float]):
        self.additionalInfo.extend(additionalInfo)

class TickSignal(Signal):
    def __init__(self, signalName: str):
        super().__init__(signalName)
        self.tick : Tick = None
        self.significance : float = 0

    def set_tick(self, tick: Tick):
        self.tick = tick

    def get_color(self):
        return 'green' if self.tick.isBuy else 'red'

class MultiBarSignal(Signal):
    def __init__(self, signalName: str):
        super().__init__(signalName)
        self.kBarList : list[FootprintBar] = []
        self.significance : float = 0
        self.additionalInfo : list[float] = []

        self.barAdditionalInfo : list[list[float]] = []
        self.colorTensor : list[list[set[int]]] = []

    def add_bar(self, bar: FootprintBar, additionalInfo: list[float], colorTensor: list[set[int]]):
        self.kBarList.append(bar)
        self.barAdditionalInfo.append(additionalInfo)
        self.colorTensor.append(colorTensor)

    def get_color_tensor(self) -> list[list[set[int]]]:
        return self.colorTensor

    def get_bars(self) -> list[FootprintBar]:
        return self.kBarList
    
    def get_timestamp(self) -> tuple[int, int]:
        return self.kBarList[0].timestamp, self.kBarList[-1].timestamp

class SingleBarSignal(Signal):
    def __init__(self, signalName: str):
        super().__init__(signalName)

    def get_signal_str(self) -> str:
        str = self.signalName + " : "
        for key, value in self.get_signal_dict().items():
            str += f"{key}: {value}, "
        return str

    @abstractmethod
    def get_signal_dict(self) -> dict:
        pass

def cal_signal_hook(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.after_calc_signal_hook(result)
        return result
    return wrapper

class GlobalNormalizer:
    def __init__(self):
        pass

    def set_max_price(self, max_price: float):
        self.max_price = max_price

    def set_min_price(self, min_price: float):
        self.min_price = min_price

    def get_max_price(self) -> float:
        return self.max_price

    def get_min_price(self) -> float:
        return self.min_price

    @staticmethod
    def _find_min_max_nested(data) -> tuple[Optional[float], Optional[float]]:
        flat_values = []

        def recursive_flatten(value):
            if value is None:
                pass
            elif isinstance(value, np.ndarray):
                flat_values.append(value.min())
                flat_values.append(value.max())
            elif isinstance(value, list):
                for item in value:
                    recursive_flatten(item)
            else:
                flat_values.append(value)
                
        recursive_flatten(data)
        
        return float(min(flat_values)) if len(flat_values) > 0 else None, float(max(flat_values)) if len(flat_values) > 0 else None

    @staticmethod
    def _normalize_value(value, normalize_func: Callable):
        if value is None:
            return None
        elif isinstance(value, list):
            return [GlobalNormalizer._normalize_value(v, normalize_func) for v in value]
        else:
            return normalize_func(value)

    @staticmethod
    def get_normalize_min_max_value(df: pd.Series) -> tuple[Optional[float], Optional[float]]:
        min_value, max_value = GlobalNormalizer._find_min_max_nested(df.tolist())
        if min_value is None or max_value is None:
            return None, None
        min_bias = random.random() * 0.1 + 0.2
        max_bias = random.random() * 0.1 + 0.2
        min_value = min_value * (1 - min_bias)
        max_value = max_value * (1 + max_bias)
        return min_value, max_value

    def min_max_normalize_for_price(self, df: pd.Series, start_row: Optional[int] = None, end_row: Optional[int] = None):
        return GlobalNormalizer.min_max_normalize(df, self.min_price, self.max_price, start_row=start_row, end_row=end_row)

    @staticmethod
    def min_max_normalize(df: pd.Series, min_value: Optional[float]=None, max_value: Optional[float]=None, 
                          start_row: Optional[int] = None, end_row: Optional[int] = None):
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = len(df) - 1
        sub_df = df.loc[start_row:end_row]

        if min_value is None or max_value is None:
            min_value, max_value = GlobalNormalizer.get_normalize_min_max_value(sub_df)

        if min_value is None or max_value is None or min_value == max_value:
            return df
        normalized_series = sub_df.apply(lambda x: GlobalNormalizer._normalize_value(x, lambda x: (x - min_value) / (max_value - min_value)))
        normalized_df = df.copy()
        normalized_df.loc[start_row:end_row] = normalized_series
        return normalized_df

    @staticmethod
    def min_max_normalize_with_negative(df: pd.Series, min_value: Optional[float]=None, max_value: Optional[float]=None, 
                                        start_row: Optional[int] = None, end_row: Optional[int] = None):
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = len(df) - 1
        sub_df = df.loc[start_row:end_row]

        if min_value is None or max_value is None:
            min_value, max_value = GlobalNormalizer.get_normalize_min_max_value(sub_df)

        if min_value is None or max_value is None or min_value == max_value:
            return df
        normalized_series = sub_df.apply(lambda x: GlobalNormalizer._normalize_value(x, lambda x: 2 * (x - min_value) / (max_value - min_value) - 1))
        normalized_df = df.copy()
        normalized_df.loc[start_row:end_row] = normalized_series
        return normalized_df

    @staticmethod
    def log_normalize_with_rows(series: pd.Series, start_row: Optional[int] = None, end_row: Optional[int] = None) -> pd.Series:
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = len(series) - 1
        sub_series = series.loc[start_row:end_row]

        normalized_series = sub_series.apply(lambda x: GlobalNormalizer._normalize_value(x, lambda x: np.log1p(x)))
        result_series = series.copy()
        result_series.loc[start_row:end_row] = normalized_series
        return GlobalNormalizer.min_max_normalize(result_series, start_row=start_row, end_row=end_row)

class Calculator(ABC):
    # _cache = {}

    # def __new__(cls, *args, **kwargs):
    #     cache_key = (cls, args, frozenset(kwargs.items()))

    #     if cache_key in cls._cache:
    #         return cls._cache[cache_key]

    #     instance = super().__new__(cls)
    #     cls._cache[cache_key] = instance
    #     return instance

    DO_NOTHING = 0
    MIN_MAX_NORMALIZE = 1
    MIN_MAX_NORMALIZE_WITH_NEGATIVE = 2
    MIN_MAX_NORMALIZE_FOR_PRICE = 3
    LOG_NORMALIZE = 4
    CUSTOM_NORMALIZE = 5

    @abstractmethod
    def calc_signal(self) -> list[Signal]:
        pass

    @abstractmethod
    def get_feature_column_name(self) -> list[str]:
        pass

    @abstractmethod
    def get_feature_name(self) -> str:
        pass

    @abstractmethod
    def get_signal_type(self) -> type[Signal]:
        pass

    @abstractmethod
    def get_normalize_type(self) -> list[int]:
        pass

    def normalize_feature(self, feature: pd.Series, name: str, normalizer: GlobalNormalizer) -> pd.DataFrame:
        expanded_df = pd.DataFrame(feature.apply(lambda x: [list(item) for item in zip(*x)]).tolist())
        expanded_df.columns = [f'{name}_{i+1}' for i in range(expanded_df.shape[1])]
        # if isinstance(self, TickSignalCalculator):
        #     expanded_df = pd.DataFrame(feature.apply(lambda x: [list(item) for item in zip(*x)]).tolist())
        #     expanded_df.columns = [f'{name}_{i+1}' for i in range(expanded_df.shape[1])]
        # else:
        #     expanded_df = pd.DataFrame(feature.tolist(), columns=[f'{name}_{i+1}' for i in range(self.get_feature_dimension())])
        normalize_type = self.get_normalize_type()
        # empty feature
        if expanded_df.shape[1] != len(normalize_type):
            return pd.DataFrame(None, index=range(expanded_df.shape[0]), columns=[f'{name}_{i+1}' for i in range(len(normalize_type))])
        for idx, normalize_type in enumerate(normalize_type):
            expanded_df[f'{name}_{idx+1}'] = self.do_simple_normalize(expanded_df[f'{name}_{idx+1}'], normalize_type, idx, normalizer)
        return expanded_df

    def do_simple_normalize(self, data: pd.Series, normalize_type: int, idx: int = -1, normalizer: GlobalNormalizer = None) -> pd.Series:
        if normalize_type == Calculator.DO_NOTHING:
            return data
        elif normalize_type == Calculator.MIN_MAX_NORMALIZE:
            return GlobalNormalizer.min_max_normalize(data)
        elif normalize_type == Calculator.MIN_MAX_NORMALIZE_WITH_NEGATIVE:
            return GlobalNormalizer.min_max_normalize_with_negative(data)
        elif normalize_type == Calculator.MIN_MAX_NORMALIZE_FOR_PRICE:
            return normalizer.min_max_normalize_for_price(data)
        elif normalize_type == Calculator.LOG_NORMALIZE:
            return GlobalNormalizer.log_normalize_with_rows(data)
        elif normalize_type == Calculator.CUSTOM_NORMALIZE:
            return self.do_custom_normalize(data, idx, normalizer)
        else:
            raise ValueError(f"Invalid normalize type: {normalize_type}")

    def get_feature_dimension(self) -> int:
        return len(self.get_feature_column_name())

    def get_color_onehot_name(self, idx: int) -> str:
        return f'{self.get_feature_name()}_color_{idx}'
    
    def get_feature_name_with_idx(self, idx: int) -> str:
        return f'{self.get_feature_name()}_{self.get_feature_column_name()[idx]}'

    def after_calc_signal_hook(self, signals: list[Signal]):
        self.signals = signals
        self.cal_finished = True
        # TODO: check color / signal / merge signal

class TickSignalCalculator(Calculator):
    def __init__(self):
        self.cal_finished = False
        self.signals : list[Signal] = []

class MultiBarSignalCalculator(Calculator):
    def __init__(self):
        self.cal_finished = False
        self.signals : list[Signal] = []

    @abstractmethod
    def get_max_color_size(self) -> int:
        pass

class SingleBarSignalCalculator(Calculator):
    def __init__(self):
        self.cal_finished = False
        self.signals : list[Signal] = []