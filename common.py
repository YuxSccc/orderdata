import json
from abc import ABC, abstractmethod
class Tick:
    def __init__(self):
        self.timestamp = 0
        self.price = 0
        self.size : float = 0
        self.isBuy : bool = False

    def __str__(self):
        return f'timestamp: {self.timestamp}, price: {self.price}, size: {self.size}, isBuy: {self.isBuy}'

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

    def normalize_price(self, price: float) -> int:
        one_price_unit = (10 ** -self.pricePrecision) * self.scale
        return round(price // one_price_unit * one_price_unit, self.pricePrecision)

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

class BarsSignal(Signal):
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

    def get_bars(self) -> list[FootprintBar]:
        return self.kBarList
    
    def get_timestamp(self) -> tuple[int, int]:
        return self.kBarList[0].timestamp, self.kBarList[-1].timestamp

class Calculator(ABC):
    pass

class TickCalculator(Calculator):

    def __init__(self):
        self.cal_finished = False
        self.signals : list[Signal] = []

    @abstractmethod
    def calc_signal(self) -> list[Signal]:
        pass

class BarCalculator(Calculator):
    def __init__(self):
        self.cal_finished = False
        self.signals : list[Signal] = []

    @abstractmethod
    def calc_signal(self) -> list[Signal]:
        pass
