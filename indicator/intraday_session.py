from common import *
import pytz
from datetime import datetime

class IntradaySessionSignal(BarStatusSignal):
    SIGNAL_NAME = "IntradaySessionStatus"
    def __init__(self):
        super().__init__(IntradaySessionSignal.SIGNAL_NAME)

    def get_signal_dict(self) -> dict:
        return {"IntradaySessionStatus": IntradaySessionCalculator.session_name[self.additionalInfo[0]]}

    def get_signal_str(self) -> str:
        return self.signalName + " : " + IntradaySessionCalculator.session_name[self.additionalInfo[0]]

class IntradaySessionCalculator(BarStatusCalculator):
    ny_tz = pytz.timezone('America/New_York')
    ldn_tz = pytz.timezone('Europe/London')
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    utc_tz = pytz.utc

    ASIA_SESSION = 0
    EU_SESSION = 1
    US_SESSION = 2
    CLOSED = 3
    enum_session = [ASIA_SESSION, EU_SESSION, US_SESSION, CLOSED]
    session_name = ["Asia", "EU", "US", "Closed"]

    def __init__(self, bars: list[FootprintBar]):
        super().__init__()
        self.bars = bars

    def _is_dst(self, dt):
        return bool(dt.astimezone(self.ny_tz).dst())

    def _get_market_session(self, timestamp: int):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.utc)
        dt_utc = dt.replace(tzinfo=self.utc_tz)

        tokyo_open = dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        tokyo_close = dt_utc.replace(hour=9, minute=0, second=0, microsecond=0)

        london_open = dt_utc.replace(hour=7, minute=0, second=0, microsecond=0)
        london_close = dt_utc.replace(hour=16, minute=0, second=0, microsecond=0)

        if self._is_dst(dt):
            ny_open = dt_utc.replace(hour=12, minute=0, second=0, microsecond=0)
            ny_close = dt_utc.replace(hour=21, minute=0, second=0, microsecond=0)
        else:
            ny_open = dt_utc.replace(hour=13, minute=0, second=0, microsecond=0)
            ny_close = dt_utc.replace(hour=22, minute=0, second=0, microsecond=0)

        if dt_utc >= ny_open and dt_utc < ny_close:
            return IntradaySessionCalculator.US_SESSION
        
        if dt_utc >= london_open and dt_utc < london_close:
            return IntradaySessionCalculator.EU_SESSION
        
        return IntradaySessionCalculator.ASIA_SESSION

    def calc_signal(self) -> list[Signal]:
        signals = []
        for bar in self.bars:
            signal = IntradaySessionSignal()
            signal.set_additional_info([self._get_market_session(bar.timestamp)])
            signals.append(signal)
        self.cal_finished = True
        self.signals = signals
        return signals