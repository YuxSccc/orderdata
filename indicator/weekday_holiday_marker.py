from common import *
import holidays
from datetime import datetime, timezone

class WeekdayHolidaySignal(BarStatusSignal):
    SIGNAL_NAME = "WeekdayHoliday"
    def __init__(self):
        super().__init__(WeekdayHolidaySignal.SIGNAL_NAME)

    def get_signal_dict(self) -> dict:
        return {"Weekday": self.additionalInfo[0], "isHoliday": self.additionalInfo[1]}

    def get_signal_str(self) -> str:
        return self.signalName + " : Weekday: " + self.additionalInfo[0] + " isHoliday: " + self.additionalInfo[1]

class WeekdayHolidayMarkerCalculator(BarStatusCalculator):
    us_holidays = holidays.US()

    def __init__(self, bars: list[FootprintBar]):
        super().__init__()
        self.bars = bars

    def get_day_info(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        weekday = dt.weekday()
        is_holiday = dt in self.us_holidays
        return weekday, is_holiday

    def calc_signal(self) -> list[Signal]:
        signals = []
        for bar in self.bars:
            weekday, is_holiday = self.get_day_info(bar.timestamp)
            signal = WeekdayHolidaySignal()
            signal.set_additional_info([weekday, is_holiday])
            signals.append(signal)
        self.cal_finished = True
        self.signals = signals
        return signals