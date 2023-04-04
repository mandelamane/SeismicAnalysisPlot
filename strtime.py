import datetime


class StrTime:
    """A class for making the time string easier to use."""

    def __init__(self, event_time: str):
        """
        Parameters
        ----------
        event_time : str
            event start time
        """

        self.origin_time = event_time
        self.year = event_time[:4]
        self.month = event_time[4:6]
        self.day = event_time[6:8]
        self.hour = event_time[8:10]
        self.minutes = event_time[10:12]
        self.sec = event_time[12:14]

    def get_time_unit(self, unit: str) -> str:
        """
        Get a part of the time string up to a specified unit.

        Parameters
        ----------
        unit : str
            the unit of time to get, such as "year", "month", "day", "hour", "minutes",
            or "sec".

        Returns
        -------
        extracted_time_unit : str
            The time string up to the specified unit
        """

        end = {
            "year": 4,
            "month": 6,
            "day": 8,
            "hour": 10,
            "minutes": 12,
            "sec": 14,
        }[unit]

        extracted_time_unit = self.origin_time[:end]
        return extracted_time_unit

    def convert_time_format(self) -> str:
        """
        Convert the time string from "YYYYMMDDHHMMSS" to "YYYY/MM/DD HH:MM:SS".

        Returns
        -------
        converted_time_string : str
            The time string in the new format
        """

        time_obj = datetime.datetime.strptime(self.origin_time, "%Y%m%d%H%M%S")

        converted_time_string = time_obj.strftime("%Y/%m/%d %H:%M:%S")

        return converted_time_string
