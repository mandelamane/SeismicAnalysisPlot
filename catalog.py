import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from strtime import StrTime


class CatalogMonitor:
    """
    A class for monitoring catalog data.
    """

    def __init__(self, cg_config: Dict[str, Any]):
        """
        Parameters
        ----------
        cg_config : Dict[str, Any]
            the parameters for the catalog data processing
        """

        self.cg_config = cg_config
        self.file_name = cg_config["file_name"]
        self.read_csv_file = cg_config["read_csv_file"]
        self.init_time = cg_config["init_time"]

        self.output_file = os.path.join(
            self.cg_config["save_dir"], f"{self.file_name}.csv"
        )

        if self.read_csv_file is None:
            print("条件検索中")
            self.event_cg = self.extract_events_from_cg()
        else:
            self.event_cg = pd.read_csv(
                self.read_csv_file, header=None, dtype=str
            )

        # Shuffle catalog data.
        self.event_cg = self.event_cg.sample(frac=1).reset_index(drop=True)

        self.total_events = len(self.event_cg)
        self.total_check = self.event_cg[5].notnull().sum()

        if self.init_time is None:
            self.cg_index = self.event_cg.iloc[[0], :].index[0]
        else:
            self.cg_index = self.event_cg[
                self.event_cg[0] == self.init_time
            ].index[0]

        self.get_event_info()

    def extract_events_from_cg(self) -> pd.DataFrame:
        """
        Extract event from interim catalog data.

        Returns
        -------
        extracted_cg : pd.DataFrame
            extracted catalog data
        """

        value_list = [
            self.cg_config["time"],
            self.cg_config["radius"],
            [],  # It's a formality in there.
            self.cg_config["dep"],
            self.cg_config["mag"],
        ]

        eq_file = self.cg_config["eq_interim_file"]
        tremor_file = self.cg_config["tremor_interim_file"]
        noise_file = self.cg_config["noise_interim_file"]

        eg_cg = pd.read_csv(eq_file, header=None, dtype=str)
        tremor_cg = pd.read_csv(tremor_file, header=None, dtype=str)
        noise_cg = pd.read_csv(noise_file, header=None, dtype=str)

        # 0: event time, 1: lat, 2: lon, 3: dep, 4, mag
        eg_cg[[1, 2, 3, 4]] = eg_cg[[1, 2, 3, 4]].applymap(lambda x: float(x))
        tremor_cg[[1, 2, 3, 4]] = tremor_cg[[1, 2, 3, 4]].applymap(
            lambda x: float(x)
        )
        noise_cg[[1, 2, 3, 4]] = noise_cg[[1, 2, 3, 4]].applymap(
            lambda x: float(x)
        )

        if "eq" in self.file_name:
            eg_cg = self.find_by_condition(eg_cg, [0, 1, 3, 4], value_list)
            event_type = "eq"
        else:
            eg_cg = self.find_by_condition(eg_cg, [0], value_list)

        if "tremor" in self.file_name:
            tremor_cg = self.find_by_condition(tremor_cg, [0, 1], value_list)
            event_type = "tremor"
        else:
            tremor_cg = self.find_by_condition(tremor_cg, [0], value_list)

        if "noise" in self.file_name:
            event_type = "noise"

        noise_cg = self.find_by_condition(noise_cg, [0], value_list)

        print(f"select event_type: {event_type}")

        # Eliminate duplication so that each event does not overlap
        getted_eg_cg = eg_cg[~eg_cg[0].isin(tremor_cg[0])]
        getted_tremor_cg = tremor_cg[~tremor_cg[0].isin(eg_cg[0])]
        getted_noise_cg = noise_cg[~noise_cg[0].isin(tremor_cg[0])]
        getted_noise_cg = getted_noise_cg[~getted_noise_cg[0].isin(eg_cg[0])]

        all_event_cg = {
            "noise": getted_noise_cg,
            "tremor": getted_tremor_cg,
            "eq": getted_eg_cg,
        }

        extracted_cg = all_event_cg[event_type]
        print(extracted_cg)

        return extracted_cg

    def find_by_condition(
        self,
        event_cg: pd.DataFrame,
        indexes: List[int],
        value_list: List[List[Union[str, int]]],
    ) -> pd.DataFrame:
        """
        Search conditions.

        Parameters
        ----------
        event_cg : pd.DataFrame
            catalog data
        indexes: List[int]
            information of event
        value_list: List[Union[str, int]]
            condition index

        Returns
        -------
        event_cg : pd.DataFrame
            catalog data retrieved by condition
        """

        values: List[Union[str, float]]

        for index in indexes:
            if index != 0:
                values = list(map(float, value_list[index]))
            else:
                values = [
                    str(input_time)
                    .replace("/", "")
                    .replace(":", "")
                    .replace(" ", "")
                    for input_time in value_list[index]
                ]

            value1, value2 = values[0], values[1]

            if (
                index == 1
            ):  # If radius is specified, the distance is calculated from lat and lon.
                dis = event_cg.apply(self.calc_dis, axis=1)
                event_cg = event_cg[(value1 <= dis) & (dis <= value2)]
            else:
                event_cg = event_cg[
                    ((value1 <= event_cg[index]) & (event_cg[index] <= value2))
                ]

        return event_cg

    def calc_dis(self, event_info: pd.Series) -> float:
        """
        Calculate the distance between the observation point and the event.

        Parameters
        ----------
        event_info: pd.Series
            information of event

        Returns
        -------
        dis: float
            Distance between observation point and event (km)
        """

        station_loc = (34.453074, 136.312884)  # Latitude and longitude of AY10
        event_loc = (event_info[1], event_info[2])
        dis = geodesic(station_loc, event_loc).km
        return dis

    def get_event_info(self):
        """
        Get information about the event
        """

        self.total_to_move = 0
        self.event_info = self.event_cg.iloc[[self.cg_index], :]
        self.event_time = StrTime(self.event_info.iloc[0, 0])
        self.lat = self.event_info.iloc[0, 1]
        self.lon = self.event_info.iloc[0, 2]
        self.dep = self.event_info.iloc[0, 3]
        self.mag = self.event_info.iloc[0, 4]
        self.chk = self.event_info.iloc[0, 5]

    def next_event(self):
        """
        move to next event.
        """

        if (
            self.cg_index == self.total_events - 1
        ):  # If you are at the end of the catalog, move to the top
            self.cg_index = 0
        else:
            self.cg_index += 1
        self.get_event_info()

    def return_event(self):
        """
        move to previous event.
        """

        if (
            self.cg_index == 0
        ):  # If you are at the beginning of the catalog, move to the end.
            self.cg_index = self.total_events - 1
        else:
            self.cg_index -= 1
        self.get_event_info()

    def check_event(self, unit_to_move: int):
        """
        check event in catalog.

        Parameters
        ----------
        unit_to_move : int
            unit to move time
        """

        date_format = "%Y%m%d%H%M%S"
        dt = datetime.strptime(self.event_time.origin_time, date_format)
        dt = dt + timedelta(seconds=int(self.total_to_move * unit_to_move))
        new_event_time = dt.strftime(date_format)

        if self.event_cg.iloc[self.cg_index, 5] == "1":
            self.total_check -= 1
            self.event_cg.iloc[self.cg_index, 5] = np.nan
        else:
            self.total_check += 1
            self.event_cg.iloc[self.cg_index, 5] = "1"

        self.event_cg.iloc[self.cg_index, 0] = new_event_time
        self.get_event_info()

    def format_event_info(self) -> str:
        """
        Format the event information as a string.

        Returns
        -------
        event_summary : str
            The event information as a string
        """

        event_summary = (
            f"Date:: {self.event_time.convert_time_format()}\n"
            f" -Lat: {self.lat}\n"
            f" -Lon: {self.lon}\n"
            f" -Dep: {self.dep}\n"
            f" -Mag: {self.mag}\n"
            f" -Chk: {self.chk}"
        )
        return event_summary

    def save_catalog(self, stations: List[str]):
        """
        save catalog data.

        Parameters
        ----------
        stations : List[str]
            the name of station
        """

        tmp_catalog = self.event_cg.copy()

        station_names = ""
        for i, station in enumerate(stations):
            if i != 0:
                station_names += ";"
            station_names += station
        tmp_catalog[6] = station_names

        print(f"SaveCatalog: {self.output_file}")
        tmp_catalog = tmp_catalog.sort_values(0)  # sort catalog in event time
        tmp_catalog.to_csv(self.output_file, header=False, index=False)
