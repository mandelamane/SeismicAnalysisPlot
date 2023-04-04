from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import yaml
from pyqtgraph import QtCore, QtWidgets
from pyqtgraph.dockarea import Dock

from catalog import CatalogMonitor
from keypress import KeyPressWindow
from sac import SeismicWaveHandler


class SeismicDataVisualizer:
    """
    A class for visualizing seismic data.
    """

    def __init__(self, config_file: str):
        """
        Parameters
        ----------
        config_file : str
            The path of the config file
        """

        self.load_config(config_file)
        self.set_attributes()
        self.create_handlers()

        # Initialize Application
        self.init_app()

        self.set_plotarea()
        self.setup_linear_region()
        self.update_plotarea()

        self.win.show()

    def load_config(self, config_file: str):
        """
        Load the config file and store it in self.config

        Parameters
        ----------
        config_file : str
            Path to config to set application
        """

        with open(config_file) as file:
            self.config = yaml.safe_load(file.read())

    def set_attributes(self):
        """
        Set the attributes from the app config
        """

        app_config = self.config["app"]
        self.width = app_config["WIDTH"]
        self.height = app_config["HEIGHT"]
        self.stations = app_config["stations"]
        self.spec_range = app_config["spec_range"]
        self.unit_to_move = app_config["unit_to_move"]
        self.cmap = self.setup_cmap(app_config["cmap"])

    def create_handlers(self):
        """
        Create the instances for data processing
        """
        # The instance for processing sac data
        sac_config = self.config["sac"]
        self.sac_handler = SeismicWaveHandler(sac_config)
        # Considering the case when the time crosses the hour boundary
        xlim = 60 + int(self.sac_handler.wave_win)
        self.x_range = (0, xlim)

        # The instance for processing catalog data
        cg_config = self.config["catalog"]
        self.event_cg = CatalogMonitor(cg_config)

    def init_app(self):
        """
        Initialize Application
        """

        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.area = KeyPressWindow()
        self.area.sigKeyPress.connect(self.key_pressed)
        self.win.setCentralWidget(self.area)
        self.win.resize(self.width, self.height)
        self.win.setWindowTitle("Seismic Analysis Plot")
        (
            min_waveform_dock,
            spectrogram_dock,
            hour_waveform_dock,
            envelope_dock,
        ) = self._add_dock()
        self.min_waveform_widgets = self._add_waveform_widget(
            min_waveform_dock
        )
        self.spectrogram_widgets = self._add_spectrogram_widget(
            spectrogram_dock
        )
        self.hour_waveform_widgets = self._add_waveform_widget(
            hour_waveform_dock
        )
        self.envelope_widgets = self._add_envelope_widget(envelope_dock)

    def setup_cmap(self, cmap_name: str, n_colors: int = 11) -> pg.ColorMap:
        """
        Set up the colors to use for the spectrogram.

        Parameters
        ----------
        cmap_name : str
            matplotlib's color map
        n_colors : int
            the number of color

        Returns
        -------
        out : pg.ColorMap
            A color map object for the spectrogram
        """

        cmap = plt.get_cmap(cmap_name)

        colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
        colors = [
            (int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            for r, g, b, a in colors
        ]

        cmap = pg.ColorMap(pos=np.linspace(0, 1.0, n_colors), color=colors)
        return cmap

    def _add_dock(self) -> Tuple[Dock, Dock, Dock, Dock]:
        """
        Add dock to area.

        Returns
        -------
        min_wave_dock : Dock
            Dock of waveform in minutes
        spectrogram_dock : Dock
            Dock of spectrogram
        hour_wave_dock : Dock
            Dock of waveform in hour
        envelope_dock : Dock
            Dock of envelope
        """

        # Calculate the width and height of each dock
        dock_width = int(self.width * 0.5)
        dock_height = int(self.height * 0.34)

        min_waveform_dock = Dock(
            "Extracted Seismic Waveform",
            size=(dock_width, dock_height),
            autoOrientation=False,
        )
        spectrogram_dock = Dock(
            "Spectrogram",
            size=(dock_width, dock_height),
            autoOrientation=False,
        )
        hour_waveform_dock = Dock(
            "Seismic Waveform",
            size=(dock_width, dock_height),
            autoOrientation=False,
        )
        envelope_dock = Dock(
            "Seismic Envelope",
            size=(dock_width, dock_height),
            autoOrientation=False,
        )

        self.area.addDock(min_waveform_dock, "left")
        self.area.addDock(spectrogram_dock, "right", min_waveform_dock)
        self.area.addDock(hour_waveform_dock, "bottom")
        self.area.addDock(envelope_dock, "right", hour_waveform_dock)

        return (
            min_waveform_dock,
            spectrogram_dock,
            hour_waveform_dock,
            envelope_dock,
        )

    def _add_waveform_widget(self, dock: Dock) -> List[pg.PlotWidget]:
        """
        Add waveform widget to dock area.

        Parameters
        ----------
        dock : pg.dockarea.Dock
            the dock for waveform

        Returns
        -------
        widgets : List[pg.PlotWidget]
            The waveform widgets of the dock area
        """

        widgets = []
        for station in self.stations:
            widget = pg.PlotWidget(title=station)
            widget.setLabel(axis="left", text="Velocity [m/s]")
            widget.setLabel(axis="bottom", text="Time [min]")
            widget.setLimits(xMin=self.x_range[0], xMax=self.x_range[1])
            widget.setRange(xRange=self.x_range, yRange=(-1, 1))
            dock.addWidget(widget)
            widgets.append(widget)
        return widgets

    def _add_envelope_widget(
        self,
        dock: pg.dockarea.Dock,
    ) -> List[pg.PlotWidget]:
        """
        Add envelope widget to dock area.

        Parameters
        ----------
        dock : pg.dockarea.Dock
            the dock for envelope

        Returns
        -------
        widgets : List[pg.PlotWidget]
            the waveform widgets of the dock area
        """

        widgets = []
        for station in self.stations:
            widget = pg.PlotWidget(title=station)
            widget.setLabel(axis="bottom", text="Time [min]")
            widget.setLimits(xMin=self.x_range[0], xMax=self.x_range[1])
            widget.setRange(xRange=self.x_range, yRange=(0, 2))
            dock.addWidget(widget)
            widgets.append(widget)
        return widgets

    def _add_spectrogram_widget(self, dock: Dock) -> pg.ImageView:
        """
        Add spectrogram widget to dock area.

        Parameters
        ----------
        dock : Dock
            the dock for spectrogram

        Returns
        -------
        widgets : pg.ImageView
            the widgets for spectrogram
        """

        widgets = []
        for station in self.stations:
            plot_item = pg.PlotItem(title=station)
            widget = pg.ImageView(view=plot_item)
            widget.view.invertY(False)
            widget.view.setLabel(axis="left", text="Frequency [Hz]")
            widget.view.setLabel(axis="bottom", text="Time [min]")
            widget.view.setLimits(
                xMin=self.x_range[0],
                xMax=self.x_range[1],
                yMin=0,
                yMax=self.sac_handler.band_width[1] + 1,
            )
            widget.view.setAspectLocked(False)
            dock.addWidget(widget)
            widgets.append(widget)
        return widgets

    def set_plotarea(self):
        """
        Plot the data on the widgets and display them on the screen
        """

        multi_data = self.sac_handler.multi_make_data(
            self.stations, self.event_cg.event_time
        )

        for i, station in enumerate(self.stations):
            (
                waveform,
                spectrogram,
                envelope,
                amp_scale,
            ) = multi_data[i]
            n_point = waveform.shape[0]

            self.min_waveform_widgets[i].plot(
                np.linspace(self.x_range[0], self.x_range[1], n_point),
                waveform,
            )
            self.min_waveform_widgets[i].plotItem.setTitle(
                f"{station} | {amp_scale:.1e} [m/s]"
            )

            self.spectrogram_widgets[i].setImage(
                spectrogram.T,
                scale=self.sac_handler.img_shape,
            )

            self.spectrogram_widgets[i].setLevels(
                self.spec_range[0], self.spec_range[1]
            )

            self.spectrogram_widgets[i].autoHistogramRange()
            self.spectrogram_widgets[i].setColorMap(self.cmap)

            self.hour_waveform_widgets[i].plot(
                np.linspace(self.x_range[0], self.x_range[1], n_point),
                waveform,
            )
            self.hour_waveform_widgets[i].plotItem.setTitle(
                f"{station} | {amp_scale:.1e} [m/s]"
            )

            self.envelope_widgets[i].plot(
                np.linspace(self.x_range[0], self.x_range[1], n_point),
                envelope,
                pen="g",
            )

            self.envelope_widgets[i].plotItem.setTitle(f"{station}")

    def setup_linear_region(self):
        """
        Link the region and the graph.
        """

        event_start, event_end = self.get_event_time()

        self.lr_w_list = []
        self.lr_e_list = []
        for i in range(len(self.min_waveform_widgets)):
            # Add region to waveform widget
            lr_w = pg.LinearRegionItem(
                [event_start, event_end],
                bounds=self.x_range,
                brush=pg.mkBrush("#f004"),
            )
            # Add region to envelope widget
            lr_e = pg.LinearRegionItem(
                [event_start, event_end],
                bounds=self.x_range,
                brush=pg.mkBrush("#f004"),
            )
            lr_w.setZValue(3)
            lr_e.setZValue(3)

            if i != 0:
                lr_w.setMovable(False)
                lr_e.setMovable(False)

            self.hour_waveform_widgets[i].addItem(lr_w)
            self.envelope_widgets[i].addItem(lr_e)

            self.lr_w_list.append(lr_w)
            self.lr_e_list.append(lr_e)

        self.lr_w_list[0].sigRegionChanged.connect(self.link_region)

    def link_region(self):
        """
        Update the display area of each widget according to the region range.
        """

        region_range = self.lr_w_list[0].getRegion()
        for i, (min_waveform_widgets, spectrogram_widgets) in enumerate(
            zip(self.min_waveform_widgets, self.spectrogram_widgets)
        ):
            min_waveform_widgets.setXRange(
                *self.lr_w_list[0].getRegion(), padding=0
            )
            spectrogram_widgets.view.setXRange(
                *self.lr_w_list[0].getRegion(), padding=0
            )

            if i != 0:
                self.lr_w_list[i].setRegion(region_range)

            self.lr_e_list[i].setRegion(region_range)

    def update_plotarea(self):
        """
        Clear and replot the plot area of each widget according to
        the selected time range
        """

        event_start, event_end = self.get_event_time()

        for i in range(len(self.min_waveform_widgets)):
            self.min_waveform_widgets[i].clear()
            self.spectrogram_widgets[i].clear()
            self.hour_waveform_widgets[i].clear()
            self.envelope_widgets[i].clear()
            self.lr_w_list[i].setRegion([event_start, event_end])
            self.lr_e_list[i].setRegion([event_start, event_end])
            self.hour_waveform_widgets[i].addItem(self.lr_w_list[i])
            self.envelope_widgets[i].addItem(self.lr_e_list[i])

        self.set_plotarea()
        self.link_region()
        self.show_console()

    def get_event_time(self) -> Tuple[float, float]:
        """
        Calculate the start and end times of the event.

        Returns
        -------
        event_start : float
            start time of the event (in minutes)
        event_end : float
            end time of the event (in minutes)
        """

        event_start = (
            float(self.event_cg.event_time.minutes)
            + float(self.event_cg.event_time.sec) / 60.0
        )
        event_end = event_start + float(self.sac_handler.wave_win)

        return event_start, event_end

    def show_console(self):
        """
        Print the event information and the summary to the console.
        """

        event_summary = self.event_cg.format_event_info()
        message = (
            "############################\n"
            f"{event_summary}\n"
            f"EventSum:: {self.event_cg.total_check}/{self.event_cg.total_events}\n"
            "############################"
        )
        print(message)

    def return_action(self):
        """
        show the previous event.
        """

        self.event_cg.return_event()
        self.update_plotarea()

    def next_action(self):
        """
        show the next event.
        """

        self.event_cg.next_event()
        self.update_plotarea()

    def quit_action(self):
        """
        quit application.
        """

        self.save_action()
        self.app.closeAllWindows()

    def check_action(self):
        """
        check the event.
        """

        self.event_cg.check_event(self.unit_to_move)
        self.show_console()

    def save_action(self):
        """
        save the catalog.
        """

        with open(self.event_cg.output_file.replace("csv", "yaml"), "w") as yf:
            yaml.dump(
                self.config, yf, default_flow_style=False, sort_keys=False
            )

        self.event_cg.save_catalog(self.stations)

    def change_region_range(self, part: str):
        """
        change the region range.

        Parameters
        ----------
        part : str
            region range（"all" or "part"）
        """
        if part == "all":
            part_range = tuple(map(float, self.x_range))
        elif part == "part":
            self.event_cg.total_to_move = 0
            part_range = self.get_event_time()
        else:
            raise ValueError(f"Invalid value for part: {part}")

        for i, (min_waveform_widgets, spectrogram_widgets) in enumerate(
            zip(self.min_waveform_widgets, self.spectrogram_widgets)
        ):
            min_waveform_widgets.setXRange(*part_range, padding=0)
            spectrogram_widgets.view.setXRange(*part_range, padding=0)
            self.lr_w_list[i].setRegion(part_range)
            self.lr_e_list[i].setRegion(part_range)

    def move_region(self, direction: str):
        """
        move the region.

        Parameters
        ----------
        direction : str
            direction to move（"left" or "right"）
        """

        part_range = self.get_event_time()

        if direction == "right":
            check_min = (
                part_range[1]
                + (self.event_cg.total_to_move + 1) * self.unit_to_move / 60.0
            )
            if check_min < self.x_range[1]:
                self.event_cg.total_to_move += 1
        elif direction == "left":
            check_min = (
                part_range[0]
                + (self.event_cg.total_to_move - 1) * self.unit_to_move / 60.0
            )
            if check_min > self.x_range[0]:
                self.event_cg.total_to_move -= 1
        else:
            raise ValueError(f"Invalid value for direction: {direction}")

        new_part_range = [
            part + self.event_cg.total_to_move * self.unit_to_move / 60.0
            for part in part_range
        ]

        for min_waveform_widgets, spectrogram_widgets, lr_w, lr_e in zip(
            self.min_waveform_widgets,
            self.spectrogram_widgets,
            self.lr_w_list,
            self.lr_e_list,
        ):
            min_waveform_widgets.setXRange(*new_part_range, padding=0)
            spectrogram_widgets.view.setXRange(*new_part_range, padding=0)
            lr_w.setRegion(new_part_range)
            lr_e.setRegion(new_part_range)

    def key_pressed(self, ev: QtCore.Qt.Key):
        """
        if you press keyboard.

        Parameters
        ----------
        ev : QtCore.Qt.Key
            key event
        """

        if ev.key() == QtCore.Qt.Key.Key_Down:
            self.next_action()
        elif ev.key() == QtCore.Qt.Key.Key_Up:
            self.return_action()
        elif ev.key() == QtCore.Qt.Key.Key_A:
            self.check_action()
        elif ev.key() == QtCore.Qt.Key.Key_Space:
            self.save_action()
        elif ev.key() == QtCore.Qt.Key.Key_Q:
            self.quit_action()
        elif ev.key() == QtCore.Qt.Key.Key_P:
            self.change_region_range("part")
        elif ev.key() == QtCore.Qt.Key.Key_O:
            self.change_region_range("all")
        elif ev.key() == QtCore.Qt.Key.Key_Left:
            self.move_region("left")
        elif ev.key() == QtCore.Qt.Key.Key_Right:
            self.move_region("right")

    def run(self):
        """
        exec application.
        """
        self.app.exec()


def main():
    config_file = "config/configure.yaml"
    app = SeismicDataVisualizer(config_file)
    app.run()


if __name__ == "__main__":
    main()
