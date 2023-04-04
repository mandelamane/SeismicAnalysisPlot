import datetime
import math
import multiprocessing
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from obspy import read
from obspy.signal.filter import bandpass
from scipy import signal

from strtime import StrTime


class SeismicWaveHandler:
    """
    A class for handling seismic wave data, such as reading, preprocessing, filtering,
    and extracting features.
    """

    def __init__(self, sac_config: Dict[str, Any]):
        """
        Parameters
        ----------
        sac_config : Dict[str, Any]
            the parameters for the SAC data processing
        """

        self.cpu_num = multiprocessing.cpu_count()

        self.amp_scale = sac_config["amp_scale"]
        self.sac_dir = sac_config["sac_dir"]
        self.component = sac_config["component"]
        self.sampling_rate = sac_config["sampling_rate"]
        self.band_width = sac_config["band_width"]
        self.fft_win = sac_config["fft_win"]
        self.wave_win = sac_config["wave_win"]
        self.overlap_rate = sac_config["overlap_rate"]

        self.nseg = self.sampling_rate * self.fft_win
        self.noverlap = self.nseg * self.overlap_rate
        self.nfft = int(2 ** len(bin(self.nseg)[2:]))

        # Number of pixels on horizontal axis
        self.overlap_bin = (
            self.sampling_rate * 60 * (60 + self.wave_win) - self.noverlap
        ) // (self.nseg - self.noverlap) + 1

        self.freq_range = np.fft.fftfreq(self.nfft, 1 / self.sampling_rate)[
            : math.ceil(self.nfft / 2)
        ]

        img_h = self.band_width[1] / len(
            self.freq_range[self.freq_range <= self.band_width[1]]
        )
        img_w = (60 + self.wave_win) / self.overlap_bin

        # image scale
        self.img_shape = (img_w, img_h)

    def preprocessing_waveform(
        self, station: str, event_time: StrTime
    ) -> np.ndarray:
        """
        read and preprocessing waveform

        Parameters
        ----------
        station : str
            the name of station
        event_time : StrTime
            event start time

        Returns
        -------
        sac_data : array_like
            waveform data
        """

        sac_file1 = os.path.join(
            self.sac_dir,
            event_time.year,
            event_time.get_time_unit("hour"),
            f"{station}.{self.component}",
        )
        sac_data1, sr1 = self.read_sac(sac_file1)

        if sac_data1 is not None:
            n_point = int(60 * sr1 * self.wave_win)

            add_time = datetime.datetime.strftime(
                datetime.datetime.strptime(
                    event_time.get_time_unit("hour"), "%Y%m%d%H"
                )
                + datetime.timedelta(hours=1),
                "%Y%m%d%H",
            )
            add_event_time = StrTime(add_time)

            sac_file2 = os.path.join(
                self.sac_dir,
                add_event_time.year,
                add_event_time.origin_time,
                f"{station}.{self.component}",
            )

            sac_data2, _ = self.read_sac(sac_file2)

            if sac_data2 is not None:
                sac_data2 = sac_data2[:n_point]
            else:
                sac_data2 = np.zeros(n_point)

            sac_data = np.append(sac_data1, sac_data2)

            if sr1 != self.sampling_rate:
                sac_data = self.decimate_sac(
                    sac_data, int(sr1 // self.sampling_rate)
                )

            sac_data = self.bandpass_sac(sac_data)
            sac_data = self.detrend_sac(sac_data)
        else:
            print(f"FileNotFoundWarning: no such file: {sac_file1}")
            sac_data = np.zeros(
                int(60 * (60 + self.wave_win) * self.sampling_rate)
            )

        return sac_data

    def read_sac(self, sac_file: str) -> Tuple[Union[np.ndarray, None], float]:
        """
        read sac file.

        Parameters
        ----------
        sac_file : str
            the path of sac file.

        Returns
        -------
        sac_data : array_like or None
            waveform data about sac file
        sr : float
            sampling rate
        """

        sac_data = None
        sr = 0.0

        try:
            sac = read(sac_file)[0]
            sr = sac.stats.sampling_rate
        except FileNotFoundError:
            print(f"FileNotFoundWarning: no such file {sac_file}")
        except TypeError:
            print(f"FormatWarning: {sac_file} is missing data")
        except Exception as e:
            print(f"ExceptionWarning: {e}: {sac_file}")
        else:
            if sac.data.shape[0] != int(3600 * sr):
                print(f"MissingValue Warning: exists missing value {sac_file}")
            else:
                sac_data = sac.data
        finally:
            return sac_data, sr

    def bandpass_sac(self, sac_data: np.ndarray) -> np.ndarray:
        """
        Apply band-pass filter to waveform data.

        Parameters
        ----------
        sac_data : np.ndarray
            waveform data

        Returns
        -------
        bd_sac_data : array_like
            Band-passed waveform dat
        """

        bd_sac_data = bandpass(
            sac_data,
            df=self.sampling_rate,
            freqmin=self.band_width[0],
            freqmax=self.band_width[1],
            zerophase=True,
        )
        return bd_sac_data

    def decimate_sac(self, sac_data: np.ndarray, factor: int) -> np.ndarray:
        """
        Decimate waveform data.

        Parameters
        ----------
        sac_data : np.ndarray
            waveform data
        factor : int
            scale factor to decimate

        Returns
        -------
        dc_sac_data : array_like
            decimated waveform data.
        """

        dc_sac_data = signal.decimate(sac_data, factor)
        return dc_sac_data

    def detrend_sac(self, sac_data: np.ndarray) -> np.ndarray:
        """
        remove trend in waveform

        Parameters
        ----------
        sac_data : np.ndarray
            waveform data

        Returns
        -------
        dc_sac_data : array_like
            waveform data
        """

        dc_sac_data = sac_data - np.mean(sac_data)
        return dc_sac_data

    def envelope_sac(self, sac_data: np.ndarray, sec: int = 6) -> np.ndarray:
        """
        calculation envelope data.

        Parameters
        ----------
        sac_data : np.ndarray
            waveform data
        sec : int, default=6
            moving window

        Returns
        -------
        envelope : array_like
            envelope data
        """

        sr = self.sampling_rate
        win = np.ones(sec * sr) / sec * sr
        envelope = np.sqrt(np.convolve(sac_data**2, win, mode="same"))
        return envelope

    def spectrogram_sac(self, sac_data: np.ndarray) -> np.ndarray:
        """
        Calculate spectrogram data.

        Parameters
        ----------
        sac_data : np.ndarray
            waveform data

        Returns
        -------
        pxx : array_like
            spectrogram data
        """

        # the type of zxx is complex
        _, _, zxx = signal.stft(
            sac_data,
            fs=self.sampling_rate,
            nperseg=self.nseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            window=self.apply_cos_taper(np.ones(self.nseg)),
        )

        pxx = self.convert_psd(zxx)
        return pxx

    def apply_cos_taper(self, x: np.ndarray) -> np.ndarray:
        """
        Apply cosine taper to waveform.

        Parameters
        ----------
        x : np.ndarray
            waveform data.

        Returns
        -------
        x : array_like
            cosine tapered waveform
        """

        n = x.shape[0]
        ntaper = int(n * 0.05)
        index = n - ntaper
        omega = np.pi / float(ntaper)
        factor = 0.5 - 0.5 * np.cos(omega * np.arange(0, ntaper, 1))
        x[:ntaper] *= factor
        x[index:] *= factor[::-1]
        return x

    def convert_psd(self, zxx: np.ndarray) -> np.ndarray:
        """
        Convert complex to power spectre density.

        Parameters
        ----------
        zxx : np.ndarray
            spectrogram data.

        Returns
        -------
        pxx : array_like
            psd's spectrogram data.
        """

        # To avoid division by zero.
        epsilon = 1e-31
        max_freq = self.freq_range[
            self.freq_range <= self.band_width[1]
        ].shape[0]
        pxx = np.log10(np.abs(zxx) ** 2 + epsilon)
        pxx = pxx[:max_freq, :]

        return pxx

    def multi_make_data(
        self, stations: str, event_time: StrTime
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Create waveform data, spectrogram, and envelope by parallel processing.

        Parameters
        ----------
        station : str
            the name of station
        event_time : StrTime
            start event time

        Returns
        -------
        multi_result : List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]
            created data by parallel processing.
        """

        multi_values = [(station, event_time) for station in stations]
        with multiprocessing.Pool(self.cpu_num) as pool:
            multi_results = pool.map(self.make_data, multi_values)
        return multi_results

    def make_data(
        self, inputs: Tuple[str, StrTime]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Create waveform data, spectrogram, and envelope

        Parameters
        ----------
        inputs: Tuple[str, StrDateTime]
            the name of station、start event time

        Returns
        -------
        result : Tuple[array_like, array_like, array_like, float]
            created data
        """

        station, event_time = inputs

        sac_data = self.preprocessing_waveform(station, event_time)
        envelope = self.envelope_sac(sac_data)

        # To make the application run faster
        dc_sac_data = self.decimate_sac(sac_data, factor=5)
        wave_scale = (
            np.sqrt(np.percentile(dc_sac_data**2, 75)) * self.amp_scale
        )

        dc_sac_data /= wave_scale + 1e-20

        dc_envelope = self.decimate_sac(envelope, factor=5)
        dc_envelope -= np.min(dc_envelope)
        envelope_scale = np.percentile(dc_envelope, 75) * 2
        dc_envelope /= envelope_scale + 1e-20

        spectrogram = self.spectrogram_sac(sac_data)
        result = (dc_sac_data, spectrogram, dc_envelope, wave_scale)

        return result
