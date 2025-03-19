import abc
import numpy as np
import tensorflow as tf

from fatigue_classifier.fatigue_classifier.fatigue_model import FatigueClassifier
from typing import final, overload, TypeVar, Generic
from collections import deque
from geometry_msgs.msg import Vector3

_SMALL_VALUE = 1e-9
_T = TypeVar("_T")

def clamp(x, _min: float, _max: float):
    return np.clip(x, _min, _max)

def map(x, in_min: float, in_max: float, out_min: float, out_max: float):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def ffc(x, raw_buf, ffc_offset):
    return abs(x - (raw_buf[-ffc_offset] if ffc_offset <= len(raw_buf) else 0.0))

class Preprocessor(abc.ABC, Generic[_T]):
    def __init__(self, fs: float, max_duration_sec: float, window_size_sec: float, stride_sec: float, mask_value: float = 0.0):
        super().__init__()

        self._fs = fs
        self._max_duration: int = int(self._fs*max_duration_sec)
        self._window = int(self._fs*window_size_sec)
        self._stride = int(self._fs*stride_sec)

        self._mask_value = mask_value
        
        self._preprocessed_data: tf.Tensor = tf.expand_dims(
            np.ones(
                FatigueClassifier.input_signal_shape(
                    self._fs, window_size_sec, stride_sec, max_duration_sec
                )
            ) * mask_value, axis=0
        )

        self._iter = 0
        self._row = 0
        self._column = 0

    @final
    def preprocess(self, sample: _T):
        # Get filtered value
        x = self._filter_data(sample)

        if x == self._mask_value: 
            # Change value slightly to not get masked
            x += _SMALL_VALUE

        # Initialize iterators at the starting row/column
        i = self._row
        j = self._column
        while i < self._preprocessed_data.shape[1] and j >= 0:
            # Continue populating our tensor with preprocessed data vectors,
            # where each row is time-shifted from the previous row by the stride
            #
            # An efficient way to do this is to populate all the rows immediately
            # as the data comes in, so no additional transformations have to be made
            
            # Copy the data point
            self._preprocessed_data[0][i][j] = x

            # Increment to the next row
            i += 1

            # Subtract by the stride
            j -= self._stride
            
        self._iter += 1
        self._column += 1

        if self._iter == self._window or \
            (self._iter > self._window and \
             ((self._iter - self._window) % self._stride) == 0):
            self._row += 1
            self._column -= self._stride

    @property
    @final
    def data(self) -> tf.Tensor:
        """
        Provide model pre-processed tensor

        Returns:
            tf.Tensor: Tensor of shape (1, W, N), 
            where W=window size and N=samples/window
        """
        # TODO(george): copy?
        return self._preprocessed_data
    
    @abc.abstractmethod
    def _filter_data(self, sample: _T) -> np.float32:
        ...

from idl_definitions.msg import (
    MindroveArmBandEightChannelMsg,
    PluxMsg
)

class EMGPreprocessor(Preprocessor[MindroveArmBandEightChannelMsg]):
    def __init__(self, fs, max_duration_sec, window_size_sec, stride_sec, mask_value = 0):
        super().__init__(fs, max_duration_sec, window_size_sec, stride_sec, mask_value)

        self._channel_raw_queues = [
            deque(maxlen=88) for _ in range(8)
        ]

        self._channel_filtered_queues = [
            deque(maxlen=88) for _ in range(8)
        ]

        self._ffc_offset = int(fs // 60)
        self._sample_index: int = None

    def set_sample_index(self, index: int):
        self._sample_index = index

    @overload
    def _filter_data(self, sample):
        assert self._sample_index is not None, \
            "Must set sample index prior to calling preprocess function"
        assert self._sample_index < sample.num_emg_samples, \
            f"Sample index exceeds number of EMG samples ({self._sample_index} < {sample.num_emg_samples})"
        
        for i in range(8):
            # Iterate across 8 EMG channels
            for dp in sample.__getattribute__(f"c{i+1}"):
                # Apply FFC filter + full-wave rectification
                # The low-pass filter (moving average) will be 
                # applied at the point of analysis

                # TODO(george): review
                # self._channel_filtered_queues[i].append(
                #     abs(dp[self._sample_index] - (self._channel_raw_queues[i][-self._ffc_offset] if self._ffc_offset <= len(self._channel_raw_queues[i]) else 0.0))
                # )
                self._channel_filtered_queues[i].append(
                    ffc(dp[self._sample_index], 
                        self._channel_raw_queues[i], 
                        self._ffc_offset)
                )

                self._channel_raw_queues[i].append(
                    map(clamp(dp[self._sample_index], -5.0, 5.0), -5.0, 5.0, -1.0, 1.0)
                )

        # Reset sample index
        self._sample_index = None

        return clamp(
            np.average([np.average(x) for x in self._channel_filtered_queues]),
            0.0,
            1.0
        )
    
class AccelerometerPreprocessor(Preprocessor[MindroveArmBandEightChannelMsg]):
    def __init__(self, fs, max_duration_sec, window_size_sec, stride_sec, mask_value = 0):
        super().__init__(fs, max_duration_sec, window_size_sec, stride_sec, mask_value)
        self._accel_channels = [deque(maxlen=8) for _ in range(3)]

    def set_sample_index(self, index: int):
        self._sample_index = index

    @overload
    def _filter_data(self, sample):
        assert self._sample_index is not None, \
            "Must set sample index prior to calling preprocess function"
        assert self._sample_index < sample.num_imu_samples, \
            f"Sample index exceeds number of IMU samples ({self._sample_index} < {sample.num_imu_samples})"
        
        accel: Vector3 = sample.accel[self._sample_index]
        for i,channel in enumerate([accel.x, accel.y, accel.z]):
            self._accel_channels[i].append(
                    map(
                    clamp(channel, -10, 10),
                    -10, 10, -1, 1
                )
            )

        # Reset sample index
        self._sample_index = None

        return np.linalg.norm(
            [np.average(x) for x in self._accel_channels], ord=2
        ) - 1.0
    
class GyroscopePreprocessor(Preprocessor[MindroveArmBandEightChannelMsg]):
    def __init__(self, fs, max_duration_sec, window_size_sec, stride_sec, mask_value = 0):
        super().__init__(fs, max_duration_sec, window_size_sec, stride_sec, mask_value)
        self._accel_channels = [deque(maxlen=8) for _ in range(3)]

    def set_sample_index(self, index: int):
        self._sample_index = index

    @overload
    def _filter_data(self, sample):
        assert self._sample_index is not None, \
            "Must set sample index prior to calling preprocess function"
        assert self._sample_index < sample.num_imu_samples, \
            f"Sample index exceeds number of IMU samples ({self._sample_index} < {sample.num_imu_samples})"
        
        accel: Vector3 = sample.gyro[self._sample_index]
        for i,channel in enumerate([accel.x, accel.y, accel.z]):
            self._accel_channels[i].append(
                    map(
                    clamp(channel, -10, 10),
                    -10, 10, -1, 1
                )
            )

        # Reset sample index
        self._sample_index = None

        return np.linalg.norm(
            [np.average(x) for x in self._accel_channels], ord=2
        )
    
from scipy.signal import find_peaks
class ECGPreprocessor(Preprocessor[PluxMsg]):
    def __init__(self, fs, max_duration_sec, window_size_sec, stride_sec, mask_value = 0, ecg_window_sec: float = 5.0, min_peaks_for_initial_estimate: int = 5):
        super().__init__(fs, max_duration_sec, window_size_sec, stride_sec, mask_value)
        
        self._ffc_offset = int(fs // 60)
        
        self._ecg_raw_queue = deque(maxlen=176)
        self._ecg_preprocessed_queue = deque(maxlen=176)
        self._ecg_filtered_queue = deque(maxlen=int(fs*min_peaks_for_initial_estimate))
        self._bpm_queue = deque(maxlen=176)
        self._initial_bpm_found = False

        self._ecg_window_sec = ecg_window_sec
        self._min_peaks_for_initial_estimate = min_peaks_for_initial_estimate

    def _filter_data(self, sample):
        self._ecg_preprocessed_queue.append(
            ffc(sample.ecg, self._ecg_raw_queue, self._ffc_offset)
        )

        self._ecg_filtered_queue.append(
            np.average(self._ecg_preprocessed_queue)
        )

        peaks, _ = find_peaks(self._ecg_filtered_queue, distance = self._fs / 4, height = np.max(self._ecg_filtered_queue)/2)
        
        if len(peaks) >= self._min_peaks_for_initial_estimate:
            x = (1/np.average(np.diff(peaks))*self._fs*60)
        elif len(self._bpm_queue) > 0:
            # TODO(george): warning
            x = self._bpm_queue[-1]
        else:
            # TODO(george): error?
            return 0.0

        self._bpm_queue.append(clamp(x, 30.0, 220.0), 30.0, 220.0, 0.0, 1.0)
        
        # Moving average of BPM
        return np.average(self._bpm_queue)