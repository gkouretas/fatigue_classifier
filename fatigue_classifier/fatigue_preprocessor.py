import abc
import numpy as np
import tensorflow as tf

from fatigue_model import (
    FatigueClassifier, 
    InputArgs,
    CompiledInputShape
)

from typing import final, TypeVar, Generic
from collections import deque
from geometry_msgs.msg import Vector3
from numpy.typing import NDArray

_SMALL_VALUE = 1e-9
_T = TypeVar("_T")

# Some helper functions
# TODO(george): use python_utils for these

def clamp(x, _min: float, _max: float):
    return np.clip(x, _min, _max)

def map(x, in_min: float, in_max: float, out_min: float, out_max: float):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def ffc(x, raw_buf, ffc_offset):
    return abs(x - (raw_buf[-ffc_offset] if ffc_offset <= len(raw_buf) else 0.0))

class Preprocessor(abc.ABC, Generic[_T]):
    """
    Abstract base class for a fatigue model datastream.

    This base class provides references for preprocessing data, which automatically reshapes the data stream to
    the required shape of the model's input, as well as data getter and reset methods

    Args:
        input_shape (`InputArgs` | `CompiledInputShape`): Input shape metadata for datastream to the model.
        Can be inputted as either `InputArgs`, which contains the necessary metadata to compile the input shape or a `CompiledInputShape`
        directly.
        mask_value (`float`, optional): Masked value to apply to unpopulated data. Defaults to 0.0.
        resampling_factor (`float` | `None`, optional): Upsampling factor to be applied directly to the data. This should be used if the sampling frequency differs from what your model was compiled to use.
        Defaults to None.

    Raises:
        TypeError: Raises a type error if input_shape is an invalid type.
    """
    @classmethod
    def from_input_args(cls, input_args, **kwargs):
        cls(input_shape=FatigueClassifier.input_signal_shape(input_args), **kwargs)

    def __init__(self, input_shape: CompiledInputShape, mask_value: float = 0.0, resampling_factor: float | None = None):
        super().__init__()
        
        self._input_shape = input_shape
        self._mask_value = mask_value

        print(self.__class__.__name__, self._input_shape.get("shape"))

        self._preprocessed_data: NDArray = np.expand_dims(
            np.ones(
                self._input_shape.get("shape")
            ) * mask_value, axis=0
        )

        if resampling_factor < 1:
            self._upsample = False
            self._resampling_factor = int(1/resampling_factor)
        elif resampling_factor > 1:
            self._upsample = True
            self._resampling_factor = int(resampling_factor)
        else:
            self._upsample = self._resampling_factor = None

        self._iter = 0
        self._row = 0
        self._column = 0
        self._last_sample: np.float32 = None

    @final
    def preprocess(self, sample: _T):
        # Get filtered value
        x = self._filter_data(sample)

        if x == self._mask_value: 
            # TODO(george): log?
            # Change value slightly to not get masked
            x += _SMALL_VALUE

        if self._upsample is None:
            self._insert_sample(x)
        elif self._upsample:
            if self._last_sample is None:
                for _ in range(self._resampling_factor):
                    # Insert N duplicates of the sample
                    self._insert_sample(x)
            else:
                for sample in reversed(np.linspace(
                    x,
                    self._last_sample,
                    self._resampling_factor,
                    endpoint=False
                )):
                    # Linearly interpolate N samples from 
                    # last sample -> current sample
                    self._insert_sample(sample)
        else:
            if ((self._iter+1) % self._resampling_factor) == 0:
                self._insert_sample(x)

    @final
    def _insert_sample(self, x: np.float32) -> None:
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
            j -= self._input_shape.get("stride")
            
        self._iter += 1
        self._column += 1

        if self._iter == self._input_shape.get("window") or \
            (self._iter > self._input_shape.get("window") and \
             ((self._iter - self._input_shape.get("window")) % self._input_shape.get("stride")) == 0):
            # TODO(george): ensure row never exceeds the shape???
            self._row += 1
            self._column -= self._input_shape.get("stride")

        self._last_sample = x

    @property
    @final
    def data(self) -> tf.Tensor:
        """
        Provide model pre-processed tensor

        Returns:
            tf.Tensor: Tensor of shape (1, W, N), 
            where W=window size and N=samples/window
        """
        return tf.convert_to_tensor(self._preprocessed_data, dtype=tf.float32)
    
    @property
    @final
    def timestep(self):
        return self._row
    
    @final
    def reset(self) -> None:
        # Reset tensor
        self._preprocessed_data = np.expand_dims(
            np.ones(
                self._input_shape.get("shape")
            ) * self._mask_value, axis=0
        )
    
    @abc.abstractmethod
    def _filter_data(self, sample: _T) -> np.float32:
        ...

from idl_definitions.msg import (
    MindroveArmBandEightChannelMsg,
    PluxMsg
)

class EMGPreprocessor(Preprocessor[MindroveArmBandEightChannelMsg]):
    def __init__(self, input_shape: InputArgs | CompiledInputShape, mask_value = 0, resampling_factor = None):
        super().__init__(input_shape, mask_value, resampling_factor)
        self._channel_raw_queues = [
            deque(maxlen=88) for _ in range(8)
        ]

        self._channel_filtered_queues = [
            deque(maxlen=88) for _ in range(8)
        ]

        self._ffc_offset = int(self._input_shape.get("fs") // 60)
        self._sample_index: int = None

    def set_sample_index(self, index: int):
        self._sample_index = index

    
    def _filter_data(self, sample):
        assert self._sample_index is not None, \
            "Must set sample index prior to calling preprocess function"
        assert self._sample_index < sample.num_emg_samples, \
            f"Sample index exceeds number of EMG samples ({self._sample_index} < {sample.num_emg_samples})"
        
        for i in range(8):
            # Iterate across 8 EMG channels
            dp = sample.__getattribute__(f"c{i+1}")

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
    def __init__(self, input_shape: InputArgs | CompiledInputShape, mask_value = 0, resampling_factor = None):
        super().__init__(input_shape, mask_value, resampling_factor)

        self._accel_channels = [deque(maxlen=8) for _ in range(3)]

    def set_sample_index(self, index: int):
        self._sample_index = index

    
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
    def __init__(self, input_shape: InputArgs | CompiledInputShape, mask_value = 0, resampling_factor = None):
        super().__init__(input_shape, mask_value, resampling_factor)
        
        self._accel_channels = [deque(maxlen=8) for _ in range(3)]

    def set_sample_index(self, index: int):
        self._sample_index = index

    
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
    def __init__(self, input_shape: InputArgs | CompiledInputShape, mask_value = 0, resampling_factor = None, ecg_window_sec: float = 5.0, min_peaks_for_initial_estimate: int = 5):
        super().__init__(input_shape, mask_value, resampling_factor)    
        self._fs = self._input_shape.get("fs")
        self._ffc_offset = int(self._fs // 60)
        
        # TODO: compute from fs
        self._ecg_raw_queue = deque(maxlen=176)
        self._ecg_preprocessed_queue = deque(maxlen=176)
        self._ecg_filtered_queue = deque(maxlen=int(self._fs*min_peaks_for_initial_estimate))
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