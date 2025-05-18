import tensorflow as tf
import threading
import rclpy

from rclpy.node import Node
from pathlib import Path

from fatigue_model import FatigueClassifier
from fatigue_preprocessor import (
    Preprocessor,
    EMGPreprocessor,
    AccelerometerPreprocessor,
    GyroscopePreprocessor,
    ECGPreprocessor
)

from fatigue_classifier_configs import *

from idl_definitions.msg import (
    ExerciseState,
    FatigueState,
    MindroveArmBandEightChannelMsg,
    PluxMsg,
    UserInputMsg
)

from fatigue_experiment_control.fatigue_experiment_control_configs import (
    FATIGUE_EXPERIMENT_CONTROL_TOPIC_NAME,
    FATIGUE_QOS_PROFILE
)

from python_utils.ros2_utils.comms.node_manager import get_realtime_qos_profile

from typing import TypeVar

_T = TypeVar("_T", bound=Preprocessor)

class RealtimeFatigueClassifier:
    def __init__(self, node: Node, path: Path):
        self._node = node
        self._model = FatigueClassifier.from_file(path)
        self._lock = threading.Lock()
        self._ordered_preprocessors: list[_T | None] = [None for _ in range(4)]

        try:
            # Data pre-processors
            # TODO: what is the sampling rate???
            self._emg_preprocessor = self._init_preprocessor(
                fs=500.0,
                name_substr="emg",
                preprocessor_cls=EMGPreprocessor
            )

            self._accelerometer_preprocessor = self._init_preprocessor(
                fs=50.0,
                name_substr="accel",
                preprocessor_cls=AccelerometerPreprocessor
            )

            self._gyroscope_preprocessor = self._init_preprocessor(
                fs=50.0,
                name_substr="gyro",
                preprocessor_cls=GyroscopePreprocessor
            )

            self._ecg_preprocessor = self._init_preprocessor(
                fs=PLUX_SAMPLING_FREQUENCY,
                name_substr="hr",
                preprocessor_cls=ECGPreprocessor
            )
        except ValueError as e:
            self._node.get_logger().error(f"Failed to initialize preprocessors (exception: {e})", exc_info=True)
            raise

        assert all(self._ordered_preprocessors)

        self._ready = False
        self._last_prediction: float = 0.0
        self._current_timestep: int = 0

        self._user_status: UserInputMsg | None = None

        # Sampling rate = 500 Hz for EMG and 50 Hz for IMU
        self._mindrove_frame_counter = 0
        self._mindrove_imu_decimation = 10

        # Subscribers
        self._mindrove_subscriber = self._node.create_subscription(
            MindroveArmBandEightChannelMsg,
            MINDROVE_ROS_TOPIC_NAME,
            self.update_armband_info,
            get_realtime_qos_profile()
        )

        self._plux_subscriber = self._node.create_subscription(
            PluxMsg,
            PLUX_ROS_TOPIC_NAME,
            self.update_plux_info,
            get_realtime_qos_profile()
        )

        self._user_input_subscriber = self._node.create_subscription(
            UserInputMsg,
            USER_INPUT_TOPIC_NAME,
            self.update_sensed_fatigue,
            get_realtime_qos_profile()
        )

        self._exercise_state_subscriber = self._node.create_subscription(
            ExerciseState,
            FATIGUE_EXPERIMENT_CONTROL_TOPIC_NAME,
            self.update_exercise_state,
            get_realtime_qos_profile()
        )

        self._fatigue_publisher = self._node.create_publisher(
            FatigueState,
            FATIGUE_OUTPUT_TOPIC_NAME,
            get_realtime_qos_profile()
        )

        self._prediction_timer = self._node.create_timer(
            FATIGUE_PREDICTION_RATE, 
            self.predict_fatigue
        )

    def _init_preprocessor(self, fs: float, name_substr: str, preprocessor_cls: type[_T], **preprocessor_kwargs) -> _T:
        input_loc, input_shape = self._model.get_input_shape(name_substr)
        if input_shape is not None and input_loc is not None:
            sampling_diff = fs/input_shape.get("fs")
            if sampling_diff != 1.0:
                self._node.get_logger().warning(
                    f"Sampling diff for {name_substr} = {sampling_diff}. Will need to resample data"
                )

            try:
                self._ordered_preprocessors[input_loc] = preprocessor_cls(input_shape=input_shape, resampling_factor=sampling_diff, **preprocessor_kwargs)
            except IndexError:
                raise ValueError(f"Unexpected input location: {input_loc}")

            return self._ordered_preprocessors[input_loc]
        else:
            raise ValueError(f"No input shape found with substring {name_substr}")

    def _clear_buffers(self):
        for preprocessor in (self._emg_preprocessor, self._accelerometer_preprocessor, self._gyroscope_preprocessor, self._ecg_preprocessor):
            preprocessor.reset()

        # Reset counter
        self._mindrove_frame_counter = 0

    def update_armband_info(self, msg: MindroveArmBandEightChannelMsg):
        self._node.get_logger().debug("Received mindrove packet")
        with self._lock:
            is_ready = self._ready

        if is_ready:
            self._mindrove_frame_counter += 1
            for i in range(msg.num_emg_samples):
                self._emg_preprocessor.set_sample_index(i)
                self._emg_preprocessor.preprocess(msg)

            if self._mindrove_frame_counter == self._mindrove_imu_decimation:
                for i in range(msg.num_imu_samples):
                    self._accelerometer_preprocessor.set_sample_index(i)
                    self._accelerometer_preprocessor.preprocess(msg)

                    self._gyroscope_preprocessor.set_sample_index(i)
                    self._gyroscope_preprocessor.preprocess(msg)
                    
                    self._mindrove_frame_counter = 0

    def update_plux_info(self, msg: PluxMsg):
        self._node.get_logger().debug("Received plux packet")

        with self._lock:
            is_ready = self._ready

        if is_ready:
            self._ecg_preprocessor.preprocess(msg)

    def update_sensed_fatigue(self, msg: UserInputMsg):
        with self._lock:
            self._user_status = msg

    def update_exercise_state(self, msg: ExerciseState):
        with self._lock:
            last_state = self._ready
            current_state = (msg.state == ExerciseState.ACTIVE)

            if last_state != current_state:
                if current_state:
                    self._node.get_logger().info("User is active, starting predictions")

                    # Enable predictions
                    self._ready = True

                    # Reset prediction timer
                    self._prediction_timer.reset()
                else:
                    self._node.get_logger().info("User not active, resetting")

                    # Disable updates
                    self._ready = False

                    # Clear the preprocessor buffers
                    self._clear_buffers()

    def predict_fatigue(self):
        self._node.get_logger().debug("Predicting fatigue")
        self._lock.acquire()

        if self._ready:
            inputs = self._get_inputs()
            ts = self._get_current_timestep()

            # Lock may be released
            self._lock.release()

            prediction = self._model.call(inputs)

            # prediction: (1, T, 1)
            self._last_prediction = float(tf.squeeze(prediction)[ts])
            self._node.get_logger().info(f"Output: {self._last_prediction}")

            # Publish the fatigue state
            # Divide the fatigue percentage by 100 so both are on a normalized 0-1 scale
            self._fatigue_publisher.publish(
                FatigueState(predicted=self._last_prediction, 
                             actual=self._user_status.fatigue_percentage / 100.0)
            )
        else:
            self._lock.release()

    def _get_inputs(self) -> tf.Tensor:
        # TODO(george): order matters...
        return [x.data for x in self._ordered_preprocessors]
    
    def _get_current_timestep(self) -> int:
        return min([x.timestep for x in \
                    (self._emg_preprocessor, 
                    self._accelerometer_preprocessor, 
                    self._gyroscope_preprocessor, 
                    self._ecg_preprocessor)])
    
def main():
    rclpy.init()
    node = Node(node_name="fatigue_classifier")
    classifier = RealtimeFatigueClassifier(node=node, path=Path("fatigue_model.keras"))
    
    rclpy.spin(node)