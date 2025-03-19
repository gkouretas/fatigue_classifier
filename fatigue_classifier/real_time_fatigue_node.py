import tensorflow as tf

from rclpy.node import Node
from pathlib import Path

from fatigue_classifier.fatigue_model import FatigueClassifier
from fatigue_classifier.fatigue_classifier_configs import *

from ur10e_custom_control.ur10e_typedefs import URService

from idl_definitions.msg import (
    MindroveArmBandEightChannelMsg,
    PluxMsg,
    UserInputMsg
)

from std_msgs.msg import Float64

class RealtimeFatigueNode:
    def __init__(self, node: Node, path: Path):
        self._node = node
        self._model = FatigueClassifier.from_file(path)

        # Subscribers
        self._mindrove_subscriber = self._node.create_subscription(
            MindroveArmBandEightChannelMsg,
            MINDROVE_ROS_TOPIC_NAME,
            self.update_armband_info,
            0
        )

        self._plux_subscriber = self._node.create_subscription(
            PluxMsg,
            PLUX_ROS_TOPIC_NAME,
            self.update_plux_info,
            0
        )

        self._user_input_subscriber = self._node.create_subscription(
            UserInputMsg,

        )

        self._fatigue_publisher = self._node.create_publisher(
            Float64,
            FATIGUE_OUTPUT_TOPIC_NAME,
            FATIGUE_OUTPUT_QOS
        )

        self._ffc_offset = int(500 // 60)

        self._last_prediction: float = 0.0
        self._current_timestep: int = 0

        # TODO(george): move to decoder
        # self._ur_speed_scaling_service = URService.init_service(
        #     self._node,
        #     URService.IOAndStatusController.SRV_SET_FORCE_MODE_PARAMS,
        #     timeout = 10
        # )

    def update_armband_info(self, msg: MindroveArmBandEightChannelMsg):
        pass

    def update_plux_info(self, msg: PluxMsg):
        pass

    def predict_fatigue(self):
        prediction = self._model.predict(
            self._get_inputs(),
            batch_size=1,
            verbose=False
        )

        # prediction: (B, T, 1)
        self._last_prediction = float(prediction[0, self._current_timestep, 0])

        self._fatigue_publisher.publish(self._last_prediction)

    def _get_inputs(self) -> tf.Tensor:
        pass