import rclpy
import threading

from rclpy.executors import MultiThreadedExecutor

from python_utils.ros2_utils.comms.node_manager import create_simple_node, get_realtime_qos_profile
from fatigue_classifier.real_time_fatigue_classifier import RealtimeFatigueClassifier
from fatigue_classifier.fatigue_classifier_configs import *

from idl_definitions.msg import (
    PluxMsg,
    MindroveArmBandEightChannelMsg,
    UserInputMsg
)

from pathlib import Path
from geometry_msgs.msg import Vector3

class FatigueSimManager:
    def __init__(self):
        self._mindrove_node = create_simple_node(MINDROVE_ROS_NODE)
        self._plux_node = create_simple_node(PLUX_ROS_NODE)
        self._user_input_node = create_simple_node(USER_INPUT_NODE_NAME)

        self._fatigue_node = create_simple_node(FATIGUE_OUTPUT_NODE)

        self._mindrove_publisher = self._mindrove_node.create_publisher(
            MindroveArmBandEightChannelMsg,
            topic=MINDROVE_ROS_TOPIC_NAME,
            qos_profile=get_realtime_qos_profile()
        )

        self._plux_publisher = self._mindrove_node.create_publisher(
            PluxMsg,
            topic=PLUX_ROS_TOPIC_NAME,
            qos_profile=get_realtime_qos_profile()
        )

        self._user_input_publisher = self._user_input_node.create_publisher(
            UserInputMsg,
            topic=USER_INPUT_TOPIC_NAME,
            qos_profile=get_realtime_qos_profile()
        )

        self._fatigue_classifier = RealtimeFatigueClassifier(
            self._fatigue_node,
            path=Path("fatigue_model.keras")
        )

        self._lock = threading.Lock()
        self._ready = False

        self._mindrove_node.create_timer(1/250.0, self._publish_mindrove_packet)
        self._plux_node.create_timer(1/1000.0, self._publish_plux_packet)

        self._user_input_node.create_timer(1/100.0, self._publish_user_input_node_packet)
        self._user_input_node.create_timer(3, self._toggle_ready_state)

    @property
    def nodes(self):
        return (self._mindrove_node, self._plux_node, self._user_input_node, self._fatigue_node)

    def _toggle_ready_state(self):
        with self._lock:
            self._ready = not self._ready

    def _publish_mindrove_packet(self):
        self._mindrove_node.get_logger().debug("Publishing packet")
        msg = MindroveArmBandEightChannelMsg()
        msg.num_emg_samples = 2
        msg.num_imu_samples = 1

        msg.c1 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c2 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c3 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c4 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c5 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c6 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c7 = [0.01 for _ in range(msg.num_emg_samples)]
        msg.c8 = [0.01 for _ in range(msg.num_emg_samples)]

        msg.accel = [Vector3(x=0.01, y=0.01, z=0.01)]
        msg.gyro = [Vector3(x=0.01, y=0.01, z=0.01)]

        self._mindrove_publisher.publish(msg)

    def _publish_plux_packet(self):
        self._plux_node.get_logger().debug("Publishing packet")
        msg = PluxMsg()
        msg.ecg = 0.01
        self._plux_publisher.publish(msg)

    def _publish_user_input_node_packet(self):
        self._user_input_node.get_logger().debug("Publishing packet")
        msg = UserInputMsg()
        with self._lock:
            msg.is_active = self._ready
        self._user_input_publisher.publish(msg)

def main():
    rclpy.init()
    manager = FatigueSimManager()
    executor = MultiThreadedExecutor()
    for node in manager.nodes:
        executor.add_node(node)

    executor.spin()
