from ros2_mindrove.mindrove_configs import (
    MINDROVE_ROS_NODE,
    MINDROVE_ROS_TOPIC_NAME,
)

from ros2_plux_biosignals.plux_configs import (
    PLUX_ROS_NODE,
    PLUX_ROS_TOPIC_NAME
)

from pi_user_input_node.user_input_node_config import (
    USER_INPUT_NODE_NAME,
    USER_INPUT_TOPIC_NAME
)

MINDROVE_ACTIVATION_SERVICE = "simple_exercise_decoder/mindrove_activation_service"
MINDROVE_DEACTIVATION_SERVICE = "simple_exercise_decoder/mindrove_deactivation_service"

MINDROVE_FILTERED_OUTPUT = "simple_exercise_decoder/mindrove_filtered_output"
MINDROVE_FILTERED_OUTPUT_QOS = 0

FATIGUE_OUTPUT_NODE = "real_time_fatigue_node"
FATIGUE_OUTPUT_TOPIC_NAME = "real_time_fatigue_node/estimated_fatigue"
FATIGUE_OUTPUT_QOS = 0

FATIGUE_PREDICTION_RATE = 1.0 # 1 Hz