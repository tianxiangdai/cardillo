from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cardillo_ros",
                executable="visualizer",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="cardillo_ros",
                executable="simulator",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="cardillo_ros",
                executable="controller",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
