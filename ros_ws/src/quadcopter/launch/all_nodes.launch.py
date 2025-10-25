from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="quadcopter",
                executable="visualizer",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="quadcopter",
                executable="controller",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="quadcopter",
                executable="simulator",
                # output="screen",
                # emulate_tty=True,
            ),
        ]
    )
