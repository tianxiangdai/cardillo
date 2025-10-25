from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cartpole",
                executable="visualizer",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="cartpole",
                executable="controller",
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="cartpole",
                executable="simulator",
                # output="screen",
                # emulate_tty=True,
            ),
        ]
    )
