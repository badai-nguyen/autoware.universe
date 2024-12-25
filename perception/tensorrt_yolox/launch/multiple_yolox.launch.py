from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    image_topics = [
        '/image_raw0',
        '/image_raw1',
        '/image_raw2',
        '/image_raw3',
        '/image_raw4',
        '/image_raw5',
        '/image_raw6',
        '/image_raw7',
    ]

    output_topic = 'rois'

    nodes = []
    for image_topic in image_topics:
        if image_topic:  # Skip empty topics
            nodes.append(
                Node(
                    package='tensorrt_yolox',
                    executable='tensorrt_yolox_node_exe',
                    name=f'yolox_{image_topic.replace("/", "")}',
                    remappings=[('input/image', image_topic), ('output/objects', output_topic)],
                )
            )

    return LaunchDescription(nodes)
