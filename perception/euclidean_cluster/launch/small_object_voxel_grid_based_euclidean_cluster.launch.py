import launch
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
import yaml


def launch_setup(context, *args, **kwargs):
    def load_composable_node_param(param_path):
        with open(LaunchConfiguration(param_path).perform(context), "r") as f:
            return yaml.safe_load(f)["/**"]["ros__parameters"]

    ns = ""
    pkg = "euclidean_cluster"

    # set lanelet pointcloud filter
    lanelet_pointcloud_filter_component = ComposableNode(
        package="pointcloud_preprocessor",
        plugin="pointcloud_preprocessor::Lanelet2MapFilterComponent",
        name="vector_map_filter",
        namespace=ns,
        remappings=[
            ("input/pointcloud", LaunchConfiguration("input_pointcloud")),
            ("input/vector_map", LaunchConfiguration("input/vector_map")),
            ("output", "vector_map_filtered/pointcloud"),
        ],
        parameters=[
            {
                "voxel_size_x": 0.4,
                "voxel_size_y": 0.4,
            }
        ],
        # cannot use intra process because vector map filter uses transient local.
        extra_arguments=[{"use_intra_process_comms": False}],
    )

    small_object_euclidean_cluster_component = ComposableNode(
        package=pkg,
        namespace=ns,
        plugin="euclidean_cluster::VoxelGridBasedEuclideanClusterNode",
        name="euclidean_cluster",
        remappings=[
            ("input", "vector_map_filtered/pointcloud"),
            ("output", LaunchConfiguration("small_object_output_clusters")),
        ],
        parameters=[
            {
                "min_cluster_size": 2,
                "max_cluster_size": 100,
                "tolerance": 0.7,
                "voxel_leaf_size": 0.3,
                "min_points_number_per_voxel": 1,
            }
        ],
    )

    container = ComposableNodeContainer(
        name="small_object_euclidean_cluster_container",
        package="rclcpp_components",
        namespace=ns,
        executable="component_container",
        composable_node_descriptions=[],
        output="screen",
    )

    small_object_cluster_loader = LoadComposableNodes(
        composable_node_descriptions=[
            lanelet_pointcloud_filter_component,
            small_object_euclidean_cluster_component,
        ],
        target_container=container,
    )

    return [container, small_object_cluster_loader]


def generate_launch_description():
    def add_launch_arg(name: str, default_value=None):
        return DeclareLaunchArgument(name=name, default_value=default_value)

    return launch.LaunchDescription(
        [
            add_launch_arg("input_pointcloud", "/perception/obstacle_segmentation/pointcloud"),
            add_launch_arg("small_object_output_clusters", "small_object_output_clusters"),
            add_launch_arg("input/vector_map", "/map/vector_map"),
            OpaqueFunction(function=launch_setup),
        ]
    )
