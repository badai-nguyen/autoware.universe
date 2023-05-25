// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "voxel_grid_based_euclidean_cluster_node.hpp"

#include "euclidean_cluster/utils.hpp"

#include <vector>

namespace euclidean_cluster
{
VoxelGridBasedEuclideanClusterNode::VoxelGridBasedEuclideanClusterNode(
  const rclcpp::NodeOptions & options)
: Node("voxel_grid_based_euclidean_cluster_node", options)
{
  const bool use_height = this->declare_parameter("use_height", false);
  const int min_cluster_size = this->declare_parameter("min_cluster_size", 1);
  const int max_cluster_size = this->declare_parameter("max_cluster_size", 500);
  const float tolerance = this->declare_parameter("tolerance", 1.0);
  const float voxel_leaf_size = this->declare_parameter("voxel_leaf_size", 0.5);
  const int min_points_number_per_voxel = this->declare_parameter("min_points_number_per_voxel", 3);
  cluster_ = std::make_shared<VoxelGridBasedEuclideanCluster>(
    use_height, min_cluster_size, max_cluster_size, tolerance, voxel_leaf_size,
    min_points_number_per_voxel);

  using std::placeholders::_1;
  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "input", rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&VoxelGridBasedEuclideanClusterNode::onPointCloud, this, _1));

  cluster_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "output", rclcpp::QoS{1});
  debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug/clusters", 1);
}

void VoxelGridBasedEuclideanClusterNode::onPointCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg)
{
  // convert ros to pcl
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pointcloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*input_msg, *raw_pointcloud_ptr);

  // clustering
  std::vector<pcl::PointCloud<pcl::PointXYZI>> clusters;
  cluster_->cluster(raw_pointcloud_ptr, clusters);

  // build output msg
  tier4_perception_msgs::msg::DetectedObjectsWithFeature output;
  convertPointCloudClusters2Msg(input_msg->header, clusters, output);
  cluster_pub_->publish(output);

  // build debug msg
  if (debug_pub_->get_subscription_count() < 1) {
    return;
  }
  {
    // pcl::PointCloud<pcl::PointXYZI> saving_cluster_pcd;

    // saving_cluster_pcd.width = raw_pointcloud_ptr->size();
    // saving_cluster_pcd.height = 1;
    // saving_cluster_pcd.is_dense = false;
    // saving_cluster_pcd.points.resize(raw_pointcloud_ptr->size());
    // // saving_cluster_pcd.reserve(raw_pointcloud_ptr->size());
    // for (auto icluster = clusters.begin(); icluster < clusters.end() ; ++icluster){
    //   for (auto point = icluster->begin(); point < icluster->end(); ++point){
    //     saving_cluster_pcd.push_back(*point);
    //   }
    // }
    // pcl::io::savePCDFileASCII(
    // "/media/autoware/MileeSSD/tasks/T4PB-24940_adverse_weather/test.pcd",saving_cluster_pcd);
    // sensor_msgs::msg::PointCloud2 cluster_pub;
    sensor_msgs::msg::PointCloud2 debug;
    convertObjectMsg2SensorMsg(output, debug);
    debug_pub_->publish(debug);
  }
}

}  // namespace euclidean_cluster

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(euclidean_cluster::VoxelGridBasedEuclideanClusterNode)
