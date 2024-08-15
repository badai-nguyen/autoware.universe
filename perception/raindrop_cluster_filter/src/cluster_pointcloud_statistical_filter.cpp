// Copyright 2024 TIER IV, Inc.
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

#include "raindrop_cluster_filter/cluster_pointcloud_statistical_filter/cluster_pointcloud_statistical_filter.hpp"


#include <autoware/universe_utils/geometry/geometry.hpp>
#include <pcl_ros/transforms.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace cluster_pointcloud_statistical_filter
{
ClusterPointcloudStatisticalFilter::ClusterPointcloudStatisticalFilter(const rclcpp::NodeOptions & node_options)
: Node("cluster_pointcloud_statistical_filter_node", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  max_x_ = declare_parameter<double>("max_x");
  min_x_ = declare_parameter<double>("min_x");
  max_y_ = declare_parameter<double>("max_y");
  min_y_ = declare_parameter<double>("min_y");

  filter_target_.UNKNOWN = declare_parameter<bool>("filter_target_label.UNKNOWN");
  filter_target_.CAR = declare_parameter<bool>("filter_target_label.CAR");
  filter_target_.TRUCK = declare_parameter<bool>("filter_target_label.TRUCK");
  filter_target_.BUS = declare_parameter<bool>("filter_target_label.BUS");
  filter_target_.TRAILER = declare_parameter<bool>("filter_target_label.TRAILER");
  filter_target_.MOTORCYCLE = declare_parameter<bool>("filter_target_label.MOTORCYCLE");
  filter_target_.BICYCLE = declare_parameter<bool>("filter_target_label.BICYCLE");
  filter_target_.PEDESTRIAN = declare_parameter<bool>("filter_target_label.PEDESTRIAN");

  nearest_points_num_ = declare_parameter<int>("nearest_points_num");
  normal_threshold_ = declare_parameter<double>("normal_threshold");
  using std::placeholders::_1;
  // Set publisher/subscriber
  object_sub_ = this->create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "input/objects", rclcpp::QoS{1},
    std::bind(&ClusterPointcloudStatisticalFilter::objectCallback, this, _1));
  object_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "output/objects", rclcpp::QoS{1});
  // initialize debug tool
  {
    using autoware::universe_utils::DebugPublisher;
    using autoware::universe_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ =
      std::make_unique<DebugPublisher>(this, this->get_name());
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }
}

void ClusterPointcloudStatisticalFilter::objectCallback(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_msg)
{
  // Guard
  stop_watch_ptr_->toc("processing_time", true);
  if (object_pub_->get_subscription_count() < 1) return;

  tier4_perception_msgs::msg::DetectedObjectsWithFeature output_object_msg;
  output_object_msg.header = input_msg->header;
  geometry_msgs::msg::TransformStamped transform_stamp;
  try {
    transform_stamp = tf_buffer_.lookupTransform(
      input_msg->header.frame_id, base_link_frame_id_, tf2_ros::fromMsg(input_msg->header.stamp));
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(get_logger(), "Failed to lookup transform: %s", ex.what());
    return;
  }
  geometry_msgs::msg::Pose min_range;
  min_range.position.x = min_x_;
  min_range.position.y = min_y_;
  geometry_msgs::msg::Pose max_pose;
  max_pose.position.x = max_x_;
  max_pose.position.y = max_y_;
  auto min_ranged_transformed = autoware::universe_utils::transformPose(min_range, transform_stamp);
  auto max_range_transformed = autoware::universe_utils::transformPose(max_pose, transform_stamp);
  geometry_msgs::msg::Pose base_link_pose;
  auto base_link_origin = autoware::universe_utils::transformPose(base_link_pose, transform_stamp);
  for (const auto & feature_object : input_msg->feature_objects) {
    const auto & object = feature_object.object;
    const auto & label = object.classification.front().label;
    const auto & feature = feature_object.feature;
    const auto & cluster = feature.cluster;
    const auto & position = object.kinematics.pose_with_covariance.pose.position;
    bool is_inside_validation_range = min_ranged_transformed.position.x < position.x &&
                                      position.x < max_range_transformed.position.x &&
                                      min_ranged_transformed.position.y < position.x &&
                                      position.y < max_range_transformed.position.y;
    int intensity_index = pcl::getFieldIndex(cluster, "intensity");
    if (
      intensity_index != -1 && filter_target_.isTarget(label) && is_inside_validation_range &&
      !isValidatedCluster(cluster, base_link_origin)) {
      continue;
    }
    output_object_msg.feature_objects.emplace_back(feature_object);
  }
  object_pub_->publish(output_object_msg);
  if (debug_publisher_ptr_ && stop_watch_ptr_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
}
bool ClusterPointcloudStatisticalFilter::isValidatedCluster(const sensor_msgs::msg::PointCloud2 & cluster, 
  const geometry_msgs::msg::Pose & base_link_origin)
{
  // convert to pcl pointcloud 
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cluster(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(cluster, *pcl_cluster);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(pcl_cluster);
  std::vector<int> pointIdxNRNSearch(nearest_points_num_);
  std::vector<float> pointRadiusSquaredDistance(nearest_points_num_);
  std::vector<double> distances;
  std::vector<Eigen::Vector3d> normals_vectors;
  std::vector<size_t> nearest_points_indices; 
  
  // go though each point in the cluster and find the nearest points
  for (size_t i = 0; i < pcl_cluster->points.size(); i++) {
    kd_tree.nearestKSearch(pcl_cluster->points[i], nearest_points_num_, pointIdxNRNSearch, pointRadiusSquaredDistance);
    nearest_points_indices.push_back(pointIdxNRNSearch[0]);
    // calculate covariance matrix
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mean_point = Eigen::Vector3d::Zero();
    for (size_t j = 0; j < pointIdxNRNSearch.size(); j++) {
      mean_point += Eigen::Vector3d(pcl_cluster->points[pointIdxNRNSearch[j]].x,
                                    pcl_cluster->points[pointIdxNRNSearch[j]].y,
                                    pcl_cluster->points[pointIdxNRNSearch[j]].z);
    }
    mean_point /= static_cast<double>(pointIdxNRNSearch.size());
    for (size_t j = 0; j < pointIdxNRNSearch.size(); j++) {
      Eigen::Vector3d point(pcl_cluster->points[pointIdxNRNSearch[j]].x,
                            pcl_cluster->points[pointIdxNRNSearch[j]].y,
                            pcl_cluster->points[pointIdxNRNSearch[j]].z);
      covariance_matrix += (point - mean_point) * (point - mean_point).transpose();
    }
    covariance_matrix /= static_cast<double>(pointIdxNRNSearch.size());
    // calculate eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance_matrix);
    Eigen::Vector3d eigen_values = eigen_solver.eigenvalues();
    normals_vectors.push_back(eigen_values);
  }

  // calculate dot product of normals of pair neighboring points
  double C_normal = 0.0;
  for (size_t i = 0; i < pcl_cluster->points.size(); i++) {
    Eigen::Vector3d curr_normal = normals_vectors[i];
    Eigen::Vector3d neighbor_normal = normals_vectors[nearest_points_indices[i]];
    C_normal += curr_normal.dot(neighbor_normal);
  }
  C_normal /= static_cast<double>(pcl_cluster->points.size());
  RCLCPP_INFO(get_logger(), "C_normal: %f", C_normal);
  if (C_normal > normal_threshold_) {
    return false;
  }
  return true;

}

}  // namespace cluster_pointcloud_statistical_filter

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(cluster_pointcloud_statistical_filter::ClusterPointcloudStatisticalFilter)
