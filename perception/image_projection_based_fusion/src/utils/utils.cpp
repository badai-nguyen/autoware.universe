// Copyright 2022 TIER IV, Inc.
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

#include "image_projection_based_fusion/utils/utils.hpp"

namespace image_projection_based_fusion
{

std::optional<geometry_msgs::msg::TransformStamped> getTransformStamped(
  const tf2_ros::Buffer & tf_buffer, const std::string & target_frame_id,
  const std::string & source_frame_id, const rclcpp::Time & time)
{
  try {
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped = tf_buffer.lookupTransform(
      target_frame_id, source_frame_id, time, rclcpp::Duration::from_seconds(0.01));
    return transform_stamped;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("image_projection_based_fusion"), ex.what());
    return std::nullopt;
  }
}

Eigen::Affine3d transformToEigen(const geometry_msgs::msg::Transform & t)
{
  Eigen::Affine3d a;
  a.matrix() = tf2::transformToEigen(t).matrix();
  return a;
}


PointCloud closest_cluster(
  const PointCloud & cluster, 
  const double cluster_threshold_radius,
  const double cluster_threshold_distance)
{
  PointCloud out_cluster;
  pcl::PointXYZ orig_point(pcl::PointXYZ(0.0, 0.0, 0.0));
  pcl::PointXYZ closest_point = getClosestPoint(cluster);
  double shortest_radius =
    tier4_autoware_utils::calcDistance2d(closest_point, pcl::PointXYZ(0.0, 0.0, 0.0));
  for (auto & point : cluster) {
    double radius = tier4_autoware_utils::calcDistance2d(point, orig_point);
    double distance = tier4_autoware_utils::calcDistance3d(point, closest_point);
    if (
      abs(radius - shortest_radius) < cluster_threshold_radius &&
      distance < cluster_threshold_distance) {
      out_cluster.push_back(point);
    }
  }
  return out_cluster;
}


geometry_msgs::msg::Point getCentroid(
  const sensor_msgs::msg::PointCloud2 & pointcloud)
{
  geometry_msgs::msg::Point centroid;
  centroid.x = 0.0f;
  centroid.y = 0.0f;
  centroid.z = 0.0f;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(pointcloud, "x"),
       iter_y(pointcloud, "y"), iter_z(pointcloud, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    centroid.x += *iter_x;
    centroid.y += *iter_y;
    centroid.z += *iter_z;
  }
  const size_t size = pointcloud.width * pointcloud.height;
  centroid.x = centroid.x / static_cast<float>(size);
  centroid.y = centroid.y / static_cast<float>(size);
  centroid.z = centroid.z / static_cast<float>(size);
  return centroid;
}

//TODO : change to template
pcl::PointXYZ getClosestPoint(const pcl::PointCloud<pcl::PointXYZ> & cluster)
{
  pcl::PointXYZ closest_point;
  double min_dist = 1e6;
  pcl::PointXYZ orig_point = pcl::PointXYZ(0.0, 0.0, 0.0);
  for (std::size_t i = 0; i < cluster.points.size(); ++i) {
    pcl::PointXYZ point = cluster.points.at(i);
    double dist_closest_point = tier4_autoware_utils::calcDistance2d(point, orig_point);
    if (min_dist > dist_closest_point) {
      min_dist = dist_closest_point;
      closest_point = pcl::PointXYZ(point.x, point.y, point.z);
    }
  }
  return closest_point;
}


}  // namespace image_projection_based_fusion
