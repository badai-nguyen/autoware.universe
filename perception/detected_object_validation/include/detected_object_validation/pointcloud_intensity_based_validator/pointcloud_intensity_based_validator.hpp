// Copyright 2024 Tier IV, Inc.
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

#ifndef DETECTED_OBJECT_VALIDATION__POINTCLOUD_INTENSITY_BASED_VALIDATOR__POINTCLOUD_INTENSITY_BASED_VALIDATOR_HPP_
#define DETECTED_OBJECT_VALIDATION__POINTCLOUD_INTENSITY_BASED_VALIDATOR__POINTCLOUD_INTENSITY_BASED_VALIDATOR_HPP_

#include <detected_object_validation/utils/utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace pointcloud_intensity_based_validator
{
class PointCloudIntensityBasedValidator : public rclcpp::Node
{
public:
  explicit PointCloudIntensityBasedValidator(const rclcpp::NodeOptions & node_options);

private:
  void onObjects(
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_objects_msg);
  bool isObjectValid(const sensor_msgs::msg::PointCloud2 & object);
  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr object_pub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr
    object_sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  double intensity_threshold_;
  double existance_probability_threshold_;
  utils::FilterTargetLabel filter_target_;
};
}  // namespace pointcloud_intensity_based_validator

#endif  // DETECTED_OBJECT_VALIDATION__POINTCLOUD_INTENSITY_BASED_VALIDATOR__POINTCLOUD_INTENSITY_BASED_VALIDATOR_HPP_
