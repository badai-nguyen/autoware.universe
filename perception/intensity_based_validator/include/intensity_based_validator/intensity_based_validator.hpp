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

#ifndef INTENSITY_BASED_VALIDATOR__INTENSITY_BASED_VALIDATOR_HPP_
#define INTENSITY_BASED_VALIDATOR__INTENSITY_BASED_VALIDATOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/ros/debug_publisher.hpp>
#include <tier4_autoware_utils/system/stop_watch.hpp>

#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <string>

namespace intensity_based_validator
{

class IntensityBasedValidator : public rclcpp::Node
{
public:
  explicit IntensityBasedValidator(const rclcpp::NodeOptions & node_options);

private:
  void objectCallback(
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_msg);
  bool isValidatedCluster(const sensor_msgs::msg::PointCloud2 & cluster);

  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr object_pub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr
    object_sub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  double intensity_threshold_;
  double existance_probability_threshold_;
  double max_x_;
  double min_x_;
  double max_y_;
  double min_y_;

  // debugger
  std::unique_ptr<tier4_autoware_utils::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_{
    nullptr};
  std::unique_ptr<tier4_autoware_utils::DebugPublisher> debug_publisher_ptr_{nullptr};
};

}  // namespace intensity_based_validator

#endif  // INTENSITY_BASED_VALIDATOR__INTENSITY_BASED_VALIDATOR_HPP_
