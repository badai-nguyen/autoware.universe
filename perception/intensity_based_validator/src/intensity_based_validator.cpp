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

#include "intensity_based_validator/intensity_based_validator.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
namespace intensity_based_validator
{
IntensityBasedValidator::IntensityBasedValidator(const rclcpp::NodeOptions & node_options)
: Node("intensity_based_validator_node", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  intensity_threshold_ = declare_parameter<double>("intensity_threshold", 2.0);
  existance_probability_threshold_ =
    declare_parameter<double>("existance_probability_threshold", 0.1);

  using std::placeholders::_1;
  // Set publisher/subscriber
  object_sub_ = this->create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "input/objects", rclcpp::QoS{1}, std::bind(&IntensityBasedValidator::objectCallback, this, _1));
  object_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "output/objects", rclcpp::QoS{1});
  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, "intensity_based_validator_node");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }
}

void IntensityBasedValidator::objectCallback(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_msg)
{
  // Guard
  stop_watch_ptr_->toc("processing_time", true);
  if (object_pub_->get_subscription_count() < 1) return;

  tier4_perception_msgs::msg::DetectedObjectsWithFeature output_object_msg;
  output_object_msg.header = input_msg->header;

  for (const auto & feature_object : input_msg->feature_objects) {
    auto const & object = feature_object.object;
    // auto const & label = object.classification.front().label;
    auto const & feature = feature_object.feature;
    auto const & cluster = feature.cluster;
    auto existance_probability = object.existence_probability;

    if (!isValidatedCluster(cluster) && existance_probability < existance_probability_threshold_) {
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
bool IntensityBasedValidator::isValidatedCluster(const sensor_msgs::msg::PointCloud2 & cluster)
{
  double mean_intensity = 0.0;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter(cluster, "intensity"); iter != iter.end();
       ++iter) {
    mean_intensity += *iter;
  }
  const size_t num_points = cluster.width * cluster.height;
  mean_intensity /= static_cast<double>(num_points);
  if (mean_intensity > intensity_threshold_) {
    return true;
  }
  return false;
}

}  // namespace intensity_based_validator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(intensity_based_validator::IntensityBasedValidator)
