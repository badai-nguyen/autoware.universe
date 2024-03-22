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

#include "detected_object_validation/pointcloud_intensity_based_validator/pointcloud_intensity_based_validator.hpp"

#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

using Label = autoware_auto_perception_msgs::msg::ObjectClassification;
namespace pointcloud_intensity_based_validator
{
PointCloudIntensityBasedValidator::PointCloudIntensityBasedValidator(
  const rclcpp::NodeOptions & node_options)
: Node("pointcloud_intensity_based_validator_node", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  using std::placeholders::_1;
  object_sub_ = this->create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "input/objects", rclcpp::QoS{1},
    std::bind(&PointCloudIntensityBasedValidator::onObjects, this, _1));
  object_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "output/objects", rclcpp::QoS{1});
  intensity_threshold_ = declare_parameter<double>("intensity_threshold");
  existance_probability_threshold_ = declare_parameter<double>("existance_probability_threshold");

  filter_target_.UNKNOWN = declare_parameter<bool>("filter_target_label.UNKNOWN");
  filter_target_.CAR = declare_parameter<bool>("filter_target_label.CAR");
  filter_target_.TRUCK = declare_parameter<bool>("filter_target_label.TRUCK");
  filter_target_.BUS = declare_parameter<bool>("filter_target_label.BUS");
  filter_target_.TRAILER = declare_parameter<bool>("filter_target_label.TRAILER");
  filter_target_.MOTORCYCLE = declare_parameter<bool>("filter_target_label.MOTORCYCLE");
  filter_target_.BICYCLE = declare_parameter<bool>("filter_target_label.BICYCLE");
  filter_target_.PEDESTRIAN = declare_parameter<bool>("filter_target_label.PEDESTRIAN");
}
void PointCloudIntensityBasedValidator::onObjects(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_objects_msg)
{
  tier4_perception_msgs::msg::DetectedObjectsWithFeature output_objects_msg;
  output_objects_msg.header = input_objects_msg->header;
  for (const auto & feature_object : input_objects_msg->feature_objects) {
    const auto & object = feature_object.object;
    const auto & label = object.classification.front().label;
    const auto & feature = feature_object.feature;
    const auto & cluster = feature.cluster;
    const auto existance_probability = object.existence_probability;
    if (
      filter_target_.isTarget(label) && existance_probability < existance_probability_threshold_) {
      if (isObjectValid(cluster)) {
        output_objects_msg.feature_objects.push_back(feature_object);
      }
    } else {
      output_objects_msg.feature_objects.push_back(feature_object);
    }
  }
  object_pub_->publish(output_objects_msg);
}
bool PointCloudIntensityBasedValidator::isObjectValid(const sensor_msgs::msg::PointCloud2 & cluster)
{
  double mean_intensity = 0.0;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cluster, "x"), iter_y(cluster, "y"),
       iter_z(cluster, "z"), iter_intensity(cluster, "intensity");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
    mean_intensity += *iter_intensity;
  }
  const size_t num_points = cluster.width * cluster.height;
  mean_intensity = mean_intensity / static_cast<double>(num_points);
  if (mean_intensity > intensity_threshold_) return true;
  return false;
}
}  // namespace pointcloud_intensity_based_validator
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
  pointcloud_intensity_based_validator::PointCloudIntensityBasedValidator)
