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

#include "pointcloud_preprocessor/outlier_filter/dynamic_statistical_outlier_filter_nodelet.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <vector>
namespace pointcloud_preprocessor
{
using PointT = pcl::PointXYZI;
ScalableStatisticalFilterComponent::ScalableStatisticalFilterComponent(
  const rclcpp::NodeOptions & options)
: Filter("DynamicStatisticalOutlierFilter", options)
{
  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, "dynamic_statistical_outlier_filter");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  // set initial parameters
  {
    mean_k_ = static_cast<int>(declare_parameter("mean_k", 5));
    std_mul_ = static_cast<double>(declare_parameter("std_mul", 0.01));
    range_mul = static_cast<double>(declare_parameter("range_mul", 0.05));
    x_max_ = static_cast<double>(declare_parameter("x_max", 60.0));
    x_min_ = static_cast<double>(declare_parameter("x_min", 0.0));
    y_max_ = static_cast<double>(declare_parameter("y_max", 25.0));
    y_min_ = static_cast<double>(declare_parameter("y_min", -25.0));
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&ScalableStatisticalFilterComponent::paramCallback, this, _1));
}

void ScalableStatisticalFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);
  if (indices) {
    RCLCPP_WARN(get_logger(), "Indices are not supported and will be ignored");
  }
  stop_watch_ptr_->toc("processing_time", true);

  pcl::PointCloud<PointT>::Ptr input_cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr filter_cloud(new pcl::PointCloud<PointT>);

  int x_offset = input->fields[pcl::getFieldIndex(*input, "x")].offset;
  int y_offset = input->fields[pcl::getFieldIndex(*input, "y")].offset;
  int z_offset = input->fields[pcl::getFieldIndex(*input, "z")].offset;
  output.data.resize(input->data.size());
  size_t output_size = 0;
  for(size_t global_offset = 0; global_offset + input->point_step <= input->data.size();
   global_offset += input->point_step){
    Eigen::Vector4f point;
    std::memcpy(&point[0], &input->data[global_offset + x_offset], sizeof(float));
    std::memcpy(&point[1], &input->data[global_offset + y_offset], sizeof(float));
    std::memcpy(&point[2], &input->data[global_offset + z_offset], sizeof(float));
    point[3] = 1;

    bool point_is_inside = point[0] > x_min_ && point[0] < x_max_ && 
                           point[1] > y_min_ && point[1] < y_max_;
    if(point_is_inside){
      filter_cloud->push_back(PointT(point[0], point[1], point[2]));
    }
    else{
      std::memcpy(&output.data[output_size + x_offset], &point[0], sizeof(float));
      std::memcpy(&output.data[output_size + y_offset], &point[1], sizeof(float));
      std::memcpy(&output.data[output_size + z_offset], &point[2], sizeof(float));
      output_size += input->point_step;
    }
  }
  pcl::KdTreeFLANN<PointT> kd_tree;
  kd_tree.setInputCloud(filter_cloud);

  std::vector<int> pointIdxNKNSearch(mean_k_);
  std::vector<float> pointNKNSquaredDistance(mean_k_);
  std::vector<float> mean_distances;

  // Go over all the points and check which doesn't have enough neighbors
  // perform filtering
  for (pcl::PointCloud<PointT>::iterator it = filter_cloud->begin(); it != filter_cloud->end();
       ++it) {
    // k nearest search
    kd_tree.nearestKSearch(*it, mean_k_, pointIdxNKNSearch  , pointNKNSquaredDistance);

    // calculate mean distance
    double dist_sum = 0;
    for (int j = 1; j < mean_k_; ++j) {
      dist_sum += sqrt(pointNKNSquaredDistance[j]);
    }
    mean_distances.push_back(static_cast<float>(dist_sum / (mean_k_ - 1)));
  }

  // Estimate the mean and the standard deviation of the distance vector
  double sum = 0, sq_sum = 0;
  for (size_t i = 0; i < mean_distances.size(); ++i) {
    sum += mean_distances[i];
    sq_sum += mean_distances[i] * mean_distances[i];
  }
  double mean = sum / static_cast<double>(mean_distances.size());
  double variance = (sq_sum - sum * sum / static_cast<double>(mean_distances.size())) /
                    (static_cast<double>(mean_distances.size()) - 1);
  double stddev = sqrt(variance);
  double distance_threshold = (mean + std_mul_ * stddev);
  // iterate through vector
  int i = 0;
  for (pcl::PointCloud<PointT>::iterator it = filter_cloud->begin(); it != filter_cloud->end();
       ++it) {
    
    // calculate distance of every point from the sensor
    float range = sqrt(pow(it->x, 2) + pow(it->y, 2) + pow(it->z, 2));
    // dynamic threshold: as a point is farther away from the sensor,
    // the threshold increases
    double dynamic_threshold = distance_threshold * range_mul * range;

    // a distance lower than the threshold is an inlier
    if (mean_distances[i] < dynamic_threshold) {
      std::memcpy(&output.data[output_size + x_offset], &it->x, sizeof(float));
      std::memcpy(&output.data[output_size + y_offset], &it->y, sizeof(float));
      std::memcpy(&output.data[output_size + z_offset], &it->z, sizeof(float));
      output_size += input->point_step;
    } 
    // update iterator
    i++;
  }

  output.data.resize(output_size);

  output.header = input->header;

  output.height = 1;
  output.fields = input->fields;
  output.is_bigendian = input->is_bigendian;
  output.point_step = input->point_step;
  output.is_dense = input->is_dense;
  output.width = static_cast<uint32_t>(output.data.size() / output.height / output.point_step);
  output.row_step = static_cast<uint32_t>(output.data.size() / output.height);
  // add processing time for debug
  if (debug_publisher_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
}

rcl_interfaces::msg::SetParametersResult ScalableStatisticalFilterComponent::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  std::scoped_lock lock(mutex_);

  mean_k_ = static_cast<int>(declare_parameter("mean_k", 5));
  std_mul_ = static_cast<double>(declare_parameter("std_mul", 0.01));
  range_mul = static_cast<double>(declare_parameter("range_mul", 0.05));

  if (get_param(p, "mean_k", mean_k_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new mean_k to: %d.", mean_k_);
  }
  if (get_param(p, "std_mul", std_mul_)) {
    RCLCPP_DEBUG(get_logger(), "Setting new std_mul to: %f.", std_mul_);
  }
  if (get_param(p, "range_mul", range_mul)) {
    RCLCPP_DEBUG(get_logger(), "Setting new range_mul to: %f.", range_mul);
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}
}  // namespace pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::ScalableStatisticalFilterComponent)
