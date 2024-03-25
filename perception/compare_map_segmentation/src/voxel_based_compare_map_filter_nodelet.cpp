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

#include "compare_map_segmentation/voxel_based_compare_map_filter_nodelet.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/segment_differences.h>

#include <string>
#include <vector>

namespace compare_map_segmentation
{
using pointcloud_preprocessor::get_param;

VoxelBasedCompareMapFilterComponent::VoxelBasedCompareMapFilterComponent(
  const rclcpp::NodeOptions & options)
: Filter("VoxelBasedCompareMapFilter", options)
{
  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ = std::make_unique<DebugPublisher>(this, "voxel_based_compare_map_filter");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  distance_threshold_ = declare_parameter<double>("distance_threshold");
  bool use_dynamic_map_loading = declare_parameter<bool>("use_dynamic_map_loading");
  double downsize_ratio_z_axis = declare_parameter<double>("downsize_ratio_z_axis");
  if (downsize_ratio_z_axis <= 0.0) {
    RCLCPP_ERROR(this->get_logger(), "downsize_ratio_z_axis should be positive");
    return;
  }
  set_map_in_voxel_grid_ = false;
  if (use_dynamic_map_loading) {
    rclcpp::CallbackGroup::SharedPtr main_callback_group;
    main_callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    voxel_grid_map_loader_ = std::make_unique<VoxelGridDynamicMapLoader>(
      this, distance_threshold_, downsize_ratio_z_axis, &tf_input_frame_, &mutex_,
      main_callback_group);
  } else {
    voxel_grid_map_loader_ = std::make_unique<VoxelGridStaticMapLoader>(
      this, distance_threshold_, downsize_ratio_z_axis, &tf_input_frame_, &mutex_);
  }
  tf_input_frame_ = *(voxel_grid_map_loader_->tf_map_input_frame_);
  RCLCPP_INFO(this->get_logger(), "tf_map_input_frame: %s", tf_input_frame_.c_str());
}
void VoxelBasedCompareMapFilterComponent::set_field_offsets(const PointCloud2ConstPtr & input)
{
  x_offset_ = input->fields[pcl::getFieldIndex(*input, "x")].offset;
  y_offset_ = input->fields[pcl::getFieldIndex(*input, "y")].offset;
  z_offset_ = input->fields[pcl::getFieldIndex(*input, "z")].offset;
  int intensity_index = pcl::getFieldIndex(*input, "intensity");
  if (intensity_index != -1) {
    intensity_offset_ = input->fields[intensity_index].offset;
  } else {
    intensity_offset_ = z_offset_ + sizeof(float);
  }
  RCLCPP_INFO(
    this->get_logger(), "x_offset: %d, y_offset: %d, z_offset: %d, intensity_offset: %d", x_offset_,
    y_offset_, z_offset_, intensity_offset_);
  offset_initialized_ = true;
}

void VoxelBasedCompareMapFilterComponent::get_point_from_global_offset(
  const PointCloud2ConstPtr & input, const size_t global_offset, pcl::PointXYZ & point)
{
  point.x = *reinterpret_cast<const float *>(&input->data[global_offset + x_offset_]);
  point.y = *reinterpret_cast<const float *>(&input->data[global_offset + y_offset_]);
  point.z = *reinterpret_cast<const float *>(&input->data[global_offset + z_offset_]);
}
void VoxelBasedCompareMapFilterComponent::faster_filter(
  const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output,
  const pointcloud_preprocessor::TransformInfo & transform_info)
{
  std::scoped_lock lock(mutex_);
  stop_watch_ptr_->toc("processing_time", true);
  if (indices) {
    RCLCPP_ERROR(this->get_logger(), "Indices are not supported in this filter");
  }
  if (!offset_initialized_) {
    set_field_offsets(input);
  }
  output.data.resize(input->data.size());
  size_t output_size = 0;
  for (size_t global_offset = 0; global_offset + input->point_step <= input->data.size();
       global_offset += input->point_step) {
    Eigen::Vector4f point;
    std::memcpy(&point[0], &input->data[global_offset + x_offset_], sizeof(float));
    std::memcpy(&point[1], &input->data[global_offset + y_offset_], sizeof(float));
    std::memcpy(&point[2], &input->data[global_offset + z_offset_], sizeof(float));
    point[3] = 1;
    if (transform_info.need_transform) {
      point = transform_info.eigen_transform * point;
    }
    if (voxel_grid_map_loader_->is_close_to_map(
          pcl::PointXYZ(point[0], point[1], point[2]), distance_threshold_)) {
      continue;
    }
    memcpy(&output.data[output_size], &input->data[global_offset], input->point_step);
    if (transform_info.need_transform) {
      std::memcpy(&output.data[output_size + x_offset_], &point[0], sizeof(float));
      std::memcpy(&output.data[output_size + y_offset_], &point[1], sizeof(float));
      std::memcpy(&output.data[output_size + z_offset_], &point[2], sizeof(float));
    }
    output_size += input->point_step;
  }
  output.data.resize(output_size);
  output.header.frame_id = tf_output_frame_;
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
void VoxelBasedCompareMapFilterComponent::filter(
  const PointCloud2ConstPtr & input, [[maybe_unused]] const IndicesPtr & indices,
  PointCloud2 & output)
{
  (void)input;
  (void)indices;
  (void)output;
}

}  // namespace compare_map_segmentation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(compare_map_segmentation::VoxelBasedCompareMapFilterComponent)
