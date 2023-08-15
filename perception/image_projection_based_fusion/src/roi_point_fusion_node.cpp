
#include "image_projection_based_fusion/roi_point_fusion_node.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/optional.hpp>

#include <cmath>
#include <string>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

namespace image_projection_based_fusion
{
RoiPointCloudFusionNode::RoiPointCloudFusionNode(const rclcpp::NodeOptions & options)
: Node("roi_pointcloud_fusion_node", options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
}

void RoiPointCloudFusionNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr input_camera_info_msg,
  const std::size_t camera_id)
{
  camera_info_map_[camera_id] = *input_camera_info_msg;
}
void RoiPointCloudFusionNode::roiCallback(
  const DetectedObjectsWithFeature::ConstSharedPtr input_roi_msg, const std::size_t roi_i)
{
  int64_t timestamp_nsec =
    (*input_roi_msg).header.stamp.sec * (int64_t)1e9 + (*input_roi_msg).header.stamp.nanosec;
  if (sub_std_pair_.second != nullptr) {
    int64_t new_stamp = sub_std_pair_.first + input_offset_ms_.at(roi_i) * (int64_t)1e6;
    int64_t interval = abs(timestamp_nsec - new_stamp);
    if (interval < match_threshold_ms_ * (int64_t)1e6 && is_fused_.at(roi_i) == false) {
      if (camera_info_map_.find(roi_i) == camera_info_map_.end()) {
        return;
      }
    }
    fuseOnSingleImage(
      *(sub_std_pair_.second), roi_i, *input_roi_msg, camera_info_map_.at(roi_i),
      *(fused_std_pair_.second));
    is_fused_.at(roi_i) = true;

    if (std::count(is_fused_.begin(), is_fused_.end(), true) == static_cast<int>(rois_number_)) {
      timer_->cancel();
      // publish(*(fused_std_pair_.second));
      std::fill(is_fused_.begin(), is_fused_.end(), false);
      sub_std_pair_.second = nullptr;
      fused_std_pair_.second = nullptr;
    }
  }
  (roi_stdmap_.at(roi_i))[timestamp_nsec] = input_roi_msg;
}

void RoiPointCloudFusionNode::fuseOnSingleImage(
  const PointCloud2 & input_cloud_msg, [[maybe_unused]] const std::size_t image_id,
  const DetectedObjectsWithFeature & input_roi_msg,
  const sensor_msgs::msg::CameraInfo & camera_info, DetectedObjectsWithFeature & output_msg)
{
  if (input_cloud_msg.data.empty()) {
    return;
  }

  std::vector<DetectedObjectWithFeature> output_objects;
  for (const auto & feature_obj : input_roi_msg.feature_objects) {
    if (fuse_unknown_only_) {
      bool is_roi_label_unknown = feature_obj.object.classification.front().label ==
                                  autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
      if (is_roi_label_unknown) {
        output_objects.push_back(feature_obj);
      }
    } else {
      output_objects.push_back(feature_obj);
    }
  }
  if (output_objects.empty()) {
    return;
  }

  Eigen::Matrix4d projection;
  projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2), camera_info.p.at(3),
    camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6), camera_info.p.at(7),
    camera_info.p.at(8), camera_info.p.at(9), camera_info.p.at(10), camera_info.p.at(11);

  geometry_msgs::msg::TransformStamped transform_stamped;
  {
    const auto transform_stamped_optional = getTransformStamped(
      tf_buffer_, input_roi_msg.header.frame_id, input_cloud_msg.header.frame_id,
      input_roi_msg.header.stamp);
    if (!transform_stamped_optional) {
      return;
    }
    transform_stamped = transform_stamped_optional.value();
  }
  sensor_msgs::msg::PointCloud2 transformed_cloud;
  tf2::doTransform(input_cloud_msg, transformed_cloud, transform_stamped);
  // int min_x(camera_info.width), min_y(camera_info.height), max_x(0), max_y(0);
  std::vector<Eigen::Vector2d> projected_points;
  projected_points.reserve(transformed_cloud.data.size());
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed_cloud, "x"),
       iter_y(transformed_cloud, "y"), iter_z(transformed_cloud, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    if (*iter_z <= 0.0) {
      continue;
    }
    Eigen::Vector4d projected_point = projection * Eigen::Vector4d(*iter_x, *iter_y, *iter_z, 1.0);
    Eigen::Vector2d normalized_projected_point = Eigen::Vector2d(
      projected_point.x() / projected_point.z(), projected_point.y() / projected_point.z());
    if (
      0 <= static_cast<int>(normalized_projected_point.x()) &&
      static_cast<int>(normalized_projected_point.x()) < static_cast<int>(camera_info.width) - 1 &&
      0 <= static_cast<int>(normalized_projected_point.y()) &&
      static_cast<int>(normalized_projected_point.y()) < static_cast<int>(camera_info.height) - 1) {
      projected_points.push_back(normalized_projected_point);
    }
  }

  if (projected_points.empty()) {
    return;
  }

  for (const auto & feature_obj : output_objects) {
    output_msg.feature_objects.push_back(feature_obj);
  }
}
}  // namespace image_projection_based_fusion
