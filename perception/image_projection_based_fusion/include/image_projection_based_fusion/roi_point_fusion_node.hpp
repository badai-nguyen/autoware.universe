#ifndef IMAGE_PROJECTION_BASED_FUSION__ROI_POINT_FUSION_NODE_HPP_
#define IMAGE_PROJECTION_BASED_FUSION__ROI_POINT_FUSION_NODE_HPP_
#include <image_projection_based_fusion/utils/utils.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace image_projection_based_fusion
{

using autoware_auto_perception_msgs::msg::DetectedObject;
using autoware_auto_perception_msgs::msg::DetectedObjects;
using sensor_msgs::msg::PointCloud2;
using tier4_perception_msgs::msg::DetectedObjectsWithFeature;
using tier4_perception_msgs::msg::DetectedObjectWithFeature;

class RoiPointCloudFusionNode : public rclcpp::Node
{
private:
  /* data */
  std::map<std::size_t, sensor_msgs::msg::CameraInfo> camera_info_map_;
  std::pair<int64_t, PointCloud2::SharedPtr> sub_std_pair_;
  std::pair<int64_t, DetectedObjectsWithFeature::SharedPtr> fused_std_pair_;
  double timeout_ms_{};
  double match_threshold_ms_{};
  bool fuse_unknown_only_{true};
  std::size_t rois_number_{1};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;

  // offsets between cameras and lidars
  std::vector<double> input_offset_ms_;

  std::vector<bool> is_fused_{1};
  std::vector<std::map<int64_t, DetectedObjectsWithFeature::ConstSharedPtr>> roi_stdmap_;

public:
  explicit RoiPointCloudFusionNode(const rclcpp::NodeOptions & options);

protected:
  void cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr input_camera_info_msg,
    const std::size_t camera_id);

  void roiCallback(
    const DetectedObjectsWithFeature::ConstSharedPtr input_roi_msg, const std::size_t roi_i);

  void fuseOnSingleImage(
    const PointCloud2 & input_cloud_msg, const std::size_t image_id,
    const DetectedObjectsWithFeature & input_roi_msg,
    const sensor_msgs::msg::CameraInfo & camera_info, DetectedObjectsWithFeature & output_msg);
};
}  // namespace image_projection_based_fusion

#endif  // IMAGE_PROJECTION_BASED_FUSION__ROI_POINT_FUSION_NODE_HPP_
