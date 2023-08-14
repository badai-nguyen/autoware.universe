#ifndef IMAGE_PROJECTION_BASED_FUSION__ROI_POINTCLOUD_FUSION__NODE_HPP_
#define IMAGE_PROJECTION_BASED_FUSION__ROI_POINTCLOUD_FUSION__NODE_HPP_

#include "image_projection_based_fusion/fusion_node.hpp"
#include <memory>
#include <sensor_msgs/msg/point_cloud2.hpp>
namespace image_projection_based_fusion
{
class RoiPointcloudFusion : public FusionNode<PointCloud2, [[maybe_unused]] PointCloud2, DetectedObjectWithFeature>
{
private:
  double cluster_tolerance_;

public:
  explicit RoiPointcloudFusion(const rclcpp::NodeOptions & options);
protected:
  void fuseOnSingleImage(
    const PointCloud2 & input_pointcloud_msg,
    const std::size_t image_id,
    const sensor_msgs::msg::CameraInfo & camera_info,
    DetectedObjectsWithFeature & output_cluster_msg) override;

};
} //namespace image_projection_based_fusion

#endif