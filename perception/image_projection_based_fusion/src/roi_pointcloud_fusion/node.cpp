#include "image_projection_based_fusion/roi_pointcloud_fusion/node.hpp"
#include "geometry_msgs/msg"
#include "image_projection_based_fusion/utils/utils.hpp"
#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#endif

namespace image_projection_based_fusion
{
RoiPointcloudFusion::RoiPointcloudFusion(const rclcpp::NodeOptioins & options)
: FusionNode<Pointcloud2, DetectedObjectWithFeature>("roi_pointcloud_fusion", options)
{
  cluster_tolerance_ = declare_parameter("cluster_tolerance",0.7);
}

void RoiPointcloudFusion::fuseOnSingleImage(
    const PointCloud2 & input_point_cloud_msg,
    const std::size_t image_id,
    const DetectedObjectsWithFeature & input_roi_msg,
    const sensor_msgs::msg::CameraInfo & camera_info,
    DetectedObjectsWithFeature & output_cluster_msg)
{
  Eigen::Maxtrix4d projection;
  projection << camera_info.p.at(0), camera_info.p.at(1), camera_info.p.at(2), camera_info.p.at(3),
    camera_info.p.at(4), camera_info.p.at(5), camera_info.p.at(6), camera_info.p.at(7),
    camera_info.p.at(8), camera_info.p.at(9), camera_info.p.at(10), camera_info.p.at(11);
  geometry_msgs::msg::TransformStamped transform_stamped;
  {
    const auto transform_stamped_optional = getTransformStamped(tf_buffer_, camera_info.header.frame_id,
      input_point_cloud_msg.header.frame_id, camera_info.header.stamp);
    if (!transform_stamped_optional){
      return;
    }
    transform_stamped = transform_stamped_optional.value();
  }
  //transform pointcloud to camera tf
  sensor_msgs::mgs::PointCloud2 transformed_pointcloud;
  tf2::doTransform(input_point_cloud_msg.points, transformed_pointcloud, transform_stamped);
  int min_x(camera_info.width),  min_y(camera_info.height), max_x(0),max_y(0);
  std::vector<Eigen::Vector2d> projected_points;
  projected_points.reverse(transformed_pointcloud.data.size());

  std:::vector<DetectedObjectWithFeature> unknown_roi_objects;
  for (const auto & feature_obj : input_roi_msg.feature_objects){
    bool is_roi_label_unknown = feature_obj.object.classification.front().label == autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
    if(is_roi_label_unknown){
      unknown_roi_objects.push_back(feature_obj);
    }
  }
  if(unknown_roi_objects.empty()){
    continue;
  }

  std::vector<pcl::PointCloud<pcl::PointXYZ>> unknown_clusters;
  unknown_clusters.reverse(unknown_roi_objects.size());

  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed_pointcloud,"x"),
    iter_y(transformed_pointcloud,"y"), iter_z(transformed_pointcloud,"z");
    iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z){
    if (*iter_z <= 0.0){
      continue;
    }
    Eigen::Vector4d project_point = projection * Eigen::Vector4d(*iter_x, *iter_y, *iter_z, 1.0);
    Eigen::Vector2d normalized_projected_point = Eigen::Vector2d(projected_point.x()/ projected_point.z(), projected_point.y() / projected_point.z());
    if (0 <= static_cast<int>(normalized_projected_point.x()) &&
     static_cast<int>(normalized_projected_point.x()) < static_cast<int>(camera_info.width) - 1 &&
     0 <= static_cast<int>(normalized_projected_point.y()) &&
     static_cast<int>(normalized_projected_point.y()) < static_cast<int>(camera_info.height) -1){
      projected_points.push_back(normalized_projected_point);
     }
    if(projected_points.empty()){
      continue;
    }
  }

}
} //namespace image_projection_based_fusion