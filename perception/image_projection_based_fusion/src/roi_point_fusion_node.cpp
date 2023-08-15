
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
  rois_number_ = static_cast<std::size_t>(declare_parameter("rois_number", 1));
  match_threshold_ms_ = declare_parameter<double>("match_threshold_ms");
  timeout_ms_ = declare_parameter<double>("timeout_ms");
  fuse_unknown_only_ = declare_parameter<bool>("fuse_unknown_only", false);

  input_rois_topics_.resize(rois_number_);
  input_camera_topics_.resize(rois_number_);
  input_camera_info_topics_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    input_rois_topics_.at(roi_i) = declare_parameter<std::string>(
      "input/rois" + std::to_string(roi_i),
      "/perception/object_recognition/detection/rois" + std::to_string(roi_i));
    input_camera_info_topics_.at(roi_i) = declare_parameter<std::string>(
      "input/camera_info" + std::to_string(roi_i),
      "/sensing/camera/camera" + std::to_string(roi_i) + "/camera_info");

    input_camera_topics_.at(roi_i) = declare_parameter<std::string>(
      "input/image" + std::to_string(roi_i),
      "/sensing/camera/camera" + std::to_string(roi_i) + "/image_rect_color");
  }

  input_offset_ms_ = declare_parameter("input_offset_ms", std::vector<double>{});
  if (!input_offset_ms_.empty() && rois_number_ > input_offset_ms_.size()) {
    throw std::runtime_error("The number of offsets does not match the number of topics.");
  }

  // sub camera info
  camera_info_subs_.resize(rois_number_);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    std::function<void(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)> fnc =
      std::bind(&RoiPointCloudFusionNode::cameraInfoCallback, this, std::placeholders::_1, roi_i);
    camera_info_subs_.at(roi_i) = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      input_camera_info_topics_.at(roi_i), rclcpp::QoS{1}.best_effort(), fnc);
  }

  // sub rois

  rois_subs_.resize(rois_number_);
  roi_stdmap_.resize(rois_number_);
  is_fused_.resize(rois_number_, false);
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    std::function<void(DetectedObjectsWithFeature::ConstSharedPtr msg)> roi_callback =
      std::bind(&RoiPointCloudFusionNode::roiCallback, this, std::placeholders::_1, roi_i);
    rois_subs_.at(roi_i) = this->create_subscription<DetectedObjectsWithFeature>(
      input_rois_topics_.at(roi_i), rclcpp::QoS{1}.best_effort(), roi_callback);
  }

  // subscribers pointcloud
  std::function<void(sensor_msgs::msg::PointCloud2::ConstSharedPtr mgs)> sub_callback =
    std::bind(&RoiPointCloudFusionNode::subCallback, this, std::placeholders::_1);
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "input", rclcpp::QoS(1).best_effort(), sub_callback);

  // publisher
  pub_ptr_ = this->create_publisher<DetectedObjectsWithFeature>("output", rclcpp::QoS(1));

  // set timer
  const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double, std::milli>(timeout_ms_));
  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&RoiPointCloudFusionNode::timer_callback, this));
}

geometry_msgs::msg::Point RoiPointCloudFusionNode::getCentroid(
  const sensor_msgs::msg::PointCloud2 & pointcloud)
{
  geometry_msgs::msg::Point centroid;
  centroid.x = 0.0f;
  centroid.y = 0.0f;
  centroid.z = 0.0f;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(pointcloud, "x"),
       iter_y(pointcloud, "y"), iter_z(pointcloud, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    centroid.x += *iter_x;
    centroid.y += *iter_y;
    centroid.z += *iter_z;
  }
  const size_t size = pointcloud.width * pointcloud.height;
  centroid.x = centroid.x / static_cast<float>(size);
  centroid.y = centroid.y / static_cast<float>(size);
  centroid.z = centroid.z / static_cast<float>(size);
  return centroid;
}

void RoiPointCloudFusionNode::timer_callback()
{
  using std::chrono_literals::operator""ms;
  timer_->cancel();
  if (mutex_.try_lock()) {
    if (sub_std_pair_.second != nullptr) {
      publish(fused_objects_);
    }
    std::fill(is_fused_.begin(), is_fused_.end(), false);
    fused_objects_.feature_objects.clear();
    mutex_.unlock();
  } else {
    try {
      std::chrono::nanoseconds period = 10ms;
      setPeriod(period.count());
    } catch (rclcpp::exceptions::RCLError & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", e.what());
    }
    timer_->reset();
  }
}

void RoiPointCloudFusionNode::setPeriod(const int64_t new_period)
{
  if (!timer_) {
    return;
  }
  int64_t old_period = 0;
  rcl_ret_t ret = rcl_timer_get_period(timer_->get_timer_handle().get(), &old_period);
  if (ret != RCL_RET_OK) {
    rclcpp::exceptions::throw_from_rcl_error(ret, "Couldn't get old period");
  }
  ret = rcl_timer_exchange_period(timer_->get_timer_handle().get(), new_period, &old_period);
  if (ret != RCL_RET_OK) {
    rclcpp::exceptions::throw_from_rcl_error(ret, "couldn't exchange period");
  }
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
  RCLCPP_INFO(get_logger(), "starting roiCallback ...");
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
    if (sub_std_pair_.second == nullptr) {
      fused_objects_.header = input_roi_msg->header;
    }
    fuseOnSingleImage(
      *(sub_std_pair_.second), roi_i, *input_roi_msg, camera_info_map_.at(roi_i), fused_objects_);
    is_fused_.at(roi_i) = true;

    if (std::count(is_fused_.begin(), is_fused_.end(), true) == static_cast<int>(rois_number_)) {
      timer_->cancel();
      // publish(*(fused_std_pair_.second));
      std::fill(is_fused_.begin(), is_fused_.end(), false);
      sub_std_pair_.second = nullptr;
      fused_objects_.feature_objects.clear();
    }
  }
  (roi_stdmap_.at(roi_i))[timestamp_nsec] = input_roi_msg;
}

void RoiPointCloudFusionNode::subCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg)
{
  RCLCPP_INFO(get_logger(), "starting subCallback ...");
  if (sub_std_pair_.second != nullptr) {
    RCLCPP_INFO(get_logger(), "subCallback publishing fused result ...");
    timer_->cancel();
    publish(fused_objects_);
    fused_objects_.feature_objects.clear();
    sub_std_pair_.second = nullptr;
    std::fill(is_fused_.begin(), is_fused_.end(), false);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double, std::milli>(timeout_ms_));
  RCLCPP_INFO(get_logger(), "subCallback timeout_ms_: %f", timeout_ms_);
  try {
    setPeriod(period.count());
  } catch (rclcpp::exceptions::RCLError & ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s", ex.what());
  }
  RCLCPP_INFO(get_logger(), "subCallback continuing 1 ...");
  timer_->reset();
  DetectedObjectsWithFeature output_msg;

  RCLCPP_INFO(get_logger(), "subCallback continuing 2 ...");
  output_msg.header = input_msg->header;

  RCLCPP_INFO(get_logger(), "subCallback continuing 3 ...");
  int64_t timestamp_nsec =
    output_msg.header.stamp.sec * (int64_t)1e9 + output_msg.header.stamp.nanosec;

  RCLCPP_INFO(get_logger(), "subCallback continuing 4 ...");
  RCLCPP_INFO(get_logger(), "subCallback timestamp_nsec: %ld", timestamp_nsec);
  // if (fused_std_pair_.second == nullptr) {
  //   RCLCPP_INFO(get_logger(), "subCallback initializing fused_std_pair_ ...");
  //   fused_std_pair_.first = timestamp_nsec;
  //   *(fused_std_pair_.second) = output_msg;
  //   RCLCPP_INFO(get_logger(), "subCallback initialized fused_std_pair_ ...");
  // }
  RCLCPP_INFO(get_logger(), "subCallback starting fusing ...");
  for (std::size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    if (camera_info_map_.find(roi_i) == camera_info_map_.end()) {
      continue;
    }
    RCLCPP_INFO(get_logger(), "subCallback fusing rois %zu", roi_i);
    if ((roi_stdmap_.at(roi_i)).size() > 0) {
      int64_t min_interval = 1e9;
      int64_t matched_stamp = -1;
      std::list<int64_t> outdate_stamps;
      for (const auto & [k, v] : roi_stdmap_.at(roi_i)) {
        RCLCPP_INFO(get_logger(), "subCallback checking roi_stdmap_...");
        int64_t new_stamp = timestamp_nsec + input_offset_ms_.at(roi_i) * (int64_t)1e6;
        int64_t interval = abs(int64_t(k) - new_stamp);
        if (interval <= min_interval && interval <= match_threshold_ms_ * (int64_t)1e6) {
          min_interval = interval;
          matched_stamp = k;
        } else if (int64_t(k) < new_stamp && interval > match_threshold_ms_ * (int64_t)1e6) {
          outdate_stamps.push_back(int64_t(k));
        }
      }

      // remove outdated stamps
      for (auto stamp : outdate_stamps) {
        (roi_stdmap_.at(roi_i)).erase(stamp);
      }

      // fuseOnSingle

      if (matched_stamp != -1) {
        RCLCPP_INFO(get_logger(), "subCallback fusing on single image");
        fuseOnSingleImage(
          *input_msg, roi_i, *((roi_stdmap_.at(roi_i))[matched_stamp]), camera_info_map_.at(roi_i),
          output_msg);
        (roi_stdmap_.at(roi_i)).erase(matched_stamp);
        is_fused_.at(roi_i) = true;
      }
    }
  }
  if (std::count(is_fused_.begin(), is_fused_.end(), false) == static_cast<int>(rois_number_)) {
    timer_->cancel();
    RCLCPP_INFO(get_logger(), "subCallback trying publishing result");
    publish(output_msg);
    sub_std_pair_.second = nullptr;
    fused_objects_.feature_objects.clear();
    std::fill(is_fused_.begin(), is_fused_.end(), false);
  } else {
    RCLCPP_INFO(get_logger(), "subCallback updating sub_std_pair_ and fused_std_pair");
    sub_std_pair_.first = int64_t(timestamp_nsec);
    sub_std_pair_.second = std::make_shared<PointCloud2>(*input_msg);
    // fused_std_pair_.first = int64_t(timestamp_nsec);
    // *(fused_std_pair_.second) = output_msg;
    fused_objects_ = output_msg;
    RCLCPP_INFO(get_logger(), "subCallback updated sub_std_pair_ and fused_std_pair");
  }
}

void RoiPointCloudFusionNode::fuseOnSingleImage(
  const PointCloud2 & input_cloud_msg, [[maybe_unused]] const std::size_t image_id,
  const DetectedObjectsWithFeature & input_roi_msg,
  const sensor_msgs::msg::CameraInfo & camera_info, DetectedObjectsWithFeature & output_msg)
{
  if (input_cloud_msg.data.empty()) {
    return;
  }
  RCLCPP_INFO(get_logger(), "starting fuseOnSingleImage ...");
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

  RCLCPP_INFO(get_logger(), "add feature objects %zu", output_objects.size());

  std::vector<PointCloud> clusters;
  clusters.resize(output_objects.size());
  RCLCPP_INFO(get_logger(), "clusters size %zu", clusters.size());
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
  // std::vector<Eigen::Vector2d> projected_points;
  // projected_points.reserve(transformed_cloud.data.size());
  RCLCPP_INFO(get_logger(), "transformed_cloud size %zu", transformed_cloud.data.size());
  // RCLCPP_INFO(get_logger(), "projected_points size %zu", projected_points.size());
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed_cloud, "x"),
       iter_y(transformed_cloud, "y"), iter_z(transformed_cloud, "z"),
       iter_orig_x(input_cloud_msg, "x"), iter_orig_y(input_cloud_msg, "y"),
       iter_orig_z(input_cloud_msg, "z");
       iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_orig_x, ++iter_orig_y, ++iter_orig_z) {
    if (*iter_z <= 0.0) {
      continue;
    }
    Eigen::Vector4d projected_point = projection * Eigen::Vector4d(*iter_x, *iter_y, *iter_z, 1.0);
    Eigen::Vector2d normalized_projected_point = Eigen::Vector2d(
      projected_point.x() / projected_point.z(), projected_point.y() / projected_point.z());
    // if (
    //   0 <= static_cast<int>(normalized_projected_point.x()) &&
    //   static_cast<int>(normalized_projected_point.x()) < static_cast<int>(camera_info.width) - 1
    //   && 0 <= static_cast<int>(normalized_projected_point.y()) &&
    //   static_cast<int>(normalized_projected_point.y()) < static_cast<int>(camera_info.height) -
    //   1) { projected_points.push_back(normalized_projected_point);
    // }

    for (std::size_t i = 0; i < output_objects.size(); ++i) {
      auto & feature_obj = output_objects.at(i);
      const auto & check_roi = feature_obj.feature.roi;
      auto & cluster = clusters.at(i);

      if (
        check_roi.x_offset <= normalized_projected_point.x() &&
        check_roi.y_offset <= normalized_projected_point.y() &&
        check_roi.x_offset + check_roi.width >= normalized_projected_point.x() &&
        check_roi.y_offset + check_roi.height >= normalized_projected_point.y()) {
        cluster.push_back(pcl::PointXYZ(*iter_orig_x, *iter_orig_y, *iter_orig_z));
      }
    }
  }
  // TODO: refine clusters before convert to PointCloud2
  RCLCPP_INFO(get_logger(), "extract clusters inside rois ");
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    const auto & cluster = clusters.at(i);
    auto & feature_obj = output_objects.at(i);
    sensor_msgs::msg::PointCloud2 ros_pc_cluster;
    pcl::toROSMsg(cluster, ros_pc_cluster);
    ros_pc_cluster.header = input_cloud_msg.header;
    feature_obj.feature.cluster = ros_pc_cluster;
    feature_obj.object.kinematics.pose_with_covariance.pose.position = getCentroid(ros_pc_cluster);

    output_msg.feature_objects.push_back(feature_obj);
  }

  RCLCPP_INFO(get_logger(), "completed fusing on single image");
}
void RoiPointCloudFusionNode::publish(const DetectedObjectsWithFeature & output_msg)
{
  if (pub_ptr_->get_subscription_count() < 1) {
    return;
  }
  pub_ptr_->publish(output_msg);
}
}  // namespace image_projection_based_fusion

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(image_projection_based_fusion::RoiPointCloudFusionNode)
