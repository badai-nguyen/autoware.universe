// Copyright 2022 Tier IV, Inc.
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

#include "obstacle_pointcloud_based_validator/obstacle_pointcloud_based_validator.hpp"

#include <object_recognition_utils/object_recognition_utils.hpp>
#include <tier4_autoware_utils/geometry/boost_polygon_utils.hpp>

#include <boost/geometry.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace
{
inline pcl::PointXY toPCL(const double x, const double y)
{
  pcl::PointXY pcl_point;
  pcl_point.x = x;
  pcl_point.y = y;
  return pcl_point;
}

inline pcl::PointXY toPCL(const geometry_msgs::msg::Point & point)
{
  return toPCL(point.x, point.y);
}

inline pcl::PointXYZ toXYZ(const pcl::PointXY & point)
{
  return pcl::PointXYZ(point.x, point.y, 0.0);
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr toXYZ(
  const pcl::PointCloud<pcl::PointXY>::Ptr & pointcloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pointcloud_xyz->reserve(pointcloud->size());
  for (const auto & point : *pointcloud) {
    pointcloud_xyz->push_back(toXYZ(point));
  }
  return pointcloud_xyz;
}

}  // namespace

namespace obstacle_pointcloud_based_validator
{
namespace bg = boost::geometry;
using Shape = autoware_auto_perception_msgs::msg::Shape;
using Polygon2d = tier4_autoware_utils::Polygon2d;

// Conductor
Validator::Validator(PointsNumThresholdParam & points_num_threshold_param)
{
  points_num_threshold_param_.min_points_num = points_num_threshold_param.min_points_num;
  points_num_threshold_param_.max_points_num = points_num_threshold_param.max_points_num;
  points_num_threshold_param_.min_points_and_distance_ratio =
    points_num_threshold_param.min_points_and_distance_ratio;
}

// pcl::PointXYZ Validator::getObjectCenter(autoware_auto_perception_msgs::msg::DetectedObject &
// object)
// {
//   auto & object_position = object.kinematics.pose_with_covariance.pose.position;
//   return pcl::PointXYZ(object_position.x, object_position.y, object_position.z);
// }

size_t Validator::getThresholdPointCloud(
  const autoware_auto_perception_msgs::msg::DetectedObject & object)
{
  const auto object_label_id = object.classification.front().label;
  const auto object_distance = std::hypot(
    object.kinematics.pose_with_covariance.pose.position.x,
    object.kinematics.pose_with_covariance.pose.position.y);
  size_t threshold_pc = std::clamp(
    static_cast<size_t>(
      points_num_threshold_param_.min_points_and_distance_ratio.at(object_label_id) /
        object_distance +
      0.5f),
    static_cast<size_t>(points_num_threshold_param_.min_points_num.at(object_label_id)),
    static_cast<size_t>(points_num_threshold_param_.max_points_num.at(object_label_id)));
  return threshold_pc;
}

// Conductor
Validator2D::Validator2D(PointsNumThresholdParam & points_num_threshold_param)
: Validator(points_num_threshold_param)
{
}

void Validator2D::setInputCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_cloud)
{
  obstacle_pointcloud_.reset(new pcl::PointCloud<pcl::PointXY>);
  pcl::fromROSMsg(*input_cloud, *obstacle_pointcloud_);
  if (obstacle_pointcloud_->empty()) {
    return;
  }

  // setup kdtree
  kdtree_ = pcl::make_shared<pcl::search::KdTree<pcl::PointXY>>(false);
  kdtree_->setInputCloud(obstacle_pointcloud_);
}
pcl::PointXYZ Validator2D::getObjectCenter(
  autoware_auto_perception_msgs::msg::DetectedObject & object)
{
  auto & object_position = object.kinematics.pose_with_covariance.pose.position;
  return pcl::PointXYZ(object_position.x, object_position.y, 0.0);
}

std::optional<size_t> Validator2D::getPointCloudWithinObject(
  const autoware_auto_perception_msgs::msg::DetectedObject & object,
  const pcl::PointCloud<pcl::PointXY>::Ptr pointcloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::Vertices> vertices_array;
  pcl::Vertices vertices;
  Polygon2d poly2d =
    tier4_autoware_utils::toPolygon2d(object.kinematics.pose_with_covariance.pose, object.shape);
  if (bg::is_empty(poly2d)) return std::nullopt;

  pcl::PointCloud<pcl::PointXYZ>::Ptr poly3d(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < poly2d.outer().size(); ++i) {
    vertices.vertices.emplace_back(i);
    vertices_array.emplace_back(vertices);
    poly3d->emplace_back(poly2d.outer().at(i).x(), poly2d.outer().at(i).y(), 0.0);
  }

  pcl::CropHull<pcl::PointXYZ> cropper;  // don't be implemented PointXY by PCL
  cropper.setInputCloud(toXYZ(pointcloud));
  cropper.setDim(2);
  cropper.setHullIndices(vertices_array);
  cropper.setHullCloud(poly3d);
  cropper.setCropOutside(true);
  cropper.filter(*cropped_pointcloud);

  return cropped_pointcloud->size();
}

bool Validator2D::validate_object(
  const autoware_auto_perception_msgs::msg::DetectedObject & transformed_object)
{
  // get neighboor_pointcloud of object
  pcl::PointCloud<pcl::PointXY>::Ptr neightbor_poincloud(new pcl::PointCloud<pcl::PointXY>);
  std::vector<int> indices;
  std::vector<float> distances;
  const auto search_radius = getMaxRadius(transformed_object);
  if (!search_radius) {
    return false;
  }
  kdtree_->radiusSearch(
    pcl::PointXY(
      transformed_object.kinematics.pose_with_covariance.pose.position.x,
      transformed_object.kinematics.pose_with_covariance.pose.position.y),
    search_radius.value(), indices, distances);
  for (const auto & index : indices) {
    neightbor_poincloud->push_back(obstacle_pointcloud_->at(index));
  }

  // add neighbor pointcloud to debug_->addNeighborPointCloud
  // get number neighbor pointcloud within object polygon or shape
  const auto num = getPointCloudWithinObject(transformed_object, neightbor_poincloud);
  if (!num) return true;
  // get the threshold validation pointcloud for object at that distance
  size_t threshold_pointcloud_num = getThresholdPointCloud(transformed_object);
  if (num.value() < threshold_pointcloud_num) {
    return true;
  }
  return false;  // remove object
}
void Validator2D::get_neighbor_pc()
{
}

std::optional<float> Validator2D::getMaxRadius(
  const autoware_auto_perception_msgs::msg::DetectedObject & object)
{
  if (object.shape.type == Shape::BOUNDING_BOX || object.shape.type == Shape::CYLINDER) {
    return std::hypot(object.shape.dimensions.x * 0.5f, object.shape.dimensions.y * 0.5f);
  } else if (object.shape.type == Shape::POLYGON) {
    float max_dist = 0.0;
    for (const auto & point : object.shape.footprint.points) {
      const float dist = std::hypot(point.x, point.y);
      max_dist = max_dist < dist ? dist : max_dist;
    }
    return max_dist;
  } else {
    return std::nullopt;
  }
}

ObstaclePointCloudBasedValidator::ObstaclePointCloudBasedValidator(
  const rclcpp::NodeOptions & node_options)
: rclcpp::Node("obstacle_pointcloud_based_validator", node_options),
  objects_sub_(this, "~/input/detected_objects", rclcpp::QoS{1}.get_rmw_qos_profile()),
  obstacle_pointcloud_sub_(
    this, "~/input/obstacle_pointcloud",
    rclcpp::SensorDataQoS{}.keep_last(1).get_rmw_qos_profile()),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  sync_(SyncPolicy(10), objects_sub_, obstacle_pointcloud_sub_)
{
  points_num_threshold_param_.min_points_num =
    declare_parameter<std::vector<int64_t>>("min_points_num");
  points_num_threshold_param_.max_points_num =
    declare_parameter<std::vector<int64_t>>("max_points_num");
  points_num_threshold_param_.min_points_and_distance_ratio =
    declare_parameter<std::vector<double>>("min_points_and_distance_ratio");

  using_2d_validator_ = declare_parameter<bool>("using_2d_validator");

  using std::placeholders::_1;
  using std::placeholders::_2;

  sync_.registerCallback(
    std::bind(&ObstaclePointCloudBasedValidator::onObjectsAndObstaclePointCloud, this, _1, _2));
  validator_ = std::make_unique<Validator2D>(points_num_threshold_param_);
  // TODO(badai-nguyen): change to single bind function

  objects_pub_ = create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>(
    "~/output/objects", rclcpp::QoS{1});

  const bool enable_debugger = declare_parameter<bool>("enable_debugger", false);
  if (enable_debugger) debugger_ = std::make_shared<Debugger>(this);
}
void ObstaclePointCloudBasedValidator::on3dObjectsAndObstaclePointCloud(
  const autoware_auto_perception_msgs::msg::DetectedObjects::ConstSharedPtr & input_objects,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_obstacle_pointcloud)
{
  autoware_auto_perception_msgs::msg::DetectedObjects output, removed_objects;
  output.header = input_objects->header;
  removed_objects.header = input_objects->header;

  // Transform to pointcloud frame
  autoware_auto_perception_msgs::msg::DetectedObjects transformed_objects;
  if (!object_recognition_utils::transformObjects(
        *input_objects, input_obstacle_pointcloud->header.frame_id, tf_buffer_,
        transformed_objects)) {
    return;
  }

  // Convert to PCL
  pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*input_obstacle_pointcloud, *obstacle_pointcloud);
  if (obstacle_pointcloud->empty()) {
    return;
  }

  // Create Kd-tree to search neighbor pointcloud to reduce cost
  pcl::search::Search<pcl::PointXYZ>::Ptr kdtree =
    pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>(false);
  kdtree->setInputCloud(obstacle_pointcloud);
  for (size_t i = 0; i < transformed_objects.objects.size(); ++i) {
    const auto & transformed_object = transformed_objects.objects.at(i);
    const auto object_label_id = transformed_object.classification.front().label;
    const auto & object = input_objects->objects.at(i);
    const auto & transformed_object_position =
      transformed_object.kinematics.pose_with_covariance.pose.position;
    const auto search_radius = getMaxRadius3D(transformed_object);
    if (!search_radius) {
      output.objects.push_back(object);
      continue;
    }
    // Search neighbor pointcloud to reduce cost
    pcl::PointCloud<pcl::PointXYZ>::Ptr neighbor_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> indices;
    std::vector<float> distances;
    pcl::PointXYZ trans_obj_position_pcl;
    trans_obj_position_pcl.x = transformed_object_position.x;
    trans_obj_position_pcl.y = transformed_object_position.y;
    trans_obj_position_pcl.z = transformed_object_position.z;
    kdtree->radiusSearch(trans_obj_position_pcl, search_radius.value(), indices, distances);
    for (const auto & index : indices) {
      neighbor_pointcloud->push_back(obstacle_pointcloud->at(index));
    }

    if (debugger_) debugger_->addNeighborPointcloud(neighbor_pointcloud);
    // Filter object that have few pointcloud in them
    const auto num = getPointCloudNumWithinShape(transformed_object, neighbor_pointcloud);
    const auto object_distance =
      std::hypot(transformed_object_position.x, transformed_object_position.y);
    size_t min_pointcloud_num = std::clamp(
      static_cast<size_t>(
        points_num_threshold_param_.min_points_and_distance_ratio.at(object_label_id) /
          object_distance +
        0.5f),
      static_cast<size_t>(points_num_threshold_param_.min_points_num.at(object_label_id)),
      static_cast<size_t>(points_num_threshold_param_.max_points_num.at(object_label_id)));
    if (num) {
      (min_pointcloud_num <= num.value()) ? output.objects.push_back(object)
                                          : removed_objects.objects.push_back(object);
    } else {
      output.objects.push_back(object);
    }
  }
  objects_pub_->publish(output);
  if (debugger_) {
    debugger_->publishRemovedObjects(removed_objects);
    debugger_->publishNeighborPointcloud(input_obstacle_pointcloud->header);
    debugger_->publishPointcloudWithinPolygon(input_obstacle_pointcloud->header);
  }
}
void ObstaclePointCloudBasedValidator::onObjectsAndObstaclePointCloud(
  const autoware_auto_perception_msgs::msg::DetectedObjects::ConstSharedPtr & input_objects,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & input_obstacle_pointcloud)
{
  autoware_auto_perception_msgs::msg::DetectedObjects output, removed_objects;
  output.header = input_objects->header;
  removed_objects.header = input_objects->header;

  // Transform to pointcloud frame
  autoware_auto_perception_msgs::msg::DetectedObjects transformed_objects;
  if (!object_recognition_utils::transformObjects(
        *input_objects, input_obstacle_pointcloud->header.frame_id, tf_buffer_,
        transformed_objects)) {
    // objects_pub_->publish(*input_objects);
    return;
  }
  validator_->setInputCloud(input_obstacle_pointcloud);

  for (size_t i = 0; i < transformed_objects.objects.size(); ++i) {
    const auto & transformed_object = transformed_objects.objects.at(i);
    const auto & object = input_objects->objects.at(i);
    const auto validity = validator_->validate_object(transformed_object);
    if (validity) {
      output.objects.push_back(object);
    } else {
      removed_objects.objects.push_back(object);
    }
  }

  objects_pub_->publish(output);
  if (debugger_) {
    debugger_->publishRemovedObjects(removed_objects);
    debugger_->publishNeighborPointcloud(input_obstacle_pointcloud->header);
    debugger_->publishPointcloudWithinPolygon(input_obstacle_pointcloud->header);
  }
}
std::optional<size_t> ObstaclePointCloudBasedValidator::getPointCloudNumWithinShape(
  const autoware_auto_perception_msgs::msg::DetectedObject & object,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_pointcloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::Vertices> vertices_array;
  pcl::Vertices vertices;

  auto const & object_position = object.kinematics.pose_with_covariance.pose.position;
  auto const object_height = object.shape.dimensions.z;
  auto z_min = object_position.z - object_height / 2.0f;
  auto z_max = object_position.z + object_height / 2.0f;

  Polygon2d poly2d =
    tier4_autoware_utils::toPolygon2d(object.kinematics.pose_with_covariance.pose, object.shape);
  if (bg::is_empty(poly2d)) return std::nullopt;

  pcl::PointCloud<pcl::PointXYZ>::Ptr poly3d(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < poly2d.outer().size(); ++i) {
    vertices.vertices.emplace_back(i);
    vertices_array.emplace_back(vertices);
    poly3d->emplace_back(poly2d.outer().at(i).x(), poly2d.outer().at(i).y(), 0.0);
  }
  pcl::CropHull<pcl::PointXYZ> cropper;
  cropper.setInputCloud(pointcloud);
  cropper.setDim(2);
  cropper.setHullIndices(vertices_array);
  cropper.setHullCloud(poly3d);
  cropper.setCropOutside(true);
  cropper.filter(*cropped_pointcloud_2d);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_pointcloud_3d(new pcl::PointCloud<pcl::PointXYZ>);
  cropped_pointcloud_3d->reserve(cropped_pointcloud_2d->size());
  for (const auto & point : *cropped_pointcloud_2d) {
    if (point.z > z_min && point.z < z_max) {
      cropped_pointcloud_3d->push_back(point);
    }
  }
  if (debugger_) debugger_->addPointcloudWithinPolygon(cropped_pointcloud_3d);
  return cropped_pointcloud_3d->size();
}

std::optional<size_t> ObstaclePointCloudBasedValidator::getPointCloudNumWithinPolygon(
  const autoware_auto_perception_msgs::msg::DetectedObject & object,
  const pcl::PointCloud<pcl::PointXY>::Ptr pointcloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::Vertices> vertices_array;
  pcl::Vertices vertices;

  Polygon2d poly2d =
    tier4_autoware_utils::toPolygon2d(object.kinematics.pose_with_covariance.pose, object.shape);
  if (bg::is_empty(poly2d)) return std::nullopt;

  pcl::PointCloud<pcl::PointXYZ>::Ptr poly3d(new pcl::PointCloud<pcl::PointXYZ>);

  for (size_t i = 0; i < poly2d.outer().size(); ++i) {
    vertices.vertices.emplace_back(i);
    vertices_array.emplace_back(vertices);
    poly3d->emplace_back(poly2d.outer().at(i).x(), poly2d.outer().at(i).y(), 0.0);
  }

  pcl::CropHull<pcl::PointXYZ> cropper;  // don't be implemented PointXY by PCL
  cropper.setInputCloud(toXYZ(pointcloud));
  cropper.setDim(2);
  cropper.setHullIndices(vertices_array);
  cropper.setHullCloud(poly3d);
  cropper.setCropOutside(true);
  cropper.filter(*cropped_pointcloud);

  if (debugger_) debugger_->addPointcloudWithinPolygon(cropped_pointcloud);
  return cropped_pointcloud->size();
}

std::optional<float> ObstaclePointCloudBasedValidator::getMaxRadius2D(
  const autoware_auto_perception_msgs::msg::DetectedObject & object)
{
  if (object.shape.type == Shape::BOUNDING_BOX || object.shape.type == Shape::CYLINDER) {
    return std::hypot(object.shape.dimensions.x * 0.5f, object.shape.dimensions.y * 0.5f);
  } else if (object.shape.type == Shape::POLYGON) {
    float max_dist = 0.0;
    for (const auto & point : object.shape.footprint.points) {
      const float dist = std::hypot(point.x, point.y);
      max_dist = max_dist < dist ? dist : max_dist;
    }
    return max_dist;
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "unknown shape type");
    return std::nullopt;
  }
}
std::optional<float> ObstaclePointCloudBasedValidator::getMaxRadius3D(
  const autoware_auto_perception_msgs::msg::DetectedObject & object)
{
  const auto max_dist_2d = getMaxRadius2D(object);
  if (!max_dist_2d) {
    return std::nullopt;
  }
  const auto object_height = object.shape.dimensions.z;
  return std::hypot(max_dist_2d.value(), object_height / 2.0);
}

}  // namespace obstacle_pointcloud_based_validator

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(
  obstacle_pointcloud_based_validator::ObstaclePointCloudBasedValidator)
