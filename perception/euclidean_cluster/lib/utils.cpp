// Copyright 2021 Tier IV, Inc.
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
#include "euclidean_cluster/utils.hpp"

#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tier4_perception_msgs/msg/detected_object_with_feature.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

float Average(std::vector<float> v)
{
  if (v.size() == 0) {
    return 0.0;
  }
  float avg = 0.0;
  for (size_t i = 0; i < v.size(); ++i) {
    avg += v[i];
  }
  return avg / static_cast<float>(v.size());
}

float Deviation(std::vector<float> v)
{
  if (v.size() == 0) {
    return 0.0;
  }
  float avg = Average(v);
  float E = 0.0;
  for (size_t i = 0; i < v.size(); ++i) {
    E += (v[i] - avg) * (v[i] - avg);
  }
  return std::sqrt(E / static_cast<float>(v.size()));
}

namespace euclidean_cluster
{
geometry_msgs::msg::Point getCentroid(const sensor_msgs::msg::PointCloud2 & pointcloud)
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

void convertPointCloudClusters2Msg(
  const std_msgs::msg::Header & header,
  const std::vector<pcl::PointCloud<pcl::PointXYZI>> & clusters,
  tier4_perception_msgs::msg::DetectedObjectsWithFeature & msg)
{
  msg.header = header;
  for (const auto & cluster : clusters) {
    sensor_msgs::msg::PointCloud2 ros_pointcloud;
    tier4_perception_msgs::msg::DetectedObjectWithFeature feature_object;
    pcl::toROSMsg(cluster, ros_pointcloud);
    ros_pointcloud.header = header;
    feature_object.feature.cluster = ros_pointcloud;
    feature_object.object.kinematics.pose_with_covariance.pose.position =
      getCentroid(ros_pointcloud);
    autoware_auto_perception_msgs::msg::ObjectClassification classification;
    classification.label = autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
    classification.probability = 1.0f;
    feature_object.object.classification.emplace_back(classification);
    msg.feature_objects.push_back(feature_object);
  }
}
void convertObjectMsg2SensorMsg(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature & input,
  sensor_msgs::msg::PointCloud2 & output)
{
  output.header = input.header;

  size_t pointcloud_size = 0;
  for (const auto & feature_object : input.feature_objects) {
    pointcloud_size += feature_object.feature.cluster.width * feature_object.feature.cluster.height;
  }
  sensor_msgs::PointCloud2Modifier modifier(output);
  modifier.setPointCloud2Fields(
    7, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32, "intensity", 1,
    sensor_msgs::msg::PointField::FLOAT32, "rgb", 1, sensor_msgs::msg::PointField::FLOAT32,
    "cluster", 1, sensor_msgs::msg::PointField::UINT32, "intensitydistribution", 1,
    sensor_msgs::msg::PointField::UINT8);
  modifier.resize(pointcloud_size);

  sensor_msgs::PointCloud2Iterator<float> iter_out_x(output, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_out_y(output, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_out_z(output, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_out_intensity(output, "intensity");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_out_r(output, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_out_g(output, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_out_b(output, "b");
  sensor_msgs::PointCloud2Iterator<uint32_t> iter_out_cluster(output, "cluster");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_out_intensitydistribution(
    output, "intensitydistribution");

  constexpr uint8_t color_data[] = {200, 0,   0, 0,   200, 0,   0, 0,   200,
                                    200, 200, 0, 200, 0,   200, 0, 200, 200};  // 6 pattern
  for (size_t i = 0; i < input.feature_objects.size(); ++i) {
    const auto & feature_object = input.feature_objects.at(i);
    std::vector<float> intensity_vec;

    sensor_msgs::PointCloud2ConstIterator<float> iter_in_x(feature_object.feature.cluster, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_in_y(feature_object.feature.cluster, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_in_z(feature_object.feature.cluster, "z");
    sensor_msgs::PointCloud2ConstIterator<float> iter_in_intensity(
      feature_object.feature.cluster, "intensity");

    for (; iter_in_intensity != iter_in_intensity.end(); ++iter_in_intensity) {
      intensity_vec.push_back(*iter_in_intensity);
    }

    float intensity_avg = Average(intensity_vec) * 5;    // for good rendering
    float intensity_std = Deviation(intensity_vec) * 5;  // for good rendering
    uint8_t avg_std = static_cast<uint8_t>(
      intensity_avg + intensity_std < 255.0 ? intensity_avg + intensity_std : 255);

    for (; iter_in_x != iter_in_x.end();
         ++iter_in_x, ++iter_in_y, ++iter_in_z, ++iter_in_intensity, ++iter_out_x, ++iter_out_y,
         ++iter_out_z, ++iter_out_intensity, ++iter_out_r, ++iter_out_g, ++iter_out_b,
         ++iter_out_cluster, ++iter_out_intensitydistribution) {
      *iter_out_x = *iter_in_x;
      *iter_out_y = *iter_in_y;
      *iter_out_z = *iter_in_z;
      *iter_out_intensity = *iter_in_intensity;
      *iter_out_r = color_data[3 * (i % 6) + 0];
      *iter_out_g = color_data[3 * (i % 6) + 1];
      *iter_out_b = color_data[3 * (i % 6) + 2];
      *iter_out_cluster = i;
      *iter_out_intensitydistribution = intensity_avg + intensity_std;
    }
  }

  output.width = pointcloud_size;
  output.height = 1;
  output.is_dense = false;
}
}  // namespace euclidean_cluster
