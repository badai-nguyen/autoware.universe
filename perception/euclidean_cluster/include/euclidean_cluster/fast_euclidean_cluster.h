
// Copyright 2023 Autoware Foundation
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

#pragma once
#ifndef EUCLIDEAN_CLUSTER__FAST_EUCLIDEAN_CLUSTER_H_
#define EUCLIDEAN_CLUSTER__FAST_EUCLIDEAN_CLUSTER_H_

#endif  // EUCLIDEAN_CLUSTER__FAST_EUCLIDEAN_CLUSTER_H_

#include <omp.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>
struct ClassifedPoint
{
  size_t point_indice;
  int class_index;
};

bool getClass(const ClassifedPoint & p0, const ClassifedPoint & p1)
{
  return p0.class_index < p1.class_index;
}

void fastClusterExtract(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud, const int min_cluster_size,
  const double tolerance, const int max_cluster_size, std::vector<pcl::PointCloud<pcl::PointXYZ>> & clusters)
{
  std::vector<pcl::PointIndices> cluster_indices;
  if (cloud->points.size() < static_cast<size_t>(min_cluster_size)) {
    cluster_indices.clear();
    clusters.clear();
    return;
  }

  cluster_indices.resize(cloud->points.size());

  pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtreeflann;
  cloud_kdtreeflann.setInputCloud(cloud);

  std::vector<ClassifedPoint> classified_points;
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    ClassifedPoint current_point;
    current_point.point_indice = i;
    current_point.class_index = 0;
    classified_points.push_back(current_point);
  }
  int class_idx = 1, temp_class_idx = -1;
  std::vector<float> nn_distances;
  std::vector<int> nn_indices;
  for (size_t i = 0; i < classified_points.size(); ++i) {
    auto * point = &classified_points[i];
    if (point->class_index == 0) {
      nn_distances.clear();
      nn_indices.clear();
      cloud_kdtreeflann.radiusSearch(cloud->points[i], tolerance, nn_indices, nn_distances, max_cluster_size);

      int min_class_idx = class_idx;
      for (size_t j = 0; j < nn_indices.size(); ++j) {
        auto * point_with_tag = &classified_points[nn_indices[j]];

        if ((point_with_tag->class_index > 0) && (point_with_tag->class_index < min_class_idx)) {
          min_class_idx = point_with_tag->class_index;
        }
      }
      for (size_t j = 0; j < nn_indices.size(); ++j) {
        auto * point_with_tag = &classified_points[nn_indices[j]];
        temp_class_idx = point_with_tag->class_index;
        if (temp_class_idx > min_class_idx) {
          for (size_t k = 0; k < classified_points.size(); ++k) {
            if (classified_points[k].class_index == temp_class_idx) {
              classified_points[k].class_index = min_class_idx;
            }
          }
        }
        classified_points[nn_indices[j]].class_index = min_class_idx;
      }
      class_idx++;
    }
  }
  std::sort(classified_points.begin(), classified_points.end(), getClass);
  int new_class_idx = 0;
  int prev_class_idx = 0;
  pcl::PointIndices::Ptr object_indices(new pcl::PointIndices);
  pcl::PointCloud<pcl::PointXYZ>::Ptr object_points(new pcl::PointCloud<pcl::PointXYZ>);
  clusters.clear();
  std::vector<pcl::PointCloud<pcl::PointXYZ>> temp_clusters;
  for (auto point = classified_points.begin(); point < classified_points.end(); point++) {
    if (point->class_index == prev_class_idx) {
      point->class_index = new_class_idx;
      object_indices->indices.push_back(point->point_indice);
      object_points->points.push_back(cloud->points[point->point_indice]);
      continue;
    }

    if (point->class_index == prev_class_idx + 1) {
      new_class_idx = point->class_index;
      prev_class_idx = point->class_index;
      //
      if (object_indices->indices.size() > 0) {
        cluster_indices.push_back(*object_indices);
        temp_clusters.push_back(*object_points);
        object_indices->indices.clear();
        object_points->points.clear();
      }
      object_indices->indices.push_back(point->point_indice);
      object_points->points.push_back(cloud->points[point->point_indice]);
      continue;
    }

    if (point->class_index > prev_class_idx + 1) {
      new_class_idx = prev_class_idx + 1;
      prev_class_idx = point->class_index;
      point->class_index = new_class_idx;

      if (object_indices->indices.size() > 0) {
        cluster_indices.push_back(*object_indices);
        temp_clusters.push_back(*object_points);
        object_points->points.clear();
        object_indices->indices.clear();
      }
      object_indices->indices.push_back(point->point_indice);
      object_points->points.push_back(cloud->points[point->point_indice]);
      continue;
    }
  }
  for (const auto & cluster : temp_clusters) {
    if (
      static_cast<size_t>(min_cluster_size) < cluster.size() &&
      cluster.size() < static_cast<size_t>(max_cluster_size)) {
      clusters.push_back(cluster);
      clusters.back().width = cluster.points.size();
      clusters.back().height = 1;
      clusters.back().is_dense = false;
    }
  }
}
