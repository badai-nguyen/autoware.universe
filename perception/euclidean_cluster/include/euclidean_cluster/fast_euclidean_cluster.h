
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
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>

struct ClassifedPoint
{
  size_t point_index;
  int point_class;
  pcl::PointXYZ * orig_point;
};

bool getClass(const ClassifedPoint & p0, const ClassifedPoint & p1)
{
  return p0.point_class < p1.point_class;
}

void fastEuclideanCluster(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const size_t min_component_size,
  const double tolerance, const int max_n, std::vector<pcl::PointIndices> & cluster_indices)
{
  if (cloud->points.size() < min_component_size) {
    cluster_indices.clear();
    return;
  }

  cluster_indices.resize(cloud->points.size());

  pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtreeflann;
  cloud_kdtreeflann.setInputCloud(cloud);
  size_t cloud_size = cloud->points.size();
  std::vector<int> marked_indices;
  marked_indices.resize(cloud_size);
  memset(marked_indices.data(), 0, sizeof(int) * cloud_size);

  std::vector<ClassifedPoint> points_with_tag;
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    ClassifedPoint current_point;
    current_point.point_index = i;
    current_point.point_class = 0;
    current_point.orig_point = &cloud->points[i];
    points_with_tag.push_back(current_point);
  }
  int tag_num = 1, temp_tag_num = -1;
  for (size_t i = 0; i < points_with_tag.size(); ++i) {
    auto * point = &points_with_tag[i];
    if (point->point_class == 0) {
      std::vector<float> nn_distances;
      std::vector<int> nn_indices;
      cloud_kdtreeflann.radiusSearch(
        *point->orig_point, tolerance, nn_indices, nn_distances, max_n);

      int min_tag_num = tag_num;
      for (size_t j = 0; j < nn_indices.size(); ++j) {
        auto * point_with_tag = &points_with_tag[nn_indices[j]];

        if ((point_with_tag->point_class > 0) && (point_with_tag->point_class < min_tag_num)) {
          min_tag_num = point_with_tag->point_class;
        }
      }
      for (size_t j = 0; j < nn_indices.size(); ++j) {
        auto * point_with_tag = &points_with_tag[nn_indices[j]];
        temp_tag_num = point_with_tag->point_class;
        if (temp_tag_num > min_tag_num) {
          for (size_t k = 0; k < points_with_tag.size(); ++k) {
            if (points_with_tag[k].point_class == temp_tag_num) {
              points_with_tag[k].point_class = min_tag_num;
            }
          }
        }
        points_with_tag[nn_indices[j]].point_class = min_tag_num;
      }
      tag_num++;
    }
  }
  std::sort(points_with_tag.begin(), points_with_tag.end(), getClass);
  int new_class = 0;
  int prev_class = 0;
  pcl::PointIndices::Ptr object_indices(new pcl::PointIndices);
  object_indices->indices.resize(points_with_tag.size());
  for (auto point = points_with_tag.begin(); point < points_with_tag.end(); point++) {
    if (point->point_class == prev_class) {
      point->point_class = new_class;
      object_indices->indices.push_back(point->point_index);
      continue;
    }

    if (point->point_class == prev_class + 1) {
      new_class = point->point_class;
      prev_class = point->point_class;
      //
      if (object_indices->indices.size() > 0) {
        cluster_indices.push_back(*object_indices);
        object_indices->indices.clear();
      }
      object_indices->indices.push_back(point->point_index);
      continue;
    }

    if (point->point_class > prev_class + 1) {
      new_class = prev_class + 1;
      prev_class = point->point_class;
      point->point_class = new_class;

      if (object_indices->indices.size() > 0) {
        cluster_indices.push_back(*object_indices);
        object_indices->indices.clear();
      }
      object_indices->indices.push_back(point->point_index);
      continue;
    }
  }
}
