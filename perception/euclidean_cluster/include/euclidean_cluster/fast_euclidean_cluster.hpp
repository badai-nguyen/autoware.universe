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
#include "euclidean_cluster/euclidean_cluster_interface.hpp"
#include "euclidean_cluster/utils.hpp"

#include <pcl/point_types.h>
#include <pcl/search/organized.h>
#include <pcl/segmentation/extract_clusters.h>

#include <vector>

namespace euclidean_cluster
{

struct PointIndex_NumberTag
{
  float nPointIndex;
  float nNumberTag;
};

bool NumberTag(const PointIndex_NumberTag & p0, const PointIndex_NumberTag & p1)
{
  return p0.nNumberTag < p1.nNumberTag;
}

template <typename PointT>
class FastExtractCluster : public pcl::EuclideanClusterExtraction<PointT>
{
protected:
  using pcl::EuclideanClusterExtraction<PointT>::input_;
  using pcl::EuclideanClusterExtraction<PointT>::cluster_tolerance_;
  using pcl::EuclideanClusterExtraction<PointT>::min_pts_per_cluster_;
  using pcl::EuclideanClusterExtraction<PointT>::max_pts_per_cluster_;
  using pcl::EuclideanClusterExtraction<PointT>::indices_;
  using pcl::EuclideanClusterExtraction<PointT>::tree_;

public:
  inline void extract(std::vector<pcl::PointIndices> & cluster_indices)
  {
    if (
      !this->initCompute() || (input_ && input_->points.empty()) ||
      (indices_ && indices_->empty())) {
      cluster_indices.clear();
      return;
    }

    unsigned long i, j;
    pcl::KdTreeFLANN<PointT> cloud_kdtreeflann;

    cloud_kdtreeflann.setInputCloud(input_);

    unsigned long cloud_size = input_->size();
    std::vector<int> marked_indices;
    marked_indices.resize(cloud_size);

    memset(marked_indices.data(), 0, sizeof(int) * cloud_size);
    std::vector<int> pointIdx;
    std::vector<float> pointquaredDistance;

    int tag_num = 1, temp_tag_num = -1;

    for (i = 0; i < cloud_size; i++) {
      // Clustering process
      if (marked_indices[i] == 0)  // reset to initial value if this point has not been manipulated
      {
        pointIdx.clear();
        pointquaredDistance.clear();
        cloud_kdtreeflann.radiusSearch(
          input_->points[i], cluster_tolerance_, pointIdx, pointquaredDistance,
          max_pts_per_cluster_);
        /**
         * All neighbors closest to a specified point with a query within a given radius
         * para.tolorance is the radius of the sphere that surrounds all neighbors
         * pointIdx is the resulting index of neighboring points
         * pointquaredDistance is the final square distance to adjacent points
         * pointIdx.size() is the maximum number of neighbors returned by limit
         */
        int min_tag_num = tag_num;
        for (j = 0; j < pointIdx.size(); j++) {
          /**
           * find the minimum label value contained in the field points, and tag it to this cluster
           * label.
           */
          if ((marked_indices[pointIdx[j]] > 0) && (marked_indices[pointIdx[j]] < min_tag_num)) {
            min_tag_num = marked_indices[pointIdx[j]];
          }
        }
        for (j = 0; j < pointIdx.size(); j++) {
          temp_tag_num = marked_indices[pointIdx[j]];

          /*
           * Each domain point, as well as all points in the same cluster, is uniformly assigned
           * this label
           */
          if (temp_tag_num > min_tag_num) {
            for (unsigned long k = 0; k < cloud_size; k++) {
              if (marked_indices[k] == temp_tag_num) {
                marked_indices[k] = min_tag_num;
              }
            }
          }
          marked_indices[pointIdx[j]] = min_tag_num;
        }
        tag_num++;
      }
    }

    std::vector<PointIndex_NumberTag> indices_tags;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    indices_tags.resize(cloud_size);

    PointIndex_NumberTag temp_index_tag;

    for (i = 0; i < cloud_size; i++) {
      /**
       * Put each point index and the corresponding tag value into the indices_tags
       */
      temp_index_tag.nPointIndex = i;
      temp_index_tag.nNumberTag = marked_indices[i];

      indices_tags[i] = temp_index_tag;
    }

    sort(indices_tags.begin(), indices_tags.end(), NumberTag);

    unsigned long begin_index = 0;
    for (i = 0; i < indices_tags.size(); i++) {
      // Relabel each cluster
      if (indices_tags[i].nNumberTag != indices_tags[begin_index].nNumberTag) {
        if ((i - begin_index) >= min_pts_per_cluster_) {
          unsigned long m = 0;
          inliers->indices.resize(i - begin_index);
          for (j = begin_index; j < i; j++) inliers->indices[m++] = indices_tags[j].nPointIndex;
          cluster_indices.push_back(*inliers);
        }
        begin_index = i;
      }
    }

    if ((i - begin_index) >= min_pts_per_cluster_) {
      for (j = begin_index; j < i; j++) {
        unsigned long m = 0;
        inliers->indices.resize(i - begin_index);
        for (j = begin_index; j < i; j++) {
          inliers->indices[m++] = indices_tags[j].nPointIndex;
        }
        cluster_indices.push_back(*inliers);
      }
    }
  }
};

class FastEuclideanCluster : public EuclideanClusterInterface
{
private:
  /* data */
public:
  FastEuclideanCluster();
  FastEuclideanCluster(bool use_height, int min_cluster_size, int max_cluster_size);
  FastEuclideanCluster(
    bool use_height, int min_cluster_size, int max_cluster_size, float tolerance);
  bool cluster(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & pointcloud,
    std::vector<pcl::PointCloud<pcl::PointXYZ>> & clusters) override;

  void setTolerance(float tolerance) { tolerance_ = tolerance; }

private:
  float tolerance_;
};

}  // namespace euclidean_cluster
