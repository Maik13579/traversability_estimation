#ifndef TRAVERSABILITY_ESTIMATION__TRAVERSABILITY_NODE_ESTIMATION_INFO_HPP_
#define TRAVERSABILITY_ESTIMATION__TRAVERSABILITY_NODE_ESTIMATION_INFO_HPP_

#include <pcl/point_cloud.h>
#include "traversability_estimation/point_types.hpp"
#include "traversability_node/params.hpp"
#include "traversability_node/filters.hpp"


#include <pcl/common/common.h>                  // for copyPointCloud
#include <pcl/filters/radius_outlier_removal.h> // for RadiusOutlierRemoval


// Traversability estimation and boundary estimation functions
void perform_traversability_estimation(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud,
                                       pcl::PointCloud<TraversablePoint>::Ptr &traversable_cloud,
                                       pcl::PointCloud<PointXYZICluster>::Ptr &obstacle_cloud,
                                       TraversabilityNodeConfig params);

void compute_boundary(pcl::PointCloud<TraversablePoint>::Ptr traversable_cloud,
                      pcl::PointCloud<TraversablePoint>::Ptr boundary_cloud,
                      TraversabilityNodeConfig params);

#endif // TRAVERSABILITY_ESTIMATION__TRAVERSABILITY_NODE_ESTIMATION_INFO_HPP_
