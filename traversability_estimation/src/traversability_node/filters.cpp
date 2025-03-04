#include "traversability_node/filters.hpp"

// Apply voxel grid downsampling to a point cloud using parameters from config
void voxel_down_sampling(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, TraversabilityNodeConfig params)
{
    // Unpack voxel downsampling parameters
    const double voxel_size = params.filters.voxel_down_sampling.voxel_size;
    const bool downsample_all_data = params.filters.voxel_down_sampling.downsample_all_data;

    // Apply voxel grid filter
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid;
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_grid.setDownsampleAllData(downsample_all_data);
    voxel_grid.setInputCloud(cloud);
    voxel_grid.filter(*cloud);
}

// Apply radius outlier removal to a point cloud using parameters from config
void radius_outlier_removal(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, TraversabilityNodeConfig params)
{
    // Unpack outlier removal parameters
    const double radius = params.filters.outlier_removal.radius;
    const int neighbors = params.filters.outlier_removal.neighbors;

    // Apply radius outlier removal filter
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlier_removal;
    outlier_removal.setRadiusSearch(radius);
    outlier_removal.setMinNeighborsInRadius(neighbors);
    outlier_removal.setInputCloud(cloud);
    outlier_removal.filter(*cloud);
}

// Perform Euclidean clustering on the input point cloud using parameters from config
template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr euclidian_clustering(typename pcl::PointCloud<PointT>::Ptr input_cloud, TraversabilityNodeConfig params)
{
    // Unpack clustering parameters
    const double eps = params.filters.clustering.eps;
    const int min_points = params.filters.clustering.min_points;
    const int max_points = std::numeric_limits<int>::max();

    // Create the KdTree object for search
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(input_cloud);

    // Perform Euclidean clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(eps);
    ec.setMinClusterSize(min_points);
    ec.setMaxClusterSize(max_points);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);
    ec.extract(cluster_indices);

    // Create output cloud with cluster IDs
    typename pcl::PointCloud<PointT>::Ptr clustered_cloud(new pcl::PointCloud<PointT>);
    uint32_t cluster_id = 0;

    for (const auto &indices : cluster_indices)
    {
        for (const auto &idx : indices.indices)
        {
            PointT point = input_cloud->points[idx];
            point.cluster_id = cluster_id; // Assign cluster ID
            clustered_cloud->points.push_back(point);
        }
        cluster_id++;
    }

    clustered_cloud->width = clustered_cloud->points.size();
    clustered_cloud->height = 1;
    clustered_cloud->is_dense = true;

    return clustered_cloud;
}

// Explicit instantiation for PointXYZICluster
template pcl::PointCloud<PointXYZICluster>::Ptr euclidian_clustering<PointXYZICluster>(
    pcl::PointCloud<PointXYZICluster>::Ptr input_cloud, TraversabilityNodeConfig params);

// Explicit instantiation for TraversablePoint
// template pcl::PointCloud<TraversablePoint>::Ptr euclidian_clustering<TraversablePoint>(
//     pcl::PointCloud<TraversablePoint>::Ptr input_cloud, TraversabilityNodeConfig params);