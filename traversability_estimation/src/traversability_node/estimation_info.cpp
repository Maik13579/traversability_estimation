#include "traversability_node/estimation_info.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/boundary.h>
#include <omp.h>
#include <cmath>


void perform_traversability_estimation(
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud,
    pcl::PointCloud<TraversablePoint>::Ptr &traversable_cloud,
    pcl::PointCloud<PointXYZICluster>::Ptr &obstacle_cloud,
    TraversabilityNodeConfig params)
{
    // Unpack parameters
    const int n_threads = params.common.n_threads;
    const double normal_radius = params.traversability_estimation.normal_vector_estimation.radius;
    const double allowed_slope_angle = params.robot.allowed_slope_angle;

    // Normal estimation using OpenMP for parallelization
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
    ne.setNumberOfThreads(n_threads);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    ne.setSearchMethod(tree);
    ne.setInputCloud(input_cloud);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);

    // Parallel region for classifying points
    #pragma omp parallel num_threads(n_threads)
    {
        // Thread-local storage for traversable and obstacle clouds
        pcl::PointCloud<TraversablePoint>::Ptr local_traversable(new pcl::PointCloud<TraversablePoint>);
        pcl::PointCloud<PointXYZICluster>::Ptr local_obstacle(new pcl::PointCloud<PointXYZICluster>);

        #pragma omp for nowait
        for (int i = 0; i < static_cast<int>(input_cloud->points.size()); ++i)
        {
            const auto &input_point = input_cloud->points[i];
            const auto &normal = normals->points[i];

            // Calculate normalized normal vector
            Eigen::Vector3f normal_vector(normal.normal_x, normal.normal_y, normal.normal_z);
            float norm = normal_vector.norm();
            if (norm > 0.0f) normal_vector /= norm;

            // Ensure the normal points upward
            if (normal_vector.z() < 0.0f) normal_vector = -normal_vector;

            // Calculate slope and classify point
            float slope_angle = std::acos(normal_vector.z()) * 180.0f / M_PI;
            if (slope_angle < allowed_slope_angle)
            {
                TraversablePoint point;
                point.x = input_point.x;
                point.y = input_point.y;
                point.z = input_point.z;
                point.intensity = input_point.intensity;
                point.normal_x = normal_vector.x();
                point.normal_y = normal_vector.y();
                point.normal_z = normal_vector.z();
                point.slope_angle = slope_angle;
                point.curvature = normal.curvature;
                local_traversable->points.push_back(point);
            }
            else
            {
                PointXYZICluster obstacle_point;
                obstacle_point.x = input_point.x;
                obstacle_point.y = input_point.y;
                obstacle_point.z = input_point.z;
                obstacle_point.intensity = input_point.intensity;
                obstacle_point.normal_x = normal_vector.x();
                obstacle_point.normal_y = normal_vector.y();
                obstacle_point.normal_z = normal_vector.z();
                obstacle_point.curvature = normal.curvature;
                local_obstacle->points.push_back(obstacle_point);
            }
        }

        // Merge local clouds into shared clouds
        #pragma omp critical
        {
            traversable_cloud->points.insert(traversable_cloud->points.end(),
                                             local_traversable->points.begin(),
                                             local_traversable->points.end());
            obstacle_cloud->points.insert(obstacle_cloud->points.end(),
                                          local_obstacle->points.begin(),
                                          local_obstacle->points.end());
        }
    }
    // Perform clustering to remove small clusters -> noise reduction
    obstacle_cloud = euclidian_clustering<PointXYZICluster>(obstacle_cloud, params);
    // traversable_cloud = euclidian_clustering<TraversablePoint>(traversable_cloud, params);
}

void compute_boundary(
    pcl::PointCloud<TraversablePoint>::Ptr traversable_cloud,
    pcl::PointCloud<TraversablePoint>::Ptr boundary_cloud,
    TraversabilityNodeConfig params)
{
    // Unpack parameters
    const double boundary_radius = params.traversability_estimation.boundary_estimation.radius;
    const double angle_threshold = params.traversability_estimation.boundary_estimation.angle_threshold;

    // Set up boundary estimation
    pcl::BoundaryEstimation<TraversablePoint, pcl::Normal, pcl::Boundary> boundary_estimation;
    boundary_estimation.setInputCloud(traversable_cloud);

    // Prepare normals for boundary estimation
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normals->points.reserve(traversable_cloud->points.size());
    for (const auto &point : traversable_cloud->points)
    {
        pcl::Normal normal;
        normal.normal_x = point.normal_x;
        normal.normal_y = point.normal_y;
        normal.normal_z = point.normal_z;
        normal.curvature = point.curvature;
        normals->points.push_back(normal);
    }

    // Perform boundary estimation
    pcl::search::KdTree<TraversablePoint>::Ptr tree(new pcl::search::KdTree<TraversablePoint>);
    boundary_estimation.setInputNormals(normals);
    boundary_estimation.setRadiusSearch(boundary_radius);
    boundary_estimation.setSearchMethod(tree);
    boundary_estimation.setAngleThreshold(angle_threshold);

    pcl::PointCloud<pcl::Boundary> boundaries;
    boundary_estimation.compute(boundaries);

    // Identify and collect boundary points
    std::vector<int> boundary_indices;
    for (size_t i = 0; i < boundaries.points.size(); ++i)
    {
        if (boundaries.points[i].boundary_point)
        {
            boundary_indices.push_back(static_cast<int>(i));
        }
    }

    // Extract boundary points and remove them from traversable cloud
    pcl::copyPointCloud(*traversable_cloud, boundary_indices, *boundary_cloud);

    // Filter out boundary points from traversable cloud
    pcl::PointCloud<TraversablePoint>::Ptr updated_traversable_cloud(new pcl::PointCloud<TraversablePoint>);
    for (size_t i = 0; i < traversable_cloud->points.size(); ++i)
    {
        if (std::find(boundary_indices.begin(), boundary_indices.end(), static_cast<int>(i)) == boundary_indices.end())
        {
            updated_traversable_cloud->points.push_back(traversable_cloud->points[i]);
        }
    }

    // Update traversable cloud to exclude boundary points
    traversable_cloud->swap(*updated_traversable_cloud);

    // Cleanup boundary points with radius outlier removal to remove noise
    pcl::RadiusOutlierRemoval<TraversablePoint> outlier_removal;
    outlier_removal.setInputCloud(boundary_cloud);
    outlier_removal.setRadiusSearch(boundary_radius * 2.0);
    outlier_removal.setMinNeighborsInRadius(2);
    outlier_removal.filter(*boundary_cloud);
}