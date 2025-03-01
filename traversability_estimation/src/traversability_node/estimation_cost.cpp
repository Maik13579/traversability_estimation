#include "traversability_node/estimation_cost.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>

// Compute slope cost for each traversable point
void compute_slope_cost(pcl::PointCloud<TraversablePoint>::Ptr traversable_cloud, TraversabilityNodeConfig params)
{
    const double slope_weight = params.costs.slope_weight;
    const double curvature_weight = params.costs.curvature_weight;
    const int n_threads = params.common.n_threads;

    if (curvature_weight == 0.0 && slope_weight == 0.0)
        return;

#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < static_cast<int>(traversable_cloud->points.size()); ++i)
    {
        auto &point = traversable_cloud->points[i];
        point.slope = std::acos(point.normal_z) / (M_PI / 2.0f); //eq. 4
        point.slope_cost = slope_weight * point.slope;
        point.curvature_cost = curvature_weight * point.curvature;
    }
}

// Compute inflation cost for traversable points based on obstacle and boundary proximity
void compute_inflation_cost(
    pcl::PointCloud<TraversablePoint>::Ptr traversable_cloud,
    pcl::PointCloud<PointXYZICluster>::Ptr obstacle_cloud,
    pcl::PointCloud<TraversablePoint>::Ptr boundary_cloud,
    TraversabilityNodeConfig params)
{
    // unpack parameters
    const float inscribed_radius = params.robot.inscribed_radius;
    const float inflation_radius = params.costs.inflation.radius;
    const float cost_scaling_factor = params.costs.inflation.cost_scaling_factor;
    const float inflation_weight = params.costs.inflation_weight;
    const float obstacle_normal_offset = params.costs.inflation.obstacle_normal_offset;
    const float boundary_normal_offset = params.costs.inflation.boundary_normal_offset;
    const int n_threads = params.common.n_threads;
    const float normal_offset = params.robot.normal_offset;

    if (inflation_radius == 0.0)
        return; // nothing to do


    // Build kd trees for fast nearest neighbor search
    pcl::KdTreeFLANN<PointXYZICluster>::Ptr kdtree_obstacle(new pcl::KdTreeFLANN<PointXYZICluster>());
    kdtree_obstacle->setInputCloud(obstacle_cloud);

    pcl::KdTreeFLANN<TraversablePoint>::Ptr kdtree_boundary(new pcl::KdTreeFLANN<TraversablePoint>());
    kdtree_boundary->setInputCloud(boundary_cloud);

    // Iterate over all points
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < static_cast<int>(traversable_cloud->points.size()); ++i)
    {
        auto &point = traversable_cloud->points[i];

        // Offset query point
        Eigen::Vector3f normal(point.normal_x, point.normal_y, point.normal_z);
        Eigen::Vector3f query_position(point.x, point.y, point.z);
        query_position += normal_offset * normal;

        float obstacle_distance = std::numeric_limits<float>::max();
        float boundary_distance = std::numeric_limits<float>::max();

        std::vector<int> obstacle_indices;
        std::vector<float> obstacle_squared_distances;
        PointXYZICluster query_point_obstacle;
        query_point_obstacle.x = query_position.x();
        query_point_obstacle.y = query_position.y();
        query_point_obstacle.z = query_position.z();

        // Find closest obstacle point
        if (kdtree_obstacle->radiusSearch(query_point_obstacle, inflation_radius, obstacle_indices, obstacle_squared_distances) > 0)
        {
            for (size_t j = 0; j < obstacle_indices.size(); ++j)
            {
                // Check if point is above offseted normal plane
                const auto &obstacle_point = obstacle_cloud->points[obstacle_indices[j]];
                Eigen::Vector3f obstacle_vector(obstacle_point.x - point.x, obstacle_point.y - point.y, obstacle_point.z - point.z);
                if (obstacle_vector.dot(normal) > obstacle_normal_offset)
                {
                    obstacle_distance = std::sqrt(obstacle_squared_distances[j]);
                    break;
                }
            }
        }

        std::vector<int> boundary_indices_local;
        std::vector<float> boundary_squared_distances;
        TraversablePoint query_point_boundary;
        query_point_boundary.x = query_position.x();
        query_point_boundary.y = query_position.y();
        query_point_boundary.z = query_position.z();

        // Find closest boundary point
        if (kdtree_boundary->radiusSearch(query_point_boundary, inflation_radius, boundary_indices_local, boundary_squared_distances) > 0)
        {
            for (size_t j = 0; j < boundary_indices_local.size(); ++j)
            {
                // Check if point is above offseted normal plane
                const auto &boundary_point = boundary_cloud->points[boundary_indices_local[j]];
                Eigen::Vector3f boundary_vector(boundary_point.x - point.x, boundary_point.y - point.y, boundary_point.z - point.z);
                if (boundary_vector.dot(normal) > boundary_normal_offset)
                {
                    boundary_distance = std::sqrt(boundary_squared_distances[j]);
                    break;
                }
            }
        }
        
        // Compute inflation cost
        float min_distance = std::min(obstacle_distance, boundary_distance);
        if (min_distance < inscribed_radius) //eq. 5
            point.inflation = 1.0;
        else if (min_distance < inflation_radius)
            point.inflation = std::exp(-cost_scaling_factor * (min_distance - inscribed_radius));
        else
            point.inflation = 0.0;
        point.inflation_cost = inflation_weight * point.inflation;
    }
}

void compute_costs(pcl::PointCloud<TraversablePoint>::Ptr traversable_cloud,
                   pcl::PointCloud<TraversablePoint>::Ptr non_traversable_cloud,
                   pcl::PointCloud<PointXYZICluster>::Ptr obstacle_cloud,
                   pcl::PointCloud<TraversablePoint>::Ptr boundary_cloud,
                   TraversabilityNodeConfig params)
{
    // Compute slope and inflation costs
    compute_slope_cost(traversable_cloud, params);
    compute_inflation_cost(traversable_cloud, obstacle_cloud, boundary_cloud, params);

    // Prepare for removal of non-traversable points
    std::mutex mtx;
    std::vector<int> indices_to_remove;

    // Identify points that exceed the inscribed cost threshold
    #pragma omp parallel for num_threads(params.common.n_threads)
    for (int i = 0; i < static_cast<int>(traversable_cloud->points.size()); ++i)
    {
        auto &point = traversable_cloud->points[i];
        point.final_cost = point.slope_cost + point.curvature_cost + point.inflation_cost;
        if (point.final_cost >= params.costs.max_cost)
        {
            // Thread-safe addition to non-traversable cloud and indices to remove
            std::lock_guard<std::mutex> lock(mtx);
            non_traversable_cloud->points.push_back(point);
            indices_to_remove.push_back(i);
        }
    }

    // Remove non-traversable points from the traversable cloud
    pcl::PointCloud<TraversablePoint>::Ptr temp_cloud(new pcl::PointCloud<TraversablePoint>);
    for (size_t i = 0; i < traversable_cloud->points.size(); ++i)
    {
        if (std::find(indices_to_remove.begin(), indices_to_remove.end(), static_cast<int>(i)) == indices_to_remove.end())
        {
            temp_cloud->points.push_back(traversable_cloud->points[i]);
        }
    }

    // Update the traversable cloud
    traversable_cloud->swap(*temp_cloud);

    // Update cloud sizes
    traversable_cloud->width = static_cast<uint32_t>(traversable_cloud->points.size());
    traversable_cloud->height = 1;
    traversable_cloud->is_dense = true;

    non_traversable_cloud->width = static_cast<uint32_t>(non_traversable_cloud->points.size());
    non_traversable_cloud->height = 1;
    non_traversable_cloud->is_dense = true;
}