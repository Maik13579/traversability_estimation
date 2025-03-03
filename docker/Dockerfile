ARG BASE_IMAGE=ros:humble
FROM ${BASE_IMAGE}

#environment variables
ENV ROS_WS /root/ros2_ws
ENV DEBIAN_FRONTEND noninteractive

#ros2 DDS implementation
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

#use bash
SHELL ["/bin/bash", "-c"]

#source ros stuff
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

#install important stuff
RUN apt update \
    && apt install -y \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-$ROS_DISTRO-rviz2 \
    ros-$ROS_DISTRO-tf-transformations \
    python3-pip \
    && pip3 install open3d numpy compas scipy python-dateutil transforms3d \
    && rm -rf /var/lib/apt/lists/*


#Copy only package.xml files to use caching and avoid reinstalling deps every time
COPY traversability_estimation/package.xml ${ROS_WS}/src/traversability_estimation/traversability_estimation/package.xml
COPY traversability_estimation_interfaces/package.xml ${ROS_WS}/src/traversability_estimation/traversability_estimation_interfaces/package.xml
WORKDIR ${ROS_WS}
RUN apt update \
    && rosdep update \
    && rosdep install -y --ignore-src --from-paths src \
    && rm -rf /var/lib/apt/lists/*

# Build
COPY traversability_estimation ${ROS_WS}/src/traversability_estimation/traversability_estimation
COPY traversability_estimation_interfaces ${ROS_WS}/src/traversability_estimation/traversability_estimation_interfaces
WORKDIR ${ROS_WS}
RUN source /opt/ros/$ROS_DISTRO/setup.bash \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
RUN echo "source ${ROS_WS}/install/setup.bash" >> /root/.bashrc

# Generate demo point cloud
COPY scripts /scripts
COPY world /world
RUN python3 /scripts/generate_cloud.py