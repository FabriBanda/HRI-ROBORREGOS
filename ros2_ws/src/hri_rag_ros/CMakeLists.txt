cmake_minimum_required(VERSION 3.8)
project(hri_rag_ros)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Ask.srv"
)
install(DIRECTORY hri_rag_ros/
  DESTINATION ${PYTHON_INSTALL_DIR}/hri_rag_ros
  USE_SOURCE_PERMISSIONS
)
install(PROGRAMS
  scripts/rag_service
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime rclpy)
ament_package()
