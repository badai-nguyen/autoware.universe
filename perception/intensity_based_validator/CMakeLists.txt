cmake_minimum_required(VERSION 3.8)
project(intensity_based_validator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# find dependencies
find_package(autoware_cmake REQUIRED)
autoware_package()

# Targets
ament_auto_add_library(intensity_based_validator SHARED
  src/intensity_based_validator.cpp
)

rclcpp_components_register_node(intensity_based_validator
  PLUGIN "intensity_based_validator::IntensityBasedValidator"
  EXECUTABLE intensity_based_validator_node)


if(BUILD_TESTING)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_uncrustify)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
)
