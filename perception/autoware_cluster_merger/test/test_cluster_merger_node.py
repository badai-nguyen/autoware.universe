# Copyright 2024 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from launch_ros.substitutions import FindPackageShare
PACKAGE_NAME="autoware_cluster_merger"
@pytest.mark.launch_test
def generate_test_description():
    """ Launch file test description.

    Returns: 
        _type_: launch.LaunchDescription
    """
    # get launch file path 
    launch_file_path = (FindPackageShare(PACKAGE_NAME).find(PACKAGE_NAME)
    + "/launch/cluster_merger.launch.xml")
    # use 
    launch_args = []
    # action to include launch file 
    test_launch_file = launch.action.IncludeLaunchDescription(
        launch.launch_description_sources.AnyLaunchDescriptionSource(launch_file_path),
        launch_arguments=launch_args
    )
    return launch.LaunchDescription(
        [
            test_launch_file,
            launch_testing.action.ReadyToTest(),
        ]
    )


# @launch_testing.post_shutdown_test()
# class TestProcessOutput()