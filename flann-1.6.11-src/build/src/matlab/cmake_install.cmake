# Install script for directory: /home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/flann/matlab" TYPE FILE FILES
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/build/src/matlab/nearest_neighbors.mexa64"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/flann_build_index.m"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/flann_free_index.m"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/flann_load_index.m"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/flann_save_index.m"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/flann_search.m"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/flann_set_distance_type.m"
    "/home/pang/software/ACG-Localizer/flann-1.6.11-src/src/matlab/test_flann.m"
    )
endif()

