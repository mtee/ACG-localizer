cmake_minimum_required (VERSION 2.6)
#prevent warnings about relative paths from hapening
cmake_policy(SET CMP0015 NEW)

if (EXISTS "${CMAKE_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
endif ()
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
find_package (LAPACK)
find_package (GMM)
find_package (OpenMesh)
#find_package(PCL 1.8.1 QUIET)
find_package( ANN REQUIRED )

FILE(GLOB_RECURSE LibFiles "include/*.h")
add_custom_target(headers SOURCES ${LibFiles})

#find_package (FLANN)
set(FLANN_INCLUDE_DIR /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/src/cpp)
#set(FLANN_LIBRARY /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib/libflann_cpp.so)
set(FLANN_LIBRARY /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib/libflann_cpp_s.a)
#set(FLANN_LIBRARY /home/mikhail/workspace/ACG-localizer/flann-1.8.4-src/build/lib/libflann_cpp_s.a)

set(FLANN_LIBRARY_DIR /home/mikhail/workspace/ACG-localizer/flann-1.6.11-src/build/lib)
# find_package(CUDA)
# if (NOT CUDA_FOUND)
#     message(STATUS "CUDA not found. Project will not be built.")
# endif(NOT CUDA_FOUND)

# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
else(NOT OpenCV_FOUND)
        message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
        message(STATUS "OpenCV Link Libraries: ${OpenCV_LIBS}")
endif(NOT OpenCV_FOUND)


# source and header of the feature library
set (features_SRC features/SIFT_loader.cc features/visual_words_handler.cc)
set (features_HDR SIFT_keypoint.hh SIFT_loader.hh visual_words_handler.hh)

# source and header of the math library
set (math_SRC math/math.cc math/matrix3x3.cc math/matrix4x4.cc math/matrixbase.cc math/projmatrix.cc math/pseudorandomnrgen.cc math/SFMT_src/SFMT.cc )
set (math_HDR 
${PROJECT_SOURCE_DIR}/include/math.hh 
${PROJECT_SOURCE_DIR}/include/matrix3x3.hh 
${PROJECT_SOURCE_DIR}/include/matrix4x4.hh 
${PROJECT_SOURCE_DIR}/include/matrixbase.hh 
${PROJECT_SOURCE_DIR}/include/projmatrix.hh  
${PROJECT_SOURCE_DIR}/include/pseudorandomnrgen.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params607.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params1279.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params2281.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params4253.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params11213.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params19937.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params44497.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params86243.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params132049.hh 
${PROJECT_SOURCE_DIR}/include/SFMT_src/SFMT-params216091.hh 
)

# source and header for the sfm functionality
set (sfm_SRC sfm/parse_bundler.cc sfm/bundler_camera.cc)
set (sfm_HDR 
${PROJECT_SOURCE_DIR}/include/parse_bundler.hh 
${PROJECT_SOURCE_DIR}/include/bundler_camera.hh
)

# source and header for the 6-point pose solver
set (solver_SRC solver/solverbase.cc solver/solverproj.cc)
set (solver_HDR include/solver/solverbase.hh include/solver/solverproj.hh)


#if(PCL_FOUND)
#    add_definitions(${PCL_DEFINITIONS})
#    link_directories (${PCL_LIBRARY_DIRS})
#    include_directories (${PCL_INCLUDE_DIRS})
#endif(PCL_FOUND)


include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${LAPACK_INCLUDE_DIR}
    ${GMM_INCLUDE_DIR}
    ${OPENMESH_INCLUDE_DIR}
    ${FLANN_INCLUDE_DIR}
    ${ANN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories (
  ${LAPACK_LIBRARY_DIR}
  ${OPENMESH_LIBRARY_DIR}
  ${ANN_LIBRARY_DIR}
 # ${FLANN_LIBRARY_DIR}
  ${OpenCV_LIBS}
)




# set sources for the executables
add_executable (Bundle2Info 
    features/SIFT_loader.cc 
  #  features/SIFT_keypoint.hh 
  #  features/SIFT_loader.hh 
    ${sfm_SRC} 
    ${sfm_HDR} 
    ${math_SRC}
    ${math_HDR}
    Bundle2Info 
)

target_include_directories(Bundle2Info PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable (compute_desc_assignments 
    compute_desc_assignments.cc 
    ${sfm_SRC} 
 #   ${sfm_HDR} 
    ${features_SRC} 
    ${math_SRC} 
#    ${math_HDR} 
  #  ${features_HDR} 
  )

add_library(acs_localizer SHARED
    acs_localizer.cc
    ${features_SRC}
 #   ${features_HDR}
    ${math_SRC}
 #   ${math_HDR}
    ${solver_SRC}
#    ${solver_HDR}
    ${sfm_SRC}
 #   ${sfm_HDR}
)
target_link_libraries (acs_localizer
    ${OpenCV_LIBS}
    ${OPENMESH_LIBRARY}
    ${LAPACK_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${GMM_LIBRARY}
    ${FLANN_LIBRARY}
    ${ANN_LIBRARIES}
)

# set libraries to link against
target_link_libraries (Bundle2Info
    ${GMM_LIBRARY}
    ${LAPACK_LIBRARY}
    ${LAPACK_LIBRARIES}
)


target_link_libraries (compute_desc_assignments
    ${OPENMESH_LIBRARY}
    ${LAPACK_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${GMM_LIBRARY}
    ${FLANN_LIBRARY}
)


add_executable(acg_localizer_active_search
    acg_localizer_active_search.cc 
#    pointcloudmapping.cpp 
#    vtkImageMatSource.cpp 
    )
target_link_libraries (acg_localizer_active_search
    acs_localizer
#    ${PCL_LIBRARIES}
)

# install the executables

#install( PROGRAMS ${CMAKE_BINARY_DIR}/src/Bundle2Info
#         DESTINATION ${CMAKE_BINARY_DIR}/bin)

#install( PROGRAMS ${CMAKE_BINARY_DIR}/src/compute_desc_assignments
#         DESTINATION ${CMAKE_BINARY_DIR}/bin) 

#install( PROGRAMS ${CMAKE_BINARY_DIR}/src/acg_localizer
#         DESTINATION ${CMAKE_BINARY_DIR}/bin) 
         
#install( PROGRAMS ${CMAKE_BINARY_DIR}/src/acg_localizer_knn
#         DESTINATION ${CMAKE_BINARY_DIR}/bin)

#install( PROGRAMS ${CMAKE_BINARY_DIR}/src/acg_localizer_active_search
#        DESTINATION ${CMAKE_BINARY_DIR}/bin) 
