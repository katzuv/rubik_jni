# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-src")
  file(MAKE_DIRECTORY "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-src")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-build"
  "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix"
  "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix/tmp"
  "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix/src/opencv_lib-populate-stamp"
  "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix/src"
  "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix/src/opencv_lib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix/src/opencv_lib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/rubik_jni/rubik_jni/build_test/_deps/opencv_lib-subbuild/opencv_lib-populate-prefix/src/opencv_lib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
