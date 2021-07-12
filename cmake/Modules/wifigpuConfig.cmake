INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_WIFIGPU wifigpu)

FIND_PATH(
    WIFIGPU_INCLUDE_DIRS
    NAMES wifigpu/api.h
    HINTS $ENV{WIFIGPU_DIR}/include
        ${PC_WIFIGPU_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    WIFIGPU_LIBRARIES
    NAMES gnuradio-wifigpu
    HINTS $ENV{WIFIGPU_DIR}/lib
        ${PC_WIFIGPU_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/wifigpuTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(WIFIGPU DEFAULT_MSG WIFIGPU_LIBRARIES WIFIGPU_INCLUDE_DIRS)
MARK_AS_ADVANCED(WIFIGPU_LIBRARIES WIFIGPU_INCLUDE_DIRS)
