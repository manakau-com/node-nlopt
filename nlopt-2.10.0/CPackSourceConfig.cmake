# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BINARY_7Z "OFF")
set(CPACK_BINARY_IFW "OFF")
set(CPACK_BINARY_NSIS "ON")
set(CPACK_BINARY_NUGET "OFF")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_BINARY_TBZ2 "ON")
set(CPACK_BINARY_TGZ "ON")
set(CPACK_BINARY_TZ "OFF")
set(CPACK_BINARY_WIX "OFF")
set(CPACK_BINARY_ZIP "OFF")
set(CPACK_BUILD_SOURCE_DIRS "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0;C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0")
set(CPACK_CMAKE_GENERATOR "Visual Studio 17 2022")
set(CPACK_COMPONENTS_ALL "Debug;Unspecified")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.22/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "nlopt built using CMake")
set(CPACK_GENERATOR "TBZ2;TGZ")
set(CPACK_IGNORE_FILES ".git;/build;.*~;")
set(CPACK_INSTALLED_DIRECTORIES "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0;/")
set(CPACK_INSTALL_CMAKE_PROJECTS "")
set(CPACK_INSTALL_PREFIX "C:/Program Files (x86)/nlopt")
set(CPACK_MODULE_PATH "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0/cmake")
set(CPACK_NSIS_DISPLAY_NAME "nlopt 2.10.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
set(CPACK_NSIS_PACKAGE_NAME "nlopt 2.10.0")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OUTPUT_CONFIG_FILE "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.22/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "nlopt built using CMake")
set(CPACK_PACKAGE_FILE_NAME "nlopt-2.10.0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "nlopt 2.10.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "nlopt 2.10.0")
set(CPACK_PACKAGE_NAME "nlopt")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION "2.10.0")
set(CPACK_PACKAGE_VERSION_MAJOR "2")
set(CPACK_PACKAGE_VERSION_MINOR "10")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_RESOURCE_FILE_LICENSE "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.22/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.22/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.22/Templates/CPack.GenericWelcome.txt")
set(CPACK_RPM_PACKAGE_SOURCES "ON")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TBZ2;TGZ")
set(CPACK_SOURCE_IGNORE_FILES ".git;/build;.*~;")
set(CPACK_SOURCE_INSTALLED_DIRECTORIES "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0;/")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0/CPackSourceConfig.cmake")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "nlopt-2.10.0")
set(CPACK_SOURCE_TOPLEVEL_TAG "win64-Source")
set(CPACK_STRIP_FILES "")
set(CPACK_SYSTEM_NAME "win64")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "win64-Source")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "C:/Users/MarcusRainbow/Projects/node-nlopt/nlopt-2.10.0/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
