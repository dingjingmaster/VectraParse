set(VECTRAPARSE_FOUND TRUE)
set(VECTRAPARSE_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")
set(VECTRAPARSE_LIBRARY vectraparse_ffi)

if(NOT TARGET VectraParse::vectraparse)
  add_library(VectraParse::vectraparse INTERFACE IMPORTED)
  set_target_properties(VectraParse::vectraparse PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${VECTRAPARSE_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${VECTRAPARSE_LIBRARY}")
endif()
