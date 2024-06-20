compile_cpp_example_with_modes(oklt_v3_external_intrinsics main.cpp)

 add_custom_target(oklt_v3_external_intrinsics_cpy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/custom_intrinsics custom_intrinsics
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/constants.h constants.h
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/scaleVec.okl scaleVec.okl)

add_dependencies(examples_cpp_oklt_v3_external_intrinsics oklt_v3_external_intrinsics_cpy)
target_sources(examples_cpp_oklt_v3_external_intrinsics 
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/scaleVec.okl
            ${CMAKE_CURRENT_SOURCE_DIR}/constants.h
            # ${CMAKE_CURRENT_SOURCE_DIR}/custom_intrinsics/api.h
            # ${CMAKE_CURRENT_SOURCE_DIR}/custom_intrinsics/cuda/impl.cpp
            # ${CMAKE_CURRENT_SOURCE_DIR}/custom_intrinsics/openmp/impl.cpp
            # ${CMAKE_CURRENT_SOURCE_DIR}/custom_intrinsics/launcher/impl.cpp

    )
