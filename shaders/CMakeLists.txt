find_program(GLSLC glslc HINTS
    ENV VULKAN_SDK
    PATH_SUFFIXES Bin bin
)

if(NOT GLSLC)
    message(WARNING "[Shaders] glslc not found. Shaders won't be compiled automatically. "
                    "Install the Vulkan SDK and ensure it is on PATH or set VULKAN_SDK.")
    add_custom_target(VulkanTemplateShaders)
    return()
endif()

set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders_build)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/default.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/default.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shadow.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/fullscreen.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/post_basic.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/cull.comp
)

set(SHADER_OUTPUTS)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT  ${OUTPUT}
        COMMAND ${GLSLC} ${SHADER} -o ${OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SHADER_OUTPUTS ${OUTPUT})
endforeach()

add_custom_target(VulkanTemplateShaders ALL DEPENDS ${SHADER_OUTPUTS})
