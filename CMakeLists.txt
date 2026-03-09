cmake_minimum_required(VERSION 3.20)
project(VulkanTemplate VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(FetchContent)

# ── volk ──────────────────────────────────────────────────────────────────────
# Must be fetched first; vk-bootstrap will link against it via VKB_USE_VOLK
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        1.3.270
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(volk)

# ── vk-bootstrap ──────────────────────────────────────────────────────────────
set(VKB_USE_VOLK ON CACHE BOOL "" FORCE)
FetchContent_Declare(vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.4.344
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(vk-bootstrap)

# ── Vulkan Memory Allocator ────────────────────────────────────────────────────
FetchContent_Declare(VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.3.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# ── GLFW ──────────────────────────────────────────────────────────────────────
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glfw)

# ── GLM ───────────────────────────────────────────────────────────────────────
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# ── stb ───────────────────────────────────────────────────────────────────────
# Single-header image loading library. No cmake targets — include dir only.
# STB_IMAGE_IMPLEMENTATION must be defined in exactly one .cpp (Texture.cpp).
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        2fb8c5a3deb2110c89669f8d6f36e5833b556b44
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

# ── nlohmann/json ─────────────────────────────────────────────────────────────
# Header-only JSON parser used by SceneLoader.
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ── ImGui ─────────────────────────────────────────────────────────────────────
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
# Tell imgui_impl_vulkan to use volk instead of linking vulkan-1
target_compile_definitions(imgui_lib PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(imgui_lib PUBLIC glfw volk)

# ── Main executable ───────────────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE src ${stb_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    volk
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glfw
    glm::glm
    imgui_lib
    nlohmann_json::nlohmann_json
    $<$<PLATFORM_ID:Windows>:winmm>  # timeBeginPeriod / timeEndPeriod
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    # volk: disable direct Vulkan symbol exports so all calls go through volk
    VK_NO_PROTOTYPES
    # GLM: Vulkan NDC (depth 0..1, Y flipped)
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    # Enable validation layers in Debug builds
    $<$<CONFIG:Debug>:VKT_VALIDATION>
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4 /WX- /utf-8 /FS
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── Shaders ───────────────────────────────────────────────────────────────────
add_subdirectory(shaders)
add_dependencies(${PROJECT_NAME} VulkanTemplateShaders)

# ── Tests (opt-in, zero runtime cost when OFF) ────────────────────────────────
option(VKT_BUILD_TESTS "Build unit tests (CPU-only, no GPU required)" OFF)
if(VKT_BUILD_TESTS)
    FetchContent_Declare(doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG        v2.4.11
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(doctest)

    # Only compile CPU-logic sources. Excluded intentionally:
    #   Application.cpp  -- defines main(), pulls in full Vulkan init
    #   Allocator.cpp    -- defines VMA_IMPLEMENTATION (must be in exactly one TU)
    #   Texture.cpp      -- defines STB_IMAGE_IMPLEMENTATION
    #   ImGuiLayer.cpp   -- requires a live GLFW/Vulkan context
    #   SceneLoader.cpp  -- exercises upload path; tested via RenderWorld directly
    add_executable(vkt_tests
        tests/main.cpp
        tests/test_renderworld.cpp
        tests/test_camera.cpp
        tests/test_frustum.cpp
        tests/test_units.cpp
        src/scene/Camera.cpp
        src/scene/RenderWorld.cpp
        src/scene/Frustum.cpp
        src/scene/Mesh.cpp
        src/scene/InstanceBuffer.cpp
        src/resources/Buffer.cpp
        src/resources/Image.cpp
    )

    target_include_directories(vkt_tests PRIVATE src ${stb_SOURCE_DIR})
    target_link_libraries(vkt_tests PRIVATE
        doctest::doctest
        volk
        glm::glm
        GPUOpen::VulkanMemoryAllocator
    )
    target_compile_definitions(vkt_tests PRIVATE
        VK_NO_PROTOTYPES
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
    )
    if(MSVC)
        target_compile_options(vkt_tests PRIVATE /W4 /WX- /utf-8)
    else()
        target_compile_options(vkt_tests PRIVATE -Wall -Wextra)
    endif()
endif()

# ── Post-build: copy assets and shaders next to the executable ────────────────
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/shaders_build"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMENT "Copying assets and shaders"
)
