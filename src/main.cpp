#include "Application.h"

#include <cstdio>
#include <exception>

int main() {
    vkt::Application app;
    try {
        app.init({
            .title  = "Vulkan Template",
            .width  = 1280,
            .height = 720,
        });
        app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[Fatal] %s\n", e.what());
        return 1;
    }
    return 0;
}
