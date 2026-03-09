#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

namespace vkt {

class Context;

// Thin RAII wrapper around VmaAllocator.
// All Buffer and Image objects borrow the raw VmaAllocator handle; the Allocator
// class just owns the lifetime.
class Allocator {
public:
    Allocator() = default;
    ~Allocator();
    Allocator(const Allocator&)            = delete;
    Allocator& operator=(const Allocator&) = delete;

    void init(const Context& ctx);
    void destroy();

    VmaAllocator handle() const { return m_allocator; }

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

} // namespace vkt
