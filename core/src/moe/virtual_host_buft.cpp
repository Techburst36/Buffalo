#include "virtual_host_buft.h"
#include "../io/platform_io.h"

#include "ggml-backend-impl.h"

#include <cstring>

namespace bmoe {

namespace {

struct VirtualHostBufCtx {
    void * base = nullptr;
    size_t reserved = 0;
};

void vhost_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<VirtualHostBufCtx *>(buffer->context);
    if (ctx) {
        if (ctx->base && ctx->reserved) pio::vm_release(ctx->base, ctx->reserved);
        delete ctx;
    }
}

void * vhost_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<VirtualHostBufCtx *>(buffer->context);
    return ctx ? ctx->base : nullptr;
}

void vhost_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {
    // Intentional no-op: keep reserved pages uncommitted (0 physical RAM).
}

void vhost_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (size == 0 || !tensor || !tensor->data || !data) return;
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

void vhost_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0 || !tensor || !tensor->data || !data) return;
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

bool vhost_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) {
    return false;
}

void vhost_clear(ggml_backend_buffer_t, uint8_t) {
    // Intentional no-op: clearing would fault every page of the reservation into physical RAM.
}

const ggml_backend_buffer_i k_vhost_buffer_iface = {
    /* .free_buffer   = */ vhost_free_buffer,
    /* .get_base      = */ vhost_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ vhost_memset_tensor,
    /* .set_tensor    = */ vhost_set_tensor,
    /* .get_tensor    = */ vhost_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ vhost_cpy_tensor,
    /* .clear         = */ vhost_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

const char * vhost_buft_get_name(ggml_backend_buffer_type_t) {
    return "BMOE_VirtualHost";
}

ggml_backend_buffer_t vhost_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (size == 0) {
        auto * ctx = new VirtualHostBufCtx{nullptr, 0};
        return ggml_backend_buffer_init(buft, k_vhost_buffer_iface, ctx, 0);
    }
    const size_t page = pio::vm_page();
    const size_t reserved = (size + page - 1) / page * page;
    void * base = pio::vm_reserve(reserved);
    if (!base) return nullptr;
    auto * ctx = new VirtualHostBufCtx{base, reserved};
    return ggml_backend_buffer_init(buft, k_vhost_buffer_iface, ctx, size);
}

size_t vhost_buft_get_alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

bool vhost_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

} // namespace

ggml_backend_buffer_type_t bmoe_virtual_host_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ vhost_buft_get_name,
            /* .alloc_buffer     = */ vhost_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ vhost_buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ vhost_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    if (!buft.device) buft.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return &buft;
}

} // namespace bmoe
