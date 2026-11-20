// Virtual-reservation host ggml_backend_buffer_type for zero-RAM, zero-disk routed experts.
//
// When booting a remote http:// MoE model via llama_model_init_from_user(), routed expert
// tensors are overridden to this buffer type. It reserves virtual address space via
// pio::vm_reserve() without committing physical RAM or touching local disk, while reporting
// is_host() == true so llama.cpp accepts it for CPU/GPU-cache host expert weights until
// ExpertStreamSource::init() rebinds tensor->data to its own streaming buffers.
#pragma once

#include "ggml-backend.h"

namespace bmoe {

ggml_backend_buffer_type_t bmoe_virtual_host_buffer_type();

} // namespace bmoe
