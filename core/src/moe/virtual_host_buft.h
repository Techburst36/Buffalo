// Virtual-reservation host ggml_backend_buffer_type for zero-RAM, zero-disk streamed weights.
//
// When booting a remote http:// model via llama_model_init_from_user(), streamed host tensors
// (routed experts and/or streamed dense weights) are overridden to this buffer type. It reserves
// virtual address space via pio::vm_reserve() without committing physical RAM or touching local
// disk, while reporting is_host() == true. Every allocated tensor is placed on its own disjoint
// OS page boundary (with tail-slack padding rounded up to pio::vm_page()), so pio::vm_evict() on
// one tensor can never zero a neighboring tensor's boundary page.
#pragma once

#include "ggml-backend.h"

namespace bmoe {

ggml_backend_buffer_type_t bmoe_virtual_host_buffer_type();

} // namespace bmoe
