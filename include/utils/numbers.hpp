#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <array>

namespace types {
    // real numbers
    using real_t = float;
    using tensor_t = torch::Tensor;

    // integer numbers
    using size_type = std::size_t;
    using dim_type = int64_t;
    using index_type = int32_t;
    using count_type = uint32_t;

    // container
    template<typename T, size_type N>
    using fixed_array = std::array<T, N>;
    using color_rgb = fixed_array<uint8_t, 3>;

    // tensor type utilities
    constexpr auto get_tensor_dtype() {
        if constexpr (std::is_same_v<real_t, float>) {
            return torch::kFloat32;
        } else if constexpr (std::is_same_v<real_t, double>) {
            return torch::kFloat64;
        }
    }
}
