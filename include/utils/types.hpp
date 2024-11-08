#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <array>
#include <random>

namespace types {
    // real numbers
    using real_t = float;
    using real_real_t = double;
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

    struct Vector2 {
        real_t a;
        real_t b;

        constexpr Vector2(real_t a_ = 0.0f, real_t b_ = 0.0f)
             : a(a_), b(b_) {}
    };

    struct Bounds2D {
        real_t min_x;
        real_t max_x;
        real_t min_y;
        real_t max_y;

        constexpr Bounds2D(real_t min_x_, real_t max_x_, real_t min_y_, real_t max_y_)
            : min_x(min_x_), max_x(max_x_), min_y(min_y_), max_y(max_y_) {}

        bool is_outside(real_t x, real_t y) const {
            return x < min_x || x > max_x || y < min_y || y > max_y;
        }

        real_t random_x(std::mt19937& gen) const {
            std::uniform_real_distribution<real_t> dist(min_x, max_x);
            return dist(gen);
        }

        real_t random_y(std::mt19937& gen) const {
            std::uniform_real_distribution<real_t> dist(min_y, max_y);
            return dist(gen);
        }
    };

	struct SACMetrics {
		real_t critic_loss1;
		real_t critic_loss2;
		real_t actor_loss;
		real_t log_pi;
		real_t q_value;
        real_t beta;
		bool is_vaild = false;
	};

	struct TrainingResult {
		std::vector<real_t> rewards;
		std::vector<SACMetrics> metrics;
	};
}
