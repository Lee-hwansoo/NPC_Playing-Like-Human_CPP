#pragma once

#include "utils/types.hpp"
#include <torch/torch.h>
#include <optional>

namespace utils {

using namespace types;

class FrenetCoordinate {
public:
    struct Result {
        tensor_t closest_point;
        real_t lateral_distance;
    };

    static int64_t getClosestWaypoint(const tensor_t& position,
                                    const tensor_t& path,
                                    const torch::Device& device) {

        auto position_dev = position.to(device);
        auto path_dev = path.to(device);

        auto expanded_position = position_dev.unsqueeze(0).expand({path_dev.size(0), -1});

        auto distances = torch::norm(path_dev - expanded_position, 2, 1);

        return torch::argmin(distances).cpu().item<dim_type>();
    }

    static std::optional<Result> getFrenetD(const tensor_t& position,
                                          const tensor_t& path,
                                          const torch::Device& device) {

        if (path.size(0) < 2) {
            return std::nullopt;
        }

        auto position_dev = position.to(device);
        auto path_dev = path.to(device);

        auto closest_wp = getClosestWaypoint(position_dev, path_dev, device);
        auto next_wp = (closest_wp + 1) % path_dev.size(0);

        auto n_vec = path_dev[next_wp] - path_dev[closest_wp];
        auto x_vec = position_dev - path_dev[closest_wp];

        auto proj_norm = torch::dot(x_vec, n_vec) / torch::dot(n_vec, n_vec);
        auto proj_vec = proj_norm * n_vec;

        auto frenet_d = torch::norm(x_vec - proj_vec).cpu().item<real_t>();

        auto x_vec_3d = torch::cat({x_vec, torch::zeros({1}, device)});
        auto n_vec_3d = torch::cat({n_vec, torch::zeros({1}, device)});
        auto d_cross = torch::cross(x_vec_3d, n_vec_3d, 0);

        if (d_cross[2].cpu().item<real_t>() > 0) {
            frenet_d = -frenet_d;
        }

        return std::optional<Result>{{path_dev[closest_wp].cpu(), frenet_d}};
    }
};

} // namespace frenet
