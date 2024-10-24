#pragma once

#include "npc/actor.hpp"
#include "npc/critic.hpp"
#include "npc/base_network.hpp"

#include <torch/torch.h>
#include <deque>
#include <random>
#include <tuple>
#include <vector>
#include <memory>

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t buffer_size, size_t batch_size);

    virtual ~ReplayBuffer() = default;

    ReplayBuffer(const ReplayBuffer&) = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    void add(const torch::Tensor& state, const torch::Tensor& action,
             const torch::Tensor& reward, const torch::Tensor& next_state,
             const torch::Tensor& done);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor> sample();

    size_t size() const { return buffer_.size(); }
    size_t batch_size() const { return batch_size_; }

private:
    std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor>> buffer_;
    size_t buffer_size_;
    size_t batch_size_;
    std::mt19937 generator_;
};

class SAC {
public:
    explicit SAC(int64_t state_dim, int64_t action_dim,
                const std::vector<float>& min_action,
                const std::vector<float>& max_action,
                torch::Device device);

    virtual ~SAC() = default;

    SAC(const SAC&) = delete;
    SAC& operator=(const SAC&) = delete;

    torch::Tensor select_action(const torch::Tensor& state);
    void update();
    std::vector<float> train(int episodes, bool render = false);
    std::vector<float> test(int episodes, bool render = true);
    void save_network_parameters(int64_t episode);
    void load_network_parameters(const std::string& timestamp, int64_t episode);

    int64_t state_dim() const { return state_dim_; }
	int64_t action_dim() const { return action_dim_; }
    torch::Device device() const { return device_; }

private:
    void update_target_networks();

    Actor actor_{nullptr};
    Critic critic1_{nullptr}, critic2_{nullptr}, critic1_target_{nullptr}, critic2_target_{nullptr};
    torch::optim::Adam actor_optimizer_, critic1_optimizer_, critic2_optimizer_;
    std::unique_ptr<ReplayBuffer> memory_;

    int64_t state_dim_, action_dim_;
    const std::vector<float>& min_action_, max_action_;
    torch::Device device_;

    double gamma_;
    double tau_;
    double alpha_;
    int64_t start_episode_;
};
