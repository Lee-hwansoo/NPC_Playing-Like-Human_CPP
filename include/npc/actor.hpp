#pragma once

#include "npc/base_network.hpp"
#include <cstdint>
#include <vector>

struct ActorImpl : public BaseNetwork {
	ActorImpl(const std::string& network_name,
			int64_t state_dim,
			int64_t action_dim,
			const std::vector<float>& min_action,
			const std::vector<float>& max_action);

	void initialize_network() override;
	void to(torch::Device device) override;

	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state);
	std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& state);

	int64_t state_dim() const { return state_dim_; }
	int64_t action_dim() const { return action_dim_; }

private:
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
	torch::nn::Linear fc_mean{ nullptr }, fc_log_std{ nullptr };
	torch::nn::Dropout dropout{ nullptr };

	int64_t state_dim_, action_dim_;
	torch::Tensor min_action_, max_action_;
};

TORCH_MODULE(Actor);
