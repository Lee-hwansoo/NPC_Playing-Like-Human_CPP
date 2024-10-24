#pragma once

#include "npc/base_network.hpp"
#include <cstdint>

struct CriticImpl : public BaseNetwork {
	CriticImpl(const std::string& network_name,
			int64_t state_dim,
			int64_t action_dim);

	void initialize_network() override;
	void to(torch::Device device) override;

	torch::Tensor forward(const torch::Tensor& state, const torch::Tensor& action);

	int64_t state_dim() const { return state_dim_; }
	int64_t action_dim() const { return action_dim_; }

private:
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr };
	torch::nn::Dropout dropout{ nullptr };

	int64_t state_dim_, action_dim_;
};

TORCH_MODULE(Critic);
