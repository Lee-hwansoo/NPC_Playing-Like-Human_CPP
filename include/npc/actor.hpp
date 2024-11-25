#pragma once

#include "npc/base_network.hpp"

using namespace types;
using namespace constants;

struct ActorImpl : public BaseNetwork {
	ActorImpl(const std::string& network_name,
			dim_type state_dim,
			dim_type action_dim,
			tensor_t min_action,
			tensor_t max_action,
			torch::Device device = torch::kCPU);

	void initialize_network(torch::Device device) override;
	void to(torch::Device device) override;

	std::tuple<tensor_t, tensor_t> forward(const tensor_t& state);
	std::tuple<tensor_t, tensor_t> sample(const tensor_t& state);

	dim_type state_dim() const { return state_dim_; }
	dim_type action_dim() const { return action_dim_; }

private:
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr }, fc6{ nullptr }, fc7{ nullptr }, fc8{ nullptr };
	torch::nn::LayerNorm ln1{ nullptr }, ln2{ nullptr }, ln3{ nullptr }, ln4{ nullptr }, ln5{ nullptr }, ln6{ nullptr }, ln7{ nullptr }, ln8{ nullptr };
	torch::nn::Linear fc_mean{ nullptr }, fc_log_std{ nullptr };

	dim_type state_dim_, action_dim_;
	tensor_t min_action_, max_action_;
};

TORCH_MODULE(Actor);
