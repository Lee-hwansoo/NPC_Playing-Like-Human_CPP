#pragma once

#include "npc/base_network.hpp"

struct CriticImpl : public BaseNetwork {
	CriticImpl(const std::string& network_name,
			dim_type state_dim,
			dim_type action_dim,
			torch::Device device = torch::kCPU);

	void initialize_network(torch::Device device) override;
	void to(torch::Device device) override;

	tensor_t forward(const tensor_t& state, const tensor_t& action);

	dim_type state_dim() const { return state_dim_; }
	dim_type action_dim() const { return action_dim_; }

private:
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr }, fc6{ nullptr }, fc7{ nullptr }, fc8{ nullptr }, fc9{ nullptr }, fc10{ nullptr };
	torch::nn::LayerNorm ln1{ nullptr }, ln2{ nullptr }, ln3{ nullptr }, ln4{ nullptr }, ln5{ nullptr }, ln6{ nullptr }, ln7{ nullptr }, ln8{ nullptr }, ln9{ nullptr };
	dim_type state_dim_, action_dim_;
};

TORCH_MODULE(Critic);
