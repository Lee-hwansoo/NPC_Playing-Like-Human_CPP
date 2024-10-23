#pragma once

#include "torch/torch.h"
#include <vector>

struct ActorImpl : torch::nn::Module {
	ActorImpl(int64_t state_dim,
			int64_t action_dim,
			const std::vector<float>& min_action,
			const std::vector<float>& max_action);

	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state);
	std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& state);
	torch::Device device() const { return device_; }
	void to(torch::Device device);

private:
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
	torch::nn::Linear fc_mean{ nullptr }, fc_log_std{ nullptr };
	torch::nn::Dropout dropout{ nullptr };

	torch::Tensor min_action_;
	torch::Tensor max_action_;
	torch::Device device_{ torch::kCPU };
};
TORCH_MODULE(Actor);
