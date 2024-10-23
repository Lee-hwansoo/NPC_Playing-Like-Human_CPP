#pragma once

#include "torch/torch.h"
#include "torch/serialize.h"
#include <vector>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <string>
#include <stdexcept>
#include <exception>

struct ActorImpl : torch::nn::Module {
	ActorImpl(int64_t state_dim,
			int64_t action_dim,
			const std::vector<float>& min_action,
			const std::vector<float>& max_action);

	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state);
	std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& state);

	void initialize_network(int64_t state_dim, int64_t action_dim);
	void to(torch::Device device);
	void save_network_parameters(int64_t episode);
	void load_network_parameters(const std::string& timestamp, int64_t episode);

	torch::Device device() const { return device_; }

private:
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
	torch::nn::Linear fc_mean{ nullptr }, fc_log_std{ nullptr };
	torch::nn::Dropout dropout{ nullptr };

	torch::Tensor min_action_;
	torch::Tensor max_action_;
	torch::Device device_{ torch::kCPU };

	std::string get_log_directory() const;
	std::string get_current_timestamp() const;
};

TORCH_MODULE(Actor);
