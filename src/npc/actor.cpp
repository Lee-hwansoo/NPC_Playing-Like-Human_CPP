#include "npc/actor.hpp"

#include <iostream>

ActorImpl::ActorImpl(const std::string& network_name,
					int64_t state_dim,
	                int64_t action_dim,
	                const std::vector<float>& min_action,
	                const std::vector<float>& max_action)
		: BaseNetwork(network_name),
		  state_dim_(state_dim),
		  action_dim_(action_dim) {

	min_action_ = torch::tensor(min_action);
	max_action_ = torch::tensor(max_action);
	initialize_network();
}

void ActorImpl::initialize_network() {
	fc1 = register_module("fc1", torch::nn::Linear(state_dim_, 256));
	fc2 = register_module("fc2", torch::nn::Linear(256, 256));
	fc3 = register_module("fc3", torch::nn::Linear(256, 256));
	fc_mean = register_module("fc_mean", torch::nn::Linear(256, action_dim_));
	fc_log_std = register_module("fc_log_std", torch::nn::Linear(256, action_dim_));
	dropout = register_module("dropout", torch::nn::Dropout(0.1));

	std::cout << "\nInitializing "<< this->network_name() << " network" << std::endl;
	int count = 0;
	for (const auto& pair : named_children()) {
		const auto& name = pair.key();
		const auto& child = pair.value();

		if (auto* linear = child->as<torch::nn::LinearImpl>()) {
			torch::nn::init::xavier_uniform_(linear->weight);
			torch::nn::init::constant_(linear->bias, 0.0);
			count++;
			std::cout << "Initializing parameters for layer " << count
				<< " (" << name << ": "
				<< linear->weight.size(1) << " -> "
				<< linear->weight.size(0) << ")" << std::endl;
		}
	}
}

void ActorImpl::to(torch::Device device) {
	if (this->device() != device) {
		BaseNetwork::to(device);
		min_action_ = min_action_.to(device);
		max_action_ = max_action_.to(device);
	}
}

std::tuple<torch::Tensor, torch::Tensor> ActorImpl::forward(const torch::Tensor& state) {
	auto x = state.device() == this->device() ? state : state.to(this->device());

	x = torch::leaky_relu(fc1->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc2->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc3->forward(x));

	auto mean = fc_mean->forward(x);
	auto log_std = torch::clamp(fc_log_std->forward(x), -20.0, 2.0);

	return std::make_tuple(mean, log_std);
}

std::tuple<torch::Tensor, torch::Tensor> ActorImpl::sample(const torch::Tensor& state) {
	auto state_device = state.device() == this->device() ? state : state.to(this->device());

	auto [mean, log_std] = forward(state_device);
	auto std = torch::exp(log_std);

	auto epsilon = torch::randn_like(mean);
	auto x_t = mean + epsilon * std;

	auto action = torch::tanh(x_t);
	action = (action + 1.0) / 2.0;
	action = action * (max_action_ - min_action_) + min_action_;

	auto log_prob = -0.5 * (
		((x_t - mean) / std).pow(2) +
		2.0 * log_std +
		std::log(2.0 * M_PI)
		);

	log_prob = log_prob - torch::log(1.0 - action.pow(2) + 1e-6);
	log_prob = log_prob.sum(1, true);

	return std::make_tuple(action, log_prob);
}
