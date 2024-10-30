#include "npc/actor.hpp"
#include <iostream>

ActorImpl::ActorImpl(const std::string& network_name,
					dim_type state_dim,
	                dim_type action_dim,
	                tensor_t min_action,
					tensor_t max_action,
					torch::Device device)
		: BaseNetwork(network_name, device),
		  state_dim_(state_dim),
		  action_dim_(action_dim),
		  min_action_(min_action),
		  max_action_(max_action) {

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
	count_type count = 0;
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

	to(this->device());
}

void ActorImpl::to(torch::Device device) {
	if (this->device() != device) {
		BaseNetwork::to(device);
	}
}

std::tuple<tensor_t, tensor_t> ActorImpl::forward(const tensor_t& state) {
	auto x = state.to(this->device());

	x = torch::leaky_relu(fc1->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc2->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc3->forward(x));

	auto mean = fc_mean->forward(x);
	auto log_std = torch::clamp(fc_log_std->forward(x), -20.0, 2.0);

	return std::make_tuple(mean, log_std);
}

std::tuple<tensor_t, tensor_t> ActorImpl::sample(const tensor_t& state) {
	auto x = state.to(this->device());

	auto [mean, log_std] = forward(x);
	auto std = torch::exp(log_std);

	auto epsilon = torch::randn_like(mean);
	auto x_t = mean + epsilon * std;

	auto action = torch::tanh(x_t);
	action = (action + 1.0) / 2.0;
	action = action * (max_action_ - min_action_) + min_action_;

	auto log_prob = -0.5 * (
		((x_t - mean) / std).pow(2) +
		2.0 * log_std +
		std::log(2.0 * constants::PI)
		);

	log_prob = log_prob - torch::log(1.0 - action.pow(2) + 1e-6);
	log_prob = log_prob.sum(1, true);

	return std::make_tuple(action, log_prob);
}
