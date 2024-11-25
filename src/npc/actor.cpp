#include "npc/actor.hpp"
#include <iostream>

ActorImpl::ActorImpl(const std::string& network_name,
					dim_type state_dim,
	                dim_type action_dim,
	                tensor_t min_action,
					tensor_t max_action,
					torch::Device device)
		: BaseNetwork(network_name),
		  state_dim_(state_dim),
		  action_dim_(action_dim),
		  min_action_(min_action),
		  max_action_(max_action) {

	initialize_network(device);
}

void ActorImpl::initialize_network(torch::Device device) {
	fc1 = register_module("fc1", torch::nn::Linear(state_dim_, 64));
	ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({64})));
	fc2 = register_module("fc2", torch::nn::Linear(64, 128));
	ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
	fc3 = register_module("fc3", torch::nn::Linear(128, 256));
	ln3 = register_module("ln3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
	fc4 = register_module("fc4", torch::nn::Linear(256, 256));
	ln4 = register_module("ln4", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
	fc5 = register_module("fc5", torch::nn::Linear(256, 256));
	ln5 = register_module("ln5", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
	fc6 = register_module("fc6", torch::nn::Linear(256, 128));
	ln6 = register_module("ln6", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
	fc7 = register_module("fc7", torch::nn::Linear(128, 64));
	ln7 = register_module("ln7", torch::nn::LayerNorm(torch::nn::LayerNormOptions({64})));
	fc_mean = register_module("fc_mean", torch::nn::Linear(64, action_dim_));
	fc_log_std = register_module("fc_log_std", torch::nn::Linear(64, action_dim_));

	std::cout << "\nInitializing "<< this->network_name() << " network" << std::endl;
	count_type count = 0;
	for (const auto& pair : named_children()) {
		const auto& name = pair.key();
		const auto& child = pair.value();

		if (auto* linear = child->as<torch::nn::LinearImpl>()) {
			if (name != "fc_mean" && name != "fc_log_std"){
				torch::nn::init::kaiming_normal_(
					linear->weight,
					std::sqrt(2.0f / (1.0f + std::pow(0.01, 2))),
					torch::kFanOut,
					torch::kLeakyReLU
				);

				torch::nn::init::constant_(linear->bias, 0.1);

				count++;
				std::cout << "Initializing parameters for layer " << count
					<< " (" << name << ": "
					<< linear->weight.size(1) << " -> "
					<< linear->weight.size(0) << ")" << std::endl;
			}
		}
	}

	// Policy mean 출력층 초기화
	torch::nn::init::kaiming_normal_(
		fc_mean->weight,
		std::sqrt(0.2f),
		torch::kFanIn,
		torch::kLeakyReLU
	);
	torch::nn::init::constant_(fc_mean->bias, 0.0);

    std::cout << "Initializing parameters for Policy mean output layer"
        << " (fc_mean: " << fc_mean->weight.size(1) << " -> "
        << fc_mean->weight.size(0) << ")" << std::endl;

	// Log std 출력층 초기화
	torch::nn::init::kaiming_normal_(
		fc_log_std->weight,
		std::sqrt(0.1f),
		torch::kFanIn,
		torch::kLeakyReLU
	);
	torch::nn::init::constant_(fc_log_std->bias, -1.0);

    std::cout << "Initializing parameters for Log std output layer"
        << " (fc_log_std: " << fc_log_std->weight.size(1) << " -> "
        << fc_log_std->weight.size(0) << ")" << std::endl;

	std::cout << "\nNetwork Weight Statistics:" << std::endl;
	for (const auto& pair : named_parameters()) {
		const auto& name = pair.key();
		const auto& param = pair.value();

		auto mean = param.mean().item<real_t>();
		auto std = param.std().item<real_t>();
		auto max_abs = param.abs().max().item<real_t>();

		std::cout << name << ":"
					<< " mean=" << mean
					<< " std=" << std
					<< " max_abs=" << max_abs << std::endl;
	}

	to(device);
}

void ActorImpl::to(torch::Device device) {
	if (this->device() != device) {
		BaseNetwork::to(device);
		min_action_ = min_action_.to(device);
		max_action_ = max_action_.to(device);
	}
}

std::tuple<tensor_t, tensor_t> ActorImpl::forward(const tensor_t& state) {
	auto x = state.to(this->device());

    x = torch::leaky_relu(ln1->forward(fc1->forward(x)), 0.01);
    x = torch::leaky_relu(ln2->forward(fc2->forward(x)), 0.01);
    x = torch::leaky_relu(ln3->forward(fc3->forward(x)), 0.01);
    x = torch::leaky_relu(ln4->forward(fc4->forward(x)), 0.01);
	x = torch::leaky_relu(ln5->forward(fc5->forward(x)), 0.01);
	x = torch::leaky_relu(ln6->forward(fc6->forward(x)), 0.01);
	x = torch::leaky_relu(ln7->forward(fc7->forward(x)), 0.01);

	auto mean = fc_mean->forward(x);
	auto log_std = torch::clamp(fc_log_std->forward(x), -20.0, 2.0);

	return std::make_tuple(mean, log_std);
}

std::tuple<tensor_t, tensor_t> ActorImpl::sample(const tensor_t& batch_state) {
	auto x = batch_state.to(this->device());

	auto [mean, log_std] = forward(x);

	const real_t epsilon = 1e-6;

	auto std = torch::exp(log_std);
	auto noise = torch::randn_like(mean);
	auto x_t = mean + noise * std;
	auto raw_action = torch::tanh(x_t);

    auto log_prob = -0.5 * (
        ((x_t - mean) / (std + epsilon)).pow(2) +
        2.0 * log_std +
        std::log(2.0 * constants::PI)
    );

	log_prob = log_prob - torch::log(1.0 - raw_action.pow(2) + epsilon);
	log_prob = log_prob.sum(1, true);

	auto action = (raw_action + 1.0) / 2.0;
	action = action * (max_action_ - min_action_) + min_action_;

	return std::make_tuple(action, log_prob);
}
