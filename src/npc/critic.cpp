#include "npc/critic.hpp"
#include <iostream>

CriticImpl::CriticImpl(const std::string& network_name,
					dim_type state_dim,
	                dim_type action_dim,
					torch::Device device)
		: BaseNetwork(network_name),
		  state_dim_(state_dim),
		  action_dim_(action_dim) {

	initialize_network(device);
}

void CriticImpl::initialize_network(torch::Device device) {
	fc1 = register_module("fc1", torch::nn::Linear(state_dim_ + action_dim_, 64));
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
    fc8 = register_module("fc8", torch::nn::Linear(64, 1));

	std::cout << "\nInitializing "<< this->network_name() << " network" << std::endl;
	count_type count = 0;
	for (const auto& pair : named_children()) {
		const auto& name = pair.key();
		const auto& child = pair.value();

		if (auto* linear = child->as<torch::nn::LinearImpl>()) {
			if (name != "fc8"){
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

	// Q-value 출력층 초기화
    torch::nn::init::kaiming_normal_(
        fc8->weight,
        std::sqrt(0.2f),
        torch::kFanIn,
        torch::kLeakyReLU
    );
	torch::nn::init::constant_(fc6->bias, 0.0);

    std::cout << "Initializing parameters for Q-value output layer"
        << " (fc8: " << fc8->weight.size(1) << " -> "
        << fc8->weight.size(0) << ")" << std::endl;

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

void CriticImpl::to(torch::Device device) {
	if (this->device() != device) {
		BaseNetwork::to(device);
	}
}

tensor_t CriticImpl::forward(const tensor_t& state, const tensor_t& action) {
	auto state_dev = state.to(this->device());
	auto action_dev = action.to(this->device());

	auto x = torch::cat({state_dev, action_dev}, 1);

    x = torch::leaky_relu(ln1->forward(fc1->forward(x)), 0.01);
    x = torch::leaky_relu(ln2->forward(fc2->forward(x)), 0.01);
    x = torch::leaky_relu(ln3->forward(fc3->forward(x)), 0.01);
    x = torch::leaky_relu(ln4->forward(fc4->forward(x)), 0.01);
	x = torch::leaky_relu(ln5->forward(fc5->forward(x)), 0.01);
	x = torch::leaky_relu(ln6->forward(fc6->forward(x)), 0.01);
	x = torch::leaky_relu(ln7->forward(fc7->forward(x)), 0.01);
	x = fc8->forward(x);

	return x;
}
