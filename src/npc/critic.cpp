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
	fc1 = register_module("fc1", torch::nn::Linear(state_dim_ + action_dim_, 128));
	ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
	fc2 = register_module("fc2", torch::nn::Linear(128, 256));
	fc3 = register_module("fc3", torch::nn::Linear(256, 256));
	fc4 = register_module("fc4", torch::nn::Linear(256, 128));
	ln4 = register_module("ln4", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
    fc5 = register_module("fc5", torch::nn::Linear(128, 1));

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

    x = torch::leaky_relu(ln1->forward(fc1->forward(x)));
    x = torch::leaky_relu(fc2->forward(x));
    x = torch::leaky_relu(fc3->forward(x));
    x = torch::leaky_relu(ln4->forward(fc4->forward(x)));
    x = fc5->forward(x);

	return x;
}
