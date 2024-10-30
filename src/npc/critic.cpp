#include "npc/critic.hpp"
#include <iostream>

CriticImpl::CriticImpl(const std::string& network_name,
					dim_type state_dim,
	                dim_type action_dim,
					torch::Device device)
		: BaseNetwork(network_name, device),
		  state_dim_(state_dim),
		  action_dim_(action_dim) {

	initialize_network();
}

void CriticImpl::initialize_network() {
	fc1 = register_module("fc1", torch::nn::Linear(state_dim_ + action_dim_, 128));
	fc2 = register_module("fc2", torch::nn::Linear(128, 256));
	fc3 = register_module("fc3", torch::nn::Linear(256, 256));
	fc4 = register_module("fc4", torch::nn::Linear(256, 256));
	fc5 = register_module("fc5", torch::nn::Linear(256, 128));
    fc6 = register_module("fc6", torch::nn::Linear(128, 1));
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

void CriticImpl::to(torch::Device device) {
	if (this->device() != device) {
		BaseNetwork::to(device);
	}
}

tensor_t CriticImpl::forward(const tensor_t& state, const tensor_t& action) {
	auto state_dev = state.to(this->device());
	auto action_dev = action.to(this->device());

	auto x = torch::cat({state_dev, action_dev}, 1);

	x = torch::leaky_relu(fc1->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc2->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc3->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc4->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc5->forward(x));
	x = dropout->forward(x);
    x = fc6->forward(x);

	return x;
}
