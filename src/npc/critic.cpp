#include "npc/critic.hpp"
#include <iostream>

CriticImpl::CriticImpl(const std::string& network_name,
					int64_t state_dim,
	                int64_t action_dim)
		: BaseNetwork(network_name),
		  state_dim_(state_dim),
		  action_dim_(action_dim) {

	initialize_network();
}

void CriticImpl::initialize_network() {
	fc1 = register_module("fc1", torch::nn::Linear(state_dim_ + action_dim_, 256));
	fc2 = register_module("fc2", torch::nn::Linear(256, 256));
	fc3 = register_module("fc3", torch::nn::Linear(256, 256));
    fc4 = register_module("fc4", torch::nn::Linear(256, 1));
	dropout = register_module("dropout", torch::nn::Dropout(0.1));

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

void CriticImpl::to(torch::Device device) {
	if (this->device() != device) {
		BaseNetwork::to(device);
	}
}

torch::Tensor CriticImpl::forward(const torch::Tensor& state, const torch::Tensor& action) {
	auto x = torch::cat({state, action}, 1);
    x = x.to(this->device());

	x = torch::leaky_relu(fc1->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc2->forward(x));
	x = dropout->forward(x);
	x = torch::leaky_relu(fc3->forward(x));
    x = fc4->forward(x);

	return x;
}
