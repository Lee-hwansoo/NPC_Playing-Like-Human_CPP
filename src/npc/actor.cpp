#include "npc/actor.hpp"

ActorImpl::ActorImpl(int64_t state_dim,
	                int64_t action_dim,
	                const std::vector<float>& min_action,
	                const std::vector<float>& max_action) {

	initialize_network(state_dim, action_dim);

	min_action_ = torch::tensor(min_action);
	max_action_ = torch::tensor(max_action);
}

void ActorImpl::initialize_network(int64_t state_dim, int64_t action_dim) {
	fc1 = register_module("fc1", torch::nn::Linear(state_dim, 256));
	fc2 = register_module("fc2", torch::nn::Linear(256, 256));
	fc3 = register_module("fc3", torch::nn::Linear(256, 256));
	fc_mean = register_module("fc_mean", torch::nn::Linear(256, action_dim));
	fc_log_std = register_module("fc_log_std", torch::nn::Linear(256, action_dim));
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

void ActorImpl::to(torch::Device device) {
	if (device_ != device) {
		device_ = device;
		torch::nn::Module::to(device);
		min_action_ = min_action_.to(device);
		max_action_ = max_action_.to(device);
	}
}

std::tuple<torch::Tensor, torch::Tensor> ActorImpl::forward(const torch::Tensor& state) {
	auto x = state.device() == device_ ? state : state.to(device_);

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
	auto state_device = state.device() == device_ ? state : state.to(device_);

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

std::string ActorImpl::get_current_timestamp() const {
	auto now = std::chrono::system_clock::now();
	auto now_time = std::chrono::system_clock::to_time_t(now);
	auto now_tm = *std::localtime(&now_time);

	std::ostringstream oss;
	oss << std::put_time(&now_tm, "%Y%m%d_%H%M%S");
	return oss.str();
}

std::string ActorImpl::get_log_directory() const {
	std::filesystem::path script_path(__FILE__);
	std::filesystem::path script_dir = script_path.parent_path();
	std::filesystem::path script_name = script_path.stem();
	std::filesystem::path log_dir = script_dir / "../../logs" / script_name;

	std::filesystem::create_directories(log_dir);
	return log_dir.string();
}
//
//void ActorImpl::save_network_parameters(int64_t episode) {
//	try {
//		std::string timestamp = get_current_timestamp();
//		std::string log_dir = get_log_directory();
//
//		std::ostringstream filename;
//		filename << timestamp << "_actor_network_episode" << episode << ".pth";
//		std::filesystem::path save_path = std::filesystem::path(log_dir) / filename.str();
//
//		torch::save(this->named_parameters(), save_path.string());
//		std::cout << "Successfully saved network parameters to: " << save_path << std::endl;
//	}
//	catch (const std::exception& e) {
//		std::cerr << "Error saving network parameters: " << e.what() << std::endl;
//		throw;
//	}
//}
//
//void ActorImpl::load_network_parameters(const std::string& timestamp, int64_t episode) {
//	try {
//		std::string log_dir = get_log_directory();
//
//		std::ostringstream filename;
//		filename << timestamp << "_actor_network_episode" << episode << ".pth";
//		std::filesystem::path load_path = std::filesystem::path(log_dir) / filename.str();
//
//		if (!std::filesystem::exists(load_path)) {
//			throw std::runtime_error("Network parameter file not found: " + load_path.string());
//		}
//
//		torch::OrderedDict<std::string, torch::Tensor> parameters;
//		torch::load(parameters, load_path.string());
//
//		auto current_params = this->named_parameters();
//
//		// Parameter 검증 및 로딩
//		for (const auto& pair : parameters) {
//			const auto& name = pair.key();
//			const auto& loaded_tensor = pair.value();
//
//			if (current_params.contains(name)) {
//				auto& current_tensor = current_params[name];
//
//				if (current_tensor.sizes() != loaded_tensor.sizes()) {
//					throw std::runtime_error(
//						"Size mismatch for parameter '" + name +
//						"': expected " + std::to_string(current_tensor.numel()) +
//						" elements, but got " + std::to_string(loaded_tensor.numel())
//					);
//				}
//
//				current_tensor.copy_(loaded_tensor.to(device_));
//			}
//			else {
//				std::cerr << "Warning: Loaded parameter '" << name
//					<< "' not found in current model" << std::endl;
//			}
//		}
//
//		start_episode_ = episode;
//		std::cout << "Loaded network parameters from episode " << episode
//			<< ". Training will continue from episode " << start_episode_ + 1
//			<< std::endl;
//	}
//	catch (const std::exception& e) {
//		std::cerr << "Error loading network parameters: " << e.what() << std::endl;
//		throw;
//	}
//}
