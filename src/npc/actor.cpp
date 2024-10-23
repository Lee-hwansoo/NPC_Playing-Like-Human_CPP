#include "npc/actor.hpp"
#include <iostream>

ActorImpl::ActorImpl(std::string network_name,
					int64_t state_dim,
	                int64_t action_dim,
	                const std::vector<float>& min_action,
	                const std::vector<float>& max_action)
					: network_name_(network_name) {

	this->initialize_network(state_dim, action_dim);

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
	log_dir = std::filesystem::absolute(log_dir).lexically_normal();
	std::filesystem::create_directories(log_dir);
	return log_dir.string();
}

void ActorImpl::save_network_parameters(int64_t episode) {
	try {
		std::string timestamp = this->get_current_timestamp();
		std::string log_dir = this->get_log_directory();

		std::ostringstream filename;
		filename << timestamp << "_" << this->network_name() << "_network_episode" << episode << ".pt";
		std::filesystem::path filepath = std::filesystem::path(log_dir) / filename.str();

		torch::jit::Module module(this->network_name());

        for (const auto& pair : this->named_parameters()) {
            module.register_parameter(pair.key(), pair.value(), false);  // false = not requiring gradient
        }

        for (const auto& pair : this->named_buffers()) {
            module.register_buffer(pair.key(), pair.value());
        }

        module.save(filepath.string());

		std::cout << "Successfully saved network parameters to: " << filepath << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error saving network parameters: " << e.what() << std::endl;
		throw;
	}
}

void ActorImpl::load_network_parameters(const std::string& timestamp, int64_t episode) {
	try {
		std::string log_dir = this->get_log_directory();

		std::ostringstream filename;
		filename << timestamp << "_" << this->network_name() << "_network_episode" << episode << ".pt";
		std::filesystem::path filepath = std::filesystem::path(log_dir) / filename.str();

		std::cout << "Loading network parameters from: " << filepath << std::endl;

		if (!std::filesystem::exists(filepath)) {
			throw std::runtime_error("Network parameter file not found: " + filepath.string());
		}else{
			std::cout << "Found parameter file: " << filepath << std::endl;
			auto fileSize = std::filesystem::file_size(filepath);
        	std::cout << "Found parameter file (size: " << fileSize << " bytes)" << std::endl;
		}

        auto model_params = this->named_parameters(true);
        auto model_buffers = this->named_buffers(true);

        torch::jit::Module loaded_model;
        try {
            loaded_model = torch::jit::load(filepath.string());
            std::cout << "Successfully loaded the model file." << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }

		for (const auto& pair : loaded_model.named_parameters()) {
			try {
				const std::string& name = pair.name;
				const torch::Tensor& value = pair.value;
				if (model_params.contains(name)) {
					torch::NoGradGuard no_grad;
					model_params[name].copy_(value);
					std::cout << "Loaded parameter: " << name << " with size "
						<< value.sizes() << std::endl;
				}
			}
			catch (const std::exception& e) {
				std::cerr << "Warning: Failed to load parameter: " << e.what() << std::endl;
				throw;
			}
		}

		for (const auto& pair : loaded_model.named_buffers()) {
			try {
				const std::string& name = pair.name;
				const torch::Tensor& value = pair.value;
				if (model_buffers.contains(name)) {
					torch::NoGradGuard no_grad;
					model_buffers[name].copy_(value);
					std::cout << "Loaded buffer: " << name << " with size "
						<< value.sizes() << std::endl;
				}
			}
			catch (const std::exception& e) {
				std::cerr << "Warning: Failed to load buffer: " << e.what() << std::endl;
				throw;
			}
		}

		std::cout << "Loaded network parameters from episode " << episode
			<< ". Training will continue from episode " << episode + 1
			<< std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Error loading network parameters: " << e.what() << std::endl;
		throw;
	}
}
