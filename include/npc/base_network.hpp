#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include <ctime>
#include <string>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <sstream>

using namespace types;
using namespace constants;

struct BaseNetwork : torch::nn::Module {
public:
    explicit BaseNetwork(const std::string& network_name, torch::Device device = torch::kCPU)
        : network_name_(network_name)
        , device_(device) {
        this->to(device_);
    }

    virtual ~BaseNetwork() = default;

    BaseNetwork(const BaseNetwork&) = delete;
    BaseNetwork& operator=(const BaseNetwork&) = delete;

    virtual void initialize_network() = 0;

    virtual void to(torch::Device device) {
        if (device_ != device) {
            device_ = std::move(device);
            torch::nn::Module::to(device);
        }
    }

    bool is_on_device(torch::Device device) const {
        return device_ == device;
    }

    torch::OrderedDict<std::string, tensor_t> state_dict() {
        torch::OrderedDict<std::string, tensor_t> state_dict;

		const auto& params = this->named_parameters();
		const auto& buffers = this->named_buffers();

        for (const auto& pair : params) {
            state_dict.insert(pair.key(), pair.value().detach().cpu());
        }

        for (const auto& pair : buffers) {
            state_dict.insert(pair.key(), pair.value().detach().cpu());
        }

        std::cout << "\nSuccessfully return " << network_name_ << " network parameters dictionary" << std::endl;
        return state_dict;
    }

    void load_state_dict(const torch::OrderedDict<std::string, tensor_t>& state_dict) {
        auto model_params = this->named_parameters(true);
        auto model_buffers = this->named_buffers(true);

        torch::NoGradGuard no_grad;

        for (const auto& pair : state_dict) {
            const auto& name = pair.key();
            const auto& tensor = pair.value();

            if (model_params.contains(name)) {
                model_params[name].copy_(tensor);
            }
            else if (model_buffers.contains(name)) {
                model_buffers[name].copy_(tensor);
            }
            else {
                std::cerr << "Warning: Key '" << name << "' in state dict was not found in the model" << std::endl;
            }
        }

        std::cout << "\nSuccessfully Loaded " << network_name_ << " network parameters" << std::endl;
    }

    void save_network_parameters(dim_type episode, bool print = true) {
        try {
            this->to(torch::kCPU);
            std::string timestamp = get_current_timestamp();
            std::string log_dir = get_log_directory();

            std::ostringstream filename;
            filename << timestamp << "_" << network_name_ << "_network_episode" << episode << ".pt";
            std::filesystem::path filepath = std::filesystem::path(log_dir) / filename.str();

            torch::serialize::OutputArchive archive;

            auto named_params = this->named_parameters(true);
            auto named_buffs = this->named_buffers(true);

            if (print){
                std::cout << "\nSaving parameters:" << std::endl;
            }

            for (const auto& pair : named_params) {
                archive.write(pair.key(), pair.value().detach().cpu());
                if (print){
                    std::cout << "Saving parameter: " << pair.key()
                            << " with size " << pair.value().sizes()
                            << " requires_grad: " << pair.value().requires_grad() << std::endl;
                }
            }

            for (const auto& pair : named_buffs) {
                archive.write(pair.key(), pair.value().detach().cpu());
                if (print){
                    std::cout << "Saving buffer: " << pair.key()
                            << " with size " << pair.value().sizes() << std::endl;
                }
            }

            if (print){
                std::cout << "\nSuccessfully saved " << network_name_ << " network parameters to: " << filepath << std::endl;
            }

            archive.save_to(filepath.string());
            this->to(device_);
        }
        catch (const std::exception& e) {
            std::cerr << "Error saving network parameters: " << e.what() << std::endl;
            throw;
        }
    }

    void load_network_parameters(const std::string& timestamp, dim_type episode) {
        try {
            std::string log_dir = get_log_directory();

            std::ostringstream filename;
            filename << timestamp << "_" << network_name_ << "_network_episode" << episode << ".pt";
            std::filesystem::path filepath = std::filesystem::path(log_dir) / filename.str();

            if (!std::filesystem::exists(filepath)) {
                throw std::runtime_error("Network parameter file not found: " + filepath.string());
            }

            auto model_params = this->named_parameters(true);
            auto model_buffers = this->named_buffers(true);

            torch::NoGradGuard no_grad;

            torch::jit::Module loaded_model;
            try {
                loaded_model = torch::jit::load(filepath.string(), device_);
                std::cout << "\nSuccessfully loaded the model file: " << filepath.string() << std::endl;

                std::cout << "\nChecking loaded " << network_name_ << " network parameters:" << std::endl;
                for (const auto& p : loaded_model.named_parameters()) {
                    std::cout << "Found parameter in loaded model: " << p.name
                            << " with size " << p.value.sizes() << std::endl;
                }

                std::cout << "\nChecking current model parameters:" << std::endl;
                auto current_params = this->named_parameters(true);
                for (const auto& p : current_params) {
                    std::cout << "Current model parameter: " << p.key()
                            << " with size " << p.value().sizes() << std::endl;
                }
            }
            catch (const c10::Error& e) {
                std::cerr << "Error loading the model: " << e.what() << std::endl;
                throw;
            }

            std::cout << "\nLoading parameters:" << std::endl;
            for (const auto& pair : loaded_model.named_parameters()) {
                try {
                    const std::string& name = pair.name;
                    const tensor_t& value = pair.value;
                    if (model_params.contains(name)) {
                        model_params[name].copy_(value);
                        std::cout << "Loaded parameter: " << name
                                << " with size " << value.sizes() << std::endl;
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
                    const tensor_t& value = pair.value;
                    if (model_buffers.contains(name)) {
                        model_buffers[name].copy_(value);
                        std::cout << "Loaded buffer: " << name
                                << " with size " << value.sizes() << std::endl;
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to load buffer: " << e.what() << std::endl;
                    throw;
                }
            }

            this->to(device_);
            std::cout << "Loaded network parameters from episode " << episode
                     << ". Training will continue from episode " << episode + 1 << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading network parameters: " << e.what() << std::endl;
            throw;
        }
    }

    void print_model_info(){
        print_model_size();
    }

    std::string network_name() const { return network_name_; }
    torch::Device device() const { return device_; }

private:
    std::string get_current_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm;

        #ifdef _WIN32
            localtime_s(&now_tm, &now_time);
        #else
            localtime_r(&now_time, &now_tm);
        #endif

        std::ostringstream oss;
        oss << std::put_time(&now_tm, "%Y%m%d_%H%M%S");
        return oss.str();
    }

    std::string get_log_directory() const {
        std::filesystem::path script_path(__FILE__);
        std::filesystem::path script_dir = script_path.parent_path();
        std::filesystem::path script_name = script_path.stem();
        std::filesystem::path log_dir = script_dir / constants::LOG_DIR / script_name;
        log_dir = std::filesystem::absolute(log_dir).lexically_normal();
        std::filesystem::create_directories(log_dir);
        return log_dir.string();
    }

    void print_model_size(){
        size_type total_params = 0;
        size_type total_memory = 0;

        std::cout << "\n" << network_name_ << " Network Memory Analysis:" << std::endl;
        for (const auto& pair : this->named_parameters(true)) {
            const auto& param = pair.value();
            size_type numel = param.numel();  // 파라미터 개수
            size_type bytes = numel * sizeof(real_t);  // 메모리 크기 (float 기준)

            total_params += numel;
            total_memory += bytes;

            std::cout << pair.key() << ": "
                    << "Parameters: " << numel << ", "
                    << "Memory: " << bytes / 1024.0 << " KB" << std::endl;
        }

        std::cout << "Total Parameters: " << total_params << std::endl;
        std::cout << "Total Memory: " << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    std::string network_name_;
    torch::Device device_{ torch::kCPU };
};
