#pragma once

#include "npc/actor.hpp"
#include "npc/critic.hpp"
#include <torch/torch.h>
#include <tuple>
#include <memory>

class ReplayBuffer {
public:
    explicit ReplayBuffer(dim_type state_dim, dim_type action_dim,
                        count_type buffer_size, index_type batch_size,
                        torch::Device device);

    virtual ~ReplayBuffer() = default;

    ReplayBuffer(const ReplayBuffer&) = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    void add(const tensor_t& state, const tensor_t& action,
             const tensor_t& reward, const tensor_t& next_state,
             const tensor_t& done);

    std::tuple<tensor_t, tensor_t, tensor_t, tensor_t, tensor_t> sample();

    size_type size() const { return current_size_; }
    size_type batch_size() const { return batch_size_; }
    torch::Device device() const { return device_; }

private:
    tensor_t states_, actions_, rewards_, next_states_, dones_;

    dim_type state_dim_, action_dim_;
    count_type buffer_size_;
    index_type batch_size_;
    count_type current_size_;
    index_type position_;
    torch::Device device_;

    tensor_t indices_;

    void preallocate_tensors();
};

class SAC {
public:
    explicit SAC(dim_type state_dim, dim_type action_dim,
                tensor_t min_action,
                tensor_t max_action,
                ReplayBuffer* memory,
                torch::Device device);

    virtual ~SAC() = default;

    SAC(const SAC&) = delete;
    SAC& operator=(const SAC&) = delete;

    void add(const tensor_t& state, const tensor_t& action,
             const tensor_t& reward, const tensor_t& next_state,
             const tensor_t& done) {
        memory_->add(state, action, reward, next_state, done);
    }

    void train() {
        actor_->train();
        critic1_->train();
        critic2_->train();
        critic1_target_->train();
        critic2_target_->train();
    }
    void eval() {
        actor_->eval();
        critic1_->eval();
        critic2_->eval();
        critic1_target_->eval();
        critic2_target_->eval();
    }

    tensor_t select_action(const tensor_t& state);
    void update(bool debug = false);
    void save_network_parameters(dim_type episode, bool print = true) {
        actor_->save_network_parameters(episode, print);
		critic1_->save_network_parameters(episode, print);
		critic2_->save_network_parameters(episode, print);
    }
    void load_network_parameters(const std::string& timestamp, dim_type episode){
        actor_->load_network_parameters(timestamp, episode);
        critic1_->load_network_parameters(timestamp, episode);
        critic2_->load_network_parameters(timestamp, episode);
        critic1_target_->load_state_dict(critic1_->state_dict());
        critic2_target_->load_state_dict(critic2_->state_dict());
    };

    void print_model_info(){
        actor_->print_model_info();
        critic1_->print_model_info();
        critic2_->print_model_info();
    }

    dim_type state_dim() const { return state_dim_; }
	dim_type action_dim() const { return action_dim_; }
    torch::Device device() const { return device_; }

private:
    void update_target_networks();

    Actor actor_{nullptr};
    Critic critic1_{nullptr}, critic2_{nullptr}, critic1_target_{nullptr}, critic2_target_{nullptr};
    torch::optim::Adam actor_optimizer_, critic1_optimizer_, critic2_optimizer_;
    ReplayBuffer* memory_;

    dim_type state_dim_, action_dim_;
    tensor_t min_action_, max_action_;
    torch::Device device_;

    real_t gamma_;
    real_t tau_;
    real_t alpha_;
};
