#include "npc/sac.hpp"
#include "utils/types.hpp"
#include <iostream>

ReplayBuffer::ReplayBuffer(dim_type state_dim, dim_type action_dim,
                         count_type buffer_size, index_type batch_size,
                         torch::Device device)
    : state_dim_(state_dim)
    , action_dim_(action_dim)
    , buffer_size_(buffer_size)
    , batch_size_(batch_size)
    , current_size_(0)
    , position_(0)
    , device_(device) {

    preallocate_tensors();
}

void ReplayBuffer::preallocate_tensors() {
    auto options = torch::TensorOptions()
                      .dtype(types::get_tensor_dtype())
                      .device(device_);

    states_ = torch::zeros({buffer_size_, state_dim_}, options);
    actions_ = torch::zeros({buffer_size_, action_dim_}, options);
    rewards_ = torch::zeros({buffer_size_, 1}, options);
    next_states_ = torch::zeros({buffer_size_, state_dim_}, options);
    dones_ = torch::zeros({buffer_size_, 1}, options);

    indices_ = torch::zeros({batch_size_},
                            torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(device_));
}

void ReplayBuffer::add(const tensor_t& state, const tensor_t& action,
                      const tensor_t& reward, const tensor_t& next_state,
                      const tensor_t& done) {

    states_[position_].copy_(state);
    actions_[position_].copy_(action);
    rewards_[position_].copy_(reward);
    next_states_[position_].copy_(next_state);
    dones_[position_].copy_(done);

    position_ = (position_ + 1) % buffer_size_;
    current_size_ = std::min(current_size_ + 1, buffer_size_);
}

std::tuple<tensor_t, tensor_t, tensor_t, tensor_t, tensor_t> ReplayBuffer::sample() {
    indices_.random_(0, current_size_);

    return std::make_tuple(
        states_.index_select(0, indices_),
        actions_.index_select(0, indices_),
        rewards_.index_select(0, indices_),
        next_states_.index_select(0, indices_),
        dones_.index_select(0, indices_)
    );
}

SAC::SAC(dim_type state_dim, dim_type action_dim,
        tensor_t min_action,
        tensor_t max_action,
        ReplayBuffer* memory,
        torch::Device device)
    : actor_("actor", state_dim, action_dim, min_action, max_action, device)
    , critic1_("critic1", state_dim, action_dim, device)
    , critic2_("critic2", state_dim, action_dim, device)
    , critic1_target_("critic1_target", state_dim, action_dim, device)
    , critic2_target_("critic2_target", state_dim, action_dim, device)
    , actor_optimizer_(actor_->parameters(), constants::NETWORK::LEARNING_RATE)
    , critic1_optimizer_(critic1_->parameters(), constants::NETWORK::LEARNING_RATE)
    , critic2_optimizer_(critic2_->parameters(), constants::NETWORK::LEARNING_RATE)
    , memory_(memory)
    , device_(device)
    , state_dim_(state_dim)
    , action_dim_(action_dim)
    , min_action_(min_action)
    , max_action_(max_action)
    , gamma_(constants::NETWORK::GAMMA)
    , tau_(constants::NETWORK::TAU)
    , alpha_(constants::NETWORK::ALPHA) {

    if (!memory_) {
        throw std::runtime_error("Memory buffer pointer is null");
    }

    critic1_target_->load_state_dict(critic1_->state_dict());
    critic2_target_->load_state_dict(critic2_->state_dict());
    std::cout << "Successfully created SAC network" << std::endl;
}

tensor_t SAC::select_action(const tensor_t& state) {
    torch::NoGradGuard no_grad;
    auto batch_state = state.unsqueeze(0);
    auto [action, _] = actor_->sample(batch_state);
    return action.squeeze(0);
}

void SAC::update() {
    if (memory_->size() < memory_->batch_size()) {
        return;
    }

    auto [states, actions, rewards, next_states, dones] = memory_->sample();

    tensor_t target_q;
    {
        torch::NoGradGuard no_grad;
        auto [next_actions, next_log_pi] = actor_->sample(next_states);
        target_q = torch::min(
            critic1_target_->forward(next_states, next_actions),
            critic2_target_->forward(next_states, next_actions)
        );
        target_q = rewards + (1 - dones) * gamma_ * (target_q - alpha_ * next_log_pi);
    }

    auto current_q1 = critic1_->forward(states, actions);
    auto current_q2 = critic2_->forward(states, actions);

    auto critic_loss1 = torch::mse_loss(current_q1, target_q.detach());
    auto critic_loss2 = torch::mse_loss(current_q2, target_q.detach());

    critic1_optimizer_.zero_grad();
    critic_loss1.backward();
    critic1_optimizer_.step();

    critic2_optimizer_.zero_grad();
    critic_loss2.backward();
    critic2_optimizer_.step();

    auto [new_actions, log_pi] = actor_->sample(states);
    auto q = torch::min(
        critic1_->forward(states, new_actions),
        critic2_->forward(states, new_actions)
    );

    auto actor_loss = (alpha_ * log_pi - q).mean();

    actor_optimizer_.zero_grad();
    actor_loss.backward();
    actor_optimizer_.step();

    update_target_networks();
}

void SAC::update_target_networks() {
    torch::NoGradGuard no_grad;

    auto params1 = critic1_->parameters();
    auto target_params1 = critic1_target_->parameters();
    auto params2 = critic2_->parameters();
    auto target_params2 = critic2_target_->parameters();

    for (size_t i = 0; i < params1.size(); ++i) {
        auto& target1 = target_params1[i];
        const auto& source1 = params1[i];
        target1.mul_(1 - tau_).add_(source1, tau_);

        auto& target2 = target_params2[i];
        const auto& source2 = params2[i];
        target2.mul_(1 - tau_).add_(source2, tau_);
    }
}
