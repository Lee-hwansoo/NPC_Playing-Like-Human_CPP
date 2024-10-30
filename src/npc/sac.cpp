#include "npc/sac.hpp"
#include <iostream>

ReplayBuffer::ReplayBuffer(size_type buffer_size, size_type batch_size, torch::Device device)
    : buffer_size_(buffer_size)
    , batch_size_(batch_size)
    , device_(device)
    , generator_(std::random_device{}()) {}

void ReplayBuffer::add(const tensor_t& state, const tensor_t& action, const tensor_t& reward, const tensor_t& next_state, const tensor_t& done) {
    if (buffer_.size() >= buffer_size_) {
        buffer_.pop_front();
    }
    buffer_.emplace_back(
        state.cpu(),
        action.cpu(),
        reward.cpu(),
        next_state.cpu(),
        done.cpu()
    );
}

std::tuple<tensor_t, tensor_t, tensor_t, tensor_t, tensor_t> ReplayBuffer::sample() {
    std::vector<size_type> indices(batch_size_);
    std::vector<tensor_t> states, actions, rewards, next_states, dones;
    states.reserve(batch_size_);
    actions.reserve(batch_size_);
    rewards.reserve(batch_size_);
    next_states.reserve(batch_size_);
    dones.reserve(batch_size_);

    for (size_type i = 0; i < batch_size_; ++i) {
        std::uniform_int_distribution<size_type> dist(0, buffer_.size() - 1);
        indices[i] = dist(generator_);

        const auto& [s, a, r, ns, d] = buffer_[indices[i]];
        states.push_back(s);
        actions.push_back(a);
        rewards.push_back(r.reshape({ 1 }));
        next_states.push_back(ns);
        dones.push_back(d.reshape({ 1 }));
    }

    return {
        torch::stack(states),
        torch::stack(actions),
        torch::stack(rewards),
        torch::stack(next_states),
        torch::stack(dones)
    };
}

SAC::SAC(dim_type state_dim, dim_type action_dim,
        tensor_t min_action,
        tensor_t max_action,
        torch::Device device)
    : actor_("actor", state_dim, action_dim, min_action, max_action, device)
    , critic1_("critic1", state_dim, action_dim, device)
    , critic2_("critic2", state_dim, action_dim, device)
    , critic1_target_("critic1_target", state_dim, action_dim, device)
    , critic2_target_("critic2_target", state_dim, action_dim, device)
    , actor_optimizer_(actor_->parameters(), constants::NETWORK::LEARNING_RATE)
    , critic1_optimizer_(critic1_->parameters(), constants::NETWORK::LEARNING_RATE)
    , critic2_optimizer_(critic2_->parameters(), constants::NETWORK::LEARNING_RATE)
    , memory_(std::make_unique<ReplayBuffer>(constants::NETWORK::BUFFER_SIZE, constants::NETWORK::BATCH_SIZE, device))
    , device_(device)
    , state_dim_(state_dim)
    , action_dim_(action_dim)
    , min_action_(min_action)
    , max_action_(max_action)
    , gamma_(constants::NETWORK::GAMMA)
    , tau_(constants::NETWORK::TAU)
    , alpha_(constants::NETWORK::ALPHA) {

    critic1_target_->load_state_dict(critic1_->state_dict());
    critic2_target_->load_state_dict(critic2_->state_dict());
    std::cout << "Successfully created SAC network" << std::endl;
}

tensor_t SAC::select_action(const tensor_t& state) {
    torch::NoGradGuard no_grad;
    auto batch_state = state.unsqueeze(0);
    auto [action, _] = actor_->sample(batch_state);
    return action.cpu().squeeze(0);
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
        auto target_q1 = critic1_target_->forward(next_states, next_actions);
        auto target_q2 = critic2_target_->forward(next_states, next_actions);
        target_q = torch::min(target_q1, target_q2) - alpha_ * next_log_pi;
        target_q = rewards + (1 - dones) * gamma_ * target_q;
    }

    auto current_q1 = critic1_->forward(states, actions);
    auto current_q2 = critic2_->forward(states, actions);

    auto critic_loss1 = torch::mse_loss(current_q1, target_q);
    auto critic_loss2 = torch::mse_loss(current_q2, target_q);

    critic1_optimizer_.zero_grad();
    critic_loss1.backward();
    critic1_optimizer_.step();

    critic2_optimizer_.zero_grad();
    critic_loss2.backward();
    critic2_optimizer_.step();

    auto [new_actions, log_pi] = actor_->sample(states);
    auto q1 = critic1_->forward(states, new_actions);
    auto q2 = critic2_->forward(states, new_actions);
    auto q = torch::min(q1, q2);

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

    auto param_it1 = params1.begin();
    auto target_it1 = target_params1.begin();
    auto param_it2 = params2.begin();
    auto target_it2 = target_params2.begin();

    while (param_it1 != params1.end()) {
        target_it1->data().copy_(tau_ * param_it1->data() + (1 - tau_) * target_it1->data());
        ++param_it1;
        ++target_it1;
    }

    while (param_it2 != params2.end()) {
        target_it2->data().copy_(tau_ * param_it2->data() + (1 - tau_) * target_it2->data());
        ++param_it2;
        ++target_it2;
    }
}
