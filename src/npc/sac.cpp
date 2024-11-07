#include "npc/sac.hpp"
#include "utils/types.hpp"
#include <iostream>

ReplayBuffer::ReplayBuffer(dim_type state_dim, dim_type action_dim,
                         count_type buffer_size, count_type batch_size,
                         torch::Device device)
    : state_dim_(state_dim)
    , action_dim_(action_dim)
    , buffer_size_(buffer_size)
    , batch_size_(batch_size)
    , current_size_(0)
    , position_(0)
    , device_(device) {

    preallocate_tensors();
    warmup();
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

void ReplayBuffer::warmup() {
	indices_.random_(0, 1);  // random_ 커널 초기화
	states_.index_select(0, indices_);  // index_select 커널 초기화
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

PrioritizedReplayBuffer::PrioritizedReplayBuffer(
    dim_type state_dim, dim_type action_dim,
    count_type buffer_size, count_type batch_size,
    torch::Device device,
    real_t alpha, real_t beta)
    : ReplayBuffer(state_dim, action_dim, buffer_size, batch_size, device)
    , alpha_(alpha)
    , beta_(beta)
    , max_priority_(1.0f) {

    initialize_priorities();
    warmup();
}

void PrioritizedReplayBuffer::initialize_priorities() {
    auto options = torch::TensorOptions()
                      .dtype(types::get_tensor_dtype())
                      .device(device());

    priorities_ = torch::ones({buffer_size()}, options);
    weights_ = torch::zeros({batch_size()}, options);
    probs_ = torch::ones({buffer_size()}, options);
    probs_ /= buffer_size();
}

void PrioritizedReplayBuffer::warmup() {
    // power 연산 웜업
    {
        torch::NoGradGuard no_grad;
        auto dummy = torch::ones({1}, device());
        dummy.pow_(alpha_);
    }

    // multinomial 샘플링 웜업
    {
        torch::NoGradGuard no_grad;
        auto dummy_probs = torch::ones({2}, device());
        dummy_probs /= dummy_probs.sum();
        torch::multinomial(dummy_probs, 1, /*replacement=*/true);
    }

    // index_copy_ 연산 웜업
    {
        torch::NoGradGuard no_grad;
        auto dummy_indices = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(device()));
        auto dummy_values = torch::ones({1}, device());
        priorities_.index_copy_(0, dummy_indices, dummy_values);
    }

    // div_ 연산 웜업
    {
        torch::NoGradGuard no_grad;
        auto dummy = torch::ones({1}, device());
        auto dummy_divisor = torch::ones({1}, device());
        dummy.div_(dummy_divisor);
    }

    // copy_ 연산 웜업
    {
        torch::NoGradGuard no_grad;
        auto dummy_src = torch::ones({1}, device());
        auto dummy_dst = torch::zeros({1}, device());
        dummy_dst.copy_(dummy_src);
    }
}

void PrioritizedReplayBuffer::add(
    const tensor_t& state, const tensor_t& action,
    const tensor_t& reward, const tensor_t& next_state,
    const tensor_t& done) {

    ReplayBuffer::add(state, action, reward, next_state, done);

    // 현재 위치의 우선순위를 최대값으로 설정
    // position은 부모 클래스의 add에서 이미 업데이트되었으므로, size()-1로 계산
    auto position = (size() - 1) % buffer_size();
    priorities_[position] = max_priority_;
}

void PrioritizedReplayBuffer::update_sampling_probabilities(count_type valid_size) {
    // valid_size까지의 우선순위 슬라이스에 대해서만 계산
    auto priorities_slice = priorities_.slice(0, 0, valid_size);
    auto probs_slice = probs_.slice(0, 0, valid_size);

    // 해당 부분의 우선순위만 업데이트
    probs_slice.copy_(priorities_slice.pow(alpha_));

    // 정규화 (valid_size 범위 내에서만)
    auto sum_probs = probs_slice.sum();
    probs_slice.div_(sum_probs);
}

void PrioritizedReplayBuffer::update_is_weights(const tensor_t& probs, const tensor_t& indices, count_type valid_size) {
    // P(i) 계산
    auto sampled_probs = probs.index_select(0, indices);

    // (1/N * 1/P(i))^β 계산하여 weights_에 직접 저장
    weights_.copy_((valid_size * sampled_probs).pow(-beta_));

    // weights 정규화
    auto max_weight = weights_.max();
    weights_.div_(max_weight);
}

std::tuple<tensor_t, tensor_t, tensor_t, tensor_t, tensor_t, tensor_t, tensor_t>PrioritizedReplayBuffer::sample_with_priorities() {
    auto valid_size = size();

    // 샘플링 확률 업데이트
    update_sampling_probabilities(valid_size);

    // 우선순위에 따른 샘플링
    indices_ = torch::multinomial(probs_, batch_size(), /*replacement=*/true);

    // IS weights 계산
    update_is_weights(probs_, indices_, valid_size);

    // 상태, 행동, 보상 등 샘플링
    return std::make_tuple(
        states_.index_select(0, indices_),
        actions_.index_select(0, indices_),
        rewards_.index_select(0, indices_),
        next_states_.index_select(0, indices_),
        dones_.index_select(0, indices_),
        weights_,
        indices_
    );
}

void PrioritizedReplayBuffer::update_priorities(const tensor_t& indices, const tensor_t& td_errors) {
    // 새로운 우선순위 계산 (TD 에러의 절대값 + 작은 상수)
    auto new_priorities = (td_errors.abs() + eplsion_);

    // 최대 우선순위 업데이트
    max_priority_ = std::max(max_priority_, new_priorities.max().item<real_t>());

    // 우선순위 업데이트
    priorities_.index_copy_(0, indices, new_priorities);
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

    warmup();

    critic1_target_->load_state_dict(critic1_->state_dict());
    critic2_target_->load_state_dict(critic2_->state_dict());
    std::cout << "Successfully created SAC network" << std::endl;
}


void SAC::warmup() {
	// 더미 데이터 생성
	auto batch_size = memory_->batch_size();
	auto dummy_states = torch::zeros({ batch_size, state_dim_ }, device_);
	auto dummy_actions = torch::zeros({ batch_size, action_dim_ }, device_);
	auto dummy_rewards = torch::zeros({ batch_size, 1 }, device_);
	auto dummy_next_states = torch::zeros({ batch_size, state_dim_ }, device_);
	auto dummy_dones = torch::zeros({ batch_size, 1 }, device_);

	// Critic networks warmup
	{
		torch::autograd::GradMode::set_enabled(true);  // 그래디언트 연산 활성화

		// Critic forward & backward
		auto current_q1 = critic1_->forward(dummy_states, dummy_actions);
		auto current_q2 = critic2_->forward(dummy_states, dummy_actions);

		auto dummy_target = torch::zeros_like(current_q1);
		auto critic_loss1 = torch::mse_loss(current_q1, dummy_target);
		auto critic_loss2 = torch::mse_loss(current_q2, dummy_target);

		critic1_optimizer_.zero_grad();
		critic_loss1.backward();
		critic1_optimizer_.zero_grad();  // 실제 파라미터 업데이트는 하지 않음

		critic2_optimizer_.zero_grad();
		critic_loss2.backward();
		critic2_optimizer_.zero_grad();
	}

	// Actor network warmup
	{
		torch::autograd::GradMode::set_enabled(true);

		// Actor forward & backward
		auto [actions, log_pi] = actor_->sample(dummy_states);
		auto q = torch::min(
			critic1_->forward(dummy_states, actions),
			critic2_->forward(dummy_states, actions)
		);
		auto actor_loss = (alpha_ * log_pi - q).mean();

		actor_optimizer_.zero_grad();
		actor_loss.backward();
		actor_optimizer_.zero_grad();
	}

	// Target networks warmup
    {
        torch::NoGradGuard no_grad;

		auto [next_actions, next_log_pi] = actor_->sample(dummy_next_states);
		auto target_q = torch::min(
			critic1_target_->forward(dummy_next_states, next_actions),
			critic2_target_->forward(dummy_next_states, next_actions)
		);
    }

	// unsqueeze/squeeze 연산 웜업
	{
        torch::NoGradGuard no_grad;

		auto single_state = torch::zeros({ state_dim_ }, device_);
		auto batch_state = single_state.unsqueeze(0);
		auto [action, _] = actor_->sample(batch_state);
		action.squeeze(0);
	}

}

tensor_t SAC::select_action(const tensor_t& state) {
    torch::NoGradGuard no_grad;
    auto batch_state = state.unsqueeze(0);
    auto [action, _] = actor_->sample(batch_state);
    return action.squeeze(0);
}

SACMetrics SAC::update(bool debug) {
    SACMetrics metrics{};
    if (memory_->size() < memory_->batch_size()) {
        return metrics;
    }

	std::chrono::duration<double, std::milli> memory_sample_time;
	std::chrono::duration<double, std::milli> actor_sample_time;
	std::chrono::duration<double, std::milli> critic_forward_time;
	std::chrono::duration<double, std::milli> critic_backward_time;
	std::chrono::duration<double, std::milli> actor_forward_time;
	std::chrono::duration<double, std::milli> actor_backward_time;
	std::chrono::duration<double, std::milli> target_update_time;

    auto start = std::chrono::high_resolution_clock::now();
    auto [states, actions, rewards, next_states, dones] = memory_->sample();
    auto end = std::chrono::high_resolution_clock::now();
    memory_sample_time = end - start;

    tensor_t target_q;
    {
        torch::NoGradGuard no_grad;

        start = std::chrono::high_resolution_clock::now();
        auto [next_actions, next_log_pi] = actor_->sample(next_states);
		end = std::chrono::high_resolution_clock::now();
		actor_sample_time = end - start;

        target_q = torch::min(
            critic1_target_->forward(next_states, next_actions),
            critic2_target_->forward(next_states, next_actions)
        );
        target_q = rewards + (1 - dones) * gamma_ * (target_q - alpha_ * next_log_pi);
    }

    start = std::chrono::high_resolution_clock::now();
    auto current_q1 = critic1_->forward(states, actions);
    end = std::chrono::high_resolution_clock::now();
    critic_forward_time = end - start;

    auto current_q2 = critic2_->forward(states, actions);

    auto critic_loss1 = torch::mse_loss(current_q1, target_q.detach());
    auto critic_loss2 = torch::mse_loss(current_q2, target_q.detach());
    metrics.critic_loss1 = critic_loss1.item<real_t>();
    metrics.critic_loss2 = critic_loss2.item<real_t>();

    start = std::chrono::high_resolution_clock::now();
    critic1_optimizer_.zero_grad();
    critic_loss1.backward();
    critic1_optimizer_.step();
    end = std::chrono::high_resolution_clock::now();
    critic_backward_time = end - start;

    critic2_optimizer_.zero_grad();
    critic_loss2.backward();
    critic2_optimizer_.step();

    start = std::chrono::high_resolution_clock::now();
    auto [new_actions, log_pi] = actor_->sample(states);
    end = std::chrono::high_resolution_clock::now();
    actor_forward_time = end - start;
    auto q = torch::min(
        critic1_->forward(states, new_actions),
        critic2_->forward(states, new_actions)
    );

    auto actor_loss = (alpha_ * log_pi - q).mean();
    metrics.actor_loss = actor_loss.item<real_t>();
    metrics.log_pi = log_pi.mean().item<real_t>();
    metrics.q_value = q.mean().item<real_t>();

    start = std::chrono::high_resolution_clock::now();
    actor_optimizer_.zero_grad();
    actor_loss.backward();
    actor_optimizer_.step();
	end = std::chrono::high_resolution_clock::now();
	actor_backward_time = end - start;

    start = std::chrono::high_resolution_clock::now();
    update_target_networks();
	end = std::chrono::high_resolution_clock::now();
	target_update_time = end - start;

    if (debug) {
	    std::cout << "\nSAC Update Timing (ms):" << std::endl;
	    std::cout << std::fixed << std::setprecision(4);
	    std::cout << "Memory Sampling: " << memory_sample_time.count() << std::endl;
	    std::cout << "Actor Sampling: " << actor_sample_time.count() << std::endl;
	    std::cout << "Critic Forward: " << critic_forward_time.count() << std::endl;
	    std::cout << "Critic Backward: " << critic_backward_time.count() << std::endl;
	    std::cout << "Actor Forward: " << actor_forward_time.count() << std::endl;
	    std::cout << "Actor Backward: " << actor_backward_time.count() << std::endl;
	    std::cout << "Target Update: " << target_update_time.count() << std::endl;

	    double total_time = memory_sample_time.count() + actor_sample_time.count() +
		    critic_forward_time.count() + critic_backward_time.count() +
		    actor_forward_time.count() + actor_backward_time.count() +
		    target_update_time.count();
	    std::cout << "Total Update Time: " << total_time << std::endl;
    }

    metrics.is_vaild = true;

    return metrics;
}

tensor_t SAC::get_critic_target_values(const tensor_t& state, const tensor_t& action) {
    torch::NoGradGuard no_grad;

    auto batch_state = state.unsqueeze(0);
    auto batch_action = action.unsqueeze(0);

    auto q = torch::min(
        critic1_target_->forward(batch_state, batch_action),
        critic2_target_->forward(batch_state, batch_action)
    );

    return q.squeeze(0);
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
