#include "npc/environment.hpp"
#include "npc/sac.hpp"
#include "utils/constants.hpp"
#include "utils/types.hpp"
#include <iostream>

namespace environment {

TrainEnvironment::TrainEnvironment(count_type width, count_type height, torch::Device device, count_type agent_count, bool init)
	: BaseEnvironment(width, height, device)
	, agent_count_(agent_count) {
	set_observation_dim(constants::Agent::FOV::RAY_COUNT + 9);
	set_action_dim(2);

	std::cout << "Environment initialized, device: " << device_ << std::endl;

	his_dir_ = get_history_directory();

	path_planner_ = std::make_unique<path_planning::RRT>();

	if (init) {
		const auto [min_action, max_action] = get_action_space();
		std::cout << "min_action: " << min_action << ", max_action: " << max_action << std::endl;

		circle_obstacles_num_ = constants::CircleObstacle::COUNT;
		rectangle_obstacles_num_ = constants::RectangleObstacle::COUNT;
		circle_obstacles_spawn_bounds_ = constants::CircleObstacle::SPAWN_BOUNDS;
		rectangle_obstacles_spawn_bounds_ = constants::RectangleObstacle::SPAWN_BOUNDS;
		goal_spawn_bounds_ = constants::Goal::SPAWN_BOUNDS;
		agents_spawn_bounds_ = constants::Agent::SPAWN_BOUNDS;
		agents_move_bounds_ = constants::Agent::MOVE_BOUNDS;

		state_ = this->init(agent_count, min_action, max_action);

		init_n_step_buffer();

		std::cout << "Finished Environment initialized" << std::endl;
	}
}

tensor_t TrainEnvironment::init(count_type agent_count, tensor_t min_action, tensor_t max_action) {
	circle_obstacles_.reserve(circle_obstacles_num_);
	for (count_type i = 0; i < circle_obstacles_num_; ++i) {
		auto obs = std::make_unique<object::CircleObstacle>(i,
			std::nullopt, std::nullopt,
			constants::CircleObstacle::RADIUS,
			circle_obstacles_spawn_bounds_,
			Display::to_sdl_color(Display::ORANGE),
			true);
		circle_obstacles_.push_back(std::move(obs));
	}

	rectangle_obstacles_.reserve(rectangle_obstacles_num_);
	for (count_type i = 0; i < rectangle_obstacles_num_; ++i) {
		auto obs = std::make_unique<object::RectangleObstacle>(i,
			std::nullopt, std::nullopt,
			std::nullopt, std::nullopt, std::nullopt,
			rectangle_obstacles_spawn_bounds_,
			Display::to_sdl_color(Display::ORANGE),
			false);
		rectangle_obstacles_.push_back(std::move(obs));
	}

	circle_obstacles_state_ = torch::zeros({ circle_obstacles_num_, 3 });
	rectangle_obstacles_state_ = torch::zeros({ rectangle_obstacles_num_, 5 });
	update_circle_obstacles_state();
	update_rectangle_obstacles_state();

	goals_.reserve(agent_count);
	agents_.reserve(agent_count);
	for (count_type i = 0; i < agent_count; ++i){
		auto goal = std::make_unique<object::Goal>(i,
			std::nullopt, std::nullopt,
			constants::Goal::RADIUS,
			goal_spawn_bounds_,
			Display::to_sdl_color(Display::GREEN),
			false);
		goals_.push_back(std::move(goal));

		auto agent = std::make_unique<object::Agent>(i,
			std::nullopt, std::nullopt,
			constants::Agent::RADIUS,
			agents_spawn_bounds_,
			agents_move_bounds_,
			Display::to_sdl_color(Display::BLUE),
			true,
			circle_obstacles_state_,
			rectangle_obstacles_state_,
			goals_[i]->get_state(),
			path_planner_.get());
		agents_.push_back(std::move(agent));
	}

	agents_state_ = torch::zeros({ agent_count, 3 });
	update_agents_state();

	// memory_ = std::make_unique<ReplayBuffer>(
	// 	get_observation_dim(),
	// 	get_action_dim(),
	// 	constants::NETWORK::BUFFER_SIZE,
	// 	constants::NETWORK::BATCH_SIZE,
	// 	device_
	// );

	memory_ = std::make_unique<PrioritizedReplayBuffer>(
		get_observation_dim(),
		get_action_dim(),
		constants::NETWORK::BUFFER_SIZE,
		constants::NETWORK::BATCH_SIZE,
		device_,
		0.6f,
		0.4f
	);

	sac_ = std::make_unique<SAC>(
		get_observation_dim(),
		get_action_dim(),
		min_action,
		max_action,
		memory_.get(),
		device_
	);

	return get_observation();
}

tensor_t TrainEnvironment::reset() {
	for (auto& obs : circle_obstacles_) {
		obs->reset();
	}
	for (auto& obs : rectangle_obstacles_) {
		obs->reset();
	}

	update_circle_obstacles_state();
	update_rectangle_obstacles_state();

    for (size_t i = 0; i < agent_count_; ++i) {
        goals_[i]->reset();
        agents_[i]->reset(std::nullopt, std::nullopt,
            circle_obstacles_state_,
            rectangle_obstacles_state_,
            goals_[i]->get_state());
    }

	update_agents_state();

	step_count_ = 0;
	terminated_ = false;
	truncated_ = false;

	return get_observation();
}

void TrainEnvironment::save(dim_type episode, bool print = true) {
	if (sac_) {
		sac_->save_network_parameters(episode, print);
	}
}

void TrainEnvironment::load(const std::string& timestamp, dim_type episode) {
	if (sac_) {
		sac_->load_network_parameters(timestamp, episode);
		start_episode_ = episode;
	}
}

void TrainEnvironment::update_circle_obstacles_state() {
	for (size_t i = 0; i < circle_obstacles_.size(); ++i) {
		circle_obstacles_state_[i] = circle_obstacles_[i]->get_state();
	}
}

void TrainEnvironment::update_rectangle_obstacles_state() {
	for (size_t i = 0; i < rectangle_obstacles_.size(); ++i) {
		rectangle_obstacles_state_[i] = rectangle_obstacles_[i]->get_state();
	}
}

void TrainEnvironment::update_agents_state() {
	for (size_t i = 0; i < agents_.size(); ++i) {
		agents_state_[i] = agents_[i]->get_raw_state();
	}
}

tensor_t TrainEnvironment::get_observation() const {
	return agents_[0]->get_state();
}

real_t TrainEnvironment::calculate_reward(const tensor_t& state, const tensor_t& action, bool debug) {
	// 중단 보상 (충돌, 경계, 최대 스텝)
	if (truncated_) {
		return -1.0f;
	}

	// 종료 보상
	if (terminated_) {
		// 빠른 도달에 대한 보너스 보상
		real_t speed_bonus = (1.0f - static_cast<real_t>(step_count_) / constants::NETWORK::MAX_STEP);
		return 2.0f + (speed_bonus);
	}

	auto state_size = state.size(0);

	auto required_state = state.slice(0, state_size-3, state_size).to(torch::kCPU);
    auto required_action = action.to(torch::kCPU);

	real_t angle_diff_cos = required_state[0].item<real_t>();
	real_t normalized_alignment = (angle_diff_cos + 1.0f) * 0.5f;
    real_t normalized_goal_dist = required_state[1].item<real_t>();
    real_t normalized_frenet_d = required_state[2].item<real_t>();

    real_t force = required_action[0].item<real_t>();
    real_t yaw_change = required_action[1].item<real_t>();

	// 보상 컴포넌트들
	real_t dist_factor = 0.7f;
	real_t path_factor = 0.3f;

	real_t dist_reward = 0.0f;
	if (normalized_goal_dist > 0.2f) {
		real_t progress = 1.0f - normalized_goal_dist;
		dist_reward = (0.5f * (progress / 0.8f)) * dist_factor;
	}
	else if (normalized_goal_dist > 0.1f) {
		real_t progress = (0.2f - normalized_goal_dist) / 0.1f;
		dist_reward = (0.5f + 0.1f * progress) * dist_factor;
	}
	else if (normalized_goal_dist > 0.05f) {
		real_t progress = (0.1f - normalized_goal_dist) / 0.05f;
		dist_reward = (0.6f + 0.1f * progress) * dist_factor;
	}
	else if (normalized_goal_dist > 0.025f) {
		real_t progress = (0.05f - normalized_goal_dist) / 0.025f;
		dist_reward = (0.7f + 0.1f * progress) * dist_factor;
	}
	else {
		real_t progress = 1.0f - (normalized_goal_dist / 0.025f);
		dist_reward = (0.8f + 0.2f * progress) * dist_factor;
	}
	real_t path_reward = std::exp(-std::abs(normalized_frenet_d) * (25.0f)) * path_factor;

	real_t reward = dist_reward +
			path_reward;

	if (debug){
		std::cout <<"\ndist: " << std::fixed << std::setprecision(5) << normalized_goal_dist
			<< ", dist_reward: " << dist_reward
			<< ", path_reward: " << path_reward
			<< std::endl;
	}

	// 기본 보상 컴포넌트들 (0.0 ~ 1.0 범위로 조정)
	return std::clamp(reward, 0.0f, 1.0f);
}

std::tuple<tensor_t, tensor_t, bool, bool> TrainEnvironment::step(const tensor_t& state, const tensor_t& action, bool debug) {
	tensor_t reward = torch::tensor(calculate_reward(state, action, debug));
	terminated_ = check_goal();
	truncated_ = check_bounds() || check_obstacle_collision() || step_count_ >= constants::NETWORK::MAX_STEP;

	if (terminated_ || truncated_) {
		return std::make_tuple(state, reward, terminated_, truncated_);
	}

	step_count_++;

	for (auto& obstacle : circle_obstacles_) {
		obstacle->update(fixed_dt_);
	}

	update_circle_obstacles_state();

	agents_[0]->update(fixed_dt_, action, circle_obstacles_state_);

	update_agents_state();

	return std::make_tuple(get_observation(), reward, terminated_, truncated_);
}

TrainingResult TrainEnvironment::train(const dim_type episodes, bool render, bool debug, bool print) {
	sac_->train();
	SDL_Event event;
	bool is_render = render;
	bool is_debug = debug;
	bool is_reward = false;
	bool is_paused = false;
    std::vector<SACResult> result_history;
	std::vector<SACMetrics> metrics_history;
    result_history.reserve(static_cast<size_t>(episodes));
	metrics_history.reserve(static_cast<size_t>(episodes * constants::NETWORK::MAX_STEP / constants::NETWORK::UPDATE_INTERVAL));

	const real_t beta_start = 0.4f;
	const real_t beta_end = 1.0f;
	const real_t decay_rate = 5.0f;
	const dim_type beta_anneal_episodes = episodes;

    // 디버그 모드용 키보드 입력 변수들
    const real_t FIXED_FORCE = 0.5f;
    const real_t FIXED_YAW = 0.5f;
    real_t force = 0.0f;
    real_t yaw_change = 0.0f;

    for (dim_type episode = start_episode_; episode < start_episode_+ episodes; ++episode) {
		real_t progress = static_cast<real_t>(episode - start_episode_) / beta_anneal_episodes;
		real_t beta = beta_end - (beta_end - beta_start) * std::exp(-decay_rate * progress);
		beta = std::min(beta_end, std::max(beta_start, beta));
		sac_->set_beta(beta);

		tensor_t state = reset();
		real_t episode_return = 0.0f;
		bool is_arrived = false;
		bool done = false;

		while (!done) {
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
					return { result_history, metrics_history };
				}
                if (event.type == SDL_KEYDOWN) {
                    switch (event.key.keysym.sym) {
                        case SDLK_r:
                            is_render = !is_render;
                            if (!is_render) {
                                SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
                                SDL_RenderClear(renderer_);
                                SDL_RenderPresent(renderer_);
                            }
                            break;
                        case SDLK_SPACE:
                            is_paused = !is_paused;
                            break;
                        case SDLK_d:
                            is_debug = !is_debug;
							force = 0.0f;
                            yaw_change = 0.0f;
                            std::cout << "\nDebug mode: " << (is_debug ? "ON" : "OFF") << std::endl;
                            break;
                        case SDLK_c:
                            is_reward = !is_reward;
							std::cout << "\nPrint Reward: " << (is_reward ? "ON" : "OFF") << std::endl;
                            break;
                    }
                    if (is_debug) {
                        switch (event.key.keysym.sym) {
                            case SDLK_UP:
                                force = FIXED_FORCE;
                                break;
                            case SDLK_DOWN:
                                force = -FIXED_FORCE;
                                break;
                            case SDLK_LEFT:
                                yaw_change = -FIXED_YAW;
                                break;
                            case SDLK_RIGHT:
                                yaw_change = FIXED_YAW;
                                break;
                        }
                    }
                } else if (event.type == SDL_KEYUP && is_debug) {
                    switch (event.key.keysym.sym) {
                        case SDLK_UP:
                        case SDLK_DOWN:
                            force = 0.0f;
                            break;
                        case SDLK_LEFT:
                        case SDLK_RIGHT:
                            yaw_change = 0.0f;
                            break;
                    }
                }
			}

            if (is_paused) {
                if (is_render) {
                    render_scene();
                }
                SDL_Delay(20);
                continue;
            }else if (is_render) {
				render_scene();
			}

			tensor_t action;
			if (is_debug) {
				action = torch::tensor({force, yaw_change}, torch::TensorOptions().device(device_).dtype(get_tensor_dtype()));
			}else {
				action = sac_->select_action(state);
			}

			auto [next_state, reward, terminated, truncated] = step(state, action, is_reward);
			done = terminated || truncated;
			tensor_t done_tensor = torch::tensor(done, get_tensor_dtype());

			if(is_debug){
				state = next_state;
			}

			if (!is_debug){
				index_type stored_idx = store_transition(state, action, reward, done_tensor);

				if (step_count_ >= n_steps_) {
					if (done) {
						// 중단된 경우 모든 이전 상태들에 대해 처리
						for (index_type i = 0; i < n_steps_ - 1; ++i) {
							index_type start_idx = (buffer_idx_ + i) % n_steps_;
							process_n_step_return(start_idx, n_steps_ - i, next_state, done_tensor);
						}
					} else {
						// 일반적인 경우 현재 상태만 처리
						process_n_step_return(buffer_idx_, n_steps_, next_state, done_tensor);
					}
				}

				if (step_count_ % constants::NETWORK::UPDATE_INTERVAL == 0) {
					auto metrics = sac_->update(print);
					if (metrics.is_vaild){
						metrics.beta = beta;
						metrics_history.push_back(metrics);
					}
				}

				episode_return += reward.item<real_t>();
				is_arrived = terminated;

				state = next_state;

				std::cout << "\rEpisode: " << episode + 1 << "/" << start_episode_+ episodes << " | Step: " << step_count_ << std::string(20, ' ') << std::flush;
			}
		}

		if (!is_debug){
			result_history.push_back({episode_return, is_arrived});
			if ((episode + 1) % constants::NETWORK::LOG_INTERVAL == 0) {
				log_statistics(result_history, episode);
				save_history(result_history, metrics_history);
				save(episode + 1, false);
			}
		}
    }

    return {result_history, metrics_history};
}

std::vector<SACResult> TrainEnvironment::test(const dim_type episodes, bool render) {
	sac_->eval();
	SDL_Event event;
	bool is_render = render;
	bool is_paused = false;
	std::vector<SACResult> result_history;
	std::vector<real_real_t> action_times;
	result_history.reserve(static_cast<size_t>(episodes));
	action_times.reserve(static_cast<size_t>(episodes) * 2000);

	for (dim_type episode = 0; episode < episodes; ++episode) {
		tensor_t state = reset();
		real_t episode_return = 0.0f;
		bool is_arrived = false;
		bool done = false;

		while (!done) {
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
					return result_history;
				}
				if (event.type == SDL_KEYDOWN){
					if(event.key.keysym.sym == SDLK_r){
						is_render = !is_render;
						if (!is_render) {
							SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
							SDL_RenderClear(renderer_);
							SDL_RenderPresent(renderer_);
						}
					}
					if(event.key.keysym.sym == SDLK_SPACE){
						is_paused = !is_paused;
					}
				}
			}

            if (is_paused) {
                if (is_render) {
                    render_scene();
                }
                SDL_Delay(20);
                continue;
            }else if (is_render) {
				render_scene();
			}

			auto start_time = std::chrono::high_resolution_clock::now();
			tensor_t action = sac_->select_action(state);
			auto end_time = std::chrono::high_resolution_clock::now();

			std::chrono::duration<real_real_t, std::milli> duration = end_time - start_time;
			action_times.push_back(duration.count());

			auto [next_state, reward, terminated, truncated] = step(state, action, false);

			done = terminated || truncated;

			episode_return += reward.item<real_t>();
			is_arrived = terminated;

			state = next_state;
		}

		std::cout << "Episode: " << episode + 1 << "/" <<  episodes << " | Reward: " << episode_return << " " << std::endl;

		result_history.push_back({episode_return, is_arrived});
	}

	double min_time = *std::min_element(action_times.begin(), action_times.end());
	double max_time = *std::max_element(action_times.begin(), action_times.end());
	double avg_time = std::accumulate(action_times.begin(), action_times.end(), 0.0) / action_times.size();

	double min_time_no_first = *std::min_element(action_times.begin() + 1, action_times.end());
	double max_time_no_first = *std::max_element(action_times.begin() + 1, action_times.end());
	double avg_time_no_first = std::accumulate(action_times.begin() + 1, action_times.end(), 0.0) /
		(action_times.size() - 1);

	std::cout << "\nAll Action Selection Times (ms):" << std::endl;
	std::cout << std::fixed << std::setprecision(4);
	for (size_t i = 0; i < action_times.size(); ++i) {
		std::cout << action_times[i];
		if (i < action_times.size() - 1) {
			std::cout << ", ";
		}
		if ((i + 1) % 20 == 0) {
			std::cout << std::endl;
		}
	}

	std::cout << "\nAction Selection Time Statistics (ms):" << std::endl;
	std::cout << "First action time: " << action_times[0] << std::endl;
	std::cout << "Including first action:" << std::endl;
	std::cout << "  Minimum: " << min_time << std::endl;
	std::cout << "  Maximum: " << max_time << std::endl;
	std::cout << "  Average: " << avg_time << std::endl;
	std::cout << "Excluding first action:" << std::endl;
	std::cout << "  Minimum: " << min_time_no_first << std::endl;
	std::cout << "  Maximum: " << max_time_no_first << std::endl;
	std::cout << "  Average: " << avg_time_no_first << std::endl;

	return result_history;
}

bool TrainEnvironment::check_goal() const { return agents_[0]->is_goal(); }
bool TrainEnvironment::check_bounds() const { return agents_[0]->is_out(); }
bool TrainEnvironment::check_obstacle_collision() const { return agents_[0]->is_collison(); }

void TrainEnvironment::render_scene() const {
    if (!renderer_) return;

    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderClear(renderer_);

    auto color = Display::to_sdl_color(Display::GREEN);
    SDL_SetRenderDrawColor(renderer_, color.r, color.g, color.b, color.a);
    SDL_RenderDrawLine(renderer_, 0, Section::GOAL_LINE, Display::WIDTH, Section::GOAL_LINE);
    SDL_RenderDrawLine(renderer_, 0, Section::START_LINE, Display::WIDTH, Section::START_LINE);

    for (const auto& obs : circle_obstacles_) obs->draw(renderer_);
    for (const auto& obs : rectangle_obstacles_) obs->draw(renderer_);

    for (size_t i = 0; i < agents_.size(); ++i) {
        goals_[i]->draw(renderer_);
        agents_[i]->draw(renderer_);
    }

    SDL_RenderPresent(renderer_);
}

void TrainEnvironment::reset_agent(size_t agent_idx) {
	goals_[agent_idx]->reset();
	agents_[agent_idx]->reset(std::nullopt, std::nullopt,
		circle_obstacles_state_,
		rectangle_obstacles_state_,
		goals_[agent_idx]->get_state());
}

tensor_t TrainEnvironment::get_agent_observation(size_t agent_idx) const {
    return agents_[agent_idx]->get_state();
}

tensor_t TrainEnvironment::get_combined_obstacles_for_agent(size_t agent_idx) const {
    if (agent_count_ == 1) {
        return circle_obstacles_state_;
    }

    tensor_t other_agents = torch::cat({
        agents_state_.slice(0, 0, agent_idx),
        agents_state_.slice(0, agent_idx + 1)
    });

    return torch::cat({circle_obstacles_state_, other_agents});
}

bool TrainEnvironment::check_agent_goal(size_t agent_idx) const { return agents_[agent_idx]->is_goal(); }
bool TrainEnvironment::check_agent_bounds(size_t agent_idx) const { return agents_[agent_idx]->is_out(); }
bool TrainEnvironment::check_agent_collision(size_t agent_idx) const { return agents_[agent_idx]->is_collison(); }

void TrainEnvironment::init_n_step_buffer() {
    // n_step_buffer의 각 행은 [state, action, reward, done] 형태
    dim_type row_dim = get_observation_dim() + get_action_dim() + 2;  // +2는 reward와 done을 위한 공간
    n_step_buffer_ = torch::zeros({n_steps_, row_dim}, torch::TensorOptions(types::get_tensor_dtype()).device(device_));
    buffer_idx_ = 0;
}

index_type TrainEnvironment::store_transition(const tensor_t& state, const tensor_t& action, const tensor_t& reward, const tensor_t& done) {
   index_type current_idx = buffer_idx_;  // 현재 인덱스 저장

    // 현재 transition을 버퍼에 저장
    dim_type state_dim = get_observation_dim();
    dim_type action_dim = get_action_dim();

    // 버퍼의 현재 위치에 transition 저장
    n_step_buffer_[buffer_idx_].slice(0, 0, state_dim).copy_(state);
    n_step_buffer_[buffer_idx_].slice(0, state_dim, state_dim + action_dim).copy_(action);
	n_step_buffer_[buffer_idx_][state_dim + action_dim].copy_(reward);
	n_step_buffer_[buffer_idx_][state_dim + action_dim + 1].copy_(done);

    // 버퍼 인덱스 업데이트
    buffer_idx_ = (buffer_idx_ + 1) % n_steps_;

	return current_idx;  // 저장된 위치 반환
}

tensor_t TrainEnvironment::calculate_n_step_return(const index_type start_idx, const index_type remaining_steps) {
    dim_type state_dim = get_observation_dim();
    dim_type action_dim = get_action_dim();

    // N-step 리턴 계산
    tensor_t n_step_return = torch::zeros({1}, torch::TensorOptions(types::get_tensor_dtype()).device(device_));
    real_t cumulative_discount = 1.0f;

    // 버퍼의 과거 데이터로부터 n-step return 계산
    for (index_type i = 0; i < remaining_steps; ++i) {
		index_type idx = (start_idx + i) % n_steps_;

        const tensor_t& reward_tensor = n_step_buffer_[idx][state_dim + action_dim];
        const tensor_t& done_tensor = n_step_buffer_[idx][state_dim + action_dim + 1];
		n_step_return += cumulative_discount * reward_tensor;

        if (done_tensor.item<real_t>() == 1.0f) {
            return n_step_return;
        }
        cumulative_discount *= gamma_;
    }

    return n_step_return;
}

void TrainEnvironment::process_n_step_return(const index_type start_idx, const index_type steps, const tensor_t& next_state, const tensor_t& done_tensor) {
    dim_type state_dim = get_observation_dim();
    dim_type action_dim = get_action_dim();

    tensor_t current_state = n_step_buffer_[start_idx].slice(0, 0, state_dim);
    tensor_t current_action = n_step_buffer_[start_idx].slice(0, state_dim, state_dim + action_dim);

    tensor_t n_step_return = calculate_n_step_return(start_idx, steps);
    sac_->add(current_state, current_action, n_step_return, next_state, done_tensor);
}

void TrainEnvironment::log_statistics(const std::vector<SACResult>& result_history, dim_type episode) const {
	// 최근 에피소드의 보상 통계 계산
    size_t start_idx = std::max(0, static_cast<int>(result_history.size()) - static_cast<int>(constants::NETWORK::LOG_INTERVAL));
    auto begin = result_history.begin() + start_idx;
    auto end = result_history.end();

	// Welford's online algorithm
    real_t mean = 0.0f;
    real_t M2 = 0.0f;  // 이차 중심적률
    size_t n = 0;
	size_t arrived_count = 0;

    for (auto it = begin; it != end; ++it) {
        n += 1;
        real_t delta = it->episode_reward - mean;
        mean += delta / n;
        real_t delta2 = it->episode_reward - mean;
        M2 += delta * delta2;

        if (it->is_arrived) {
            arrived_count++;
        }
    }

    // n-1로 나누어 표본표준편차 계산
    real_t variance = n > 1 ? M2 / (n - 1) : 0.0f;
    real_t std = std::sqrt(variance);

    std::vector<real_t> recent_rewards;
	recent_rewards.reserve(n);
	for (auto it = begin; it != end; ++it) {
		recent_rewards.push_back(it->episode_reward);
	}

    size_t mid = n / 2;
	std::nth_element(recent_rewards.begin(), recent_rewards.begin() + mid, recent_rewards.end());
    real_t median = recent_rewards[mid];

    // 도착 성공률 계산
    real_t arrival_rate = n > 0 ? (static_cast<real_t>(arrived_count) / n) * 100.0f : 0.0f;

    // 현재 시간 가져오기
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto local_time = std::localtime(&now_time);

    // 시간 포맷팅
    char time_buf[100];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", local_time);

 	std::cout << "\r" << std::string(100, ' ') << "\r"
			<< "[" << time_buf << "] "
			<< "Episode " << episode + 1
			<< ", Average Reward: " << std::fixed << std::setprecision(2) << mean
			<< ", Median Reward: " << median
			<< ", std: " << std
			<< ", Arrival Rate: " << std::setprecision(1) << arrival_rate << "%"
			<< " (" << arrived_count << "/" << n << ")" << std::endl;
}

void TrainEnvironment::save_history(const std::vector<SACResult>& result_history, const std::vector<SACMetrics>& metrics_history) const {
	try {
		// 보상 데이터 저장
		{
			std::filesystem::path result_path = std::filesystem::path(his_dir_) / "train_results.csv";
			std::ofstream result_file(result_path);
			if (!result_file.is_open()) {
				throw std::runtime_error("Could not open " + result_path.string());
			}
			result_file << "episode,reward,arrived\n";
			for (size_t i = 0; i < result_history.size(); ++i) {
				result_file << i << ","
					<< result_history[i].episode_reward << ","
					<< result_history[i].is_arrived << "\n";
			}
		}

		// 메트릭 데이터 저장
		{
			std::filesystem::path metrics_path = std::filesystem::path(his_dir_) / "train_metrics.csv";
			std::ofstream metrics_file(metrics_path);
			if (!metrics_file.is_open()) {
				throw std::runtime_error("Could not open " + metrics_path.string());
			}
			metrics_file << "step,critic_loss1,critic_loss2,actor_loss,log_pi,q_value,beta\n";
			for (size_t i = 0; i < metrics_history.size(); ++i) {
				metrics_file << i << ","
					<< metrics_history[i].critic_loss1 << ","
					<< metrics_history[i].critic_loss2 << ","
					<< metrics_history[i].actor_loss << ","
					<< metrics_history[i].log_pi << ","
					<< metrics_history[i].q_value << ","
					<< metrics_history[i].beta << "\n";
			}
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error saving history: " << e.what() << std::endl;
	}

}

MultiAgentEnvironment::MultiAgentEnvironment(count_type width, count_type height, torch::Device device, count_type agent_count)
	: TrainEnvironment(width, height, device, agent_count, false) {

	const auto [min_action, max_action] = get_action_space();
	std::cout << "min_action: " << min_action << ", max_action: " << max_action << std::endl;

	circle_obstacles_num_ = 0;
	rectangle_obstacles_num_ = 5;
	circle_obstacles_spawn_bounds_ = Bounds2D(0, constants::Display::WIDTH, 0, constants::Display::HEIGHT);;
	rectangle_obstacles_spawn_bounds_ = Bounds2D(0, constants::Display::WIDTH, 0, constants::Display::HEIGHT);;
	goal_spawn_bounds_ = Bounds2D(0, constants::Display::WIDTH, 0, constants::Display::HEIGHT);
	agents_spawn_bounds_ = Bounds2D(0, constants::Display::WIDTH, 0, constants::Display::HEIGHT);;
	agents_move_bounds_ = Bounds2D(0, constants::Display::WIDTH, 0, constants::Display::HEIGHT);;

	state_ = this->init(agent_count, min_action, max_action);

	std::cout << "Finished Environment initialized" << std::endl;
}

void MultiAgentEnvironment::test(bool render) {
	sac_->eval();
	SDL_Event event;
	bool is_render = render;
	bool is_paused = false;

	reset();
	bool done = false;

	while (!done) {
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
				return;
			}
			if (event.type == SDL_KEYDOWN){
				if(event.key.keysym.sym == SDLK_r){
					is_render = !is_render;
					if (!is_render) {
						SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
						SDL_RenderClear(renderer_);
						SDL_RenderPresent(renderer_);
					}
				}
				if(event.key.keysym.sym == SDLK_SPACE){
					is_paused = !is_paused;
				}
			}
		}

		if (is_paused) {
			if (is_render) {
				render_scene();
			}
			SDL_Delay(20);
			continue;
		}else if (is_render) {
			render_scene();
		}

		for (count_type i = 0; i < agent_count_; ++i){
			if (check_agent_goal(i) || check_agent_bounds(i) || check_agent_collision(i)) {
				reset_agent(i);
			}

			tensor_t agent_obs = get_agent_observation(i);

            tensor_t action = sac_->select_action(agent_obs);

            agents_[i]->update(fixed_dt_, action, get_combined_obstacles_for_agent(i));
		}

		update_agents_state();
	}
}

void MultiAgentEnvironment::render_scene() const {
	if (!renderer_) return;

	SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
	SDL_RenderClear(renderer_);

	for (const auto& obs : circle_obstacles_) obs->draw(renderer_);
	for (const auto& obs : rectangle_obstacles_) obs->draw(renderer_);

	for (size_t i = 0; i < agents_.size(); ++i) {
		goals_[i]->draw(renderer_);
		agents_[i]->draw(renderer_);
	}

	SDL_RenderPresent(renderer_);
}

MazeAgentEnvironment::MazeAgentEnvironment(count_type width, count_type height, torch::Device device, count_type agent_count)
	: TrainEnvironment(width, height, device, agent_count, false) {

	const auto [min_action, max_action] = get_action_space();
	std::cout << "min_action: " << min_action << ", max_action: " << max_action << std::endl;

	circle_obstacles_num_ = 0;
	rectangle_obstacles_num_ = 11;
	circle_obstacles_spawn_bounds_ = constants::CircleObstacle::SPAWN_BOUNDS;
	rectangle_obstacles_spawn_bounds_ = constants::RectangleObstacle::SPAWN_BOUNDS;
	goal_spawn_bounds_ = constants::Goal::SPAWN_BOUNDS;
	agents_spawn_bounds_ = constants::Agent::SPAWN_BOUNDS;
	agents_move_bounds_ = constants::Agent::MOVE_BOUNDS;

	init_maze();
    state_ = init_maze_environment(agent_count, min_action, max_action);

	std::cout << "Finished Environment initialized" << std::endl;
}

void MazeAgentEnvironment::init_maze() {
    // 벡터 공간 할당 및 초기화
    circle_obstacles_.clear();
    circle_obstacles_.reserve(circle_obstacles_num_);
    circle_obstacles_state_ = torch::zeros({ circle_obstacles_num_, 3 });

    rectangle_obstacles_.clear();
    rectangle_obstacles_.reserve(rectangle_obstacles_num_);
    rectangle_obstacles_state_ = torch::zeros({ rectangle_obstacles_num_, 5 });

    // 상단 벽
    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        0, width_/8, height_/8, (width_ - width_/4), WALL_THICKNESS, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    // 하단 벽
    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        1, width_ - ((width_ - width_/4) / 2 + WALL_THICKNESS), height_ - height_/8, (width_ - width_/4) / 2, WALL_THICKNESS, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        2, WALL_THICKNESS, height_ - height_/8, (width_ - width_/4) / 2, WALL_THICKNESS, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    // 좌측 벽
    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        3, WALL_THICKNESS, height_ - height_/8 + WALL_THICKNESS, WALL_THICKNESS, height_ - height_/4 + WALL_THICKNESS, constants::PI,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    // 우측 벽
    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        4, width_ - 1, height_ - height_/8 + WALL_THICKNESS, WALL_THICKNESS, height_ - height_/4 + WALL_THICKNESS, constants::PI,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    // 미로의 내부 벽 생성
	// 수직 벽
	rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        5, width_/4, height_/3, WALL_THICKNESS, height_/3, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        6, width_/2, height_/2, WALL_THICKNESS, height_/4, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        7, (width_/4)*3, height_/3, WALL_THICKNESS, height_/3, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    // 수평 벽
    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        8, width_/4, height_/3, width_/4, WALL_THICKNESS, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        9, width_/2, height_/2, width_/4, WALL_THICKNESS, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    rectangle_obstacles_.push_back(std::make_unique<object::RectangleObstacle>(
        10, width_/3, (height_/4)*3, width_/3, WALL_THICKNESS, 0,
        rectangle_obstacles_spawn_bounds_, Display::to_sdl_color(Display::ORANGE), false
    ));

    update_rectangle_obstacles_state();
}

tensor_t MazeAgentEnvironment::init_maze_environment(count_type agent_count, tensor_t min_action, tensor_t max_action) {
    // 메모리와 SAC 초기화
    memory_ = std::make_unique<PrioritizedReplayBuffer>(
        get_observation_dim(),
        get_action_dim(),
        constants::NETWORK::BUFFER_SIZE,
        constants::NETWORK::BATCH_SIZE,
        device_,
        0.6f,
        0.4f
    );

    sac_ = std::make_unique<SAC>(
        get_observation_dim(),
        get_action_dim(),
        min_action,
        max_action,
        memory_.get(),
        device_
    );

    // 목표지점과 에이전트 초기화
    goals_.reserve(agent_count);
    agents_.reserve(agent_count);

    for (count_type i = 0; i < agent_count; ++i) {
        auto goal = std::make_unique<object::Goal>(i,
            std::nullopt, std::nullopt,
            constants::Goal::RADIUS,
            goal_spawn_bounds_,
            Display::to_sdl_color(Display::GREEN),
            false);
        goals_.push_back(std::move(goal));

        auto agent = std::make_unique<object::Agent>(i,
            std::nullopt, std::nullopt,
            constants::Agent::RADIUS,
            agents_spawn_bounds_,
            agents_move_bounds_,
            Display::to_sdl_color(Display::BLUE),
            true,
            circle_obstacles_state_,
            rectangle_obstacles_state_,
            goals_[i]->get_state(),
            path_planner_.get());
        agents_.push_back(std::move(agent));
    }

    agents_state_ = torch::zeros({ agent_count, 3 });
    update_agents_state();

    return get_observation();
}

void MazeAgentEnvironment::test(bool render) {
	sac_->eval();
	SDL_Event event;
	bool is_render = render;
	bool is_paused = false;

	bool done = false;

	while (!done) {
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
				return;
			}
			if (event.type == SDL_KEYDOWN){
				if(event.key.keysym.sym == SDLK_r){
					is_render = !is_render;
					if (!is_render) {
						SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
						SDL_RenderClear(renderer_);
						SDL_RenderPresent(renderer_);
					}
				}
				if(event.key.keysym.sym == SDLK_SPACE){
					is_paused = !is_paused;
				}
			}
		}

		if (is_paused) {
			if (is_render) {
				render_scene();
			}
			SDL_Delay(20);
			continue;
		}else if (is_render) {
			render_scene();
		}

		for (count_type i = 0; i < agent_count_; ++i){
			if (check_agent_goal(i) || check_agent_bounds(i) || check_agent_collision(i)) {
				reset_agent(i);
			}

			tensor_t agent_obs = get_agent_observation(i);

            tensor_t action = sac_->select_action(agent_obs);

            agents_[i]->update(fixed_dt_, action, get_combined_obstacles_for_agent(i));
		}

		update_agents_state();

		if (is_render) {
			render_scene();
		}
	}
}

void MazeAgentEnvironment::render_scene() const {
	if (!renderer_) return;

	SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
	SDL_RenderClear(renderer_);

	for (const auto& obs : circle_obstacles_) obs->draw(renderer_);
	for (const auto& obs : rectangle_obstacles_) obs->draw(renderer_);

	for (size_t i = 0; i < agents_.size(); ++i) {
		goals_[i]->draw(renderer_);
		agents_[i]->draw(renderer_);
	}

	SDL_RenderPresent(renderer_);
}

} // namespace environment
