#include "npc/environment.hpp"
#include "utils/constants.hpp"
#include "utils/types.hpp"

namespace environment {

TrainEnvironment::TrainEnvironment(count_type width, count_type height, torch::Device device)
	: BaseEnvironment(width, height, device) {
	set_observation_dim(constants::Agent::FOV::RAY_COUNT + 9);
	set_action_dim(2);

	std::cout << "Environment initialized, device: " << device << std::endl;

	const auto [min_action, max_action] = get_action_space();
	std::cout << "min_action: " << min_action << ", max_action: " << max_action << std::endl;

	sac_ = std::make_unique<SAC>(
		get_observation_dim(),
		get_action_dim(),
		min_action,
		max_action,
		device_
		);

	state_ = init_objects();

	std::cout << "Finished Environment initialized" << std::endl;
}

tensor_t TrainEnvironment::init_objects() {
	circle_obstacles_.reserve(constants::CircleObstacle::COUNT);
	for (count_type i = 0; i < constants::CircleObstacle::COUNT; ++i) {
		auto obs = std::make_unique<object::CircleObstacle>(std::nullopt, std::nullopt, constants::CircleObstacle::RADIUS, constants::CircleObstacle::SPAWN_BOUNDS, Display::to_sdl_color(Display::ORANGE), true);
		circle_obstacles_.push_back(std::move(obs));
	}

	rectangle_obstacles_.reserve(constants::RectangleObstacle::COUNT);
	for (count_type i = 0; i < constants::RectangleObstacle::COUNT; ++i) {
		auto obs = std::make_unique<object::RectangleObstacle>(std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, constants::RectangleObstacle::SPAWN_BOUNDS, Display::to_sdl_color(Display::ORANGE), false);
		rectangle_obstacles_.push_back(std::move(obs));
	}

	circle_obstacles_state_ = torch::zeros({ constants::CircleObstacle::COUNT, 3 });
	rectangle_obstacles_state_ = torch::zeros({ constants::RectangleObstacle::COUNT, 5 });

	update_circle_obstacles_state();
	update_rectangle_obstacles_state();

	goal_ = std::make_unique<object::Goal>(std::nullopt, std::nullopt, constants::Goal::RADIUS, constants::Goal::SPAWN_BOUNDS, Display::to_sdl_color(Display::GREEN), false);
	agent_ = std::make_unique<object::Agent>(std::nullopt, std::nullopt, constants::Agent::RADIUS, constants::Agent::SPAWN_BOUNDS, constants::Agent::MOVE_BOUNDS, Display::to_sdl_color(Display::BLUE), true, circle_obstacles_state_, rectangle_obstacles_state_, goal_->get_state());

	return get_observation();
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

tensor_t TrainEnvironment::get_observation() const {
	return agent_->get_state();
}

bool TrainEnvironment::check_goal() const { return agent_->is_goal(); }
bool TrainEnvironment::check_bounds() const { return agent_->is_out(); }
bool TrainEnvironment::check_obstacle_collision() const { return agent_->is_collison(); }

tensor_t TrainEnvironment::reset() {
	for (auto& obs : circle_obstacles_) {
		obs->reset();
	}
	for (auto& obs : rectangle_obstacles_) {
		obs->reset();
	}

	update_circle_obstacles_state();
	update_rectangle_obstacles_state();

	goal_->reset();
	agent_->reset(std::nullopt, std::nullopt, circle_obstacles_state_, rectangle_obstacles_state_, goal_->get_state());

	step_count_ = 0;
	terminated_ = false;
	truncated_ = false;

	return get_observation();
}

std::tuple<tensor_t, tensor_t, bool, bool> TrainEnvironment::step(const tensor_t& action) {
	terminated_ = check_goal();
	truncated_ = check_bounds() || check_obstacle_collision() || step_count_ >= constants::NETWORK::MAX_STEP;

	tensor_t reward = torch::tensor(calculate_reward(state_, action));

	if (terminated_ || truncated_) {
		tensor_t current_state = get_observation();
		bool is_terminated = terminated_;
		bool is_truncated = truncated_;
		reset();
		return std::make_tuple(current_state, reward, is_terminated, is_truncated);
	}

	step_count_++;

	for (auto& obstacle : circle_obstacles_) {
		obstacle->update(fixed_dt_);
	}

	update_circle_obstacles_state();

	state_ = agent_->update(fixed_dt_, action, circle_obstacles_state_);

	return std::make_tuple(state_, reward, terminated_, truncated_);
}

real_t TrainEnvironment::calculate_reward(const tensor_t& state, const tensor_t& action) {
	auto state_size = state.size(0);

	real_t normalized_goal_dist = state[state_size-4].item<real_t>();
	real_t normalized_angle_diff = state[state_size-3].item<real_t>();
	real_t goal_in_fov = state[state_size-2].item<real_t>();
	real_t frenet_d = state[state_size-1].item<real_t>();

	real_t force = action[0].item<real_t>();
	real_t yaw_change = action[1].item<real_t>();

	real_t goal_reward = (1.0f - normalized_goal_dist);
	real_t fov_reward = (1.0f - std::abs(normalized_angle_diff)) * goal_in_fov * 0.4f;
	real_t angle_reward = (1.0f - std::abs(normalized_angle_diff)) * 0.4f;
	real_t turn_penalty = -std::abs(yaw_change) * 0.2f;
	real_t path_delta_penalty = -std::abs(frenet_d) * 0.5f;
	real_t time_penalty = -0.0005f * static_cast<real_t>(step_count_);

	real_t terminal_reward = 0.0f;
	if (terminated_) {
		terminal_reward = terminal_reward + 100.0f;
	}
	if (truncated_) {
		terminal_reward = terminal_reward -100.0f;
	}

	real_t reward = goal_reward +
		fov_reward +
		angle_reward +
		turn_penalty +
		path_delta_penalty +
		time_penalty +
		terminal_reward;

	return reward;
}

void TrainEnvironment::save(dim_type episode) {
	if (sac_) {
		sac_->save_network_parameters(episode);
	}
}

void TrainEnvironment::load(const std::string& timestamp, dim_type episode) {
	if (sac_) {
		sac_->load_network_parameters(timestamp, episode);
		start_episode_ = episode;
	}
}

std::vector<real_t> TrainEnvironment::train(const dim_type episodes, bool render) {
	sac_->train();
	SDL_Event event;
    std::vector<real_t> reward_history;
    reward_history.reserve(static_cast<size_t>(episodes));

    for (dim_type episode = start_episode_; episode < start_episode_+ episodes; ++episode) {
		real_t episode_return = 0.0f;
		tensor_t state = reset();
		bool done = false;

		while (!done) {
			tensor_t action = sac_->select_action(state);
			auto [next_state, reward, terminated, truncated] = step(action);

			done = terminated || truncated;
			sac_->add(state, action, reward, next_state, torch::tensor(done, get_tensor_dtype()));
			sac_->update();

			episode_return += reward.item<real_t>();
			state = next_state;

			if (render) {
				while (SDL_PollEvent(&event)) {
					if (event.type == SDL_QUIT ||
						(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
						return reward_history;;
					}
				}

				SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
				SDL_RenderClear(renderer_);

				auto color = Display::to_sdl_color(Display::GREEN);
				SDL_SetRenderDrawColor(renderer_, color.r, color.g, color.b, color.a);
				SDL_RenderDrawLine(renderer_, 0, Section::GOAL_LINE, Display::WIDTH, Section::GOAL_LINE);
				SDL_RenderDrawLine(renderer_, 0, Section::START_LINE, Display::WIDTH, Section::START_LINE);

				for (auto& obs : circle_obstacles_) {
					obs->draw(renderer_);
				}
				for (auto& obs : rectangle_obstacles_) {
					obs->draw(renderer_);
				}
				goal_->draw(renderer_);
				agent_->draw(renderer_);

				SDL_RenderPresent(renderer_);
			}

			std::cout << "\rEpisode: " << episode + 1 << "/" << episodes << " | Step: " << step_count_ << " " << std::flush;
		}

		reward_history.push_back(episode_return);

		if ((episode + 1) % constants::NETWORK::INTERVAL == 0) {
			log_statistics(reward_history, episode);
			save(episode + 1);
		}
    }

    return reward_history;
}

std::vector<real_t> TrainEnvironment::test(const dim_type episodes, bool render) {
	sac_->eval();

	std::vector<real_t> reward_history;
	reward_history.reserve(static_cast<size_t>(episodes));

	for (dim_type episode = 0; episode < episodes; ++episode) {
		real_t episode_return = 0.0f;
		tensor_t state = reset();
		bool done = false;

		while (!done) {
			// 행동 선택 및 환경 스텝
			tensor_t action = sac_->select_action(state);
			auto [next_state, reward, terminated, truncated] = step(action);

			done = terminated || truncated;
			episode_return += reward.item<real_t>();
			state = next_state;

			// 렌더링 수행
            if (render) {
				for (auto& obs : circle_obstacles_) {
					obs->draw(renderer_);
				}
				for (auto& obs : rectangle_obstacles_) {
					obs->draw(renderer_);
				}
				goal_->draw(renderer_);
				agent_->draw(renderer_);
            }
		}

		reward_history.push_back(episode_return);
	}

	return reward_history;
}

void TrainEnvironment::log_statistics(const std::vector<real_t>& reward_history, dim_type episode) const {
	// 최근 10개 에피소드의 보상 통계 계산
	size_t start_idx = std::max(0, static_cast<int>(reward_history.size()) - 10);
	std::vector<real_t> recent_rewards(reward_history.begin() + start_idx, reward_history.end());

	// 평균, 중앙값, 표준편차 계산
	real_t mean_reward = std::accumulate(recent_rewards.begin(), recent_rewards.end(), 0.0f) / recent_rewards.size();

	std::vector<real_t> sorted_rewards = recent_rewards;
	std::sort(sorted_rewards.begin(), sorted_rewards.end());
	real_t median_reward = sorted_rewards[sorted_rewards.size() / 2];

	real_t variance = 0.0f;
	for (const auto& reward : recent_rewards) {
		variance += std::pow(reward - mean_reward, 2);
	}
	real_t std_dev = std::sqrt(variance / recent_rewards.size());

	std::cout << "Episode " << episode + 1
			<< ", Average Reward: " << std::fixed << std::setprecision(2) << mean_reward
			<< ", Median Reward: " << median_reward
			<< ", std: " << std_dev << std::endl;
}

} // namespace environment
