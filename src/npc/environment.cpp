#include "npc/environment.hpp"
#include "utils/constants.hpp"
#include "utils/types.hpp"

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

	memory_ = std::make_unique<ReplayBuffer>(
		get_observation_dim(),
		get_action_dim(),
		constants::NETWORK::BUFFER_SIZE,
		constants::NETWORK::BATCH_SIZE,
		device_
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

real_t TrainEnvironment::calculate_reward(const tensor_t& state, const tensor_t& action) {
	auto state_size = state.size(0);

	auto required_state = state.slice(0, state_size-4, state_size).to(torch::kCPU);
    auto required_action = action.to(torch::kCPU);

    real_t normalized_goal_dist = required_state[0].item<real_t>();
    real_t normalized_angle_diff = required_state[1].item<real_t>();
    real_t goal_in_fov = required_state[2].item<real_t>();
    real_t frenet_d = required_state[3].item<real_t>();

    real_t force = required_action[0].item<real_t>();
    real_t yaw_change = required_action[1].item<real_t>();

	real_t goal_reward = (1.0f - normalized_goal_dist);
	real_t fov_reward = (1.0f - std::abs(normalized_angle_diff)) * goal_in_fov * 0.3f;
	real_t angle_reward = (1.0f - std::abs(normalized_angle_diff)) * 0.8f;
	real_t turn_penalty = -(std::abs(yaw_change) * 0.3f);
	real_t path_delta_penalty = -(std::abs(frenet_d) * 0.8f);
	real_t time_penalty = -(0.001f * static_cast<real_t>(step_count_));
	real_t terminal_reward = 0.0f;
	if (terminated_) {
		real_t speed_bonus = std::max(0.0f, (constants::NETWORK::MAX_STEP - step_count_) * 0.5f);
		terminal_reward = terminal_reward + 100.0f + speed_bonus;
	}
	if (truncated_) {
		terminal_reward = terminal_reward - 200.0f;
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

	state_ = agents_[0]->update(fixed_dt_, action, circle_obstacles_state_);

	update_agents_state();

	return std::make_tuple(state_, reward, terminated_, truncated_);
}

TrainingResult TrainEnvironment::train(const dim_type episodes, bool render, bool debug) {
	sac_->train();
	SDL_Event event;
    std::vector<real_t> reward_history;
	std::vector<SACMetrics> metrics_history;
    reward_history.reserve(static_cast<size_t>(episodes));
	metrics_history.reserve(static_cast<size_t>(episodes * constants::NETWORK::MAX_STEP / constants::NETWORK::UPDATE_INTERVAL));

    for (dim_type episode = start_episode_; episode < start_episode_+ episodes; ++episode) {
		real_t episode_return = 0.0f;
		tensor_t state = reset();
		bool done = false;

		while (!done) {
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT ||
					(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
					return { reward_history, metrics_history };
				}
			}

			tensor_t action = sac_->select_action(state);
			auto [next_state, reward, terminated, truncated] = step(action);

			done = terminated || truncated;
			sac_->add(state, action, reward, next_state, torch::tensor(done, get_tensor_dtype()));

			if (step_count_ % constants::NETWORK::UPDATE_INTERVAL == 0) {
				auto metrics = sac_->update(debug);
				if (metrics.is_vaild){
					metrics_history.push_back(metrics);
				}
			}

			episode_return += reward.item<real_t>();
			state = next_state;

			if (render) {
				render_scene();
			}

			std::cout << "\rEpisode: " << episode + 1 << "/" << start_episode_+ episodes << " | Step: " << step_count_ << " " << std::flush;
		}

		reward_history.push_back(episode_return);
		if ((episode + 1) % constants::NETWORK::LOG_INTERVAL == 0) {
			log_statistics(reward_history, episode);
			save_history(reward_history, metrics_history);
			save(episode + 1, false);
		}
    }

    return {reward_history, metrics_history};
}

std::vector<real_t> TrainEnvironment::test(const dim_type episodes, bool render) {
	sac_->eval();
	SDL_Event event;
	std::vector<real_t> reward_history;
	std::vector<real_real_t> action_times;
	reward_history.reserve(static_cast<size_t>(episodes));
	action_times.reserve(static_cast<size_t>(episodes) * 2000);

	for (dim_type episode = 0; episode < episodes; ++episode) {
		real_t episode_return = 0.0f;
		tensor_t state = reset();
		bool done = false;

		while (!done) {
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT ||
					(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
					return reward_history;;
				}
			}

			auto start_time = std::chrono::high_resolution_clock::now();
			tensor_t action = sac_->select_action(state);
			auto end_time = std::chrono::high_resolution_clock::now();

			std::chrono::duration<real_real_t, std::milli> duration = end_time - start_time;
			action_times.push_back(duration.count());

			auto [next_state, reward, terminated, truncated] = step(action);

			done = terminated || truncated;
			episode_return += reward.item<real_t>();
			state = next_state;

            if (render) {
				render_scene();
            }
		}

		std::cout << "Episode: " << episode + 1 << "/" <<  episodes << " | Reward: " << episode_return << " " << std::endl;

		reward_history.push_back(episode_return);
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

	return reward_history;
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

void TrainEnvironment::log_statistics(const std::vector<real_t>& reward_history, dim_type episode) const {
	// 최근 10개 에피소드의 보상 통계 계산
    size_t start_idx = std::max(0, static_cast<int>(reward_history.size()) - 10);
    auto begin = reward_history.begin() + start_idx;
    auto end = reward_history.end();

	// Welford's online algorithm
    real_t mean = 0.0f;
    real_t M2 = 0.0f;  // 이차 중심적률
    size_t n = 0;

    for (auto it = begin; it != end; ++it) {
        n += 1;
        real_t delta = *it - mean;
        mean += delta / n;
        real_t delta2 = *it - mean;
        M2 += delta * delta2;
    }

    // n-1로 나누어 표본표준편차 계산
    real_t variance = M2 / (n - 1);
    real_t std = std::sqrt(variance);

    std::vector<real_t> recent_rewards(begin, end);
    size_t mid = n / 2;
	std::nth_element(recent_rewards.begin(), recent_rewards.begin() + mid, recent_rewards.end());
    real_t median = recent_rewards[mid];

	std::cout << "Episode " << episode + 1
			<< ", Average Reward: " << std::fixed << std::setprecision(2) << mean
			<< ", Median Reward: " << median
			<< ", std: " << std << std::endl;
}

void TrainEnvironment::save_history(const std::vector<real_t>& reward_history, const std::vector<SACMetrics>& metrics_history) const {
	try {
		// 보상 데이터 저장
		{
			std::filesystem::path reward_path = std::filesystem::path(his_dir_) / "train_rewards.csv";
			std::ofstream reward_file(reward_path);
			if (!reward_file.is_open()) {
				throw std::runtime_error("Could not open " + reward_path.string());
			}
			reward_file << "episode,reward\n";
			for (size_t i = 0; i < reward_history.size(); ++i) {
				reward_file << i << "," << reward_history[i] << "\n";
			}
		}

		// 메트릭 데이터 저장
		{
			std::filesystem::path metrics_path = std::filesystem::path(his_dir_) / "train_metrics.csv";
			std::ofstream metrics_file(metrics_path);
			if (!metrics_file.is_open()) {
				throw std::runtime_error("Could not open " + metrics_path.string());
			}
			metrics_file << "step,critic_loss1,critic_loss2,actor_loss,log_pi,q_value\n";
			for (size_t i = 0; i < metrics_history.size(); ++i) {
				metrics_file << i << ","
					<< metrics_history[i].critic_loss1 << ","
					<< metrics_history[i].critic_loss2 << ","
					<< metrics_history[i].actor_loss << ","
					<< metrics_history[i].log_pi << ","
					<< metrics_history[i].q_value << "\n";
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
	reset();
	bool done = false;

	while (!done) {
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT ||
				(event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
				return;
			}
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

		if (render) {
			render_scene();
		}
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

} // namespace environment
