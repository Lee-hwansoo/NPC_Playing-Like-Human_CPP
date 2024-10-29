#include "npc/environment.hpp"

TrainEnvironment::TrainEnvironment(count_type width, count_type height)
	: BaseEnvironment(width, height) {
	set_observation_dim(constants::Agent::FOV::RAY_COUNT + 10);
	set_action_dim(2);

	const auto [min_action, max_action] = get_action_space();
	sac_ = std::make_unique<SAC>(
		get_observation_dim(),
		get_action_dim(),
		std::vector<real_t>(min_action.data_ptr<real_t>(), min_action.data_ptr<real_t>() + min_action.numel()),
		std::vector<real_t>(max_action.data_ptr<real_t>(), max_action.data_ptr<real_t>() + max_action.numel()),
		device_
		);

	state_ = init_objects();
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

	circle_obstacles_state_ = torch::zeros({ constants::CircleObstacle::COUNT, 3 }, torch::TensorOptions().dtype(get_tensor_dtype()).device(device_));
	rectangle_obstacles_state_ = torch::zeros({ constants::RectangleObstacle::COUNT, 5 }, torch::TensorOptions().dtype(get_tensor_dtype()).device(device_));

	update_circle_obstacles_state();
	update_rectangle_obstacles_state();

	goal_ = std::make_unique<object::Goal>(std::nullopt, std::nullopt, constants::Goal::RADIUS, constants::Goal::SPAWN_BOUNDS, Display::to_sdl_color(Display::GREEN), false);
	agent_ = std::make_unique<object::Agent>(std::nullopt, std::nullopt, constants::Agent::RADIUS, constants::Agent::SPAWN_BOUNDS, constants::Agent::MOVE_BOUNDS, Display::to_sdl_color(Display::BLUE), true, circle_obstacles_state_, rectangle_obstacles_state_, goal_->get_state());

	return get_observation();
}

void TrainEnvironment::update_circle_obstacles_state() const {
	for (size_t i = 0; i < circle_obstacles_.size(); ++i) {
		circle_obstacles_state_[i] = circle_obstacles_[i]->get_state();
	}
}

void TrainEnvironment::update_rectangle_obstacles_state() const {
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

std::tuple<tensor_t, real_t, bool, bool> TrainEnvironment::step(const tensor_t& action) {
	terminated_ = check_goal();
	truncated_ = check_bounds() || check_obstacle_collision() || step_count_ >= constants::SAC::MAX_STEP;

	real_t reward = calculate_reward(state_, action);

	if (terminated_ || truncated_) {
		tensor_t current_state = get_observation();
		reset();
		return std::make_tuple(current_state, reward, terminated_, truncated_);
	}

	step_count_++;

	for (auto& obstacle : circle_obstacles_) {
		obstacle->update(fixed_dt_);
	}

	update_circle_obstacles_state();

	state_ = agent_->update(fixed_dt_, action, circle_obstacles_state_, goal_->get_state());

	return std::make_tuple(state_, reward, terminated_, truncated_);
}

real_t TrainEnvironment::calculate_reward(const tensor_t& state, const tensor_t& action) {
	real_t normalized_goal_dist = state[-4].item<real_t>();
	real_t normalized_angle_diff = state[-3].item<real_t>();
	real_t goal_in_fov = state[-2].item<real_t>();
	real_t frenet_d = state[-1].item<real_t>();

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
	sac_->save_network_parameters(episode);
}

void TrainEnvironment::load(const std::string& timestamp, dim_type episode) {
	sac_->load_network_parameters(timestamp, episode);
}