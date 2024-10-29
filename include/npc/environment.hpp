#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include "utils/utils.hpp"
#include "npc/sac.hpp"
#include "npc/object.hpp"
#include <SDL.h>

class BaseEnvironment {
public:
	explicit BaseEnvironment(count_type width = Display::WIDTH,
							 count_type height = Display::HEIGHT,
							 dim_type observation_dim = 0,
							 dim_type action_dim = 0)
		: width_(width)
		, height_(height)
		, observation_dim_(observation_dim)
		, action_dim_(action_dim)
		, device_(utils::get_device()) {}

	virtual ~BaseEnvironment() = default;

	BaseEnvironment(const BaseEnvironment&) = delete;
	BaseEnvironment& operator=(const BaseEnvironment&) = delete;

	virtual tensor_t reset() = 0;
	virtual std::tuple<tensor_t, real_t, bool, bool> step(const tensor_t& action) = 0;
	virtual void render(SDL_Renderer* renderer) const = 0;

	virtual tensor_t get_observation_space() const {
		return torch::full({ observation_dim_ }, std::numeric_limits<real_t>::infinity(),
			torch::TensorOptions().dtype(get_tensor_dtype()).device(device_));
	}

	virtual std::pair<tensor_t, tensor_t> get_action_space() const {
		auto min_action = torch::tensor({ (constants::Agent::VELOCITY_LIMITS.a / constants::Agent::VELOCITY_LIMITS.b), -(constants::Agent::YAW_CHANGE_LIMIT / constants::Agent::YAW_CHANGE_LIMIT) }, torch::TensorOptions().dtype(get_tensor_dtype()).device(device_));
		auto max_action = torch::tensor({ (constants::Agent::VELOCITY_LIMITS.b / constants::Agent::VELOCITY_LIMITS.b), (constants::Agent::YAW_CHANGE_LIMIT / constants::Agent::YAW_CHANGE_LIMIT) }, torch::TensorOptions().dtype(get_tensor_dtype()).device(device_));
		return { min_action, max_action };
	}

	dim_type get_observation_dim() const { return observation_dim_; }
	dim_type get_action_dim() const { return action_dim_; }
	tensor_t get_state() const { return state_; }
	count_type get_step_count() const { return step_count_; }
	bool is_terminated() const { return terminated_; }
	bool is_truncated() const { return truncated_; }
	const torch::Device& get_device() const { return device_; }

protected:
	const count_type width_;
	const count_type height_;
	dim_type observation_dim_;
	dim_type action_dim_;
	count_type step_count_{ 0 };
	real_t fixed_dt_{ 1.0f / static_cast<real_t>(Display::FPS) };
	bool terminated_{ false };
	bool truncated_{ false };
	tensor_t state_;

	std::random_device rd_;
	std::mt19937 gen_{ rd_() };

	const torch::Device device_;

	virtual tensor_t get_observation() const = 0;
	virtual real_t calculate_reward(const tensor_t& state, const tensor_t& action) = 0;
	virtual bool check_goal() const = 0;
	virtual bool check_bounds() const = 0;
	virtual bool check_obstacle_collision() const = 0;
};


class BasicEnvironment : public BaseEnvironment {
public:


protected:

private:
	std::vector<std::unique_ptr<object::CircleObstacle>> circle_obstacles_;
	std::vector<std::unique_ptr<object::RectangleObstacle>> rectangle_obstacles_;
	std::unique_ptr<object::Goal> goal_;
	std::unique_ptr<object::Agent> agent_;

	std::unique_ptr<SAC> sac_;
};