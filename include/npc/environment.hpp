#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include "npc/sac.hpp"
#include "npc/object.hpp"
#include <SDL.h>
#include <string>

namespace environment {

class BaseEnvironment {
public:
	explicit BaseEnvironment(count_type width = Display::WIDTH,
							 count_type height = Display::HEIGHT,
							 torch::Device device = torch::kCPU)
		: width_(width)
		, height_(height)
		, device_(device) {}

	virtual ~BaseEnvironment() = default;

	BaseEnvironment(const BaseEnvironment&) = delete;
	BaseEnvironment& operator=(const BaseEnvironment&) = delete;

	virtual tensor_t reset() = 0;
	virtual std::tuple<tensor_t, tensor_t, bool, bool> step(const tensor_t& action) = 0;
	virtual void render(SDL_Renderer* renderer) const {}

	virtual tensor_t get_observation_space() const {
		return torch::full({ observation_dim_ }, std::numeric_limits<real_t>::infinity());
	}

	virtual std::pair<tensor_t, tensor_t> get_action_space() const {
		auto min_action = torch::tensor({ (constants::Agent::VELOCITY_LIMITS.a / constants::Agent::VELOCITY_LIMITS.b), -(constants::Agent::YAW_CHANGE_LIMIT / constants::Agent::YAW_CHANGE_LIMIT) });
		auto max_action = torch::tensor({ (constants::Agent::VELOCITY_LIMITS.b / constants::Agent::VELOCITY_LIMITS.b), (constants::Agent::YAW_CHANGE_LIMIT / constants::Agent::YAW_CHANGE_LIMIT) });
		return { min_action, max_action };
	}

	void set_observation_dim(dim_type observation_dim) { observation_dim_ = observation_dim; }
	void set_action_dim(dim_type action_dim) { action_dim_ = action_dim; }
	void set_render(SDL_Renderer* renderer) { renderer_ = renderer; }

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
	dim_type observation_dim_{ 0 };
	dim_type action_dim_{ 0 };
	real_t fixed_dt_{ 1.0f / static_cast<real_t>(Display::FPS) };

	tensor_t state_;
	bool terminated_{ false };
	bool truncated_{ false };
	count_type step_count_{ 0 };

	std::random_device rd_;
	std::mt19937 gen_{ rd_() };

	const torch::Device device_;

	SDL_Renderer* renderer_;

	virtual tensor_t get_observation() const = 0;
	virtual real_t calculate_reward(const tensor_t& state) { return 0.0f; }
	virtual bool check_goal() const = 0;
	virtual bool check_bounds() const = 0;
	virtual bool check_obstacle_collision() const = 0;
};


class TrainEnvironment : public BaseEnvironment {
public:
	TrainEnvironment(count_type width = Display::WIDTH,
					 count_type height = Display::HEIGHT,
					 torch::Device device = torch::kCPU);

	tensor_t reset() override;
	std::tuple<tensor_t, tensor_t, bool, bool> step(const tensor_t& action) override;
	void save(dim_type episode, bool print);
	void load(const std::string& timestamp, dim_type episode);
	std::vector<real_t> train(const dim_type episodes, bool render = false);
	std::vector<real_t> test(const dim_type episodes, bool render = false);

protected:
	tensor_t get_observation() const override;
	real_t calculate_reward(const tensor_t& state, const tensor_t& action);
	bool check_goal() const override;
	bool check_bounds() const override;
	bool check_obstacle_collision() const override;

private:
	std::vector<std::unique_ptr<object::CircleObstacle>> circle_obstacles_;
	std::vector<std::unique_ptr<object::RectangleObstacle>> rectangle_obstacles_;
	tensor_t circle_obstacles_state_;
	tensor_t rectangle_obstacles_state_;
	std::unique_ptr<object::Goal> goal_;
	std::unique_ptr<object::Agent> agent_;

	std::unique_ptr<SAC> sac_;

	dim_type start_episode_{ 0 };

	tensor_t init_objects();
	void update_circle_obstacles_state();
	void update_rectangle_obstacles_state();
	void log_statistics(const std::vector<real_t>& reward_history, dim_type episode) const;
};

}  // namespace environment
