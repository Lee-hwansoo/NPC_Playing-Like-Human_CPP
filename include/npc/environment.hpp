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
	std::string get_history_directory() const {
		std::filesystem::path script_path(__FILE__);
		std::filesystem::path script_dir = script_path.parent_path();
		std::filesystem::path script_name = script_path.stem();
		std::filesystem::path his_dir = script_dir / constants::HIS_DIR / script_name;
		his_dir = std::filesystem::absolute(his_dir).lexically_normal();
		std::filesystem::create_directories(his_dir);
		return his_dir.string();
	}
};

class TrainEnvironment : public BaseEnvironment {
public:
	TrainEnvironment(count_type width = Display::WIDTH,
					 count_type height = Display::HEIGHT,
					 torch::Device device = torch::kCPU,
					 count_type agent_count = 1,
					 bool init = true);

	tensor_t reset() override;
	std::tuple<tensor_t, tensor_t, bool, bool> step(const tensor_t& action) override;
	void save(dim_type episode, bool print);
	void load(const std::string& timestamp, dim_type episode);
	TrainingResult train(const dim_type episodes, bool render = false, bool debug = false);
	std::vector<real_t> test(const dim_type episodes, bool render = false);

protected:
	count_type circle_obstacles_num_;	 // 원형 장애물 수
	count_type rectangle_obstacles_num_; // 사각형 장애물 수
	std::vector<std::unique_ptr<object::CircleObstacle>> circle_obstacles_;
	std::vector<std::unique_ptr<object::RectangleObstacle>> rectangle_obstacles_;
	tensor_t circle_obstacles_state_;    // [num_circles, 3]
	tensor_t rectangle_obstacles_state_; // [num_rectangles, 5]
	Bounds2D circle_obstacles_spawn_bounds_ { constants::CircleObstacle::SPAWN_BOUNDS };
	Bounds2D rectangle_obstacles_spawn_bounds_ { constants::RectangleObstacle::SPAWN_BOUNDS };

	std::vector<std::unique_ptr<object::Goal>> goals_;
	Bounds2D goal_spawn_bounds_ = constants::Goal::SPAWN_BOUNDS;

    count_type agent_count_;             // 총 에이전트 수
	std::vector<std::unique_ptr<object::Agent>> agents_;
	tensor_t agents_state_;             // [num_agents, 3]
	Bounds2D agents_spawn_bounds_ { constants::Agent::SPAWN_BOUNDS };
	Bounds2D agents_move_bounds_ { constants::Agent::MOVE_BOUNDS };

	std::unique_ptr<path_planning::RRT> path_planner_;

	std::unique_ptr<ReplayBuffer> memory_;
	std::unique_ptr<SAC> sac_;

	dim_type start_episode_{ 0 };

	tensor_t init(count_type agent_count, tensor_t min_action, tensor_t max_action);
	void update_circle_obstacles_state();
	void update_rectangle_obstacles_state();
	void update_agents_state();

	tensor_t get_observation() const override;

	real_t calculate_reward(const tensor_t& state, const tensor_t& action);

	bool check_goal() const override;
	bool check_bounds() const override;
	bool check_obstacle_collision() const override;

	virtual void render_scene() const;

	void reset_agent(size_t agent_idx);
	tensor_t get_agent_observation(size_t agent_idx) const;
    tensor_t get_combined_obstacles_for_agent(size_t agent_idx) const;
    bool check_agent_goal(size_t agent_idx) const;
    bool check_agent_bounds(size_t agent_idx) const;
    bool check_agent_collision(size_t agent_idx) const;

private:
	std::string his_dir_;

	void log_statistics(const std::vector<real_t>& reward_history, dim_type episode) const;
	void save_history(const std::vector<real_t>& reward_history, const std::vector<SACMetrics>& metrics_history) const;
};

class MultiAgentEnvironment : public TrainEnvironment {
public:
	MultiAgentEnvironment(count_type width = Display::WIDTH,
						count_type height = Display::HEIGHT,
						torch::Device device = torch::kCPU,
						count_type agent_count = 1);

	void test(bool render = true);

protected:
	void render_scene() const override;

};


}  // namespace environment
