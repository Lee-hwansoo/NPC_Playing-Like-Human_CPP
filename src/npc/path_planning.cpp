#include "npc/path_planning.hpp"
#include "utils/constants.hpp"
#include <cmath>
#include <iostream>

namespace path_planning {

RRT::RRT(const tensor_t& start,
         const types::Bounds2D& space,
	     const tensor_t& circle_obstacles_state,
	     const tensor_t& rectangle_obstacles_state,
         const tensor_t& goal_state,
         const torch::Device device)
    : space_(space)
    , circle_obstacles_state_(circle_obstacles_state.to(device))
    , rectangle_obstacles_state_(rectangle_obstacles_state.to(device))
    , device_(device)
    , max_iter_(constants::RRT::MAX_ITER)
    , goal_sample_rate_(constants::RRT::GOAL_SAMPLE_RATE)
    , min_u_(constants::RRT::MIN_U)
    , max_u_(constants::RRT::MAX_U)
    , success_dist_threshold_(constants::RRT::SUCCESS_DIST_THRESHOLD)
    , collision_check_step_(constants::RRT::COLLISION_CHECK_STEP)
    , step_size_(constants::RRT::STEP_SIZE)
    , gen_(std::random_device{}())
    , unit_dist_(0.0f, 1.0f)
{
    start_node_ = std::make_shared<Node>(start[0].item<real_t>(), start[1].item<real_t>());
    goal_node_ = std::make_shared<Node>(goal_state[0].item<real_t>(), goal_state[1].item<real_t>());

    node_list_.reserve(max_iter_);
}

void RRT::update(const tensor_t& start, const tensor_t& circle_obstacles_state, const tensor_t& rectangle_obstacles_state, const tensor_t& goal_state) {
    start_node_ = std::make_shared<Node>(start[0].item<real_t>(), start[1].item<real_t>());
    goal_node_ = std::make_shared<Node>(goal_state[0].item<real_t>(), goal_state[1].item<real_t>());
    circle_obstacles_state_ = circle_obstacles_state.to(device_);
    rectangle_obstacles_state_ = rectangle_obstacles_state.to(device_);
}

tensor_t RRT::plan() {
    std::cout << "Start planning" << std::endl;

    count_type attempt = 0;
    while (attempt < constants::RRT::MAX_ATTEMPTS) {
        bool path_found = false;
        node_list_.clear();
        node_list_.push_back(start_node_);

        for (count_type i = 0; i < max_iter_; ++i) {
            std::cout << "Iteration " << i + 1 << std::endl;
            auto rand_node = get_random_node();
            auto nearest_node = find_nearest_node(node_list_, rand_node);
            real_t u = step_size_ * get_random_input(min_u_, max_u_);
            auto new_node = create_child_node(nearest_node, rand_node, u);

            if (is_collide(new_node)) {
                continue;
            }
            if (is_path_collide(nearest_node, new_node)) {
                continue;
            }

            new_node->set_parent(nearest_node);
            node_list_.push_back(new_node);

            if (check_goal(new_node)) {
                std::cout << "Finished planning" << std::endl;
                path_found = true;
                return backtrace_path(new_node);
            }
        }

        if (!path_found) {
            std::cout << "Attempt " << attempt + 1 << " failed to find path" << std::endl;
            attempt++;

            step_size_ *= 1.5f;
            success_dist_threshold_ *= 1.2f;
        }
    }

    std::cout << "Failed to find path after " << constants::RRT::MAX_ATTEMPTS << " attempts" << std::endl;

    std::vector<real_t> fallback_path;
    fallback_path.push_back(start_node_->x());
    fallback_path.push_back(start_node_->y());
    fallback_path.push_back(goal_node_->x());
    fallback_path.push_back(goal_node_->y());

    return torch::from_blob(fallback_path.data(), {2, 2}, torch::TensorOptions().dtype(get_tensor_dtype()).device(device_)).clone();
}

std::shared_ptr<Node> RRT::get_random_node() {
    if (unit_dist_(gen_) > goal_sample_rate_) {
        return std::make_shared<Node>(space_.random_x(gen_), space_.random_y(gen_));
    }
    return goal_node_;
}

std::shared_ptr<Node> RRT::find_nearest_node(
    const std::vector<std::shared_ptr<Node>>& node_list,
    const std::shared_ptr<Node>& rand_node) const
{
    index_type min_index = 0;
    real_t min_dist = std::numeric_limits<real_t>::max();

	for (size_t i = 0; i < node_list.size(); ++i) {
		const auto& node = node_list[i];
		real_t dx = rand_node->x() - node->x();
		real_t dy = rand_node->y() - node->y();
		real_t dist = std::sqrt(dx * dx + dy * dy);

		if (dist < min_dist) {
			min_dist = dist;
			min_index = i;
		}
	}

    return node_list[min_index];
}

real_t RRT::get_random_input(real_t min_u, real_t max_u) {
    std::uniform_real_distribution<real_t> dist(min_u, max_u);
    return dist(gen_);
}

std::shared_ptr<Node> RRT::create_child_node(
    const std::shared_ptr<Node>& nearest_node,
    const std::shared_ptr<Node>& rand_node,
    const real_t u) const
{
    real_t dx = rand_node->x() - nearest_node->x();
    real_t dy = rand_node->y() - nearest_node->y();
    real_t mag = std::sqrt(dx*dx + dy*dy);

    real_t direction_x = dx/mag;
    real_t direction_y = dy/mag;

    real_t new_x = nearest_node->x() + u * direction_x;
    real_t new_y = nearest_node->y() + u * direction_y;

    return std::make_shared<Node>(new_x, new_y);
}

bool RRT::is_same_node(
    const std::shared_ptr<Node>& node1,
    const std::shared_ptr<Node>& node2) const
{
    return (std::abs(node1->x() - node2->x()) < constants::EPSILON) &&
           (std::abs(node1->y() - node2->y()) < constants::EPSILON);
}

bool RRT::is_collide(const std::shared_ptr<Node>& node) const {
    auto position = torch::tensor({node->x(), node->y()}, torch::TensorOptions().dtype(get_tensor_dtype()).device(device_));

    auto distances = torch::norm(circle_obstacles_state_.slice(1, 0, 2) - position.unsqueeze(0), 2, 1);

    return torch::any(distances < circle_obstacles_state_.select(1, 2)).item<bool>();
}

bool RRT::is_path_collide(
    const std::shared_ptr<Node>& node_from,
    const std::shared_ptr<Node>& node_to) const
{
    real_t dx = node_to->x() - node_from->x();
    real_t dy = node_to->y() - node_from->y();
    real_t length = std::sqrt(dx*dx + dy*dy);

	if (length < collision_check_step_) {
		return is_collide(node_from) || is_collide(node_to);
	}

    count_type n_points = static_cast<count_type>(std::ceil(length / collision_check_step_)) + 1;
    tensor_t t = torch::linspace(0, 1, n_points, device_);
    tensor_t start_pos = torch::tensor({ node_from->x(), node_from->y() }, torch::TensorOptions().device(device_));
    tensor_t end_pos = torch::tensor({ node_to->x(), node_to->y() }, torch::TensorOptions().device(device_));

    tensor_t t_expanded = t.unsqueeze(1);
    tensor_t points = start_pos * (1 - t_expanded) + end_pos * t_expanded;
    tensor_t points_expanded = points.unsqueeze(1);

    tensor_t obstacles_pos = circle_obstacles_state_.slice(1, 0, 2).unsqueeze(0);

    tensor_t distances = torch::norm(points_expanded - obstacles_pos, 2, 2);
    tensor_t radius = circle_obstacles_state_.select(1, 2).unsqueeze(0);

    tensor_t collisions = distances <= radius;

    return collisions.any().item<bool>();
}

bool RRT::check_goal(const std::shared_ptr<Node>& node) const {
    real_t dx = node->x() - goal_node_->x();
    real_t dy = node->y() - goal_node_->y();
    real_t dist = std::sqrt(dx*dx + dy*dy);

    return (dist < success_dist_threshold_);
}

tensor_t RRT::backtrace_path(const std::shared_ptr<Node>& node) const {
    std::vector<real_t> path_points;
    auto current_node = node;

    while (current_node) {
        path_points.push_back(current_node->x());
        path_points.push_back(current_node->y());
        current_node = current_node->parent();
    }

    auto options = torch::TensorOptions()
        .dtype(get_tensor_dtype())
        .device(device_);

    auto path_tensor = torch::from_blob(path_points.data(),
                                      {static_cast<int64_t>(path_points.size() / 2), 2},
                                      options);

    return path_tensor.flip(0).clone();
}

} // namespace path_planning
