#pragma once

#include "utils/types.hpp"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <random>

namespace path_planning {

using namespace types;

class Node {
public:
    Node(real_t x, real_t y) : x_(x), y_(y), parent_(nullptr) {}

    real_t x() const { return x_; }
    real_t y() const { return y_; }

    void set_parent(const std::shared_ptr<Node>& parent) { parent_ = parent; }
    std::shared_ptr<Node> parent() const { return parent_; }

private:
    real_t x_;
    real_t y_;
    std::shared_ptr<Node> parent_;  // 부모 노드에 대한 shared_ptr
};

class RRT {
public:
    RRT(const tensor_t& start,
        const Bounds2D& space,
        const tensor_t& circle_obstacles_state,
        const tensor_t& rectangle_obstacles_state,
        const tensor_t& goal_state);

    void update(const tensor_t& start, const tensor_t& circle_obstacles_state, const tensor_t& rectangle_obstacles_state, const tensor_t& goal_state);
    tensor_t plan();
    const std::vector<std::shared_ptr<Node>>& get_node_list() const {
        return node_list_;
    }

private:
    std::shared_ptr<Node> get_random_node();
    std::shared_ptr<Node> find_nearest_node(const std::vector<std::shared_ptr<Node>>& node_list,
                                          const std::shared_ptr<Node>& rand_node) const;
    real_t get_random_input(real_t min_u, real_t max_u);
    std::shared_ptr<Node> create_child_node(const std::shared_ptr<Node>& nearest_node,
                                          const std::shared_ptr<Node>& rand_node,
                                          real_t u) const;
    bool is_same_node(const std::shared_ptr<Node>& node1,
                     const std::shared_ptr<Node>& node2) const;
    bool is_collide(const std::shared_ptr<Node>& node) const;
    bool is_path_collide(const std::shared_ptr<Node>& node_from,
                        const std::shared_ptr<Node>& node_to) const;
    bool check_goal(const std::shared_ptr<Node>& node) const;
    tensor_t backtrace_path(const std::shared_ptr<Node>& node) const;

private:
    std::shared_ptr<Node> start_node_;
    std::shared_ptr<Node> goal_node_;
    Bounds2D space_;
    tensor_t circle_obstacles_state_;
    tensor_t rectangle_obstacles_state_;
    std::vector<std::shared_ptr<Node>> node_list_;

    count_type max_iter_;
    real_t goal_sample_rate_;
    real_t min_u_;
    real_t max_u_;
    real_t success_dist_threshold_;
    real_t collision_check_step_;
    real_t step_size_;

    std::mt19937 gen_;
    std::uniform_real_distribution<real_t> unit_dist_;

    bool check_circle_collision(const tensor_t& position) const;
    bool check_circle_path_collision(const tensor_t& points) const;
    bool check_rectangle_collision(const tensor_t& position) const;
    bool check_rectangle_path_collision(const tensor_t& points) const;
};

} // namespace path_planning
