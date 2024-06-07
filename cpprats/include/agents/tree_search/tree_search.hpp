#pragma once

#include "utils.hpp"
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include "pareto_uct/pareto_curves.hpp"

namespace rats {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/

/**
 * @brief Forward declaration of action_node template structure
 * 
 * @tparam S_ State type
 * @tparam A_ Action type
 * @tparam DATA_ Common data type
 * @tparam V_ V value type
 * @tparam Q_ Q value type
 */
template<typename S_, typename A_, typename DATA_, typename V_, typename Q_>
struct action_node;

/**
 * @brief Tree search state node
 * 
 * @tparam S_ State type
 * @tparam A_ Action type
 * @tparam DATA_ Common data type
 * @tparam V_ V value type
 * @tparam Q_ Q value type
 */
template<typename S_, typename A_, typename DATA_, typename V_, typename Q_>
struct state_node {
    using S = S_;
    using A = A_;
    using DATA = DATA_;
    using V = V_;
    using Q = Q_;
    using action_node_t = action_node<S, A, DATA, V, Q>;

public:
    S state;  ///< The state represented by this node
    action_node_t *parent;  ///< Pointer to the parent action node
    std::vector<action_node_t> children;  ///< List of children action nodes
    std::vector<A> actions;  ///< List of possible actions from this state
    
    int depth = 0;  ///< Depth of the node in the tree
    float observed_reward = 0;  ///< Reward observed on the incoming edge
    float observed_penalty = 0;  ///< Penalty observed on the incoming edge
    float rollout_reward = 0;  ///< Reward obtained from rollouts (rollout estimate of the state v value)
    float rollout_penalty = 0;  ///< Penalty obtained from rollouts (rollout estimate of the state v value)
    bool terminal = false;  ///< Whether the node represents a terminal state
    bool leaf = true;  ///< Whether the node is a leaf node
    size_t num_visits = 0;  ///< Number of visits to this node

    V v;  ///< V value of the node (MCTS estimate of the state v value)
    DATA* common_data;  ///< Common data shared across the tree

public:
    /**
     * @brief Get a child action node by index
     * 
     * @param a_idx Index of the action node
     * @return Pointer to the child action node
     */
    action_node_t* get_child(size_t a_idx) { return &children[a_idx]; }
    
    /**
     * @brief Get a reference to the parent action node pointer
     * 
     * @return Reference to the parent action node pointer
     */
    action_node_t*& get_parent() { return parent; }
    
    /**
     * @brief Get the number of visits to this node
     * 
     * @return Number of visits
     */
    size_t get_num_visits() const { return num_visits; }
    
    /**
     * @brief Get the depth of the node in the tree
     * 
     * @return Depth of the node
     */
    int node_depth() const { return depth; }
    
    /**
     * @brief Check if the node is a terminal node
     * 
     * @return True if the node is terminal, false otherwise
     */
    bool is_terminal() const { return terminal; }
    
    /**
     * @brief Check if the node is the root node
     * 
     * @return True if the node is the root, false otherwise
     */
    bool is_root() const { return parent == nullptr; }
    
    /**
     * @brief Check if the node is a leaf node
     * 
     * @return True if the node is a leaf, false otherwise
     */
    bool is_leaf() const { return children.empty(); }
    
    /**
     * @brief Check if the node represents a leaf state
     * 
     * @return True if the node represents a leaf state, false otherwise
     */
    bool is_leaf_state() const { return leaf; }
};


/**
 * @brief Tree search action node
 * 
 * @tparam S_ State type
 * @tparam A_ Action type
 * @tparam DATA_ Common data type
 * @tparam V_ V value type
 * @tparam Q_ Q value type
 */
template<typename S_, typename A_, typename DATA_, typename V_, typename Q_>
struct action_node {
    using S = S_;
    using A = A_;
    using DATA = DATA_;
    using V = V_;
    using Q = Q_;
    using state_node_t = state_node<S, A, DATA, V, Q>;

public:
    A action;  ///< The action represented by this node
    state_node_t *parent;  ///< Pointer to the parent state node
    std::map<S, std::unique_ptr<state_node_t>> children;  ///< Map of child state nodes
    std::map<S, size_t> child_idx;  ///< Index of child nodes

    size_t num_visits = 0;  ///< Number of visits to this node

    float rollout_reward = 0;  ///< Reward obtained from rollouts (rollout estimate of the action q value)
    float rollout_penalty = 0;  ///< Penalty obtained from rollouts (rollout estimate of the action q value)

    DATA* common_data;  ///< Common data shared across the tree
    Q q;  ///< Q value of the node (MCTS estimate of the action q value)

public:
    /**
     * @brief Get a child state node by state
     * 
     * @param s State of the child node
     * @return Pointer to the child state node
     */
    state_node_t* get_child(S s) { return children[s].get(); }
    
    /**
     * @brief Get a unique pointer to a child state node by state
     * 
     * @param s State of the child node
     * @return Unique pointer to the child state node
     */
    std::unique_ptr<state_node_t>&& get_child_unique_ptr(S s) { return std::move(children[s]); }
    
    /**
     * @brief Get a reference to the parent state node pointer
     * 
     * @return Reference to the parent state node pointer
     */
    state_node_t*& get_parent() { return parent; }
    
    /**
     * @brief Get the number of visits to this node
     * 
     * @return Number of visits
     */
    size_t get_num_visits() const { return num_visits; }
};


/*********************************************************************
 * TREE SEARCH
 *********************************************************************/

/**
 * @brief Expand a state node by generating its children action nodes
 * 
 * @tparam SN State node type
 * @param sn Pointer to the state node to expand
 */
template<typename SN>
void expand_state(SN* sn) {
    // Get possible actions player can take in this state
    sn->actions = sn->common_data->handler.possible_actions();
    // Clear the children list
    sn->children.clear();
    // Generate action nodes for each action
    std::transform(sn->actions.begin(), sn->actions.end(), std::back_inserter(sn->children), [sn](auto a) {
        typename SN::action_node_t an;
        an.action = a;
        an.parent = sn;
        an.common_data = sn->common_data;
        return an;
    });
}


/**
 * @brief Expand an action node by adding a child state node.
 * 
 * @tparam AN Action node type
 * @param an Pointer to the action node to expand
 * @param s State to transition to
 * @param r Reward observed on the transition
 * @param p Penalty observed on the transition
 * @param t Whether the state is terminal
 * @param future_r Future reward predicted by the predictor
 * @param future_p Future penalty predicted by the predictor
 */
template<typename AN>
void expand_action(
    AN* an, typename AN::S s, float r, float p, bool t, float future_r, float future_p
) {
    using state_node_t = typename AN::state_node_t;

    an->parent->leaf = false;
    an->child_idx[s] = an->children.size();

    std::unique_ptr<state_node_t>& new_sn = an->children[s] = std::make_unique<state_node_t>();
    new_sn->state = s;
    new_sn->parent = an;
    new_sn->common_data = an->common_data;
    new_sn->observed_reward = r;
    new_sn->observed_penalty = p;
    new_sn->rollout_reward = future_r;
    new_sn->rollout_penalty = future_p;
    new_sn->terminal = t;
    new_sn->depth = an->parent->depth + 1;
}


/**
 * @brief Fully expand an action node by adding all successor states known to the predictor.
 * 
 * @tparam AN Action node type
 * @param an Pointer to the action node to expand
 * @param s State to transition to
 * @param r Reward observed on the transition
 * @param p Penalty observed on the transition
 * @param t Whether the state is terminal 
*/
template<typename AN>
void full_expand_action(AN* an) {
    using state_node_t = typename AN::state_node_t;

    auto& predictor = an->common_data->predictor;
    typename AN::A a1 = an->action;
    typename AN::S s0 = an->parent->state;
    for (auto& [s1, signals] : predictor.predict_signals(s0, a1)) {
        if (an->children.find(s1) == an->children.end()) {
            auto [r1, p1, t1, fr1, fp1] = signals;
            expand_action(an, s1, r1, p1, t1, fr1, fp1);
        }
    }
}

/**
 * @brief Update the predictor with the observed transition.
 * 
 * @tparam AN Action node type
 * @param an Pointer to the action node
 * @param s State to transition to
 * @param r Reward observed on the transition
 * @param p Penalty observed on the transition
 * @param t Whether the state is terminal
 */
template<typename SN>
void update_predictor(SN* sn, typename SN::A a, typename SN::S s, float r, float p, bool t) {
    sn->common_data->predictor.add(sn->state, a, s, r, p, t);
}

/**
 * @brief Start at the root node and select actions until a leaf node is reached or a maximum depth is reached.
 * Fully expand each node along the way.
 * Update the predictor with the observed transitions.
 * 
 * @tparam SN State node type
 * 
 * @param root Pointer to the root node
 * @param explore Whether to explore the tree
 * @param max_depth Maximum depth to explore (0 for root node, 1 for root's children, etc.)
*/
template<typename SN, typename select_t, typename descend_cb_t>
SN* select_leaf(SN* root, bool explore = true, int max_depth = 10) {
    using state_node_t = SN;
    using action_node_t = typename SN::action_node_t;
    using A = typename SN::A;
    constexpr static auto select = select_t();
    constexpr static auto descend_cb = descend_cb_t();

    state_node_t* sn = root;
    auto& handler = sn->common_data->handler;

    int depth = 0;

    while (!sn->is_leaf() && !handler.is_over() && !sn->is_terminal() && depth < max_depth) {
        // Select an action
        size_t a_idx = select(sn, explore);
        action_node_t *an = sn->get_child(a_idx);
        A action = an->action;
        // Simulate the action
        auto [s, r, p, t] = handler.sim_action(action);
        // Update the predictor
        update_predictor(sn, action, s, r, p, t);
        // Fully expand the action node
        full_expand_action(an);
        // Descend to the next state node
        state_node_t* new_sn = an->get_child(s);
        descend_cb(sn, action, an, s, new_sn);
        depth++;
        sn = new_sn;
    }
    return sn;
}


/**
 * @brief Propagate the observed rewards and penalties up the tree.
 * 
 * @tparam SN State node type
 * 
 * @param leaf Pointer to the leaf node to propagate from
 */
template<typename SN, typename prop_v_t, typename prop_q_t>
void propagate(SN* leaf) {
    using state_node_t = SN;
    using action_node_t = typename SN::action_node_t;

    state_node_t* current_sn = leaf;
    auto common_data = current_sn->common_data;

    // Initialize the discounted rewards and penalties by the estimated V values
    float disc_r = leaf->rollout_reward;
    float disc_p = leaf->rollout_penalty;

    while (!current_sn->is_root()) {
        // Update the state V value
        prop_v_t()(current_sn,  disc_r, disc_p);
        disc_r = current_sn->observed_reward + common_data->gamma * disc_r;
        disc_p = current_sn->observed_penalty + common_data->gammap * disc_p;
        action_node_t* current_an = current_sn->get_parent();
        // Update the action Q value
        prop_q_t()(current_an, disc_r, disc_p);
        current_sn = current_an->get_parent();
    }

    // Update the root state V value
    prop_v_t()(current_sn, disc_r, disc_p);
}

template <typename S, typename A>
class predictor_manager {
    struct state_record {
        size_t count;
        float reward;
        float penalty;
        bool terminal;
        float future_reward;
        float future_penalty;
    };

    struct action_record {
        size_t count;
        std::map<S, state_record> records;
    };

    std::map<std::pair<S, A>, action_record> records;
    std::map<S, relu_pareto_curve> value_curves;

    std::map<S, float> future_rewards;
    std::map<S, float> future_penalties;
public:
    void add(S s0, A a1, S s1, float r, float p, bool t) {
        action_record& ar = records[{s0, a1}];
        ++ar.count;
        state_record& sr = ar.records[s1];
        ++sr.count;
        sr.reward = r;
        sr.penalty = p;
        sr.terminal = t;
    }

    void add_future(S s, float fr, float fp) {
        future_rewards[s] = fr;
        future_penalties[s] = fp;
    }

    void add_value(float r, float p, S s) {
        value_curves[s].update(r, p);
    }

    float predict_value(float p, S s) {
        auto it = value_curves.find(s);
        if (it == value_curves.end()) {
            return 0;
        } else {
            return it->second.predict(p);
        }
    }

    std::map<S, float> predict_probs(S s0, A a1) {
        std::map<S, float> probs;
        action_record& ar = records[{s0, a1}];

        for (auto& [s1, sr] : ar.records) {
            probs[s1] = static_cast<float>(sr.count) / ar.count;
        }
        return probs;
    }

    std::map<S, std::tuple<float, float, float, float, float>> predict_signals(S s0, A a1) {
        std::map<S, std::tuple<float, float, float, float, float>> signals;
        action_record& ar = records[{s0, a1}];

        for (auto& [s1, sr] : ar.records) {
            signals[s1] = {sr.reward, sr.penalty, sr.terminal, future_rewards[s1], future_penalties[s1]};
        }
        return signals;
    }
};


/*********************************************************************
 * CALLBACKS AND PLUGIN FUNCTIONS
 *********************************************************************/

using point_value = std::pair<float, float>;

template<typename... Args>
struct void_fn {
    void operator()(Args...) const {}
};


/**
 * @brief Estimate the V value of a state node by sampling rollouts
 * 
 * @tparam SN State node type
 * 
 * @param sn Pointer to the state node to estimate
 */
template<typename SN>
std::pair<float, float> v_rollout_sample(SN* sn) {
    using state_node_t = SN;
    using A = typename SN::A;

    auto common_data = sn->common_data;
    auto& handler =  common_data->handler;
    int num_steps = 10;

    float disc_r = 0;
    float disc_p = 0;
    float gamma_pow = 1.0;
    float gammap_pow = 1.0;
    bool terminal = sn->is_terminal();

    while (!terminal && (num_steps--) > 0) {
        A action = handler.get_action(
            rng::unif_int(handler.num_actions())
        );
        auto [s, r, p, t] = handler.sim_action(action);
        terminal = t;
        disc_r += r * gamma_pow;
        disc_p += p * gammap_pow;
        gamma_pow *= common_data->gamma;
        gammap_pow *= common_data->gammap;
    }

    return {disc_r, disc_p};
}


/**
 * @brief Monte carlo rollout function
 * Perform a Monte Carlo rollout from the given leaf node. Update its rollout reward and penalty.
 * Store the estimated future rewards and penalties in the predictor.
 * 
 * @param sn A leaf node to rollout
 * @param penalty Whether to estimate penalties
 * @param num_sim Number of simulations to perform
 */
template<typename SN>
void rollout(SN* sn, bool penalty = false, int num_sim = 10) {
    float mean_r = 0;
    float mean_p = 0;

    auto* common_data = sn->common_data;
    auto& handler = common_data->handler;
    handler.make_checkpoint(1);

    for (int i = 0; i < num_sim; ++i) {
        auto [disc_r, disc_p] = v_rollout_sample(sn);
        mean_r += disc_r;
        mean_p += disc_p;
        handler.restore_checkpoint(1);
    }

    mean_r /= num_sim;
    mean_p /= num_sim;
    sn->rollout_reward = mean_r;
    sn->rollout_penalty = penalty ? mean_p : 0;

    common_data->predictor.add_future(sn->state, mean_r, mean_p);
}

/** Propagate by point **/

/**
 * @brief Update the MCTS V value estimate in a MCTS tree based on a discounted reward sample.
 * The resulting value is the average of the sampled rewards (respectively penalties).
 * 
 * @tparam SN State node type
 * 
 * @param leaf Pointer to the leaf node to propagate from
 * @param disc_r Discounted reward
 * @param disc_p Discounted penalty
 */
template<typename SN>
struct uct_prop_v_value {
    void operator()(SN* sn, float disc_r, float disc_p) const {
        sn->num_visits++;
        sn->v.first += (disc_r - sn->v.first) / sn->num_visits;
        sn->v.second += (disc_p - sn->v.second) / sn->num_visits;
    }
};


/**
 * @brief Update the MCTS Q value estimate in a MCTS tree based on a discounted reward sample.
 * The resulting value is the average of the sampled rewards (respectively penalties).
 * 
 * @tparam AN Action node type
 * 
 * @param an Pointer to the action node to propagate from
 * @param disc_r Discounted reward
 * @param disc_p Discounted penalty
 */
template<typename AN>
struct uct_prop_q_value {
    void operator()(AN* an, float disc_r, float disc_p) const {
        an->num_visits++;
        an->q.first += (disc_r - an->q.first) / an->num_visits;
        an->q.second += (disc_p - an->q.second) / an->num_visits;
    }
};

/** Propagate by probability **/

/**
 * @brief Update the MCTS V value estimate in a MCTS tree based on a discounted reward sample.
 * The resulting value is a sum of the child Q values weighted by the visitation probabilities.
 * 
 * @tparam SN State node type
 * 
 * @param sn Pointer to the state node to propagate from
 * @param disc_r Discounted reward
 * @param disc_p Discounted penalty
 */
template<typename SN>
struct uct_prop_v_value_prob {
    void operator()(SN* sn, float disc_r, float disc_p) const {
        sn->num_visits++;
        if (sn->children.size()) {
            float r = 0;
            float p = 0;
            for (auto& child : sn->children) {
                float prob = child.num_visits / static_cast<float>(sn->num_visits);
                r += prob * child.q.first;
                p += prob * child.q.second;
            }
            sn->v.first = r;
            sn->v.second = p;
        }
    }
};


/**
 * @brief Update the MCTS Q value estimate in a MCTS tree based on a discounted reward sample.
 * The resulting value is a sum of the child V values weighted by the visitation probabilities.
 * 
 * @tparam AN Action node type
 * 
 * @param an Pointer to the action node to propagate from
 * @param disc_r Discounted reward
 * @param disc_p Discounted penalty
 */
template<typename AN>
struct uct_prop_q_value_prob {
    void operator()(AN* an, float disc_r, float disc_p) const {
        an->num_visits++;
        float gamma = an->common_data->gamma;
        float gammap = an->common_data->gammap;
        if (an->children.size()) {
            float r = 0;
            float p = 0;
            auto probs = an->common_data->predictor.predict_probs(an->parent->state, an->action);
            for (auto& [s, child] : an->children) {
                float prob = probs[s];
                r += prob * (child->v.first * gamma + child->observed_reward);
                p += prob * (child->v.second * gammap + child->observed_penalty);
            }
            an->q.first = r;
            an->q.second = p;
        }
    }
};


} // namespace ts
} // namespace rats
