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

template<typename S_, typename A_, typename DATA_, typename V_, typename Q_>
struct action_node;

/**
 * @brief Tree search state node
 * 
 * @tparam S State type
 * @tparam A Action type
 * @tparam DATA Common data type
 * @tparam V V value type
 * @tparam Q Q value type
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
    S state;
    action_node_t *parent;
    std::vector<action_node_t> children;
    std::vector<A> actions;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    float rollout_reward = 0;
    float rollout_penalty = 0;
    bool terminal = false;
    bool leaf = true;
    size_t num_visits = 0;

    V v;
    DATA* common_data;
public:
    action_node_t* get_child(size_t a_idx) {return &children[a_idx];}
    action_node_t*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
    bool is_terminal() const {return terminal;}
    bool is_root() const {return parent == nullptr;}
    bool is_leaf() const {return children.empty();}
};

template<typename T>
struct is_state_node {
    static constexpr bool value = false;
};

template<typename... Args>
struct is_state_node<state_node<Args...>> {
    static constexpr bool value = true;
};


/**
 * @brief Tree search action node
 * 
 * @tparam S State type
 * @tparam A Action type
 * @tparam DATA Common data type
 * @tparam V V value type
 * @tparam Q Q value type
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
    A action;
    state_node_t *parent;
    std::map<S, std::unique_ptr<state_node_t>> children;
    std::map<S, size_t> child_idx;

    size_t num_visits = 0;

    float rollout_reward = 0;
    float rollout_penalty = 0;

    DATA* common_data;
    Q q;

public:
    state_node_t* get_child(S s) {return children[s].get();}
    std::unique_ptr<state_node_t>&& get_child_unique_ptr(S s) {return std::move(children[s]);}
    state_node_t*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
};

template<typename T>
struct is_action_node {
    static constexpr bool value = false;
};

template<typename... Args>
struct is_action_node<action_node<Args...>> {
    static constexpr bool value = true;
};


/*********************************************************************
 * TREE SEARCH
 *********************************************************************/
template<typename SN>
void expand_state(SN* sn) {
    sn->actions = sn->common_data->handler.possible_actions();
    sn->children.clear();
    std::transform(sn->actions.begin(), sn->actions.end(), std::back_inserter(sn->children), [sn](auto a) {
        typename SN::action_node_t an;
        an.action = a;
        an.parent = sn;
        an.common_data = sn->common_data;
        return an;
    });
}

template<typename AN>
void expand_action(
    AN* an, typename AN::S s, float r, float p, bool t
) {
    using state_node_t = AN::state_node_t;

    an->parent->leaf = false;
    an->child_idx[s] = an->children.size();

    std::unique_ptr<state_node_t>& new_sn = an->children[s] = std::make_unique<state_node_t>();
    new_sn->state = s;
    new_sn->parent = an;
    new_sn->common_data = an->common_data;
    new_sn->observed_reward = r;
    new_sn->observed_penalty = p;
    new_sn->terminal = t;
}

template<typename AN>
void full_expand_action(
    AN* an, typename AN::S s, float r, float p, bool t
) {
    using state_node_t = AN::state_node_t;

    auto& predictor = an->common_data->predictor;
    typename AN::A a1 = an->action;
    typename AN::S s0 = an->parent->state;
    predictor.add(an->parent->state, an->action, s, r, p, t);
    for (auto& [s1, signals] : predictor.predict_signals(s0, a1)) {
        if (an->children.find(s1) == an->children.end()) {
            auto [r1, p1, t1] = signals;
            expand_action(an, s1, r1, p1, t1);
        }
    }
}

/**
 * @brief Start at the root node and select actions until a leaf node is reached or a maximum depth is reached.
 * 
 * Depth is the number of steps. If depth is 0, then the root node is returned.
*/
template<typename SN, typename select_t, typename descend_cb_t>
SN* select_leaf(SN* root, bool explore = true, int max_depth = 10) {
    using state_node_t = SN;
    using action_node_t = SN::action_node_t;
    using A = SN::A;
    constexpr static auto select = select_t();
    constexpr static auto descend_cb = descend_cb_t();

    state_node_t* sn = root;

    int depth = 0;

    while (!sn->is_leaf() && depth < max_depth && !sn->is_terminal()) {
        size_t a_idx = select(sn, explore);
        action_node_t *an = sn->get_child(a_idx);
        A action = an->action;
        auto [s, r, p, t] = sn->common_data->handler.sim_action(action);
        full_expand_action(an, s, r, p, t);
        state_node_t* new_sn = an->get_child(s);
        descend_cb(sn, action, an, s, new_sn);
        depth++;
        sn = new_sn;
    }
    return sn;
}

template<typename SN, typename prop_v_t, typename prop_q_t>
void propagate(SN* leaf) {
    using state_node_t = SN;
    using action_node_t = SN::action_node_t;

    state_node_t* current_sn = leaf;
    auto common_data = current_sn->common_data;

    float disc_r = leaf->rollout_reward;
    float disc_p = leaf->rollout_penalty;

    while (!current_sn->is_root()) {
        prop_v_t()(current_sn,  disc_r, disc_p);
        disc_r = current_sn->observed_reward + common_data->gamma * disc_r;
        disc_p = current_sn->observed_penalty + disc_p;
        action_node_t* current_an = current_sn->get_parent();
        prop_q_t()(current_an, disc_r, disc_p);
        current_sn = current_an->get_parent();
    }

    prop_v_t()(current_sn, disc_r, disc_p);
}

template <typename S, typename A>
class predictor_manager {
    struct state_record {
        size_t count;
        float reward;
        float penalty;
        bool terminal;
    };

    struct action_record {
        size_t count;
        std::map<S, state_record> records;
    };

    std::map<std::pair<S, A>, action_record> records;
    std::map<S, relu_pareto_curve> value_curves;
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

    std::map<S, std::tuple<float, float, float>> predict_signals(S s0, A a1) {
        std::map<S, std::tuple<float, float, float>> signals;
        action_record& ar = records[{s0, a1}];

        for (auto& [s1, sr] : ar.records) {
            signals[s1] = {sr.reward, sr.penalty, sr.terminal};
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
 * @brief Monte carlo rollout function
 * 
 * @param sn A leaf node to rollout
 * 
 * Do a Monte Carlo rollout from the given leaf node. Update its rollout reward and penalty.
 */
template<typename SN>
void rollout(SN* sn, bool penalty = false) {
    using state_node_t = SN;
    using A = SN::A;

    if (sn->is_terminal()) {
        sn->rollout_reward = 0;
        sn->rollout_penalty = 0;
        return;
    }

    auto common_data = sn->common_data;
    auto& handler =  common_data->handler;
    int num_steps = 10;
    int num_sim = 10;
    float mean_r = 0;
    float mean_p = 0;

    state_node_t* current_sn = sn;
    common_data->handler.make_checkpoint(1);

    for (int i = 0; i < num_sim; ++i) {
        float disc_r = 0;
        float disc_p = 0;
        float gamma_pow = 1.0;
        float gammap_pow = 1.0;
        bool terminal = current_sn->is_terminal();

        while (!terminal && (num_steps--) > 0) {
            gamma_pow *= common_data->gamma;
            gammap_pow *= common_data->gammap;
            A action = handler.get_action(
                rng::unif_int(handler.num_actions())
            );
            auto [s, r, p, t] = handler.sim_action(action);
            terminal = t;
            disc_r = r + disc_r * gamma_pow;
            disc_p = p + disc_p * gammap_pow;
        }

        mean_r += disc_r;
        mean_p += disc_p;
        handler.restore_checkpoint(1);
    }

    mean_r /= num_sim;
    mean_p /= num_sim;
    current_sn->rollout_reward = mean_r;
    // to disable rollout_pen add if (common_data->gammap > 0)
    current_sn->rollout_penalty = penalty ? mean_p : 0;
}


/**
 * @brief Monte carlo rollout function
 * 
 * @param sn A leaf node to rollout
*/
template<typename SN, typename SN::A a, int N>
void constant_rollout_state(SN* sn) {
    using state_node_t = SN;

    float mean_r = 0;
    // float mean_p = 0;
    auto common_data = sn->common_data;

    common_data->handler.make_checkpoint(1);

    for (int i = 0; i < N; ++i) {
        state_node_t* current_sn = sn;
        float disc_r = 0;
        // float disc_p = 0;
        float gamma_pow = 1.0;
        bool terminal = current_sn->is_terminal();

        while (!terminal) {
            gamma_pow *= common_data->gamma;
            auto [s, r, p, t] = common_data->handler.sim_action(a);
            terminal = t;
            disc_r = r + disc_r * gamma_pow;
            // disc_p = p + disc_p * gamma_pow;
        }

        mean_r += disc_r;
        // mean_p += disc_p;
        common_data->handler.restore_checkpoint(1);
    }

    sn->rollout_reward = mean_r / N;
    float probs[] = {1.000000f,0.428571f,0.183673f,0.078717f,0.033736f,0.014458f,0.006196f,0.002656f,0.001138f,0.000488f,0.000209f,0.000090f,0.000038f,0.000016f,0.000007f,0.000003f,0.000001f,0.000001f,0.000000f,0.000000f,0.000000f};
    int s = sn->state;
    s = std::min(s, 20);
    s = std::max(s, 0);
    sn->rollout_penalty = probs[s];
}


template<typename AN, typename AN::A a, int N>
void constant_rollout_action(AN* an) {
    using A = AN::A;

    float mean_r = 0;
    float mean_p = 0;
    auto common_data = an->common_data;

    common_data->handler.make_checkpoint(1);

    for (int i = 0; i < N; ++i) {
        A initial_action = an->action;
        auto [s, r, p, t] = common_data->handler.sim_action(initial_action);
        bool terminal = t;
        float disc_r = r;
        float disc_p = p;
        float gamma_pow = 1.0;

        while (!terminal) {
            gamma_pow *= common_data->gamma;
            auto [s_, r_, p_, t_] = common_data->handler.sim_action(a);
            terminal = t_;
            disc_r = r_ + disc_r * gamma_pow;
            disc_p = p_ + disc_p * gamma_pow;
        }

        mean_r += disc_r;
        mean_p += disc_p;
        common_data->handler.restore_checkpoint(1);
    }

    an->rollout_reward = mean_r / N;
    an->rollout_penalty = mean_p / N;
}

/** Propagate by point **/
template<typename SN>
struct uct_prop_v_value {
    void operator()(SN* sn, float disc_r, float disc_p) const {
        sn->num_visits++;
        sn->v.first += (disc_r - sn->v.first) / sn->num_visits;
        sn->v.second += (disc_p - sn->v.second) / sn->num_visits;
    }
};


template<typename AN>
struct uct_prop_q_value {
    void operator()(AN* an, float disc_r, float disc_p) const {
        an->num_visits++;
        an->q.first += (disc_r - an->q.first) / an->num_visits;
        an->q.second += (disc_p - an->q.second) / an->num_visits;
    }
};

/** Propagate by probability **/
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
        sn->v.first += sn->observed_reward;
        sn->v.second += sn->observed_penalty;
    }
};

template<typename AN>
struct uct_prop_q_value_prob {
    void operator()(AN* an, float disc_r, float disc_p) const {
        an->num_visits++;
        if (an->children.size()) {
            float r = 0;
            float p = 0;
            auto probs = an->common_data->predictor.predict_probs(an->parent->state, an->action);
            for (auto& [s, child] : an->children) {
                float prob = probs[s];
                r += prob * child->v.first;
                p += prob * child->v.second;
            }
            an->q.first = r;
            an->q.second = p;
        }
        float gamma = an->common_data->gamma;
        an->q.first *= gamma;
        an->q.second *= gamma;
    }
};


} // namespace ts
} // namespace rats
