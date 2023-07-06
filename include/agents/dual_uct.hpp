#pragma once
#include "rand.hpp"
#include "tree_search/tree_search.hpp"
#include "tree_search/uct.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

template <typename S, typename A>
struct dual_uct_data {
    float risk_thd;
    float sample_risk_thd;
    float lambda;
    float exploration_constant;
    environment_handler<S, A>& handler;
};


/*********************************************************************
 * @brief dual uct agent
 * 
 * @tparam S State type
 * @tparam A Action type
 *********************************************************************/

template <typename S, typename A>
class dual_uct : public agent<S, A> {
    using data_t = dual_uct_data<S, A>;
    constexpr static int mode = DUAL;
private:
    int num_sim;
    float risk_thd;
    float lr;
    data_t common_data;
    tree_search<S, A, uct_state<S, A, data_t, mode>, uct_action<S, A, data_t, mode>, data_t> ts;
public:
    dual_uct(int _max_depth, int _num_sim, float _risk_thd, float _gamma, float _exploration_constant = 5.0)
    : agent<S, A>()
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , lr(0.01f)
    , common_data({_risk_thd, _risk_thd, 0, _exploration_constant, agent<S, A>::handler})
    , ts(_max_depth, _gamma, &common_data)
    {
        reset();
    }

    void play() override;
    void reset() override;

    std::string name() const override {
        return "dual_uct";
    }
};

} // namespace ts
} // namespace gym

#include "dual_uct.ipp"
