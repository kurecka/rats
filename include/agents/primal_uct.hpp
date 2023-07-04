#pragma once
#include "tree_search/tree_search.hpp"
#include "tree_search/uct.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

template <typename S, typename A>
struct uct_data {
    float risk_thd;
    environment_handler<S, A>& handler;
};


/*********************************************************************
 * @brief primal uct agent
 * 
 * @tparam S State type
 * @tparam A Action type
 *********************************************************************/

template <typename S, typename A>
class primal_uct : public agent<S, A> {
    using data_t = uct_data<S, A>;
    constexpr static int mode = PRIMAL_DETERMINISTIC;
private:
    int num_sim;
    float risk_thd;
    data_t common_data;
    tree_search<S, A, uct_state<S, A, data_t, mode>, uct_action<S, A, data_t, mode>, data_t> ts;
public:
    primal_uct(int _max_depth, int _num_sim, float _risk_thd, float _gamma)
    : agent<S, A>()
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , common_data({_risk_thd, agent<S, A>::handler})
    , ts(_max_depth, _risk_thd, _gamma, &common_data)
    {
        reset();
    }

    void play() override;
    void reset() override;

    std::string name() const override {
        return "primal_uct";
    }
};

} // namespace ts
} // namespace gym

#include "primal_uct.ipp"
