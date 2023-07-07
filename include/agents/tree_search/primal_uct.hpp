#pragma once
#include "uct.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/

/**
 * @brief Primal UCT state node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename AN, typename DATA>
struct primal_uct_state : public uct_state<S, A, AN, DATA> {    
public:
    A select_action(bool explore);
};

} // namespace ts
} // namespace gym

#include "primal_uct.ipp"
