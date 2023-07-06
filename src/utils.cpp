#include "utils.hpp"
#include <algorithm>
#include <tuple>


std::tuple<size_t, float, size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd) {
    if (rs.size() == 1) {
        return {0, 1.f, 0};
    }

    std::vector<std::tuple<float, float, size_t>> action_values(rs.size());
    for (size_t i = 0; i < action_values.size(); ++i) {
        std::get<0>(action_values[i]) = ps[i];
        std::get<1>(action_values[i]) = rs[i];
        std::get<2>(action_values[i]) = i;
    }
    std::sort(action_values.begin(), action_values.end());
    
    std::vector<size_t> hull;
    for (size_t i = 0; i < action_values.size(); ++i) {
        auto [p, r, idx] = action_values[i];
        while (hull.size() >= 2) {
            size_t a = hull[hull.size() - 2];
            size_t b = hull[hull.size() - 1];
            auto [pa, ra, idxa] = action_values[a];
            auto [pb, rb, idxb] = action_values[b];
            
            if ((rb - ra) * (p - pa) < (r - ra) * (pb - pa)) {
                hull.pop_back();
            }
        }
        hull.push_back(idx);
    }

    auto r = std::upper_bound(hull.begin(), hull.end(), thd);
    if (r == hull.end()) {
        return {hull.back(), 1., hull.back()};
    } else if (r == hull.begin()) {
        return {hull.front(), 1., hull.front()};
    } else {
        auto l = r;
        --l;
        float pl = std::get<0>(action_values[*l]);
        float pr = std::get<0>(action_values[*r]);
        return {*l, (thd - pl) / (pr - pl), *r};
    }
}
