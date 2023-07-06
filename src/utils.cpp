#include "utils.hpp"
#include <algorithm>
#include <tuple>

std::tuple<size_t, float, size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd) {
    if (rs.size() == 1) {
        return {0, 1.f, 0};
    }

    std::vector<std::pair<float, size_t>> action_values(rs.size());
    for (size_t i = 0; i < action_values.size(); ++i) {
        action_values[i] = {ps[i], i};
    }
    std::sort(action_values.begin(), action_values.end());
    
    std::vector<size_t> hull;
    for (size_t i = 0; i < action_values.size(); ++i) {
        auto [p, idx] = action_values[i];
        float r = rs[idx];
        while (hull.size() >= 2) {
            size_t idxa = hull[hull.size() - 2];
            size_t idxb = hull[hull.size() - 1];

            if ((r - rs[idxa]) * (ps[idxb] - ps[idxa]) < (ps[idx] - ps[idxa]) * (rs[idxb] - rs[idxa])) {
                hull.pop_back();
            }
        }

        if (hull.size() == 0 || r > rs[hull.back()]) {
            hull.push_back(idx);
        }
    }

    std::vector<float> hull_ps(hull.size());
    for (size_t i = 0; i < hull.size(); ++i) {
        hull_ps[i] = ps[hull[i]];
    }

    auto r = std::upper_bound(hull_ps.begin(), hull_ps.end(), thd);
    if (r == hull_ps.end()) {
        return {hull.back(), 1., hull.back()};
    } else if (r == hull_ps.begin()) {
        return {hull.front(), 1., hull.front()};
    } else {
        auto l = r;
        --l;
        size_t li = static_cast<size_t>(l - hull_ps.begin());
        return {hull[li], (thd - *l) / (*r - *l), hull[li + 1]};
    }
}
