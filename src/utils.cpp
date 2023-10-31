#include "utils.hpp"
#include <algorithm>
#include <tuple>


mixture<size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd) {
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
            } else {
                break;
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
        return {hull.back(), hull.back(), 0., 1., true};
    } else if (r == hull_ps.begin()) {
        return {hull.front(), hull.front(), 1., 0., true};
    } else {
        auto l = r;
        --l;
        size_t li = static_cast<size_t>(l - hull_ps.begin());
        float p2 = (thd - *l) / (*r - *l);
        return {hull[li], hull[li + 1], 1 - p2, p2, false};
    }
}
