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


std::tuple<size_t, size_t> common_tangent(const std::vector<std::pair<float, float>>& v1, const std::vector<std::pair<float, float>>& v2) {
    size_t idx1 = 0;
    size_t idx2 = v2.size() - 1;

    size_t* idx = &idx1;
    auto* v = &v1;

    size_t without_change = 0;
    size_t total_trials = 0;
    do {
        bool change = false;

        float slope = (v1[idx1].first - v2[idx2].first) / (v1[idx1].second - v2[idx2].second);
        if (*idx + 1 < v->size() && (*v)[*idx + 1].first - (*v)[*idx].first > slope * ((*v)[*idx + 1].second - (*v)[*idx].second)) {
            ++*idx;
            change = true;
        }
        if (*idx >= 1 && ((*v)[*idx].first - (*v)[*idx - 1].first) < slope * ((*v)[*idx].second - (*v)[*idx - 1].second)) {
            --*idx;
            change = true;
        }

        idx = idx == &idx1 ? &idx2 : &idx1;
        v = v == &v1 ? &v2 : &v1;
        without_change = change ? 0 : without_change + 1;
        ++total_trials;
    } while (total_trials < v1.size()*v2.size() + 1 && without_change < 2);

    if (total_trials > v1.size()*v2.size()) {
        return {v1.size(), v2.size()};
    }

    return {idx1, idx2};
}
