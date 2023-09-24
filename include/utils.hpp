#pragma once

#include <vector>
#include <tuple>

std::tuple<size_t, float, size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd);
std::tuple<size_t, size_t> common_tangent(const std::vector<std::pair<float, float>>& v1, const std::vector<std::pair<float, float>>& v2);

template<typename T>
std::vector<std::tuple<float, float, T>> upper_hull(std::vector<std::tuple<float, float, T>> points) {
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
        auto& [y1, x1, t1] = a;
        auto& [y2, x2, t2] = b;
        if (x1 < x2) {
            return true;
        } else if (x1 > x2) {
            return false;
        } else {
            return y1 > y2;
        }
    });

    std::vector<std::tuple<float, float, T>> hull = {points[0]};

    for (size_t i = 1; i < points.size(); ++i) {
        auto& [y, x, t] = points[i];
        while (hull.size() >= 2) {
            auto& [y1, x1, t1] = hull[hull.size() - 1];
            auto& [y2, x2, t2] = hull[hull.size() - 2];

            if ((x - x2) * (y1 - y2) < (y - y2) * (x1 - x2)) {
                hull.pop_back();
            } else {
                break;
            }
        }
        if (std::get<0>(points[i]) > std::get<0>(hull.back())) {
            hull.push_back(points[i]);
        }
    }

    return hull;
}
