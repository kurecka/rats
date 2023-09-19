#pragma once

#include <vector>
#include <tuple>

std::tuple<size_t, float, size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd);
std::tuple<size_t, size_t> common_tangent(const std::vector<std::pair<float, float>>& v1, const std::vector<std::pair<float, float>>& v2);

template<typename T>
std::vector<std::tuple<float, float, T>> upper_hull(std::vector<std::tuple<float, float, T>> points) {
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
        auto [x1, y1, t1] = a;
        auto [x2, y2, t2] = b;
        if (y1 < y2) {
            return true;
        } else if (y1 > y2) {
            return false;
        } else {
            return x1 > x2;
        }
    });

    std::vector<std::tuple<float, float, T>> hull = {points[0]};

    for (size_t i = 1; i < points.size(); ++i) {
        auto [x, y, t] = points[i];
        while (hull.size() >= 2) {
            auto [x1, y1, t1] = hull[hull.size() - 1];
            auto [x2, y2, t2] = hull[hull.size() - 2];

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
