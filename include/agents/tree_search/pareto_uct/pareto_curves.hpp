#pragma once

#include <Eigen/Dense>
#include "utils.hpp"


namespace gym {
namespace ts {

struct EPC {
    size_t num_samples = 0;
    std::vector<std::pair<float, float>> points;

public:
    EPC() = default;

    EPC(float r, float p)
    : num_samples(1)
    , points({{r, p}})
    {}
};


std::string to_string(const EPC& curve) {
    std::string s = "[";
    for (auto [r, p] : curve.points) {
        s += "(" + std::to_string(r) + ", " + std::to_string(p) + "), ";
    }
    s += "]";
    return s;
}


std::pair<EPC, std::vector<size_t>> convex_hull_merge(std::vector<EPC*> curves) {
    if (curves.size() == 1) {
        return *curves[0];
    }

    std::vector<std::tuple<float, float, size_t>> points;
    for (size_t i = 0; i < curves.size(); ++i) {
        for (auto [r, p] : curves[i]->points) {
            points.push_back({r, p, i});
        }
    }

    std::vector<std::tuple<float, float, size_t>> hull = upper_hull(points);
    std::vector<std::pair<float, float>> hull_points(hull.size());
    std::vector<size_t> hull_indices(hull.size());
    for (size_t i = 0; i < hull.size(); ++i) {
        auto [r, p, idx] = hull[i];
        hull_points[i] = {r, p};
        hull_indices[i] = idx;
    }
    EPC curve;
    curve.points = hull_points;
    curve.num_samples = 0;
    for (auto c : curves) {
        curve.num_samples += c->num_samples;
    }
    return curve, hull_indices;
}

std::pair<EPC, std::vector<size_t>> weighted_merge(std::vector<EPC*> curves) {
    if (curves.size() == 1) {
        return *curves[0];
    }

    std::vector<std::vector<std::pair<float, float>>> points(curves.size());
    size_t total_samples = 0;
    size_t total_points = 0;
    for (size_t i = 0; i < curves.size(); ++i) {
        total_samples += curves[i]->num_samples;
        total_points += curves[i]->points.size();
    }
    for (size_t i = 0; i < curves.size(); ++i) {
        float f = static_cast<float>(curves[i]->num_samples) / total_samples;
        for (auto [r, p] : curves[i]->points) {
            points[i].push_back({f * r, f * p});
        }
    }

    std::vector<size_t> idxs(curves.size(), 0);
    std::vector<std::pair<float, float>> merged_points(1 + total_points - curves.size());
    merged_points[0] = {0, 0};
    for (size_t i = 0; i < curves.size(); ++i) {
        merged_points[0].first += points[i][0].first;
        merged_points[0].second += points[i][0].second;
    }

    std::vector<size_t> idxs(merged_points-1);


    for (size_t i = 1; i < merged_points.size(); ++i) {
        double max_grad = -1;
        size_t max_idx = 0;

        for (size_t j = 0; j < curves.size(); ++j) {
            if (idxs[j] + 1 < points[j].size()) {
                double grad = (points[j][idxs[j] + 1].first - points[j][idxs[j]].first) / static_cast<double>(points[j][idxs[j] + 1].second - points[j][idxs[j]].second);
                if (grad > max_grad) {
                    max_grad = grad;
                    max_idx = j;
                }
            }
        }

        merged_points[i] = merged_points[i - 1];
        merged_points[i].first += points[max_idx][idxs[max_idx] + 1].first - points[max_idx][idxs[max_idx]].first;
        merged_points[i].second += points[max_idx][idxs[max_idx] + 1].second - points[max_idx][idxs[max_idx]].second;
        idxs[i] = max_idx;
    }

    EPC curve;
    curve.points = merged_points;
    curve.num_samples = total_samples;
    return curve;
}


/**
 * @brief A class to represent a quadratic approximation of a pareto curve
 * 
 * 
 * The quadratic approximation is of the form:
 * r = beta[2] * p^2 + beta[1] * p + beta[0]
 */
class quad_pareto_curve {
private:
    size_t num_samples = 0;
    Eigen::Matrix3d moments = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mean_xy = Eigen::Vector3d::Zero();

    Eigen::Vector3f beta = Eigen::Vector3f::Zero();
public:
    void update(float r, float p) {
        double _r = static_cast<double>(r);
        double _p = static_cast<double>(p);
        ++num_samples;
        Eigen::Vector3d x = {1, _p, _p*_p};
        moments += (x * x.transpose() - moments) / num_samples;
        mean_xy += (_r * x - mean_xy) / num_samples;

        auto QR = moments.fullPivHouseholderQr();
        beta = QR.solve(mean_xy).cast<float>();

        if (beta[2] > 0) {
            beta[2] = 0;
        }
        if (beta[2] > -1e-6f && beta[1] < 0) {
            beta[1] = 0;
        }
    }

    float eval(float p, float uct = 0) const {
        return static_cast<float>(beta[2] * p * p + beta[1] * p + beta[0]) + uct;
    }

    std::array<float, 3> get_beta() const {
        return {beta[0], beta[1], beta[2]};
    }

    std::pair<float, float> r_bounds() const {
        float right_r = eval(1);
        float left_r = eval(std::min(min_p() + 0.01f, 1.f));
        return {left_r, right_r};
    }

    float min_p() const {
        float root;
        if (beta[2] > -1e-6f && beta[2] < 1e-6f) {
            root = -beta[0] / beta[1];
        } else {
            root = (-beta[1] - sqrt(beta[1] * beta[1] - 4 * beta[2] * beta[0])) / (2 * beta[2]);
        }

        return std::max(0.f, std::min(1.f, root));
    }

    /**
     * @brief Get p such that c'(p) = d, where c is the curve and d is the derivative.
     * 
     * d = c'(p) = 2 * beta[2] * p + beta[1]
     * p = (d - beta[1]) / (2 * beta[2])
     * 
     * @param d derivative
     * @return float
     */
    float inverse_derivative(float d) const {
        return (d - beta[1]) / (2 * beta[2]);
    }

    float derivative(float p) const {
        return 2 * beta[2] * p + beta[1];
    }
};


std::string to_string(const quad_pareto_curve& c) {
    auto beta = c.get_beta();
    return std::to_string(beta[2]) + " * p^2 + " + std::to_string(beta[1]) + " * p + " + std::to_string(beta[0]);
}


// class softmax_pareto_curve {
// private:
//     std::vector<float> rs;
//     std::vector<float> ps;
//     float min_r = std::numeric_limits<float>::infinity();
//     float max_r = -std::numeric_limits<float>::infinity();
//     float l = 5;
// public:
//     float eval(float p, float uct = 0) const {
//         double coeff_sum = 0;
//         double sum = 0;
//         for (size_t i = 0; i < rs.size(); ++i) {
//             double coeff = exp(- l * abs(p - ps[i]));
//             coeff_sum += coeff;
//             sum += coeff * rs[i];
//         }
//         return static_cast<float>(sum / coeff_sum) + uct;
//     }

//     void update(float r, float p) {
//         rs.push_back(r);
//         ps.push_back(p);
//         min_r = std::min(min_r, r);
//         max_r = std::max(max_r, r);
//     }

//     std::pair<float, float> r_bounds() const {
//         if (rs.size() < 2) {
//             return {0, 1};
//         }
//         return {min_r, max_r};
//     }
// };

} // namespace ts
} // namespace gym
