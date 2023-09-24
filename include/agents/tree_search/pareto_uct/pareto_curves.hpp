#pragma once

#include <Eigen/Dense>
#include "utils.hpp"


namespace rats {
namespace ts {

struct outcome_support {
    // outcome index, curve vertex index
    std::vector<std::pair<size_t, size_t>> support;

    outcome_support() = default;
    outcome_support(size_t o, size_t vtx) : support({{o, vtx}}) {}

    outcome_support& operator+=(const outcome_support& other) {
        support.insert(support.end(), other.support.begin(), other.support.end());
        return *this;
    }
};

struct EPC {
    size_t num_samples = 0;
    std::vector<std::tuple<float, float, outcome_support>> points;

public:
    EPC() : num_samples(0), points({{0, 0, {}}}) {}
    EPC(EPC&&) = default;
    EPC(const EPC&) = default;
    EPC& operator=(EPC&&) = default;
    EPC& operator=(const EPC&) = default;

    std::pair<float, float> r_bounds() const {
        return {std::get<0>(points.front()), std::get<0>(points.back())};
    }

    EPC& operator+=(std::pair<float, float> p) {
        for (auto& [r, prob, supp] : points) {
            r += p.first;
            prob += p.second;
        }
        return *this;
    }

    EPC& add_and_fix(float dr, float dp) {
        for (auto& [r, prob, supp] : points) {
            r += dr;
            prob = std::max(0.f, prob + dp);
        }
        return *this;
    }

    EPC operator*=(float f) {
        for (auto& [r, prob, supp] : points) {
            r *= f;
        }
        return *this;
    }
};


std::string to_string(const EPC& curve) {
    std::string s = "[";
    for (auto [r, p, supp] : curve.points) {
        s += "(" + std::to_string(r) + ", " + std::to_string(p) + "), ";
    }
    s += "]";
    return s;
}


EPC convex_hull_merge(std::vector<EPC*> curves) {
    std::vector<std::tuple<float, float, outcome_support>> points;
    for (size_t i = 0; i < curves.size(); ++i) {
        for (size_t j = 0; j < curves[i]->points.size(); ++j) {
            auto [r, p, supp] = curves[i]->points[j];
            points.push_back({r, p, {i, j}});
        }
    }

    std::vector<std::tuple<float, float, outcome_support>> hull = upper_hull(points);
    EPC curve;
    curve.points = hull;
    return curve;
}

EPC weighted_merge(std::vector<EPC*> curves, std::vector<float> weights) {
    if (curves.size() == 1) {
        return *curves[0];
    }

    std::vector<std::vector<std::tuple<float, float, outcome_support>>> points(curves.size());
    size_t total_points = 0;
    for (size_t i = 0; i < curves.size(); ++i) {
        total_points += curves[i]->points.size();
    }
    // Estimate outcome probabilities
    for (size_t i = 0; i < curves.size(); ++i) {
        float w = weights[i];
        for (size_t j = 0; j < curves[i]->points.size(); ++j) {
            auto [r, p, supp] = curves[i]->points[j];
            points[i].push_back({w * r, w * p, {i, j}});
        }
    }

    std::vector<std::tuple<float, float, outcome_support>> merged_points(1 + total_points - curves.size());
    // Initialize first point as a sum of the first points of all curves
    merged_points[0] = {0, 0, {}};
    for (size_t i = 0; i < curves.size(); ++i) {
        std::get<0>(merged_points[0]) += std::get<0>(points[i][0]);
        std::get<1>(merged_points[0]) += std::get<1>(points[i][0]);
    }
    // Initialize indexes of the last processed points of each curve
    std::vector<size_t> point_idxs(curves.size(), 0);

    for (size_t i = 1; i < merged_points.size(); ++i) {
        // find curve with the highest gradient
        double max_grad = -std::numeric_limits<double>::infinity();
        size_t max_idx = 0;

        for (size_t j = 0; j < curves.size(); ++j) {
            if (point_idxs[j] + 1 < points[j].size()) {
                auto [r1, p1, supp1] = points[j][point_idxs[j]];
                auto [r2, p2, supp2] = points[j][point_idxs[j] + 1];
                double grad = static_cast<double>(r2 - r1) / static_cast<double>(p2 - p1);
                if (grad > max_grad) {
                    max_grad = grad;
                    max_idx = j;
                }
            }
        }

        auto& [r, p, supp] = merged_points[i] = merged_points[i - 1];
        auto& [r1, p1, supp1] = points[max_idx][point_idxs[max_idx]];
        auto& [r2, p2, supp2] = points[max_idx][point_idxs[max_idx] + 1];
        r += r2 - r1;
        p += p2 - p1;
        point_idxs[max_idx] += 1;
    }

    EPC curve;
    curve.points = merged_points;
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
} // namespace rats
