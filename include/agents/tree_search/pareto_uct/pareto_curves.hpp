#pragma once

#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include "utils.hpp"
#include "../../../string_utils.hpp"
// #include "string_utils.hpp"


namespace rats {
namespace ts {

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


/**
 * @brief Enumeration of possible outcomes of a stochastic event associated with a playable penalty threshold
 */
struct outcome_support {
    // outcome index, new_thd
    std::vector<std::pair<size_t, float>> support;

    outcome_support() = default;
    outcome_support(size_t o, float penalty) : support({{o, penalty}}) {}

    outcome_support& operator+=(const outcome_support& other) {
        support.insert(support.end(), other.support.begin(), other.support.end());
        return *this;
    }

    float penalty_at_outcome(size_t o) const {
        for (auto& [o_, penalty] : support) {
            if (o_ == o) {
                return penalty;
            }
        }
        return 0;
    }

    size_t num_outcomes() const {
        return support.size();
    }

    size_t get_first_outcome() const {
        return support.front().first;
    }
};


/**
 * @brief A class to represent an exact estimate of the pareto curve, both for state and action curves.
 * 
 * The estimate is exact in the sense that it precisely satisfies the Bellman equations.
 * 
 * The state curve 
*/
struct EPC {
    size_t num_samples = 0;
    // A vector of a points on a curve. Each point is a tuple of (reward, penalty, outcome_support)
    // If the curve is a state curve, the outcome_support is a single outcome (the played action) with a penalty threshold
    // If the curve is an action curve, the outcome_support is a set of outcomes (the possible destination states) with their penalty thresholds
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
            for (auto& [o, thd] : supp.support) {
                thd = std::max(0.f, thd + dp);
            }
        }
        return *this;
    }

    EPC operator*=(float f) {
        for (auto& [r, prob, supp] : points) {
            r *= f;
        }
        return *this;
    }

    EPC operator*=(std::pair<float, float> fs) {
        for (auto& [r, prob, supp] : points) {
            r *= fs.first;
            prob *= fs.second;
        }
        return *this;
    }

    /**
     * @brief Select a mixture of two points on the curve that satisfies the given threshold
    */
    template<bool is_state_curve = false>
    mixture select_vertex(float thd) {
        size_t idx;
        // Find idx of first point with risk > risk_thd or idx = points.size()
        for (idx = 0; idx < points.size() && std::get<1>(points[idx]) <= thd; ++idx);

        // If idx = 0, chose 0-th point for the left vertex. Otherwise, chose the last point with risk <= risk_thd
        size_t point_idx1 = idx > 0 ? idx-1 : idx;

        // If idx = points.size(), chose the last point for the right vertex. Otherwise, chose the first point with risk > risk_thd
        size_t point_idx2 = idx == points.size() ? idx-1 : idx;

        auto& [reward1, penalty1, supp1] = points[point_idx1];
        auto& [reward2, penalty2, supp2] = points[point_idx2];
        if (is_state_curve) {
            if (!supp1.num_outcomes()) {
                return mixture(0, 0, 0, 0, thd);
            }
            // Mixture of the two actions and their assigned penalty thresholds
            return mixture(supp1.get_first_outcome(), supp2.get_first_outcome(), penalty1, penalty2, thd);
        } else {
            // Mixture of the two points on the curve
            return mixture(point_idx1, point_idx2, penalty1, penalty2, thd);
        }
    }

    template<bool is_state_curve = false>
    mixture select_vertex_by_lambda(float thd, float lambda, float epsilon = 0.1) {
        std::vector<float> lagrangians(points.size());
        size_t max_idx;
        float max_lagrangian = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < points.size(); ++i) {
            auto& [reward, penalty, supp] = points[i];
            lagrangians[i] = reward - lambda * penalty;
            if (lagrangians[i] >= max_lagrangian) {
                max_lagrangian = lagrangians[i];
                max_idx = i;
            }
        }
        size_t point_idx1;
        for (point_idx1 = 0; point_idx1 < points.size()-1 && lagrangians[point_idx1] < max_lagrangian - epsilon; ++point_idx1);
        while (point_idx1 + 1 < points.size() && std::get<1>(points[point_idx1 + 1]) <= thd) {
            ++point_idx1;
        }
        size_t point_idx2;
        for (point_idx2 = points.size() - 1; point_idx2 > 0 && lagrangians[point_idx2] < max_lagrangian - epsilon; --point_idx2);
        while (point_idx2 > 0 && std::get<1>(points[point_idx2 - 1]) >= thd) {
            --point_idx2;
        }

        auto& [reward1, penalty1, supp1] = points[point_idx1];
        auto& [reward2, penalty2, supp2] = points[point_idx2];
        if (is_state_curve) {
            if (!supp1.num_outcomes()) {
                return mixture(0, 0, 0, 0, thd);
            }
            return mixture(supp1.get_first_outcome(), supp2.get_first_outcome(), penalty1, penalty2, thd);
        } else {
            return mixture(point_idx1, point_idx2, penalty1, penalty2, thd);
        }
    }

    std::pair<float, float> const reward_bounds() const {
        return {std::get<0>(points.front()), std::get<0>(points.back())};
    }

    size_t num_outcomes() const {
        return std::get<2>(points.front()).num_outcomes();
    }

    size_t get_first_outcome(size_t point_idx) const {
        return std::get<2>(points[point_idx]).get_first_outcome();
    }
};


/**
 * @brief Merge a set of action EPCs into a single state EPC by taking the convex hull of the points.
 * 
 * The incomong reward and penalty values are not considered in the merging process.
 * I.e. the outcome curve describes the reward-penalty thresholds achievable from the state.
 */
EPC convex_hull_merge(std::vector<EPC*> curves) {
    // Each point is a tuple of (reward, penalty_threshold, (action_index, penalty_threshold))
    std::vector<std::tuple<float, float, outcome_support>> points;
    for (size_t i = 0; i < curves.size(); ++i) {
        for (size_t j = 0; j < curves[i]->points.size(); ++j) {
            auto [r, p, supp] = curves[i]->points[j];
            points.push_back({r, p, {i, p}});
        }
    }

    std::random_shuffle(points.begin(), points.end());
    std::vector<std::tuple<float, float, outcome_support>> hull = upper_hull(points);
    EPC curve;
    curve.points = hull;
    return curve;
}


/**
 * @brief Merge a set of state EPCs into a single action EPC by taking the weighted average of the points.
 * 
 * The input curves are assumed to contain information about the incoming reward and penalty values.
 * Hence the outcome curve describes the expected reward and penalty values achievable after taking the action.
 */
EPC weighted_merge(std::vector<EPC*> curves, std::vector<float> weights, std::vector<size_t> state_refs) {
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
            points[i].push_back({w * r, w * p, {state_refs[i], p}});
        }
    }

    std::vector<std::tuple<float, float, outcome_support>> merged_points(1 + total_points - curves.size());
    // Initialize first point as a sum of the first points of all curves
    merged_points[0] = {0, 0, {}};
    for (size_t i = 0; i < curves.size(); ++i) {
        std::get<0>(merged_points[0]) += std::get<0>(points[i][0]);
        std::get<1>(merged_points[0]) += std::get<1>(points[i][0]);
        std::get<2>(merged_points[0]) += std::get<2>(points[i][0]);
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
        ++point_idxs[max_idx];
        supp.support[max_idx] = supp2.support.front();
    }

    EPC curve;
    curve.points = merged_points;
    return curve;
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


/**
 * @brief A general linear model class
 */
class linear_model {
private:
    size_t num_samples;
    Eigen::MatrixXf moments;
    Eigen::VectorXf mean_xy;
    Eigen::VectorXf beta;
    bool invalid_beta;

public:
    linear_model(size_t n = 1)
    : num_samples(0)
    , moments(Eigen::MatrixXf::Zero(n, n))
    , mean_xy(Eigen::VectorXf::Zero(n))
    , beta(Eigen::VectorXf::Zero(n))
    , invalid_beta(false)
    {}

    void update(float y, Eigen::VectorXf x) {
        invalid_beta = true;
        ++num_samples;
        moments += (x * x.transpose() - moments) / num_samples;
        mean_xy += (y * x - mean_xy) / num_samples;

        std::string s = "";
        for (size_t i = 0; i < x.size(); ++i) {
            s += to_string(x[i]) + ", ";
        }
        spdlog::trace("x: {}", s);
        s = "";
        for (size_t i = 0; i < beta.size(); ++i) {
            s += to_string(beta[i]) + ", ";
        }
        spdlog::trace("beta: {}", s);
    }

    void set_samples(Eigen::VectorXf y, Eigen::MatrixXf x) {
        invalid_beta = true;
        num_samples = x.rows();
        moments = x.transpose() * x / num_samples;
        mean_xy = x.transpose() * y / num_samples;
    }

    float predict(Eigen::VectorXf x) {
        if (invalid_beta) {
            auto QR = moments.fullPivHouseholderQr();
            beta = QR.solve(mean_xy);
            for (float& b : beta) b = std::max(0.f, b);
            invalid_beta = false;
        }
        return beta.dot(x);
    }
};


class relu_pareto_curve {
private:
    std::vector<std::pair<float, float>> samples;
    std::vector<float> thds;
    linear_model model;

    float min_thd = 1;
    float max_thd = 0;
public:
    relu_pareto_curve()
    : model()
    {}

    float rrelu(float p, float thd) const {
        return 1-std::max(0.f, thd - p);
    }

    Eigen::VectorXf get_features(float p) const {
        // Eigen::VectorXf x(thds.size() + 1);
        Eigen::VectorXf x(thds.size());
        for (size_t i = 0; i < thds.size(); ++i) {
            x[i] = rrelu(p, thds[i]);
        }
        // x[thds.size()] = 1;
        return x;
    }

    void update(float r, float p) {
        samples.push_back({r, p});
        if (p >= min_thd && p <= max_thd) {
            model.update(r, get_features(p));   
        } else {
            min_thd = std::min(min_thd, p);
            max_thd = std::max(max_thd, p);
            if (min_thd + 0.001 < max_thd) {
                thds = {(min_thd + max_thd) / 2, max_thd};
            } else {
                thds = {max_thd};
            }
        }
    }

    float predict(float p) {
        if (p < min_thd) {
            return 0;
        } else {
            return model.predict(get_features(p));
        }
    }

    void set_thresholds(std::vector<float> thds) {
        this->thds = thds;
        model = linear_model(thds.size() + 1);
        for (auto [r, p] : samples) {
            model.update(r, get_features(p));
        }
    }
};


// /**
//  * @brief A class to represent a quadratic approximation of a pareto curve
//  * 
//  * 
//  * The quadratic approximation is of the form:
//  * r = beta[2] * p^2 + beta[1] * p + beta[0]
//  */
// class quad_pareto_curve {
// private:
//     size_t num_samples = 0;
//     Eigen::Matrix3d moments = Eigen::Matrix3d::Zero();
//     Eigen::Vector3d mean_xy = Eigen::Vector3d::Zero();

//     Eigen::Vector3f beta = Eigen::Vector3f::Zero();
// public:
//     void update(float r, float p) {
//         double _r = static_cast<double>(r);
//         double _p = static_cast<double>(p);
//         ++num_samples;
//         Eigen::Vector3d x = {1, _p, _p*_p};
//         moments += (x * x.transpose() - moments) / num_samples;
//         mean_xy += (_r * x - mean_xy) / num_samples;

//         auto QR = moments.fullPivHouseholderQr();
//         beta = QR.solve(mean_xy).cast<float>();

//         if (beta[2] > 0) {
//             beta[2] = 0;
//         }
//         if (beta[2] > -1e-6f && beta[1] < 0) {
//             beta[1] = 0;
//         }
//     }

//     float eval(float p, float uct = 0) const {
//         return static_cast<float>(beta[2] * p * p + beta[1] * p + beta[0]) + uct;
//     }

//     std::array<float, 3> get_beta() const {
//         return {beta[0], beta[1], beta[2]};
//     }

//     std::pair<float, float> r_bounds() const {
//         float right_r = eval(1);
//         float left_r = eval(std::min(min_p() + 0.01f, 1.f));
//         return {left_r, right_r};
//     }

//     float min_p() const {
//         float root;
//         if (beta[2] > -1e-6f && beta[2] < 1e-6f) {
//             root = -beta[0] / beta[1];
//         } else {
//             root = (-beta[1] - sqrt(beta[1] * beta[1] - 4 * beta[2] * beta[0])) / (2 * beta[2]);
//         }

//         return std::max(0.f, std::min(1.f, root));
//     }

//     /**
//      * @brief Get p such that c'(p) = d, where c is the curve and d is the derivative.
//      * 
//      * d = c'(p) = 2 * beta[2] * p + beta[1]
//      * p = (d - beta[1]) / (2 * beta[2])
//      * 
//      * @param d derivative
//      * @return float
//      */
//     float inverse_derivative(float d) const {
//         return (d - beta[1]) / (2 * beta[2]);
//     }

//     float derivative(float p) const {
//         return 2 * beta[2] * p + beta[1];
//     }
// };


// std::string to_string(const quad_pareto_curve& c) {
//     auto beta = c.get_beta();
//     return to_string(beta[2]) + " * p^2 + " + to_string(beta[1]) + " * p + " + to_string(beta[0]);
// }


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

//check whther T contain a curve in the compile time

std::string to_string(const ts::EPC& curve) {
    std::string s = "[";
    for (auto& [r, p, supp] : curve.points) {
        s += "(" + to_string(r) + ", " + to_string(p) + "), ";
    }
    s += "]";
    return s;
}

} // namespace rats
