#pragma once

#include <Eigen/Dense>


namespace gym {
namespace ts {

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
            root = -(beta[1] - sqrt(beta[1] * beta[1] - 4 * beta[2] * beta[0])) / (2 * beta[2]);
        }

        return std::max(0.f, std::min(1.f, root));
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
