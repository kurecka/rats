#pragma once
#include "unittest.hpp"
#include "agents/tree_search/tuct/pareto_curves.hpp"

#include <algorithm>


rats::ts::EPC get_curve(std::vector<std::pair<float, float>>&& points) {
    std::vector<rats::ts::EPC> leaves;
    for (auto& [r, p] : points) {
        rats::ts::EPC leaf;
        leaf += {r, p};
        leaves.push_back(leaf);
    }
    std::vector<rats::ts::EPC*> lead_ptrs;
    for (auto& leaf : leaves) {
        lead_ptrs.push_back(&leaf);
    }
    return convex_hull_merge(lead_ptrs);
}


UTest(convex_merge, simple) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.2};
    rats::ts::EPC leaf2;
    leaf2 += {1, 0.1};
    rats::ts::EPC leaf3;
    leaf3 += {2.5, 0.4};

    std::vector leaves = {&leaf1, &leaf2, &leaf3};
    rats::ts::EPC merged = convex_hull_merge(leaves);

    std::vector<std::pair<float, float>> expected_vals = {{1, 0.1}, {2, 0.2}, {2.5, 0.4}};
    std::vector<size_t> expected_action = {1, 0, 2};

    ExpectEQ(merged.points.size(), expected_vals.size());

    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
        ExpectEQ(supp.support[0].first, expected_action[i]);
    }
}


UTest(convex_merge, reduce_point_out_of_three) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.1};
    rats::ts::EPC leaf2;
    leaf2 += {1, 0.2};
    rats::ts::EPC leaf3;
    leaf3 += {2.5, 0.4};

    std::vector leaves = {&leaf1, &leaf2, &leaf3};
    rats::ts::EPC merged = convex_hull_merge(leaves);

    std::vector<std::pair<float, float>> expected_vals = {{2, 0.1}, {2.5, 0.4}};
    std::vector<size_t> expected_action = {0, 2};

    ExpectEQ(merged.points.size(), expected_vals.size());

    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
        ExpectEQ(supp.support[0].first, expected_action[i]);
    }
}


UTest(convex_merge, reduce_two_points_to_one) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.1};
    rats::ts::EPC leaf2;
    leaf2 += {2, 0.2};

    std::vector leaves = {&leaf1, &leaf2};
    rats::ts::EPC merged = convex_hull_merge(leaves);

    std::vector<std::pair<float, float>> expected_vals = {{2, 0.1}};
    std::vector<size_t> expected_action = {0};

    ExpectEQ(merged.points.size(), expected_vals.size());

    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
        ExpectEQ(supp.support[0].first, expected_action[i]);
    }
}


UTest(convex_merge, reduce_three_points_to_one) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.2};
    rats::ts::EPC leaf2;
    leaf2 += {2, 0.1};
    rats::ts::EPC leaf3;
    leaf3 += {1.5, 0.4};

    std::vector leaves = {&leaf1, &leaf2, &leaf3};
    rats::ts::EPC merged = convex_hull_merge(leaves);

    std::vector<std::pair<float, float>> expected_vals = {{2, 0.1}};
    std::vector<size_t> expected_action = {1};


    ExpectEQ(merged.points.size(), expected_vals.size());

    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
        ExpectEQ(supp.support[0].first, expected_action[i]);
    }
}


UTest(convex_merge, merge_complex_curves) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.3};
    rats::ts::EPC leaf2;
    leaf2 += {1, 0.1};
    rats::ts::EPC leaf3;
    leaf3 += {2.5, 0.4};
    rats::ts::EPC leaf4;
    leaf4 += {2, 0.2};

    std::vector curves = {&leaf1, &leaf2};
    rats::ts::EPC merged1 = convex_hull_merge(curves);
    curves = {&leaf3, &leaf4};
    rats::ts::EPC merged2 = convex_hull_merge(curves);
    curves = {&merged2, &merged1};
    rats::ts::EPC merged = convex_hull_merge(curves);

    std::vector<std::pair<float, float>> expected_vals = {{1, 0.1}, {2, 0.2}, {2.5, 0.4}};
    std::vector<size_t> expected_action = {1, 0, 0};

    ExpectEQ(merged.points.size(), expected_action.size());
    for (int i = 0; i < expected_action.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
        ExpectEQ(supp.support[0].first, expected_action[i]);
    }
}


UTest(convex_merge, remove_duplicate_point) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.3};
    rats::ts::EPC leaf2;
    leaf2 += {2, 0.3};
    rats::ts::EPC leaf3;
    leaf3 += {3, 0.35};

    std::vector curves = {&leaf1, &leaf2, &leaf3};
    rats::ts::EPC merged = convex_hull_merge(curves);

    std::vector<std::pair<float, float>> expected_vals = {{2, 0.3}, {3, 0.35}};

    ExpectEQ(merged.points.size(), expected_vals.size());
    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
    }
}


UTest(convex_merge, single_point) {
    rats::ts::EPC leaf1;
    leaf1 += {2, 0.3};

    std::vector curves = {&leaf1};
    rats::ts::EPC merged = convex_hull_merge(curves);

    std::vector<std::pair<float, float>> expected_vals = {{2, 0.3}};
    std::vector<size_t> expected_action = {0};

    ExpectEQ(merged.points.size(), expected_vals.size());
    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), static_cast<size_t>(1));
        ExpectEQ(supp.support[0].first, expected_action[i]);
    }
}


UTest(upper_hull, error_in_manhattan) {
    std::vector<std::tuple<float, float, size_t>> points = {{0, 0, 0}, {0, 0, 1}, {0.999, 1, 2}, {0.999, 1, 3}};
    std::vector<int> perm = {0, 1, 2, 3};

    while (std::next_permutation(perm.begin(), perm.end())) {
        std::vector<std::tuple<float, float, size_t>> perm_points(4);
        for (int i = 0; i < 4; i++) {
            perm_points[i] = points[perm[i]];
        }

        std::vector<std::tuple<float, float, size_t>> hull = rats::ts::upper_hull(perm_points);
        std::vector<std::pair<float, float>> expected_vals = {{0, 0}, {0.999, 1}};

        ExpectEQ(hull.size(), expected_vals.size());
        for (int i = 0; i < expected_vals.size(); i++) {
            auto& [r, p, _] = hull[i];
            AreClose(r, expected_vals[i].first);
            AreClose(p, expected_vals[i].second);
        }
    }
}




UTest(weighted_merge, simple) {
    rats::ts::EPC curve1 = get_curve({{1, 1}});
    rats::ts::EPC curve2 = get_curve({{0, 0}, {2, 2}, {3, 4}});

    std::vector curve_ptrs = {&curve1, &curve2};
    std::vector weights = {0.5f, 0.5f};
    std::vector<size_t> state_refs = {0, 1};

    rats::ts::EPC merged = weighted_merge(curve_ptrs, weights, state_refs);

    std::vector<std::pair<float, float>> expected_vals = {{0.5, 0.5}, {1.5, 1.5}, {2., 2.5}};
    std::vector<std::vector<size_t>> expected_state_refs = {{0, 1}, {0, 1}, {0, 1}};

    ExpectEQ(merged.points.size(), expected_vals.size());
    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), expected_state_refs[i].size());
        for (int j = 0; j < expected_state_refs[i].size(); j++) {
            ExpectEQ(supp.support[j].first, expected_state_refs[i][j]);
        }
    }
}


UTest(weighted_merge, single_points) {
    rats::ts::EPC curve1 = get_curve({{1, 1}});
    rats::ts::EPC curve2 = get_curve({{1, 1}});
    rats::ts::EPC curve3 = get_curve({{1, 1}});

    std::vector curve_ptrs = {&curve1, &curve2, &curve3};
    std::vector weights = {0.2f, 0.3f, 0.5f};
    std::vector<size_t> state_refs = {0, 2, 1};

    rats::ts::EPC merged = weighted_merge(curve_ptrs, weights, state_refs);

    std::vector<std::pair<float, float>> expected_vals = {{1, 1}};
    std::vector<std::vector<size_t>> expected_state_refs = {{0, 2, 1}};

    ExpectEQ(merged.points.size(), expected_vals.size());
    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(r, expected_vals[i].first);
        AreClose(p, expected_vals[i].second);
        ExpectEQ(supp.support.size(), expected_state_refs[i].size());
        for (int j = 0; j < expected_state_refs[i].size(); j++) {
            ExpectEQ(supp.support[j].first, expected_state_refs[i][j]);
        }
    }
}


UTest(weighted_merge, complex_curves) {
    rats::ts::EPC curve1 = get_curve({{1, 1}, {4, 2}, {5, 3}, {6, 5}});
    rats::ts::EPC curve2 = get_curve({{1, 1}, {3, 2}, {4.5, 3}, {5, 5}});
    rats::ts::EPC curve3 = get_curve({{1, 1}, {5, 2}});

    std::vector curve_ptrs = {&curve1, &curve2, &curve3};
    std::vector weights = {0.4f, 0.4f, 0.2f};
    std::vector<size_t> state_refs = {0, 1, 2};

    rats::ts::EPC merged = weighted_merge(curve_ptrs, weights, state_refs);

    std::vector<std::vector<float>> expected_thds = {{1, 1, 1}, {1, 1, 2}, {2, 1, 2}, {2, 2, 2}, {2, 3, 2}, {3, 3, 2}, {5, 3, 2}, {5, 5, 2}};
    std::vector<float> expected_vals;
    for (auto& thds : expected_thds) {
        float thd = 0;
        for (int i = 0; i < thds.size(); i++) {
            thd += thds[i] * weights[i];
        }
        expected_vals.push_back(thd);
    }

    ExpectEQ(merged.points.size(), expected_vals.size());
    for (int i = 0; i < expected_vals.size(); i++) {
        auto& [r, p, supp] = merged.points[i];
        AreClose(p, expected_vals[i]);
        ExpectEQ(supp.support.size(), static_cast<size_t>(3));
        for (int j = 0; j < 3; j++) {
            ExpectEQ(supp.support[j].first, state_refs[j]);
        }
    }
}


void register_pareto_curves_tests() {
    RegisterTest(convex_merge, simple);
    RegisterTest(convex_merge, reduce_point_out_of_three);
    RegisterTest(convex_merge, reduce_two_points_to_one);
    RegisterTest(convex_merge, reduce_three_points_to_one);
    RegisterTest(convex_merge, merge_complex_curves);
    RegisterTest(convex_merge, remove_duplicate_point);
    RegisterTest(upper_hull, error_in_manhattan);

    RegisterTest(weighted_merge, simple);
    RegisterTest(weighted_merge, single_points);
    RegisterTest(weighted_merge, complex_curves);

}
