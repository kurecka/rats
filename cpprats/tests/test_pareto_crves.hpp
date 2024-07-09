#pragma once
#include "unittest.hpp"
#include "agents/tree_search/pareto_uct/pareto_curves.hpp"


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


void register_pareto_curves_tests() {
    RegisterTest(convex_merge, simple);
    RegisterTest(convex_merge, reduce_point_out_of_three);
    RegisterTest(convex_merge, reduce_two_points_to_one);
    RegisterTest(convex_merge, reduce_three_points_to_one);
    RegisterTest(convex_merge, merge_complex_curves);
}
