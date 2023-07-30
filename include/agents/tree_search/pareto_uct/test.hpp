namespace gym {
namespace ts {

#ifdef ENABLE_DOCTEST_IN_LIBRARY
#include "doctest/doctest.h"

TEST_CASE("Test quad pareto curve: constant")
{
    quad_pareto_curve c;
    c.update(1.f, 0.2f);
    c.update(1.f, 0.4f);
    c.update(1.f, 0.5f);
    auto beta = c.get_beta();
    CHECK(static_cast<double>(beta[0]) == doctest::Approx(1.0));
    CHECK(static_cast<double>(beta[1]) == doctest::Approx(0.0));
    CHECK(static_cast<double>(beta[2]) == doctest::Approx(0.0));
}

TEST_CASE("Test quad pareto curve: linear")
{
    quad_pareto_curve c;
    c.update(1, 0.2f);
    c.update(2, 0.4f);
    c.update(3, 0.6f);
    CHECK(static_cast<double>(c.eval(0.2f)) == doctest::Approx(1.0));
    CHECK(static_cast<double>(c.eval(0.4f)) == doctest::Approx(2.0));
    CHECK(static_cast<double>(c.eval(0.6f)) == doctest::Approx(3.0));
    CHECK(static_cast<double>(c.eval(0.8f)) == doctest::Approx(4.0));
}

TEST_CASE("Test quad pareto curve: quadratic")
{
    quad_pareto_curve c;
    c.update(1, 0.2f);
    c.update(2, 0.4f);
    c.update(2.2f, 0.6f);
    CHECK(static_cast<double>(c.eval(0.2f)) == doctest::Approx(1.0));
    CHECK(static_cast<double>(c.eval(0.4f)) == doctest::Approx(2.0));
    CHECK(static_cast<double>(c.eval(0.6f)) == doctest::Approx(2.2));
}

TEST_CASE("Test quad pareto curve: min linear")
{
    quad_pareto_curve c;
    c.update(1, 0.2f);
    c.update(2, 0.4f);
    c.update(3, 0.6f);
    CHECK(static_cast<double>(c.eval(c.min_p())) == doctest::Approx(0.0));
}

TEST_CASE("Test quad pareto curve: min quadratic")
{
    quad_pareto_curve c;
    c.update(1, 0.2f);
    c.update(2, 0.4f);
    c.update(2.5f, 0.6f);
    CHECK(static_cast<double>(c.eval(c.min_p())) == doctest::Approx(0.0));
}

TEST_CASE("Test common tangent - no overlap (parabola, parabola)")
{
    std::vector<std::pair<float, float>> v1 = {{1, 0.1f}, {2, 0.2f}, {1, 0.3f}};
    std::vector<std::pair<float, float>> v2 = {{1, 0.5f}, {2, 0.7f}, {1, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 1);
    CHECK(idx2 == 1);
}

TEST_CASE("Test common tangent - no overlap (parabola, line)")
{
    std::vector<std::pair<float, float>> v1 = {{1, 0.1f}, {2, 0.2f}, {1, 0.3f}};
    std::vector<std::pair<float, float>> v2 = {{1, 0.5f}, {2, 0.7f}, {3, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 1);
    CHECK(idx2 == 2);
}

TEST_CASE("Test common tangent - no overlap (parabola, line2)")
{
    std::vector<std::pair<float, float>> v1 = {{1, 0.1f}, {2, 0.2f}, {1, 0.3f}};
    std::vector<std::pair<float, float>> v2 = {{3, 0.5f}, {2, 0.7f}, {1, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 1);
    CHECK(idx2 == 0);
}

TEST_CASE("Test common tangent - overlap (parabola, parabola)")
{
    std::vector<std::pair<float, float>> v1 = {{1, 0.1f}, {1.2f, 0.2f}, {1, 0.3f}, {0.5f, 0.4f}};
    std::vector<std::pair<float, float>> v2 = {{0.1f, 0.25f}, {3, 0.5f}, {2, 0.7f}, {1, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 0);
    CHECK(idx2 == 1);
}

TEST_CASE("Test common tangent - overlap (parabola, parabola2)")
{
    std::vector<std::pair<float, float>> v1 = {{1, 0.1f}, {1.2f, 0.2f}, {1, 0.3f}, {0.5f, 0.4f}};
    std::vector<std::pair<float, float>> v2 = {{2, 0.25f}, {3, 0.5f}, {2, 0.7f}, {1, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 0);
    CHECK(idx2 == 0);
}

TEST_CASE("Test common tangent - overlap (parabola3, parabola2)")
{
    std::vector<std::pair<float, float>> v1 = {{0, 0}, {1, 0.1f}, {1.2f, 0.2f}, {1, 0.3f}, {0.5f, 0.4f}};
    std::vector<std::pair<float, float>> v2 = {{2, 0.25f}, {3, 0.5f}, {2, 0.7f}, {1, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 1);
    CHECK(idx2 == 0);
}

TEST_CASE("Test common tangent - overlap (parabola3, line)")
{
    std::vector<std::pair<float, float>> v1 = {{0, 0}, {1, 0.1f}, {1.2f, 0.2f}, {1, 0.3f}, {0.5f, 0.4f}};
    std::vector<std::pair<float, float>> v2 = {{1, 0.25f}, {1, 0.5f}, {1, 0.7f}, {1, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 2);
    CHECK(idx2 == 3);
}

TEST_CASE("Test common tangent - overlay")
{
    std::vector<std::pair<float, float>> v1 = {{1, 0.1f}, {1.2f, 0.2f}, {1, 0.3f}, {0.5f, 0.4f}};
    std::vector<std::pair<float, float>> v2 = {{2, 0}, {2, 0.5f}, {2, 0.7f}, {2, 0.9f}};

    auto [idx1, idx2] = common_tangent(v1, v2);
    CHECK(idx1 == 4);
    CHECK(idx2 == 4);
}

#endif

} // namespace ts
} // namespace gym
