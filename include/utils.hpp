#pragma once

#include <vector>
#include <tuple>

std::tuple<size_t, float, size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd);
