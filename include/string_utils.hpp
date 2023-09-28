#pragma once
#include <string>


namespace rats {

template<typename T>
std::string to_string(T t) {
    return std::to_string(t);
}

template<typename T1, typename T2>
std::string to_string(const std::pair<T1, T2>& p) {
    return "(" + to_string(p.first) + "," + to_string(p.second) + ")";
}

}; // end namespace rats
