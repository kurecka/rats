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

template<typename T1, typename T2>
std::string to_string(const std::map<T2, T2>& p) {
    std::string str = "{";

    for ( const auto &[k, v] : p ) {
        str += to_string(k) + " : " + to_string(v) + ", ";
    }

    return str + "}";
}

// temporary solution for MANHATTAN env, add a template or something later
std::string to_string(const std::tuple<std::string, std::map<std::string, float>, bool>& p) {
    std::string str = "( ";
    str += std::get<0>(p) + ", ";

    for ( const auto &[k, v] : std::get<1>(p) ) {
        str += k + " : " + to_string(v) + ", ";
    }

    str += ", ";
    str += std::get<2>(p);
    str += " )";

    return str;
}


}; // end namespace rats
