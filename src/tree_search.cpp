#include "tree_search.hpp"

template <typename T>
struct node {
    T data;
    std::vector<node<T>> children;
};

template <typename T>
class search_tree {
    private:
        node<T> root;
    public:

};