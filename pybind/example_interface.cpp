#include <pybind11/pybind11.h>
// #include "envs/investor_env.hpp"

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

struct Pet {
    Pet(const std::string &name_) : name(name_) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
    bool dead = false;
    Pet* other = nullptr;

    void kill_other() {
        if (other != nullptr) {
            other->dead = true;
        }
    }
};

std::string to_str(const Pet &a) {
    return "example.Pet named '" + a.name + "'";
}

PYBIND11_MODULE(rats, m) {
    m.def("add", &add, "A function which adds two numbers");

    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def("__repr__",
        [](const Pet &a) {
            return "<example.Pet named '" + a.name + "'>";
        })
        .def("__str__", &to_str)
        .def_readwrite("name", &Pet::name)
        .def("kill_other", &Pet::kill_other)
        .def("is_dead", [](const Pet &a) { return a.dead; })
        .def("set_other", [](Pet &a, Pet &b) { a.other = &b; });
}
