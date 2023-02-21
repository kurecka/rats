#pragma once

#include <iostream>
#include <map>

class _logger {
public:
    _logger() = default;
    ~_logger() = default;

    enum level {
        DEBUG = 0,
        INFO = 1,
        WARN = 2,
        ERROR = 3,
        FATAL = 4
    };

    std::string level_names[5] = {
        "DEBUG",
        " INFO",
        " WARN",
        "ERROR",
        "FATAL"
    };

    std::map<std::string, int> level_map = {
        {"DEBUG", DEBUG},
        {"INFO", INFO},
        {"WARN", WARN},
        {"ERROR", ERROR},
        {"FATAL", FATAL}
    };

    int level = DEBUG;

    void set_level(int level) {
        this->level = level;
    }

    void set_level(std::string level) {
        this->level = level_map[level];
    }

    void log(const std::string& msg, int level = INFO) {
        if (level < this->level) {
            return;
        }
        std::cout << "[" << level_names[level] << "] " << msg << std::endl;
    }

    void log(std::string&& msg, int level = INFO) {
        if (level < this->level) {
            return;
        }
        std::cout << "[" << level_names[level] << "] " << msg << std::endl;
    }

    void debug(const std::string& msg) {
        log(msg, DEBUG);
    }

    void debug(std::string&& msg) {
        log(msg, DEBUG);
    }

    void info(const std::string& msg) {
        log(msg, INFO);
    }

    void info(std::string&& msg) {
        log(msg, INFO);
    }

    void warn(const std::string& msg) {
        log(msg, WARN);
    }

    void warn(std::string&& msg) {
        log(msg, WARN);
    }

    void error(const std::string& msg) {
        log(msg, ERROR);
    }

    void error(std::string&& msg) {
        log(msg, ERROR);
    }

    void fatal(const std::string& msg) {
        log(msg, FATAL);
    }

    void fatal(std::string&& msg) {
        log(msg, FATAL);
    }
};

extern _logger logger;