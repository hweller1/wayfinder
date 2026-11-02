// Fixture: top-level declarations of a C++ translation unit.
#include <string>
#include <vector>

class Stack {
public:
    void push(const std::string &value) { items_.push_back(value); }

    std::string pop()
    {
        std::string top = items_.back();
        items_.pop_back();
        return top;
    }

private:
    std::vector<std::string> items_;
};

template <typename T>
T largest(const std::vector<T> &values)
{
    // Assumes a non-empty vector; the caller checks before asking for a maximum.
    T best = values.front();
    for (const T &v : values) {
        if (best < v) best = v;
    }
    return best;
}
