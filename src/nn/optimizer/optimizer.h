#pragma once

#include "optimizer_entry.h"
namespace nn {

struct Optimizer {

    protected:
    std::vector<OptimizerEntry> entries {};

    public:
    explicit Optimizer(const std::vector<OptimizerEntry>& entries)
        : entries(entries) {}

    void step(float lr) {
        for (int i = 0; i < entries.size(); i++) {
            this->step(entries[i], i, lr);
        }
    }

    void compile() {
        for (int i = 0; i < entries.size(); i++) {
            this->add_field(entries[i]);
        }
    }

    protected:
    virtual void add_field(OptimizerEntry& entry)    = 0;
    virtual void step(OptimizerEntry& entry, int idx, float lr) = 0;
};

}    // namespace nn