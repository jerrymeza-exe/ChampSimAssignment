#include <vector>
#include <cassert>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <string>

#include "cache.h"

// Constants
const int INITIAL_TABLE_SIZE = 256;
const int MAX_WEIGHT = 31; // Saturating counters max value
const int MIN_WEIGHT = -32; // Saturating counters min value
const int THRESHOLD = 3; // Threshold for prediction
const float LEAKY_RELU_ALPHA = 0.01f; // Leaky ReLU parameter

// Weight Table for Features
struct WeightTable {
    std::vector<int8_t> weights;
    int size;

    WeightTable(int table_size) : size(table_size), weights(table_size, 0) {}

    void resize(int new_size) {
        std::vector<int8_t> new_weights(new_size, 0);
        for (int i = 0; i < size; ++i) {
            new_weights[i % new_size] = weights[i];
        }
        weights = new_weights;
        size = new_size;
    }
};

// Perceptron Predictor
class PerceptronPredictor {
public:
    std::unordered_map<std::string, WeightTable> feature_tables;
    std::unordered_map<std::string, int> table_usage_count;

    PerceptronPredictor() {
        feature_tables["pc"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["mem_instr_pc"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["mem_addr_bits"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["data_content"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["time_ref_count"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["cycles_not_accessed"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["read_write"] = WeightTable(INITIAL_TABLE_SIZE);
        feature_tables["block_age"] = WeightTable(INITIAL_TABLE_SIZE);
    }

    int compute_prediction(const std::unordered_map<std::string, uint64_t>& features, uint64_t pc);
    void update_weights(const std::unordered_map<std::string, uint64_t>& features, uint64_t pc, bool is_correct);
    void dynamic_resize();
};

// Hash function to index weight tables
uint32_t hash_feature(uint64_t feature, uint64_t pc, int table_size) {
    return (feature ^ (pc & 0xFFFF)) % table_size;
}

// Compute prediction
int PerceptronPredictor::compute_prediction(const std::unordered_map<std::string, uint64_t>& features, uint64_t pc) {
    int32_t yout = 0;

    for (const auto& [feature_name, feature_value] : features) {
        auto& table = feature_tables[feature_name];
        uint32_t index = hash_feature(feature_value, pc, table.size);
        yout += table.weights[index];
    }

    // Apply ReLU or Leaky ReLU
    return (yout > 0) ? yout : static_cast<int>(LEAKY_RELU_ALPHA * yout);
}

// Update weights
void PerceptronPredictor::update_weights(const std::unordered_map<std::string, uint64_t>& features, uint64_t pc, bool is_correct) {
    int adjustment = is_correct ? -1 : 1;

    for (const auto& [feature_name, feature_value] : features) {
        auto& table = feature_tables[feature_name];
        uint32_t index = hash_feature(feature_value, pc, table.size);

        table.weights[index] = std::clamp(table.weights[index] + adjustment, MIN_WEIGHT, MAX_WEIGHT);
        table_usage_count[feature_name]++;
    }
}

// Dynamically resize tables based on workload
void PerceptronPredictor::dynamic_resize() {
    for (auto& [feature_name, table] : feature_tables) {
        if (table_usage_count[feature_name] > table.size * 10) {
            table.resize(table.size * 2); // Expand
        } else if (table_usage_count[feature_name] < table.size / 10 && table.size > INITIAL_TABLE_SIZE) {
            table.resize(table.size / 2); // Shrink
        }

        table_usage_count[feature_name] = 0; // Reset usage count
    }
}

// Global Predictor Instance
static PerceptronPredictor perceptron;

// Extract features for the predictor
std::unordered_map<std::string, uint64_t> extract_features(const BLOCK& block, uint64_t current_cycle) {
    return {
        {"pc", block.pc},
        {"mem_instr_pc", block.pc}, // Assuming block.pc holds memory instruction PC
        {"mem_addr_bits", block.full_addr >> 4}, // Use tag bits
        {"data_content", block.data_hash}, // Assume block contains a hash of its data
        {"time_ref_count", block.ref_count}, // Reference count
        {"cycles_not_accessed", current_cycle - block.last_access_cycle},
        {"read_write", block.is_write ? 1 : 0},
        {"block_age", current_cycle - block.creation_cycle}
    };
}

// Find victim block for eviction
uint32_t CACHE::find_victim(uint32_t set, const BLOCK* current_set, uint64_t current_pc) {
    uint32_t victim = 0;
    int32_t min_yout = INT32_MAX;

    for (uint32_t way = 0; way < NUM_WAY; ++way) {
        auto features = extract_features(current_set[way], current_cycle);
        int32_t yout = perceptron.compute_prediction(features, current_pc);

        if (yout > THRESHOLD && yout < min_yout) {
            min_yout = yout;
            victim = way;
        }
    }

    return victim;
}

// Update replacement state
void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit) {
    auto features = extract_features(cache_sets[set][way], current_cycle);
    bool is_correct = hit || (type == access_type::WRITE); // Example correctness condition
    perceptron.update_weights(features, ip, is_correct);

    // Optionally, resize tables dynamically
    perceptron.dynamic_resize();
}
