#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <string>
#include <cassert>
#include <climits>

// Include necessary ChampSim headers
#include "cache.h"

// Constants
const int INITIAL_TABLE_SIZE = 256;
const int MAX_WEIGHT = 31; // Saturating counters max value
const int MIN_WEIGHT = -32; // Saturating counters min value
const float LEAKY_RELU_ALPHA = 0.01f; // Leaky ReLU parameter

// Weight Table for Features
struct WeightTable {
    std::vector<int8_t> weights;
    int size;

    // Default constructor
    WeightTable() : weights(INITIAL_TABLE_SIZE, 0), size(INITIAL_TABLE_SIZE) {}

    // Constructor with table size
    WeightTable(int table_size) : weights(table_size, 0), size(table_size) {}

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
        // Initialize feature tables with default size
        feature_tables["pc"] = WeightTable();
        feature_tables["setIndex"] = WeightTable();
        feature_tables["is_write"] = WeightTable();
        // Add other features as needed
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

    // Apply Leaky ReLU
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

// Metadata structure
struct BlockMetadata {
    uint64_t pc = 0;
    bool is_write = false;
    // Add other features as needed
};

// Number of sets and ways (adjust these based on your cache configuration)
constexpr uint32_t NUM_SET = 2048;  // Example: 2048 sets
constexpr uint32_t NUM_WAY = 16;    // Example: 16 ways

// Metadata array indexed by set and way
BlockMetadata metadata_array[NUM_SET][NUM_WAY];

// Extract features for the predictor
std::unordered_map<std::string, uint64_t> extract_features(const BlockMetadata& metadata, uint32_t setIndex) {
    return {
        {"pc", metadata.pc},
        {"setIndex", setIndex},
        {"is_write", metadata.is_write ? 1 : 0}
        // Add other features as needed
    };
}

// **Define the member functions of CACHE class**

void CACHE::repl_replacementDpcn_initialize_replacement() {
    // Initialize the perceptron predictor if needed
    perceptron = PerceptronPredictor();
}

uint32_t CACHE::repl_replacementDpcn_find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t setIndex, const BLOCK* current_set, uint64_t ip,
                                                 uint64_t full_addr, uint32_t type) {
    uint32_t victim = 0;
    int32_t min_yout = INT32_MAX;

    // Loop over all ways in the set
    for (uint32_t way = 0; way < NUM_WAY; ++way) {
        const BlockMetadata& metadata = metadata_array[setIndex][way];

        auto features = extract_features(metadata, setIndex);
        int32_t yout = perceptron.compute_prediction(features, ip);

        if (yout < min_yout) {
            min_yout = yout;
            victim = way;
        }
    }

    return victim;
}

void CACHE::repl_replacementDpcn_update_replacement_state(uint32_t triggering_cpu, uint32_t setIndex, uint32_t wayID, uint64_t full_addr, uint64_t ip,
                                                          uint64_t victim_addr, uint32_t type, uint8_t hit) {
    // Update the block metadata
    BlockMetadata& metadata = metadata_array[setIndex][wayID];
    metadata.pc = ip;
    metadata.is_write = (static_cast<access_type>(type) == access_type::WRITE);

    // Extract features
    auto features = extract_features(metadata, setIndex);

    // Call perceptron update
    bool is_correct = hit;
    perceptron.update_weights(features, ip, is_correct);

    // Optionally, resize tables dynamically
    perceptron.dynamic_resize();
}

void CACHE::repl_replacementDpcn_replacement_final_stats() {
    // Any final statistics or cleanup code
}

