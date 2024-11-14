#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include "cache.h"

namespace
{
// Replaced last_used_cycles with perceptron weights
std::map<CACHE*, std::vector<int>> perceptron_weights;
const int threshold_value = 3;
//saturating counter
//Max and Min weight values for a 6-bit saturating counter
const int max_weight = 31;
const int min_weight = -32;
}

void CACHE::initialize_replacement() { ::perceptron_weights[this] = std::vector<uint64_t>(NUM_SET * NUM_WAY, 0); }

uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type)
{
  //begin points to the start of the weights for the current set
  //end points to the end of the weights for the current set
  auto begin = std::next(perceptron_weights[this].begin(), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the cache block with the lowest prediction score
  auto victim = std::min_element(begin, end, [](int weight1, int weight2) {
    //the first condition checks if the weight is below the threshold value
    //the second condition checks if the weight is less than the weight of the other block
    //this selects the one with the smallest weight
    return (weight1 < threshold_value) && (weight < weight2);
    });

  assert(begin <= victim);
  assert(victim < end);

  return static_cast<uint32_t>(std::distance(begin, victim)); // cast protected by prior asserts
}

void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  // Mark the way as being used on the current cycle
  if (!hit || access_type{type} != access_type::WRITE) // Skip this for writeback hits
    ::last_used_cycles[this].at(set * NUM_WAY + way) = current_cycle;
}

void CACHE::replacement_final_stats() {}
