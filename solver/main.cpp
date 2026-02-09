#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <cstring>
#include <sstream>
#include <numeric>
#include <mutex>
#include "toulbar2lib.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <atomic>
#include "sysinfo.h"
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <thread>
#include <cctype>
#include <execution>
#include <cmath>
#include <unordered_set>

#ifdef __linux__
#include <sys/prctl.h>
#include <signal.h>
#elif defined(__APPLE__)

#include <chrono>

#endif

enum WeightTuner {
    GLOBAL,
    NODES,
    LAGRANGE,
};

std::ostream &operator<<(std::ostream &os, __int128_t value)
{
  if (value == 0)
    return os << "0";

  bool negative = value < 0;
  if (negative)
    value = -value;

  std::string result;
  while (value > 0)
  {
    result += '0' + (value % 10);
    value /= 10;
  }

  if (negative)
    result += '-';

  std::reverse(result.begin(), result.end());
  return os << result;
}

std::string read_file(const std::string &filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open())
  {
    throw std::runtime_error("Error: Unable to open file " + filename);
  }
  std::streamsize size = file.tellg();
  if (size < 0)
  {
    throw std::runtime_error("Error: Unable to determine file size for " + filename);
  }
  std::string content(static_cast<size_t>(size), '\0');
  file.seekg(0);
  file.read(&content[0], size);
  return content;
}

std::string remove_whitespace_single_threaded(const std::string &input)
{
  std::string result = input;
  result.erase(
      std::remove_if(result.begin(), result.end(), [](unsigned char c)
                     { return std::isspace(c); }),
      result.end());
  return result;
}

std::string remove_whitespace_multi_threaded(const std::string &input, int num_threads)
{
  const size_t n = input.size();
  if (n == 0)
    return std::string();

  std::vector<size_t> chunk_counts;
  std::vector<size_t> chunk_offsets;

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
#pragma omp single
    {
      num_threads = omp_get_num_threads();
      chunk_counts.resize(num_threads, 0);
      chunk_offsets.resize(num_threads, 0);
    }
    const size_t chunk_size = (n + num_threads - 1) / num_threads;
    const size_t start = tid * chunk_size;
    const size_t end = std::min(n, start + chunk_size);
    size_t count = 0;
    for (size_t i = start; i < end; i++)
    {
      if (!std::isspace(static_cast<unsigned char>(input[i])))
        count++;
    }
    chunk_counts[tid] = count;
  }

  size_t total = 0;
  for (int i = 0; i < num_threads; i++)
  {
    chunk_offsets[i] = total;
    total += chunk_counts[i];
  }

  std::string result;
  result.resize(total);

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    const size_t chunk_size = (n + num_threads - 1) / num_threads;
    const size_t start = tid * chunk_size;
    const size_t end = std::min(n, start + chunk_size);
    size_t pos = chunk_offsets[tid];
    for (size_t i = start; i < end; i++)
    {
      if (!std::isspace(static_cast<unsigned char>(input[i])))
        result[pos++] = input[i];
    }
  }
  return result;
}

std::string remove_whitespace(const std::string &input)
{
  if (input.size() < 10000000)
  {
    return remove_whitespace_single_threaded(input);
  }
  else
  {
    int num_threads = std::max(std::min(int(PHYSICAL_CORE_COUNT), 64), 8);
    return remove_whitespace_multi_threaded(input, num_threads);
  }
}

inline int64_t custom_strtoll(const char *str, char **end_ptr)
{
  int64_t result = 0;
  bool is_negative = false;
  while (*str == ' ' || *str == '\t' || *str == '\n')
  {
    ++str;
  }
  if (*str == '-')
  {
    is_negative = true;
    ++str;
  }
  else if (*str == '+')
  {
    ++str;
  }
  while (*str >= '0' && *str <= '9')
  {
    const unsigned char c = *str - '0';
    result = result * 10 + c;
    ++str;
  }
  if (end_ptr)
  {
    *end_ptr = const_cast<char *>(str);
  }
  return is_negative ? -result : result;
}

std::vector<int64_t> extract_flatten_vector(const std::string &json_string, const std::string &s = "intervals")
{
  std::vector<int64_t> intervals;
  const char *json_cstr = json_string.c_str();
  std::string _key = "\"" + s + "\":[";
  const char *key = _key.c_str();
  const char *start = strstr(json_cstr, key);
  if (!start)
  {
    std::cerr << "Key \"" << s << "\" not found!" << std::endl;
    return intervals;
  }
  start += strlen(key);
  const char *pos = start;
  int open_brackets = 1;
  while (*pos != '\0' && open_brackets > 0)
  {
    while (*pos == ' ' || *pos == ',' || *pos == '\n' || *pos == '\t')
    {
      pos++;
    }
    if (isdigit(*pos) || *pos == '-')
    {
      char *end_ptr;
      int64_t value = custom_strtoll(pos, &end_ptr);
      intervals.push_back(value);
      pos = end_ptr;
    }
    else if (*pos == '[')
    {
      open_brackets++;
      pos++;
    }
    else if (*pos == ']')
    {
      open_brackets--;
      pos++;
    }
    else
    {
      pos++;
    }
  }
  if (open_brackets != 0)
  {
    std::cerr << "Error: mismatched brackets in JSON!" << std::endl;
  }
  return intervals;
}

int64_t extract_usage_limit(const std::string &json_string)
{
  std::string key = "\"usage_limit\":";
  size_t start = json_string.find(key);
  if (start == std::string::npos)
  {
    std::cerr << "Key \"usage_limit\" not found!" << std::endl;
    return -1;
  }
  start += key.length();
  size_t end = start;
  while (end < json_string.size() && std::isdigit(json_string[end]))
  {
    end++;
  }
  return std::stoll(json_string.substr(start, end - start));
}

struct Node_Cost
{
  std::vector<int64_t> strategies;
  std::vector<int64_t> ranges;
};

Node_Cost extract_node_costs(const std::string &json_string, const std::string &s = "nodes")
{
  Node_Cost result;
  const char *json_cstr = json_string.c_str();
  auto key = "\"" + s + "\":{";
  const char *s_key = key.c_str();
  const char *nodes_start = strstr(json_cstr, s_key);
  if (!nodes_start)
  {
    std::cerr << "Key \"" << s << "\" not found!" << std::endl;
    return result;
  }
  const char *costs_key = "\"costs\":[";
  const char *costs_start = strstr(nodes_start, costs_key);
  if (!costs_start)
  {
    std::cerr << R"(Key "costs" not found in "nodes"!)" << std::endl;
    return result;
  }
  costs_start += strlen(costs_key);
  const char *pos = costs_start;
  result.ranges.push_back(0);
  while (*pos != '\0')
  {
    const char *node_start = strchr(pos, '[');
    if (!node_start || *(node_start + 1) == ']')
    {
      break;
    }
    const char *node_end = node_start + 1;
    while (*node_end != '\0' && *node_end != ']')
    {
      while (*node_end == ' ' || *node_end == ',' || *node_end == '\t' || *node_end == '\n')
      {
        node_end++;
      }
      if (isdigit(*node_end) || *node_end == '-')
      {
        char *value_end;
        int64_t value = custom_strtoll(node_end, &value_end);
        result.strategies.push_back(value);
        node_end = value_end;
      }
      else
      {
        node_end++;
      }
    }
    if (*node_end == ']')
    {
      node_end++;
    }
    auto cost_end_idx = static_cast<int64_t>(result.strategies.size());
    result.ranges.push_back(cost_end_idx);
    if (*node_end == ']')
    {
      break;
    }
    pos = node_end;
  }
  return result;
}

struct Edge_Cost
{
  std::vector<int64_t> strategies_combinations;
  std::vector<int64_t> ranges;
};

Edge_Cost extract_edge_costs(const std::string &json_string)
{
  auto costs = extract_node_costs(json_string, "edges");
  return {std::move(costs.strategies), std::move(costs.ranges)};
}

struct Problem
{
  int64_t usage_limit = 0;
  int64_t intervals_min = std::numeric_limits<int64_t>::max();
  int64_t intervals_max = std::numeric_limits<int64_t>::min();
  std::vector<int64_t> intervals;
  Node_Cost node_costs;
  std::vector<int64_t> usages;
  std::vector<int64_t> edges;
  Edge_Cost edge_costs;
  std::vector<std::pair<int64_t, int64_t>> forward_edges;
  std::vector<std::pair<int64_t, int64_t>> inverted_edges;

  Problem()
  {
    const int64_t reserved_size = 1024 * 16;
    intervals.reserve(reserved_size);
    node_costs.ranges.reserve(reserved_size);
    node_costs.strategies.reserve(reserved_size);
    usages.reserve(reserved_size);
    edges.reserve(reserved_size);
    edge_costs.ranges.reserve(reserved_size);
    edge_costs.strategies_combinations.reserve(reserved_size);
    inverted_edges.reserve(reserved_size);
  }
};

std::vector<int64_t> coordinate_compress(const std::vector<int64_t> &arr)
{
  std::vector<int64_t> temp(arr.begin(), arr.end());
  std::sort(temp.begin(), temp.end());
  // remove duplicates
  temp.erase(std::unique(temp.begin(), temp.end()), temp.end());
  // 'temp' is now a sorted list of unique values. We'll use it for binary search.
  // prepare output vector
  std::vector<int64_t> result;
  result.reserve(arr.size());
  for (int64_t val : arr)
  {
    auto it = std::lower_bound(temp.begin(), temp.end(), val);
    int64_t compressed_index = static_cast<int64_t>(it - temp.begin());
    result.push_back(compressed_index);
  }
  return result;
}

void verify_problem(const Problem &problem)
{
  for (size_t i = 0; i < problem.node_costs.ranges.size(); i += 2)
  {
    if (problem.intervals[i] > problem.intervals[i + 1])
    {
      throw std::runtime_error(
          "Error: intervals must be ascending like [2, 3] and not descending like [3, 2].");
    }
  }
  if (problem.usages.size() != problem.node_costs.strategies.size())
  {
    throw std::runtime_error(
        "Error: size of costs and usages must be the same.");
  }
  if (int64_t(problem.intervals.size()) != (int64_t(problem.node_costs.ranges.size()) - 1) * 2)
  {
    throw std::runtime_error(
        "Error: number nodes must correspond to the number of intervals.");
  }
  if (int64_t(problem.edges.size()) != (int64_t(problem.edge_costs.ranges.size()) - 1) * 2)
  {
    throw std::runtime_error(
        "Error: number of edges must correspond to the number of its costs.");
  }
  const auto num_nodes = int64_t(problem.intervals.size() / 2);
  for (auto val : problem.edges)
  {
    if (val < 0)
    {
      throw std::runtime_error(
          "Error: nodes in the edges must be positive.");
    }
    if (val >= num_nodes)
    {
      throw std::runtime_error(
          "Error: edges must reference existing nodes.");
    }
  }
  // check validity correspondence between cost strategies and edges connecting them
  for (size_t i = 0; i < problem.edges.size(); i += 2)
  {
    int64_t e1 = problem.edges[i];
    int64_t e2 = problem.edges[i + 1];
    int64_t edge_size = problem.edge_costs.ranges[i / 2 + 1] - problem.edge_costs.ranges[i / 2];
    int64_t n1_strategies = problem.node_costs.ranges[e1 + 1] - problem.node_costs.ranges[e1];
    int64_t n2_strategies = problem.node_costs.ranges[e2 + 1] - problem.node_costs.ranges[e2];
    if (edge_size != n1_strategies * n2_strategies)
    {
      throw std::runtime_error(
          "Error: edge costs must match nodes strategies.");
    }
  }
}

Problem get_problem_from_path(const std::string &path)
{
  Problem problem;
  std::string content = remove_whitespace(read_file(path));
#pragma omp parallel if (content.size() > 100000)
  {
#pragma omp sections
    {
#pragma omp section
      problem.edge_costs = extract_edge_costs(content);
#pragma omp section
      problem.node_costs = extract_node_costs(content);
#pragma omp section
      {
        problem.intervals = extract_flatten_vector(content);
        if (problem.intervals.size() % 2 != 0)
        {
          throw std::runtime_error(
              "Error: intervals must be pairs of numbers like [2, 3].");
        }
        problem.intervals = coordinate_compress(problem.intervals);
        for (const auto val : problem.intervals)
        {
          problem.intervals_min = std::min(problem.intervals_min, val);
          problem.intervals_max = std::max(problem.intervals_max, val);
        }
      }
#pragma omp section
      problem.usages = extract_flatten_vector(content, "usages");
#pragma omp section
      {
        problem.edges = extract_flatten_vector(content, "nodes");
        for (size_t i = 0; i < problem.edges.size(); i += 2)
        {
          if (problem.edges[i] == problem.edges[i + 1])
          {
            throw std::runtime_error(
                "Error: self-edges like [1, 1] are not supported.");
          }
        }

        if (problem.edges.size() % 2 != 0)
        {
          throw std::runtime_error(
              "Error: edges must be pairs of numbers like [2, 3].");
        }

#pragma omp task
        {
          problem.inverted_edges.resize(problem.edges.size() / 2);
          for (int64_t i = 0; i < problem.edges.size(); i += 2)
          {
            const int64_t e2 = problem.edges[i + 1];
            problem.inverted_edges[i / 2] = {e2, i};
          }
          std::sort(begin(problem.inverted_edges), end(problem.inverted_edges),
                    [](const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b)
                    {
                      return a.first < b.first;
                    });
        }
#pragma omp task
        {
          problem.forward_edges.resize(problem.edges.size() / 2);
          for (int64_t i = 0; i < problem.edges.size(); i += 2)
          {
            const int64_t e1 = problem.edges[i];
            problem.forward_edges[i / 2] = {e1, i};
          }
          std::sort(begin(problem.forward_edges), end(problem.forward_edges),
                    [](const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b)
                    {
                      return a.first < b.first;
                    });
        }
      }
#pragma omp section
      problem.usage_limit = extract_usage_limit(content);
    }
  }
  for (int64_t i = 0; i < int64_t(problem.node_costs.ranges.size()) - 1; ++i)
  {
    const int64_t b = problem.node_costs.ranges[i];
    const int64_t e = problem.node_costs.ranges[i + 1];
    auto start_interval = problem.intervals[2 * i];
    auto end_interval = problem.intervals[2 * i + 1];
    if (end_interval == start_interval)
    {
      for (int64_t j = b; j < e; ++j)
      {
        problem.usages[j] = 0;
      }
    }
  }
  verify_problem(problem);

  return problem;
}

std::vector<int>
min_ressource_solution(const Problem &problem)
{
  std::vector<int> mrs;
  if (problem.node_costs.ranges.size() < 2)
  {
    throw std::runtime_error(
        "Error: node_costs_ranges must be >= 2 but is " + std::to_string(problem.node_costs.ranges.size()));
  }
  mrs.resize(problem.node_costs.ranges.size() - 1);
#pragma omp parallel for if (problem.node_costs.ranges.size() > 100000)
  for (int64_t i = 1; i < problem.node_costs.ranges.size(); ++i)
  {
    const int64_t b = problem.node_costs.ranges[i - 1];
    const int64_t e = problem.node_costs.ranges[i];
    int64_t min_val = std::numeric_limits<int64_t>::max();
    int idx_min = 0;
    for (int64_t j = b; j < e; ++j)
    {
      if (problem.usages[j] < min_val)
      {
        min_val = problem.usages[j];
        idx_min = int(j - b);
      }
    }
    mrs[i - 1] = idx_min;
  }
  return mrs;
}

inline int64_t get_edge_cost(const Problem &problem, const std::vector<int> &solution, const int64_t i)
{
  const int64_t e1 = problem.edges[i];
  const int64_t e2 = problem.edges[i + 1];
  const int64_t cols = problem.node_costs.ranges[e2 + 1] - problem.node_costs.ranges[e2];
  const int64_t offset = solution[e1] * cols + solution[e2];
  const int64_t strategies_begin = problem.edge_costs.ranges[i / 2];
  return problem.edge_costs.strategies_combinations[strategies_begin + offset];
}

struct Eval_Result
{
  std::vector<int64_t> usage_at_time;
  int64_t max_usage_at_time;
  __int128_t cost;
};

Eval_Result eval_solution(const Problem &problem, const std::vector<int> &solution)
{
  if (solution.size() + 1 != problem.node_costs.ranges.size())
  {
    throw std::runtime_error(
        "Error: number of solutions is wrong!");
  }
  __int128_t costs_nodes = 0;
  std::vector<int64_t> usage_at_time(problem.intervals_max);

  for (int64_t i = 0; i < int64_t(solution.size()); ++i)
  {
    const int64_t idx_strategy = problem.node_costs.ranges[i] + solution[i];
    costs_nodes += problem.node_costs.strategies[idx_strategy];
    for (int64_t j = problem.intervals[i * 2]; j < problem.intervals[i * 2 + 1]; ++j)
    {
      usage_at_time[j] += problem.usages[idx_strategy];
    }
  }
  int64_t max_usage_at_time = *std::max_element(begin(usage_at_time), end(usage_at_time));
  __int128_t costs_edges = 0;

  for (int64_t i = 0; i < problem.edges.size(); i += 2)
  {
    costs_edges += get_edge_cost(problem, solution, i);
  }

  return {usage_at_time, max_usage_at_time, costs_nodes + costs_edges};
}

struct Option_Helper
{

  std::vector<std::pair<int64_t, int64_t>> start_pos_nodes_edges;
  std::vector<std::pair<int64_t, int64_t>> start_pos_nodes_inverted_edges;

  explicit Option_Helper(const Problem &problem)
  {

    start_pos_nodes_edges.resize(problem.node_costs.ranges.size() - 1, {std::numeric_limits<int64_t>::max(), -1});
    start_pos_nodes_inverted_edges.resize(problem.node_costs.ranges.size() - 1,
                                          {std::numeric_limits<int64_t>::max(), -1});

    for (int64_t i = 0; i < problem.edges.size(); i += 2)
    {
      const int64_t e1 = problem.forward_edges[i / 2].first;
      const int64_t e2 = problem.inverted_edges[i / 2].first;

      start_pos_nodes_edges[e1] = {std::min(start_pos_nodes_edges[e1].first, i / 2),
                                   std::max(start_pos_nodes_edges[e1].second, i / 2)};

      start_pos_nodes_inverted_edges[e2] = {std::min(start_pos_nodes_inverted_edges[e2].first, i / 2),
                                            std::max(start_pos_nodes_inverted_edges[e2].second, i / 2)};
    }
  }
};

struct Greedy_Result
{
  std::vector<int> chosen_strategies;
  Eval_Result eval_result;
};

class Xoroshiro128Plus
{
  uint64_t state[2]{};

  static inline uint64_t rotl(const uint64_t x, int k)
  {
    return (x << k) | (x >> (64 - k));
  }

  inline uint64_t next()
  {
    const uint64_t s0 = state[0];
    uint64_t s1 = state[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    state[1] = rotl(s1, 37);

    return result;
  }

public:
  explicit Xoroshiro128Plus(uint64_t seed = 0)
  {
    state[0] = (12345678901234567ULL + seed) | 0x9008104809202929ULL;
    state[1] = (98765432109876543ULL + seed) | 0x4033209202929020ULL;
    for (int i = 0; i < 10; ++i)
      next(); // warm up the PRNG
  }

  double rand_double()
  {
    uint64_t result = (next() >> 12) | 0x3FF0000000000000ULL;
    union
    {
      uint64_t i;
      double d;
    } u{result};
    return u.d - 1.0;
  }

  int64_t rand_int(int64_t min, int64_t max)
  {
    if (min > max)
      std::swap(min, max);
    uint64_t range = static_cast<uint64_t>(max - min) + 1;
    return min + static_cast<int64_t>(next() % range);
  }

  void shuffle_vector(std::vector<int64_t> &vec)
  {
    for (auto i = int64_t(vec.size()); i > 1; --i)
    {
      std::swap(vec[i - 1], vec[rand_int(0, i - 1)]);
    }
  }
};

inline __int128_t
compute_option_cost(const Problem &problem, const Option_Helper &option_helper, const Greedy_Result &gr,
                    const int64_t node, const int64_t b, const int current_option)
{
  __int128_t current_option_cost = problem.node_costs.strategies[b + current_option];

  if (option_helper.start_pos_nodes_edges[node].second != -1)
  {
    for (int64_t i = option_helper.start_pos_nodes_edges[node].first;
         i <= option_helper.start_pos_nodes_edges[node].second; ++i)
    {
      int64_t e2 = problem.edges[problem.forward_edges[i].second + 1];
      const int64_t cols = problem.node_costs.ranges[e2 + 1] - problem.node_costs.ranges[e2];
      const int64_t offset = current_option * cols + gr.chosen_strategies[e2];
      const int64_t strategies_begin = problem.edge_costs.ranges[problem.forward_edges[i].second / 2];
      current_option_cost += problem.edge_costs.strategies_combinations[strategies_begin + offset];
    }
  }

  if (option_helper.start_pos_nodes_inverted_edges[node].second != -1)
  {
    for (int64_t i = option_helper.start_pos_nodes_inverted_edges[node].first;
         i <= option_helper.start_pos_nodes_inverted_edges[node].second; ++i)
    {
      int64_t e1 = problem.edges[problem.inverted_edges[i].second];
      int64_t e2 = problem.edges[problem.inverted_edges[i].second + 1];

      const int64_t cols = problem.node_costs.ranges[e2 + 1] - problem.node_costs.ranges[e2];
      const int64_t offset = gr.chosen_strategies[e1] * cols + current_option;
      const int64_t strategies_begin = problem.edge_costs.ranges[problem.inverted_edges[i].second / 2];
      current_option_cost += problem.edge_costs.strategies_combinations[strategies_begin + offset];
    }
  }

  return current_option_cost;
}

inline bool tune_node(const Problem &problem, const Option_Helper &option_helper, Greedy_Result &gr, int64_t node,
                      std::vector<int> &valid_options, std::vector<__int128_t> &options_costs, std::vector<int> &sample,
                      double cooling, Xoroshiro128Plus &prng, double threshold_random)
{
  int64_t b = problem.node_costs.ranges[node];
  int64_t e = problem.node_costs.ranges[node + 1];
  int num_options = int(e - b);
  if (num_options <= 1)
    return false;
  const int current_option = gr.chosen_strategies[node];
  const int64_t max_usage_in_range = *std::max_element(
      begin(gr.eval_result.usage_at_time) + problem.intervals[node * 2],
      begin(gr.eval_result.usage_at_time) + problem.intervals[node * 2 + 1]);
  const int64_t allowed_max_usage_of_node =
      problem.usage_limit - max_usage_in_range + problem.usages[b + current_option];
  valid_options.clear();

  for (int i = 0; i < num_options; ++i)
  {
    if (i != current_option && problem.usages[b + i] <= allowed_max_usage_of_node)
    {
      valid_options.push_back(i);
    }
  }

  if (valid_options.empty())
  {
    return false;
  }

  __int128_t current_option_cost = compute_option_cost(problem, option_helper, gr, node, b, current_option);

  options_costs.clear();
  sample.clear();

  int i = 0;
  double random_value = prng.rand_double();

  for (int option : valid_options)
  {
    __int128_t cost = compute_option_cost(problem, option_helper, gr, node, b, option);
    options_costs.push_back(cost - current_option_cost);
    if ((cost < current_option_cost) - (random_value < cooling))
    {
      sample.push_back(i);
    }
    i++;
  }

  if (sample.empty())
  {
    return false;
  }

  int idx_sample = 0;
  if (random_value < threshold_random)
  {
    double _min = std::numeric_limits<double>::max();
    for (int j = 0; j < sample.size(); ++j)
    {
      auto id = sample[j];
      if (double(options_costs[id]) * double(problem.usages[b + valid_options[id]]) < _min)
      {
        _min = double(options_costs[id]) * double(problem.usages[b + valid_options[id]]);
        idx_sample = j;
      }
    }
  }
  else
  {
    idx_sample = int(prng.rand_int(0, int64_t(sample.size() - 1)));
  }

  int new_option = valid_options[sample[idx_sample]];
  gr.chosen_strategies[node] = new_option;
  gr.eval_result.cost += options_costs[sample[idx_sample]];
  int64_t usages_diff = problem.usages[b + new_option] - problem.usages[b + current_option];
  for (int64_t j = problem.intervals[node * 2]; j < problem.intervals[node * 2 + 1]; ++j)
  {
    gr.eval_result.usage_at_time[j] += usages_diff;
  }
  return true;
}

static void write_node_usage_weights_binary(const std::vector<double> &weights,
                                            const std::string &filename)
{
  try
  {
    const std::string tmp = filename + ".tmp";
    std::ofstream ofs(tmp, std::ios::binary | std::ios::out);
    if (!ofs)
      return;
    if (!weights.empty())
    {
      ofs.write(reinterpret_cast<const char *>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(double)));
    }
    ofs.close();
    std::rename(tmp.c_str(), filename.c_str()); // atomic on POSIX
  }
  catch (...)
  {
    // best-effort write; ignore errors to avoid crashing the solver
  }
}

Greedy_Result
greedy_optimize(const Problem &problem, const std::vector<int> &v, const Eval_Result &er,
                const Option_Helper &option_helper, uint64_t seed, const double cool_alpha,
                const double time_out_in_s)
{
  const double tic = omp_get_wtime();
  Greedy_Result gr = Greedy_Result{v, er};
  std::vector<int64_t> order(int64_t(problem.node_costs.ranges.size()) - 1);
  std::iota(begin(order), end(order), 0);

  std::vector<int> valid_options;
  std::vector<__int128_t> options_costs;
  std::vector<int> sample;

  bool is_running = true;
  double cooling = cool_alpha;
  Xoroshiro128Plus prng(seed);

  double threshold_random = 0.5 + (prng.rand_double() < 0.5) * 0.5;
  while (is_running)
  {
    prng.shuffle_vector(order);
    is_running = false;

    for (int64_t node : order)
    {
      const bool is_change = tune_node(problem, option_helper, gr, node, valid_options, options_costs, sample, cooling,
                                       prng, threshold_random);
      is_running |= is_change;
    }
    if (omp_get_wtime() - tic > time_out_in_s)
    {
      gr.eval_result.max_usage_at_time = *std::max_element(begin(gr.eval_result.usage_at_time),
                                                           end(gr.eval_result.usage_at_time));
      return gr;
    }
    cooling *= cool_alpha;
  }

  gr.eval_result.max_usage_at_time = *std::max_element(begin(gr.eval_result.usage_at_time),
                                                       end(gr.eval_result.usage_at_time));
  return gr;
}

Greedy_Result get_min_resource_solution(const Problem &problem)
{
  std::vector<int> mrs = min_ressource_solution(problem);
  return {mrs, eval_solution(problem, mrs)};
}

static std::atomic<bool> stop_print = false;
static std::atomic<bool> stop_greedy = false;
static std::mutex best_solution_mutex;
static Greedy_Result best_solution;
static int iterations = 1;

static std::atomic<int> best_pid = -1;
static __int128_t previous_best_solution_cost = std::numeric_limits<__int128_t>::max();

void update_best(const Problem &problem, const Greedy_Result &gr, const int pid)
{
  if (!gr.chosen_strategies.empty() && gr.eval_result.max_usage_at_time <= problem.usage_limit)
  {
    best_solution_mutex.lock();
    if (gr.eval_result.cost <= best_solution.eval_result.cost && !stop_greedy && pid != -1)
    {
      stop_greedy = true;
    }
    if (gr.eval_result.cost < best_solution.eval_result.cost)
    {
      best_solution = gr;
      best_pid = pid;
    }
    best_solution_mutex.unlock();
  }
}

Greedy_Result
run_greedy(const Problem &problem, const Greedy_Result &start_gr, const Option_Helper &option_helper,
           const int64_t iterations, double time_out_in_s, int64_t seed, int threads = 0, bool verbose = false,
           bool is_update_best = false)
{
  const double tic = omp_get_wtime();
  int64_t _seed =
      seed ^ Xoroshiro128Plus(seed).rand_int(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
  __int128_t best = start_gr.eval_result.cost;
  if (start_gr.eval_result.max_usage_at_time <= problem.usage_limit)
  {
    best = std::numeric_limits<__int128_t>::max();
  }
  std::mutex m;
  Greedy_Result gr_best = start_gr;

  if ((threads < 1) || (threads > PHYSICAL_CORE_COUNT))
  {
    threads = int(PHYSICAL_CORE_COUNT);
  }

  if (stop_print)
  {
    return gr_best;
  }

#pragma omp parallel for schedule(dynamic) num_threads(threads) if (iterations > 1 && threads > 1)
  for (int64_t i = 0; i < iterations; ++i)
  {
    if (stop_print || (is_update_best && stop_greedy))
    {
      continue;
    }
    if (omp_get_wtime() - tic > time_out_in_s)
    {
      continue;
    }
    Greedy_Result gr;
    Xoroshiro128Plus prng(_seed + i);

    if (i == 0)
    {
      gr = greedy_optimize(problem, start_gr.chosen_strategies, start_gr.eval_result, option_helper, _seed, 0.0,
                           time_out_in_s - (omp_get_wtime() - tic));
    }
    else if (prng.rand_double() > 0.05)
    {
      m.lock();
      auto local_cs = gr_best.chosen_strategies;
      m.unlock();
      if (prng.rand_double() > 0.5)
      {
        double t = prng.rand_double() * 0.05;
        bool is_complete_random_update = prng.rand_double() < 0.5;
        for (int64_t j = 0; j < start_gr.chosen_strategies.size(); ++j)
        {
          if (prng.rand_double() < t)
          {
            if (is_complete_random_update)
            {
              local_cs[j] = int(prng.rand_int(0, problem.node_costs.ranges[j + 1] - problem.node_costs.ranges[j] - 1));
            }
            else
            {
              local_cs[j] = start_gr.chosen_strategies[j];
            }
          }
        }
        gr = greedy_optimize(problem, local_cs, eval_solution(problem, local_cs), option_helper,
                             (4449551 + _seed) * i + i, 0.05, time_out_in_s - (omp_get_wtime() - tic));
      }
      else
      {
        gr = greedy_optimize(problem, local_cs, eval_solution(problem, local_cs), option_helper,
                             (4449551 + _seed) * i + i, 0.01, time_out_in_s - (omp_get_wtime() - tic));
      }
    }
    else
    {
      gr = greedy_optimize(problem, start_gr.chosen_strategies, start_gr.eval_result, option_helper,
                           (4449551 + _seed) * i + i, 0.03, time_out_in_s - (omp_get_wtime() - tic));
    }
    if (gr.eval_result.cost < best)
    {
      m.lock();
      if (gr.eval_result.cost < best && (gr.eval_result.max_usage_at_time <= problem.usage_limit))
      {
        best = gr.eval_result.cost;
        gr_best = gr;
        if (verbose)
        {
          std::cout << "best: " << i << ": " << best << std::endl;
        }
      }
      m.unlock();
    }
    if (is_update_best && i != 0 && i % 2 == 0)
    {
      m.lock();
      best_solution_mutex.lock();

      if (!gr.chosen_strategies.empty() && gr.eval_result.max_usage_at_time <= problem.usage_limit)
      {
        if (gr.eval_result.cost < best_solution.eval_result.cost)
        {
          best_solution = gr;
          best_pid = -1;
        }
      }
      else if (best_solution.eval_result.cost < gr_best.eval_result.cost)
      {
        gr_best = best_solution;
        best = gr_best.eval_result.cost;
      }

      best_solution_mutex.unlock();
      m.unlock();
    }
  }
  return gr_best;
}

void init_solver_once(int seed)
{
  static bool initialized = ([seed]
                             {
    tb2init();
    ToulBar2::verbose = -1;
    ToulBar2::vnsInitSol = LS_INIT_INF;
    ToulBar2::lds = 4;
    ToulBar2::vac = 0;
    ToulBar2::seed = seed;
    ToulBar2::useRASPS = false;
    ToulBar2::decimalPoint = 0;
    initCosts();

    return true; })();
}

struct Int128Split
{
  int64_t hi;
  int64_t lo;
};

inline Int128Split splitInt128(__int128 val)
{
  int64_t hi = static_cast<int64_t>(val >> 64);
  int64_t lo = static_cast<int64_t>(val & 0xFFFFFFFFFFFFFFFFULL);
  return {hi, lo};
}

inline __int128 combineInt128(int64_t hi, int64_t lo)
{
  __int128 val = static_cast<__int128>(hi) << 64;
  val |= static_cast<uint64_t>(lo);
  return val;
}

bool robust_write(int fd, const void *buffer, size_t bytesToWrite)
{
  const char *ptr = static_cast<const char *>(buffer);
  while (bytesToWrite > 0)
  {
    ssize_t written = write(fd, ptr, bytesToWrite);
    if (written < 0)
    {
      return false;
    }
    ptr += written;
    bytesToWrite -= written;
  }
  return true;
}

bool robust_read(int fd, void *buffer, size_t bytesToRead)
{
  char *ptr = static_cast<char *>(buffer);
  while (bytesToRead > 0)
  {
    ssize_t nread = read(fd, ptr, bytesToRead);
    if (nread < 0)
    {
      return false;
    }
    if (nread == 0)
    {
      return false;
    }
    ptr += nread;
    bytesToRead -= nread;
  }
  return true;
}

struct WCSP_result
{
  int broken = -1;
  Greedy_Result gr;

  [[nodiscard]] bool serialize(int fd) const
  {
    if (!robust_write(fd, &broken, sizeof(broken)))
      return false;
    auto chosenSize = (int32_t)gr.chosen_strategies.size();
    if (!robust_write(fd, &chosenSize, sizeof(chosenSize)))
      return false;
    if (chosenSize > 0)
    {
      if (!robust_write(fd, gr.chosen_strategies.data(), chosenSize * sizeof(int)))
        return false;
    }
    auto usageSize = (int32_t)gr.eval_result.usage_at_time.size();
    if (!robust_write(fd, &usageSize, sizeof(usageSize)))
      return false;
    if (usageSize > 0)
    {
      if (!robust_write(fd, gr.eval_result.usage_at_time.data(), usageSize * sizeof(int64_t)))
        return false;
    }
    if (!robust_write(fd, &gr.eval_result.max_usage_at_time, sizeof(gr.eval_result.max_usage_at_time)))
      return false;
    Int128Split c = splitInt128(gr.eval_result.cost);
    if (!robust_write(fd, &c, sizeof(c)))
      return false;
    return true;
  }

  static WCSP_result deserialize(int fd)
  {
    WCSP_result result;
    if (!robust_read(fd, &result.broken, sizeof(result.broken)))
    {
      throw std::runtime_error("Failed to read 'broken'");
    }
    int32_t chosenSize = 0;
    if (!robust_read(fd, &chosenSize, sizeof(chosenSize)))
    {
      throw std::runtime_error("Failed to read chosenSize");
    }
    result.gr.chosen_strategies.resize(chosenSize);
    if (chosenSize > 0)
    {
      if (!robust_read(fd, result.gr.chosen_strategies.data(), chosenSize * sizeof(int)))
      {
        throw std::runtime_error("Failed to read chosen_strategies");
      }
    }
    int32_t usageSize = 0;
    if (!robust_read(fd, &usageSize, sizeof(usageSize)))
    {
      throw std::runtime_error("Failed to read usageSize");
    }
    result.gr.eval_result.usage_at_time.resize(usageSize);
    if (usageSize > 0)
    {
      if (!robust_read(fd, result.gr.eval_result.usage_at_time.data(), usageSize * sizeof(int64_t)))
      {
        throw std::runtime_error("Failed to read usage_at_time");
      }
    }

    if (!robust_read(fd, &result.gr.eval_result.max_usage_at_time, sizeof(result.gr.eval_result.max_usage_at_time)))
    {
      throw std::runtime_error("Failed to read max_usage_at_time");
    }

    Int128Split c{};
    if (!robust_read(fd, &c, sizeof(c)))
    {
      throw std::runtime_error("Failed to read cost split");
    }
    result.gr.eval_result.cost = combineInt128(c.hi, c.lo);

    return result;
  }
};

enum
{
  IS_BEST = -3,
  IS_OPTIMAL = -4
};

double tic_global;

void print_best_solution_thread(const double finish_time, bool print_solution)
{
  while (true)
  {
    if (stop_print)
    {
      return;
    }
    best_solution_mutex.lock();
    if (best_solution.eval_result.cost < previous_best_solution_cost)
    {
      auto pid_info =
          best_pid != -1 ? " (pid: " +
                               (best_pid == IS_BEST ? "B" : best_pid == IS_OPTIMAL ? "*"
                                                                                   : std::to_string(best_pid)) +
                               ")"
                         : " (greedy)";
      std::ostringstream oss;
      double time_point = omp_get_wtime() - tic_global;
      oss << "# Cost: " << std::scientific << std::setprecision(8)
          << static_cast<double>(best_solution.eval_result.cost)
          << pid_info << " " << std::fixed << std::setprecision(2) << time_point << " s (iteration: "  << iterations << ')';
      if (print_solution)
      {
        oss << '\n'
            << '[';
        for (size_t i = 0; i < best_solution.chosen_strategies.size(); ++i)
        {
          oss << best_solution.chosen_strategies[i];
          if (i != best_solution.chosen_strategies.size() - 1)
          {
            oss << ", ";
          }
        }
        oss << ']';
      }
      std::cout << oss.str() << std::endl;
      previous_best_solution_cost = best_solution.eval_result.cost;
    }
    best_solution_mutex.unlock();
    if (omp_get_wtime() > finish_time)
    {
      exit(0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void print_and_exit_invalid_max_usage(const __int128_t max_usage_allowed, const __int128_t minimal_usage)
{
  if (minimal_usage > max_usage_allowed)
  {
    std::ostringstream oss;
    oss << "# Cost: " << std::scientific << std::setprecision(8)
        << std::numeric_limits<double>::infinity() << " (ERROR: Usage limit is " << max_usage_allowed
        << ", but minimum usage is "
        << minimal_usage << ".)\n[]" << std::endl;
    std::cout << oss.str();
    exit(0);
  }
}

void back_ground_greedy(const Problem &problem, const Greedy_Result &mrs, int64_t max_iterations, double time_out_in_s,
                        int threads, int64_t seed)
{
  Option_Helper option_helper(problem);
  Greedy_Result gr_best = run_greedy(problem, mrs, option_helper, max_iterations, time_out_in_s, seed, threads, false,
                                     true);
}

#define TOP_LIMIT 100000000000000000ll

void add_nodes_and_edges(const Problem &problem, WeightedCSP &csp, const std::vector<int64_t> &min_usages,
                         const vector<double> &node_usage_weight, int seed)
{
  Xoroshiro128Plus rng(seed);
  for (int64_t i = 1; i < problem.node_costs.ranges.size(); ++i)
  {
    const int64_t b = problem.node_costs.ranges[i - 1];
    const int64_t e = problem.node_costs.ranges[i];
    const int64_t num_strats = e - b;
    const int index = csp.makeEnumeratedVariable(to_string("n") + to_string(i), 0, int(num_strats) - 1);
    std::vector<Cost> table(num_strats);
    Double min_val = std::numeric_limits<Double>::max();
    Double max_val = 0;
    for (int64_t j = b; j < e; ++j)
    {
      Cost val = 0;
      double penalty = node_usage_weight[i - 1] * (double(problem.usages[j]) - double(min_usages[i - 1]));
      const double wval = ceil(double(problem.node_costs.strategies[j]) + penalty);
      if (wval + 2 > double(LONGLONG_MAX))
      {
        val = LONGLONG_MAX;
      }
      else
      {
        val = Cost(wval);
      }
      table[j - b] = val;
      if (double(val) < min_val)
      {
        min_val = double(val);
      }
      if (double(val) > max_val)
      {
        max_val = double(val);
      }
    }
    if (min_val == max_val)
    {
      csp.postNullaryConstraint(min_val);
    }
    else
    {
      csp.postUnaryConstraint(index, table);
    }
  }
  for (int64_t i = 0; i < problem.edges.size(); i += 2)
  {
    const int n_1 = int(problem.edges[i]);
    const int n_2 = int(problem.edges[i + 1]);
    const int64_t b = problem.edge_costs.ranges[i / 2];
    const int64_t e = problem.edge_costs.ranges[i / 2 + 1];
    vector<Cost> costs(problem.edge_costs.strategies_combinations.begin() + b,
                       problem.edge_costs.strategies_combinations.begin() + e);
    const Cost min_cost = *std::min_element(costs.begin(), costs.end());
    const Cost max_cost = *std::max_element(costs.begin(), costs.end());

    if (min_cost == max_cost)
    {
      csp.postNullaryConstraint(min_cost);
    }
    else
    {
      csp.postBinaryConstraint(n_1, n_2, costs);
    }
  }
}

struct Processes_Running
{
  int pid;
  int total;
};

enum
{
  TIME_ERROR = -1,
  SOLVER_CRASH = -2
};

void set_usage_constraints(const Problem &problem, WeightedCSP &csp)
{
  int num_nodes = static_cast<int>(problem.node_costs.ranges.size()) - 1;
  vector<int64_t> at_times(problem.intervals_max);
  std::iota(begin(at_times), end(at_times), 0);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> nodeOffsets(num_nodes);
  for (int node = 0; node < num_nodes; ++node)
  {
    int64_t b = problem.node_costs.ranges[node];
    int64_t e = problem.node_costs.ranges[node + 1];
    for (int64_t k = b; k < e; ++k)
    {
      int64_t u = problem.usages[k];
      int64_t domainVal = k - b;
      int64_t usageDelta = -(u);
      nodeOffsets[node].emplace_back(domainVal, usageDelta);
    }
  }
  std::vector<std::unordered_set<int>> vec_of_sets(at_times.size());
  const auto &intervals = problem.intervals;
  for (int i = 0; i < static_cast<int>(intervals.size()); i += 2)
  {
    int node_idx = i / 2;
    int64_t start = intervals[i];
    int64_t end = intervals[i + 1];
    if (end <= start)
    {
      continue;
    }
    for (int j = 0; j < static_cast<int>(at_times.size()); ++j)
    {
      int64_t t = at_times[j];
      if (t >= start && t < end)
      {
        vec_of_sets[j].insert(node_idx);
      }
    }
  }
  for (int i = 0; i < static_cast<int>(vec_of_sets.size()); i++)
  {
    const auto &u_set = vec_of_sets[i];
    std::vector<int> scope(u_set.begin(), u_set.end());
    std::ostringstream oss;
    oss << -problem.usage_limit;
    for (int node : scope)
    {
      auto &offsets = nodeOffsets[node];
      if (!offsets.empty())
      {
        oss << ' ' << offsets.size();
        for (auto &pair : offsets)
        {
          oss << ' ' << pair.first << ' ' << pair.second;
        }
      }
    }
    std::string params = oss.str();
    csp.postKnapsackConstraint(scope, params, false, 1, false);
  }
}

void init_solver_optimal_once(int seed)
{
  static bool initialized = ([seed]
                             {
    tb2init();
    ToulBar2::verbose = -1;
    ToulBar2::vac = 1;
    ToulBar2::seed = seed;
    ToulBar2::decimalPoint = 0;
    initCosts();
    return true; })();
}

WCSP_result optimal_wcsp(const Problem &problem,
                         const vector<int64_t> &min_usages, int timeout)
{
  double tic = omp_get_wtime();
  const Cost top = TOP_LIMIT;
  WeightedCSPSolver *solver = WeightedCSPSolver::makeWeightedCSPSolver(top);
  const std::vector<double> weights(problem.node_costs.ranges.size() - 1, 0);
  add_nodes_and_edges(problem, *solver->getWCSP(), min_usages, weights, -1);
  set_usage_constraints(problem, *solver->getWCSP());
  solver->getWCSP()->sortConstraints();
  tb2checkOptions();
  double toc = omp_get_wtime();

  ToulBar2::startCpuTime = cpuTime();
  ToulBar2::startRealTime = realTime();

  // correct timeout (remove from it preprocessing)
  timeout = std::max(timeout - int(toc - tic + 1), 1);

  signal(SIGINT, timeOut);
  timer(timeout);

  vector<Value> sol;
  Greedy_Result mrs;
  int broken = 0;
  if (solver->solve())
  {
    solver->getSolution(sol);
    auto sol_cost = solver->getSolutionCost();
    mrs = {sol, eval_solution(problem, sol)};
    for (int64_t usage_at_time : mrs.eval_result.usage_at_time)
    {
      if (usage_at_time > problem.usage_limit)
      {
        broken++;
      }
    }
  }
  else
  {
    delete solver;
    return WCSP_result{TIME_ERROR, {}};
  }
  delete solver;
  return {broken, mrs};
}

WCSP_result deep_greedy_wcsp(const Problem &problem, const Greedy_Result &min_usage_rs, int seed, double tic_global,
                             const double total_seconds_running_time, const vector<int64_t> &min_usages,
                             const vector<double> &node_usage_weight,
                             const Processes_Running process_info,
                             std::function<void(const WCSP_result &)> progress_callback,
                             int time_out, bool is_best,
                             const Option_Helper &option_helper)
{
  Xoroshiro128Plus prng(seed);

  if (process_info.pid == IS_OPTIMAL)
  {
    init_solver_optimal_once(seed);
    double tic = omp_get_wtime();
    WCSP_result result = optimal_wcsp(problem, min_usages, time_out);
    double toc = omp_get_wtime();
    if (toc - tic + 1 > time_out)
    {
      if (result.broken == 0)
      {
        result.gr = run_greedy(problem, result.gr, option_helper, 1, 1, prng.rand_int(0, 1000000000));
      }
    }
    return result;
  }
  init_solver_once(seed);

  int broken = 1;
  Greedy_Result mrs;
  const Cost top = TOP_LIMIT;

  vector<Value> sol;
  WeightedCSPSolver *solver = WeightedCSPSolver::makeWeightedCSPSolver(top);

  int ne_seed = -1;
  add_nodes_and_edges(problem, *solver->getWCSP(), min_usages, node_usage_weight, ne_seed);
  solver->getWCSP()->sortConstraints();
  tb2checkOptions();
  double time_left_in_seconds = total_seconds_running_time - (omp_get_wtime() - tic_global) - 1;
  time_out = std::max(std::min<int>(int(time_left_in_seconds), time_out), 1);
  ToulBar2::startCpuTime = cpuTime();
  ToulBar2::startRealTime = realTime();
#ifndef __WIN32__
  signal(SIGINT, timeOut);
  if (time_out > 0)
    timer(time_out);
#endif
  double solver_tic = omp_get_wtime();
  if (solver->solve())
  {
    solver->getSolution(sol);
    auto solver_toc = omp_get_wtime();
    mrs = Greedy_Result{sol, eval_solution(problem, sol)};
    broken = 0;
    for (int64_t usage_at_time : mrs.eval_result.usage_at_time)
    {
      if (usage_at_time > problem.usage_limit)
      {
        broken++;
      }
    }
    if (broken == 0)
    {
        auto prevCost = mrs.eval_result.cost;
        auto tic = omp_get_wtime();
        mrs = run_greedy(problem, mrs, option_helper, 1, 1, prng.rand_int(0, 1000000000));
        auto toc = omp_get_wtime();
        double time_point = omp_get_wtime() - tic_global;
        cout << "# Solver time:" << solver_toc - solver_tic << ", Greedy time: "
            << toc - tic << ", before greedy: " << prevCost << ", after greedy: "
            << mrs.eval_result.cost << " in iteration: " << iterations << " after: " << time_point << " s" << endl;
    }
    delete solver;
    return WCSP_result{broken, mrs};
  }
  else
  {
    delete solver;
    return WCSP_result{TIME_ERROR, {}};
  }
}

WCSP_result compute_deep_greedy_wcsp_forked(const Problem &problem,
                                            const Greedy_Result &min_usage_rs,
                                            int seed,
                                            bool is_best,
                                            double tic_global,
                                            const double total_seconds_running_time,
                                            const vector<int64_t> &min_usages,
                                            const vector<double> &node_usage_weight,
                                            Processes_Running process_info,
                                            int time_out,
                                            const Option_Helper &option_helper)
{
  static std::mutex fork_mutex;
  fork_mutex.lock();
  int finalPipe[2];
  int progressPipe[2];

  if (pipe(finalPipe) == -1 || pipe(progressPipe) == -1)
  {
    fork_mutex.unlock();
    return WCSP_result{};
  }

  pid_t pid = fork();
  fork_mutex.unlock();

  if (pid == -1)
  {
    return WCSP_result{};
  }

  if (pid == 0)
  {
    close(finalPipe[0]);
    close(progressPipe[0]);
#ifdef __linux__
    // ensure the child dies if the parent dies.
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1)
    {
      _exit(EXIT_FAILURE);
    }
#elif defined(__APPLE__)
    std::thread([parent = getppid()]()
                {
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (getppid() == 1) {
          _exit(EXIT_FAILURE);
        }
      } })
        .detach();
#endif
    if (getppid() == 1)
    {
      _exit(EXIT_FAILURE);
    }
    auto progress_callback = [&](const WCSP_result &progress_result)
    {
      auto r = progress_result.serialize(progressPipe[1]);
    };
    WCSP_result finalResult;

    try
    {
      finalResult = deep_greedy_wcsp(problem, min_usage_rs,
                                     seed, tic_global,
                                     total_seconds_running_time,
                                     min_usages,
                                     node_usage_weight,
                                     process_info,
                                     progress_callback,
                                     time_out,
                                     is_best,
                                     option_helper);
    }
    catch (...)
    {
      finalResult.broken = SOLVER_CRASH;
    }

    auto fr = finalResult.serialize(finalPipe[1]);
    close(finalPipe[1]);
    close(progressPipe[1]);
    _exit(0);
  }
  else
  {
    close(finalPipe[1]);
    close(progressPipe[1]);

    std::atomic<bool> childDone{false};
    std::thread progressListener([&]()
                                 {
      while (!childDone) {
        try {
          WCSP_result progress = WCSP_result::deserialize(progressPipe[0]);
          update_best(problem, progress.gr, process_info.pid);
        } catch (...) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      } });
    WCSP_result finalResult;
    try
    {
      finalResult = WCSP_result::deserialize(finalPipe[0]);
    }
    catch (...)
    {
      finalResult.broken = SOLVER_CRASH;
    }
    childDone = true;
    progressListener.join();
    close(finalPipe[0]);
    close(progressPipe[0]);
    waitpid(pid, nullptr, 0);
    update_best(problem, finalResult.gr, process_info.pid);
    return finalResult;
  }
}

struct Const_Data
{
  vector<int64_t> min;
  std::vector<double> node_usage_weight;
};

struct Usages
{
  vector<double> node_usage_weight_lb;
  vector<double> node_usage_weight_ub;
  vector<vector<double>> node_usage_weights;
};

std::pair<Const_Data, Usages> make_usages_func(const Problem &problem, int num_usage_intervals, int seed)
{
  Xoroshiro128Plus prng(seed);

  const int start_offset = 0;
  const auto n = problem.node_costs.ranges.size() - 1;
  vector<int64_t> min_usages(n);
  vector<int64_t> max_usage_at_time(problem.intervals_max - problem.intervals_min + 1, problem.usage_limit);
  std::vector<__int128_t> maximal_possible_usage_at_time(problem.intervals_max, 0);

  for (int64_t i = 0; i < n; ++i)
  {
    const int64_t b = problem.node_costs.ranges[i];
    const int64_t e = problem.node_costs.ranges[i + 1];
    auto max_usage = *std::max_element(problem.usages.begin() + b, problem.usages.begin() + e);
    auto min_usage = *std::min_element(problem.usages.begin() + b, problem.usages.begin() + e);
    min_usages[i] = min_usage;
    auto start_interval = problem.intervals[2 * i];
    auto end_interval = problem.intervals[2 * i + 1];
    for (int64_t j = start_interval; j < end_interval; ++j)
    {
      max_usage_at_time[j - problem.intervals_min] -= min_usage;
      maximal_possible_usage_at_time[j] += max_usage;
    }
  }
  for (int64_t i = 0; i < n; ++i)
  {
    auto start_interval = problem.intervals[2 * i];
    auto end_interval = problem.intervals[2 * i + 1];
    const int64_t b = problem.node_costs.ranges[i];
    const int64_t e = problem.node_costs.ranges[i + 1];
    vector<int64_t> usages(problem.usages.begin() + b, problem.usages.begin() + e);
  }
  vector<double> node_usage_weight(n, 1.0);
  vector<double> node_usage_weight_lb(n, 0.0);
  vector<double> node_usage_weight_ub;
  node_usage_weight_ub = vector<double>(n, pow(10, num_usage_intervals - num_usage_intervals / 2 - 1 + start_offset));

  for (int64_t i = 0; i < node_usage_weight.size(); i++)
  {
    const int64_t b = problem.node_costs.ranges[i];
    auto start_interval = problem.intervals[2 * i];
    auto end_interval = problem.intervals[2 * i + 1];
    auto max_possible_usage = *std::max_element(maximal_possible_usage_at_time.begin() + start_interval,
                                                maximal_possible_usage_at_time.begin() + end_interval);
    if (max_possible_usage <= problem.usage_limit)
    {
      node_usage_weight[i] = 0;
      node_usage_weight_ub[i] = 0;
    }
  }
  vector<vector<double>> node_usage_weights(num_usage_intervals, vector<double>(n, 1));

  for (int64_t i = 0; i < num_usage_intervals; ++i)
  {
    double start_weight = pow(10, -1 * num_usage_intervals / 2 + i + start_offset);
    ;
    for (int64_t j = 0; j < n; ++j)
    {
      node_usage_weights[i][j] = node_usage_weight[j] * start_weight;
    }
  }

  if (seed != -1)
  {
    for (size_t i = 0; i < node_usage_weight_lb.size(); ++i)
    {
      node_usage_weight_lb[i] += node_usage_weight_lb[i] * (prng.rand_double() - 0.5) * 0.25;
      node_usage_weight_ub[i] += node_usage_weight_ub[i] * (prng.rand_double() - 0.5) * 0.25;
    }
    for (auto &v : node_usage_weights)
    {
      for (size_t i = 0; i < v.size(); ++i)
      {
        v[i] += v[i] * (prng.rand_double() - 0.5) * 0.25;
        while (v[i] > node_usage_weight_ub[i])
        {
          v[i] -= v[i] * prng.rand_double() * 0.25;
        }
      }
    }
  }

  return {{min_usages, node_usage_weight},
          {node_usage_weight_lb, node_usage_weight_ub, node_usage_weights}};
}

void update_intervall(const Problem &problem, vector<std::vector<int64_t>> &node_max_concurrent_usages,
                      const WCSP_result &result, int j)
{
  if (result.broken == SOLVER_CRASH || result.broken == TIME_ERROR)
  {
    return;
  }
  for (size_t i = 0; i < node_max_concurrent_usages[j].size(); i++)
  {
    const int64_t b = problem.node_costs.ranges[i];
    auto start_interval = problem.intervals[2 * i];
    auto end_interval = problem.intervals[2 * i + 1];
    node_max_concurrent_usages[j][i] = *std::max_element(
        result.gr.eval_result.usage_at_time.begin() + start_interval,
        result.gr.eval_result.usage_at_time.begin() + end_interval);
  }
}

inline double decayFunction(double t, double initial_value = 1.75, double final_value = 1.5, double decay_rate = 0.25)
{
  return final_value + (initial_value - final_value) * exp(-decay_rate * t);
}

struct Broken_State
{
  int iterations = 0;
  std::vector<bool> had_broken_solution;
  std::vector<bool> had_unbroken_solution;
  std::vector<int8_t> last_all_state;
  std::vector<int> consecutive_kept_all_state;

  explicit Broken_State(int64_t n)
  {
    had_broken_solution = std::vector<bool>(n, false);
    had_unbroken_solution = std::vector<bool>(n, false);
    last_all_state = std::vector<int8_t>(n, 0);
    consecutive_kept_all_state = std::vector<int>(n, 0);
  }
};

void update_usages_func(const Problem &problem, const Const_Data &cd, int num_usage_intervals, Usages &u,
                        const vector<WCSP_result> &results,
                        const vector<std::vector<int64_t>> &node_max_concurrent_usages, Broken_State &bs, WeightTuner tuner)
{
  const auto n = problem.node_costs.ranges.size() - 1;
  iterations++;

  int64_t num_errors = 0;
  for (int64_t j = 0; j < num_usage_intervals; j++)
  {
    if (results[j].broken == TIME_ERROR)
    {
      return;
    }
    if (results[j].broken == SOLVER_CRASH)
    {
      num_errors++;
    }
  }
  if (num_errors == num_usage_intervals)
  {
    return;
  }

  ++bs.iterations;

  if (tuner == WeightTuner::GLOBAL) {
      bool found_unbroken = false;
      bool found_broken = false;
      bool had_broken_after_unbroken = false;
      std::vector<double> max_weight_per_result(num_usage_intervals, 0.0);
      for (int64_t j = 0; j < num_usage_intervals; ++j) {
          const auto &weights_for_result = u.node_usage_weights[j];
          if (!weights_for_result.empty()) {
              max_weight_per_result[j] = *std::max_element(weights_for_result.begin(), weights_for_result.end());
          }
      }

      double max_of_maxes = 0.0;
      double min_of_maxes = 0.0;
      if (!max_weight_per_result.empty()) {
          max_of_maxes = *std::max_element(max_weight_per_result.begin(), max_weight_per_result.end());
          min_of_maxes = *std::min_element(max_weight_per_result.begin(), max_weight_per_result.end());
      }
      double lb = min_of_maxes;
      double ub = max_of_maxes;

      // std::cout << "# Old lb " << lb << " Old ub " << ub << std::endl;

      for (int64_t j = 0; j < num_usage_intervals; j++)
      {
        if (results[j].broken == TIME_ERROR || results[j].broken == SOLVER_CRASH)
        {
            continue;
        }
        // Found broken one and no unbroken before --> New lower bound
        if (results[j].broken > 0 && !found_unbroken){
            found_broken = true;
            lb = max_weight_per_result[j];
        }
        // Found unbroken one, but now its broken, weights a fuzzy reset of upper bound
        if (results[j].broken > 0 && found_unbroken){
            had_broken_after_unbroken = true;
            found_unbroken = false;
            ub = max_of_maxes;
        }
        if (results[j].broken == 0 && !found_unbroken){
            found_unbroken = true;
            ub = max_weight_per_result[j];
        }
      }
      // Lower bound way to high, scale down heavily
      if (!found_broken) {
          lb = lb / 1000.0;
      }
      // Decreased upper bound too much, scale up
      if (!found_unbroken) {
          ub = ub * 2;
      }

      // std::cout << "# New lb " << lb << " New ub " << ub << std::endl;
      for (int64_t j = 0; j < num_usage_intervals; j++)
      {
          for (int64_t i = 0; i < n; i++)
          {
              u.node_usage_weights[j][i] = (lb +
                                      double(j + 1) * (ub - lb) /
                                          double(num_usage_intervals));
          }
      }
  } else if (tuner == WeightTuner::LAGRANGE) {
      throw std::runtime_error("Not implemented yet");
  } else {
    for (int64_t i = 0; i < n; i++)
    {
        bool found_unbroken = false;
        bool found_broken = false;
        bool had_broken_after_unbroken = false;
        if (cd.node_usage_weight[i] == 0)
        {
        continue;
        }

        __int128_t best_unbroken = std::numeric_limits<__int128_t>::max();

        const int64_t b = problem.node_costs.ranges[i];
        const int64_t e = problem.node_costs.ranges[i + 1];
        for (int64_t j = 0; j < num_usage_intervals; j++)
        {
            if (results[j].broken == TIME_ERROR || results[j].broken == SOLVER_CRASH)
            {
                continue;
            }
            auto max_usage = node_max_concurrent_usages[j][i];
            if (max_usage > problem.usage_limit)
            {
                bs.had_broken_solution[i] = true;

                if (!found_unbroken)
                {
                u.node_usage_weight_lb[i] = u.node_usage_weights[j][i];
                }
                had_broken_after_unbroken = true;
                found_broken = true;
            }
            else
            {
                bs.had_unbroken_solution[i] = true;
                if (!found_unbroken || had_broken_after_unbroken || results[j].gr.eval_result.cost < best_unbroken)
                {
                u.node_usage_weight_ub[i] = u.node_usage_weights[j][i];
                best_unbroken = results[j].gr.eval_result.cost;
                }
                found_unbroken = true;
                had_broken_after_unbroken = false;
            }
        }

        if (had_broken_after_unbroken)
        {
            found_unbroken = false;
        }
        if (bs.had_unbroken_solution[i])
        {
            if (!found_unbroken)
            {
                if (bs.last_all_state[i] == 1)
                {
                bs.consecutive_kept_all_state[i]++;
                }
                else
                {
                bs.last_all_state[i] = 1;
                bs.consecutive_kept_all_state[i] = 1;
                }

                double ub_factor = 1.35;
                if (bs.consecutive_kept_all_state[i] > 1)
                {
                ub_factor = std::min(
                    std::pow(1.35, decayFunction(bs.iterations - 1, double(bs.consecutive_kept_all_state[i]), 1.0, 0.175)),
                    100.0);
                }
                u.node_usage_weight_ub[i] = u.node_usage_weight_ub[i] * ub_factor;
            }
            double factor = 1.5;
            if (!bs.had_broken_solution[i])
            {
                factor = 1000.0;
            }
            if (!found_broken)
            {
                if (bs.last_all_state[i] == -1)
                {
                bs.consecutive_kept_all_state[i]++;
                }
                else
                {
                bs.last_all_state[i] = -1;
                bs.consecutive_kept_all_state[i] = 1;
                }
                if (bs.consecutive_kept_all_state[i] > 1)
                {
                factor = std::min(std::pow(1.5,
                                            decayFunction(bs.iterations - 1, 1.25 * double(bs.consecutive_kept_all_state[i]),
                                                        1.0, 0.175)),
                                    1000.0);
                }

                u.node_usage_weight_lb[i] = u.node_usage_weight_lb[i] / factor;
            }
        }
        else
        {
        u.node_usage_weight_ub[i] = u.node_usage_weight_ub[i] * 10000;
        }

        for (int64_t j = 0; j < num_usage_intervals; j++)
        {
        auto lb = u.node_usage_weight_lb[i];
        auto ub = u.node_usage_weight_ub[i];
        u.node_usage_weights[j][i] = (lb +
                                        double(j + 1) * (ub - lb) /
                                            double(num_usage_intervals));
        }
    }
  }
}

struct Task
{
  std::vector<double> weights;
  int id{};
  int seed{};
  int timeout{};
};

struct Problem_Metrics
{
  int64_t nodes;
  int64_t strategies;
  int64_t edges;
  int64_t intervals;
};

Problem_Metrics get_problem_metrics(const Problem &problem)
{
  int64_t nodes = static_cast<int64_t>(problem.node_costs.ranges.size()) - 1;
  int64_t strategies = static_cast<int64_t>(problem.node_costs.strategies.size());
  int64_t edges = static_cast<int64_t>(problem.edge_costs.ranges.size()) - 1;
  int64_t intervals = problem.intervals_max;
  return {nodes, strategies, edges, intervals};
}

std::string get_problem_metrics_str(const Problem &problem)
{
  Problem_Metrics metrics = get_problem_metrics(problem);
  std::ostringstream oss;
  oss << "# " << "nodes: " << metrics.nodes << " edges: " << metrics.edges
      << "\n# strategies: " << metrics.strategies << " intervals: " << metrics.intervals;
  return oss.str();
}

bool is_try_optimal(const Problem &problem)
{
  auto num_nodes = static_cast<int64_t>(problem.node_costs.ranges.size()) - 1;
  auto num_strats = problem.node_costs.strategies.size();
  auto num_intervals = problem.intervals_max;

  int64_t strat_squared_sizes_sum = 0;

  for (int64_t i = 0; i < num_nodes; ++i)
  {
    auto b = problem.node_costs.ranges[i];
    auto e = problem.node_costs.ranges[i + 1];
    strat_squared_sizes_sum += (e - b) * (e - b);
  }
  if (num_nodes < 2000 && num_strats < 50000 && num_intervals < 1600 && strat_squared_sizes_sum < 6000000)
  {
    return true;
  }
  return false;
}

void save_exit()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(110));
  stop_print = true;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  exit(0);
}

class Usage_Manager
{
public:
  int64_t global_seed = 0;
  WeightTuner tuner;
  std::string problem_name;
  Usage_Manager(const Problem &problem, int num_usage_intervals, int best_threads, int64_t seed, int timout_start, int manager_id,
                double tic_global, double total_seconds_running_time, WeightTuner tuner, std::string problem_name) : problem(problem),
                                                                        pnrg(seed),
                                                                        num_usage_intervals(
                                                                            num_usage_intervals),
                                                                        best_threads(best_threads),
                                                                        tic_global(tic_global),
                                                                        total_seconds_running_time(
                                                                            total_seconds_running_time),
                                                                        bs(problem.node_costs.ranges.size() - 1)
  {

    timout = timout_start;
    max_timeout = timout_start * 2 + 1;

    this->tuner = tuner;
    this->global_seed = seed;
    this->problem_name = problem_name;

    if (manager_id != 0)
    {
      std::tie(allowed, usages) = make_usages_func(problem, num_usage_intervals,
                                                   int(pnrg.rand_int(INT32_MIN, INT32_MAX)));
    }
    else
    {
      std::tie(allowed, usages) = make_usages_func(problem, num_usage_intervals, -1);
    }
    const auto n = problem.node_costs.ranges.size() - 1;
    node_max_concurrent_usages = vector<std::vector<int64_t>>(num_usage_intervals, vector<int64_t>(n));
    best_weights.resize(allowed.node_usage_weight.size(), 5.0);
    results = vector<WCSP_result>(num_usage_intervals, WCSP_result(TIME_ERROR, {}));
    reinit_tasks();
  }

  Task get_task()
  {
    Task task;
    if (other_running >= best_threads || !stop_greedy || num_updates == 0)
    {
      while (true)
      {
        task_mutex.lock();
        if (!free_tasks.empty())
        {
          int seed = int(pnrg.rand_int(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max()));
          int id = int(free_tasks.back());
          free_tasks.pop_back();
          task = Task{usages.node_usage_weights[id], id, seed, timout};
          task_mutex.unlock();
          return task;
        }
        task_mutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    else
    {
      task_mutex.lock();
      int seed = int(pnrg.rand_int(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max()));
      if (!free_tasks.empty())
      {
        int id = int(free_tasks.back());
        free_tasks.pop_back();
        task = Task{usages.node_usage_weights[id], id, seed, timout};
      }
      else
      {
        other_running++;
        if (!optimal_pid_executed && is_try_optimal(problem))
        {
          int time_left_in_seconds = std::max<int>(int(total_seconds_running_time - (omp_get_wtime() - tic_global)) - 1,
                                                   1);
          optimal_finish_time = omp_get_wtime() + time_left_in_seconds - 1;
          task = Task{best_weights, IS_OPTIMAL, seed, time_left_in_seconds};
        }
        else
        {
          task = Task{best_weights, IS_BEST, seed, timout};
        }
        optimal_pid_executed = true;
      }
      task_mutex.unlock();
      return task;
    }
  }

  void update_usage(const WCSP_result &wr, const Task &t)
  {
    if (t.id >= 0)
    {
      update_intervall(problem, node_max_concurrent_usages, wr, t.id);
      results[t.id] = wr;
    }
    task_mutex.lock();
    if (wr.gr.eval_result.cost == best_cost)
    {
      if (++same_best >= 100)
      {
        stop = true;
        if (!is_try_optimal(problem))
        {
          save_exit();
        }
      }
    }
    else
    {
      same_best = 0;
    }
    if (wr.gr.eval_result.max_usage_at_time <= problem.usage_limit)
    {
      if (wr.gr.eval_result.cost < best_cost)
      {
        best_cost = wr.gr.eval_result.cost;
        best_weights = t.weights;
        std::string tuner_string = "";
        switch (tuner)
        {
        case WeightTuner::NODES:
            tuner_string = "nodes";
            break;
        case WeightTuner::LAGRANGE:
            tuner_string = "lagrange";
            break;
        case WeightTuner::GLOBAL:
            tuner_string = "global";
            break;
        }
        std::string file_name = this->problem_name+ "_" + tuner_string + "_" + std::to_string(this->global_seed) + "_best_weights.bin";
        write_node_usage_weights_binary(best_weights, file_name);
      }
    }
    is_timeout |= (wr.broken == TIME_ERROR);
    if (t.id >= 0)
    {
      completed++;
    }
    else
    {
      if (t.id == IS_OPTIMAL)
      {
        bool is_found_optimal = omp_get_wtime() < optimal_finish_time;
        if (is_found_optimal && best_cost == wr.gr.eval_result.cost)
        {
          stop = true;
          save_exit();
        }
      }
      other_running--;
    }
    if (completed == num_usage_intervals && free_tasks.empty())
    {
      update_usages_func(problem, allowed, num_usage_intervals, usages, results, node_max_concurrent_usages, bs, this->tuner);
      reinit_tasks();
      num_updates++;
      if (is_timeout)
      {
        timout += 5;
        if (timout >= max_timeout)
        {
          max_timeout += 5;
        }
      }
      else
      {
        timout = std::min(timout + 1, max_timeout);
      }
      is_timeout = false;
    }
    task_mutex.unlock();
  }

  const Const_Data &limits()
  {
    return allowed;
  }

  bool is_stop()
  {
    return stop;
  }

private:
  std::mutex task_mutex;
  const Problem &problem;
  Xoroshiro128Plus pnrg;
  int num_usage_intervals;
  Usages usages;
  Const_Data allowed;
  vector<double> best_weights;
  std::vector<int64_t> free_tasks;
  __int128_t best_cost = std::numeric_limits<__int128_t>::max();
  vector<WCSP_result> results;
  int completed = 0;
  std::atomic<bool> stop = false;
  std::atomic<int64_t> num_updates = 0;
  std::atomic<int64_t> same_best = 0;
  std::atomic<int64_t> other_running = 0;
  int best_threads = 0;
  vector<std::vector<int64_t>> node_max_concurrent_usages;
  int timout = 12;
  int max_timeout = 25;
  bool is_timeout = false;
  bool optimal_pid_executed = false;
  const double tic_global;
  const double total_seconds_running_time;
  double optimal_finish_time = 0;
  Broken_State bs;

  void reinit_tasks()
  {
    completed = 0;
    free_tasks.resize(num_usage_intervals);
    std::iota(begin(free_tasks), end(free_tasks), 0);
    pnrg.shuffle_vector(free_tasks);
  }
};

void greedy_wcsp(const Problem &problem, const Greedy_Result &min_usage_rs, int global_seed, int timout_start, double tic_global,
                 const double total_seconds_running_time, int num_of_process_forks, WeightTuner tuner, string problem_name)
{

  int num_usage_intervals = std::max(num_of_process_forks - 1, 7);
  int num_other_treads = std::max(num_of_process_forks - num_usage_intervals, 0);
  // if you have at least two processes available try optimal if optimal possible
  if (num_of_process_forks >= 2 && is_try_optimal(problem))
  {
    num_other_treads = std::max(num_other_treads, 1);
  }
  int manager_id = 1; // if id == 0 then no randomness in initialization of weights (we want randomness)
  Usage_Manager usage_manager(problem, num_usage_intervals, num_other_treads, global_seed, timout_start, manager_id,
                              tic_global, total_seconds_running_time, tuner, problem_name);
  const Const_Data &limits = usage_manager.limits();
  const Option_Helper option_helper(problem);

#pragma omp parallel num_threads(num_of_process_forks)
  {
    while (true)
    {
      WCSP_result result;
      int pid = omp_get_thread_num();
      Task task = usage_manager.get_task();
      Processes_Running process_info{task.id, omp_get_num_threads()};
      if (task.id >= 0)
      { // for tuning weights it is good to have the same seed for the solver
        task.seed = global_seed;
      }
      result = compute_deep_greedy_wcsp_forked(problem, min_usage_rs, task.seed,
                                               task.id == IS_BEST, tic_global,
                                               total_seconds_running_time,
                                               limits.min,
                                               task.weights,
                                               process_info,
                                               task.timeout,
                                               option_helper);
      usage_manager.update_usage(result, task);
      if (usage_manager.is_stop())
      {
        break;
      }
    }
  }
}



struct CmdOptions
{
  std::string problemPath;
  double timeout = 0;
  int seed = 1;
  int solver_timeout_start = 12;
  //  int numForks = int(PHYSICAL_CORE_COUNT);
  int numForks = 8;
  bool silent = false;
  bool help = false;
  WeightTuner tuner = NODES;
};

std::string getExecutableName(const char *path)
{
  std::string fullPath(path);
  size_t pos = fullPath.find_last_of("/\\");
  if (pos != std::string::npos)
    return fullPath.substr(pos + 1);
  else
    return fullPath;
}

void printHelp(const char *progName)
{
  std::string exeName = getExecutableName(progName);
  std::cout << "Usage: ./" << exeName << " <input_file_path> <timeout_in_seconds> [options]\n"
            << "Positional Arguments:\n"
            << "  input_problem_path     Path to the input JSON file\n"
            << "  timeout_in_seconds     Timeout duration in seconds\n\n"
            << "Options:\n"
            << "  -h                     Show this help message and exit\n"
            << "  -s <seed>              Set the seed value (default: 1)\n"
            << "  -t <timeoutWCSP>       Set the timeout in seconds for internal WCSP solver (default: 12)\n"
            << "  -j <numForks>          Set the number of forks for parallelism (default: 8)\n"
            << "  -q                     Quiet mode (do not print node strategies)\n"
            << "  \n"
            << "Usage Example: ./" << exeName << " example.json 60 -s 42 -t 10 -j 4 -q  \n";
}

CmdOptions parseCommandLine(int argc, char *argv[])
{
  CmdOptions opts;

  if (argc < 3)
  {
    std::cerr << "Error: Missing required positional arguments.\n";
    printHelp(argv[0]);
    std::exit(EXIT_FAILURE);
  }

  opts.problemPath = argv[1];
  opts.timeout = std::stod(argv[2]);

  int remaining = argc - 3;
  if (remaining > 0)
  {
    int flagArgc = remaining + 1;
    char **flagArgv = new char *[flagArgc];
    flagArgv[0] = argv[0];
    for (int i = 3; i < argc; i++)
    {
      flagArgv[i - 2] = argv[i];
    }

    int opt;
    while ((opt = getopt(flagArgc, flagArgv, "hs:t:j:qw:")) != -1)
    {
      switch (opt)
      {
      case 'h':
        opts.help = true;
        break;
      case 's':
        opts.seed = std::atoi(optarg);
        break;
      case 't':
        opts.solver_timeout_start = std::atoi(optarg);
        break;
      case 'j':
        opts.numForks = std::atoi(optarg);
        break;
      case 'q':
        opts.silent = true;
        break;
      case 'w':
        std::cout << "# Weight tuner option set to: " << optarg << std::endl;
        if (std::string(optarg) == "global")
        {
          opts.tuner = WeightTuner::GLOBAL;
        }
        else if (std::string(optarg) == "nodes")
        {
          opts.tuner = WeightTuner::NODES;
        }
        else if (std::string(optarg) == "lagrange")
        {
          opts.tuner = WeightTuner::LAGRANGE;
        }
        else
        {
          opts.help = true;
        }
        break;
      default:
        opts.help = true;
        break;
      }
    }
    delete[] flagArgv;
  }

  if (opts.help)
  {
    printHelp(argv[0]);
    std::exit(EXIT_SUCCESS);
  }
  return opts;
}

std::string filename_without_ext(const std::string &path) {
    const auto slash = path.find_last_of("/\\");
    const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const auto dot = base.rfind('.');
    // No dot, or dot is first char (treat as dotfile => no extension removal)
    if (dot == std::string::npos || dot == 0) {
        return base;
    }
    return base.substr(0, dot);
}

int main(int argc, char *argv[])
{

  CmdOptions options = parseCommandLine(argc, argv);

  const double total_seconds_running_time = options.timeout;
  if (options.timeout < 0)
  {
    std::cerr << "Timeout value is negative. Terminating program." << std::endl;
    exit(1);
  }
  int global_seed = options.seed;
  if (options.numForks <= 0)
  {
    std::cerr << "Number of forks for parallelism is negative or zero. Terminating program." << std::endl;
    exit(1);
  }
  int num_of_process_forks = 8;
  if (options.numForks != 8)
  {
    num_of_process_forks = std::max<int>(std::min<int>(options.numForks, int(PHYSICAL_CORE_COUNT)), 1);
  }
  const int solver_timout_start = options.solver_timeout_start;
  if (options.solver_timeout_start < 1)
  {
    std::cerr << "Timeout for internal solver is < 1. Terminating program." << std::endl;
    exit(1);
  }
  bool is_print_vector = !options.silent;
  std::string path = options.problemPath;

 std::string problem_name = filename_without_ext(path);

  tic_global = omp_get_wtime();

  Problem problem = get_problem_from_path(path);
  auto toc = omp_get_wtime();
  std::cout << "# Preprocessing: " << toc - tic_global << " s\n";
  std::cout << get_problem_metrics_str(problem) << std::endl;

  const Greedy_Result min_usage_rs = get_min_resource_solution(problem);
  best_solution = min_usage_rs;
  print_and_exit_invalid_max_usage(problem.usage_limit, min_usage_rs.eval_result.max_usage_at_time);

  double max_seconds_left = total_seconds_running_time - (omp_get_wtime() - tic_global) - 0.1;
  std::thread print_solution_thread(print_best_solution_thread, omp_get_wtime() + max_seconds_left, is_print_vector);

  std::thread back_ground_greedy_thread(back_ground_greedy, std::ref(problem), std::ref(min_usage_rs), 10000,
                                        max_seconds_left, 1, global_seed);

  greedy_wcsp(problem, min_usage_rs, global_seed, solver_timout_start, tic_global, total_seconds_running_time, num_of_process_forks, options.tuner, problem_name);

  std::this_thread::sleep_for(std::chrono::milliseconds(110));
  stop_print = true;
  print_solution_thread.join();
  back_ground_greedy_thread.join();
  return 0;
}
