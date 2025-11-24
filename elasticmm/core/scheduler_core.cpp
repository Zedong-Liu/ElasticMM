#include "scheduler_core.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

namespace elasticmm {
namespace core {

// SchedulerCore implementation
SchedulerCore::SchedulerCore(int ping_seconds) 
    : rr_prefill_(0), rr_decode_(0), ping_seconds_(ping_seconds) {
}

void SchedulerCore::heartbeat(const std::string& http_address, 
                             const std::string& zmq_address, 
                             const std::string& role) {
    std::lock_guard<std::mutex> lock(lock_);
    
    garbage_collect();
    
    double deadline = get_current_time() + ping_seconds_;
    InstanceInfo info(http_address, zmq_address, role, deadline);
    
    if (role == "P") {
        prefill_instances_[http_address] = info;
    } else if (role == "D") {
        decode_instances_[http_address] = info;
    }
}

void SchedulerCore::remove_instance(const std::string& http_address) {
    std::lock_guard<std::mutex> lock(lock_);
    prefill_instances_.erase(http_address);
    decode_instances_.erase(http_address);
}

void SchedulerCore::garbage_collect() {
    double now = get_current_time();
    
    // Clean up expired prefill instances
    auto prefill_it = prefill_instances_.begin();
    while (prefill_it != prefill_instances_.end()) {
        if (prefill_it->second.last_heartbeat_deadline < now) {
            prefill_it = prefill_instances_.erase(prefill_it);
        } else {
            ++prefill_it;
        }
    }
    
    // Clean up expired decode instances
    auto decode_it = decode_instances_.begin();
    while (decode_it != decode_instances_.end()) {
        if (decode_it->second.last_heartbeat_deadline < now) {
            decode_it = decode_instances_.erase(decode_it);
        } else {
            ++decode_it;
        }
    }
}

std::optional<InstanceInfo> SchedulerCore::select_prefill() {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    return select_from_pool(prefill_instances_, rr_prefill_);
}

std::optional<InstanceInfo> SchedulerCore::select_decode() {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    return select_from_pool(decode_instances_, rr_decode_);
}

std::vector<InstanceInfo> SchedulerCore::select_prefills(int k) {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    return select_multiple_from_pool(prefill_instances_, k, rr_prefill_);
}

std::vector<InstanceInfo> SchedulerCore::select_decodes(int k) {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    return select_multiple_from_pool(decode_instances_, k, rr_decode_);
}

std::optional<InstanceInfo> SchedulerCore::get_decode_with_lowest_utilization() {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    
    auto instances = get_instance_list(decode_instances_);
    if (instances.empty()) {
        return std::nullopt;
    }
    
    // Find instance with lowest load
    auto min_it = std::min_element(instances.begin(), instances.end(),
        [](const InstanceInfo& a, const InstanceInfo& b) {
            return a.load < b.load;
        });
    
    return *min_it;
}

std::vector<InstanceInfo> SchedulerCore::get_all_decode_nodes() {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    return get_instance_list(decode_instances_);
}

ModalityType SchedulerCore::route_request(const Request& request) {
    return request.modality_type;
}

std::vector<Request> SchedulerCore::filter_requests_by_stage(
    const std::vector<Request>& requests, InferenceStage stage) {
    
    std::vector<Request> filtered;
    
    for (const auto& request : requests) {
        bool should_process = false;
        
        switch (stage) {
            case InferenceStage::ENCODE:
                should_process = (request.modality_type == ModalityType::MULTIMODAL && 
                                request.has_images);
                break;
            case InferenceStage::PREFILL:
                should_process = true;
                break;
            case InferenceStage::DECODE:
                should_process = true;
                break;
        }
        
        if (should_process) {
            filtered.push_back(request);
        }
    }
    
    return filtered;
}

std::pair<std::vector<InstanceInfo>, std::vector<InstanceInfo>> 
SchedulerCore::list_instances() {
    std::lock_guard<std::mutex> lock(lock_);
    garbage_collect();
    
    return {get_instance_list(prefill_instances_), 
            get_instance_list(decode_instances_)};
}

int SchedulerCore::get_total_instances() const {
    std::lock_guard<std::mutex> lock(lock_);
    return prefill_instances_.size() + decode_instances_.size();
}

double SchedulerCore::get_system_load() const {
    std::lock_guard<std::mutex> lock(lock_);
    
    double total_load = 0.0;
    int total_instances = 0;
    
    for (const auto& instance : prefill_instances_) {
        total_load += instance.second.load;
        total_instances++;
    }
    
    for (const auto& instance : decode_instances_) {
        total_load += instance.second.load;
        total_instances++;
    }
    
    return total_instances > 0 ? total_load / total_instances : 0.0;
}

void SchedulerCore::update_instance_load(const std::string& http_address, double load) {
    std::lock_guard<std::mutex> lock(lock_);
    
    auto prefill_it = prefill_instances_.find(http_address);
    if (prefill_it != prefill_instances_.end()) {
        prefill_it->second.load = load;
        return;
    }
    
    auto decode_it = decode_instances_.find(http_address);
    if (decode_it != decode_instances_.end()) {
        decode_it->second.load = load;
    }
}

double SchedulerCore::calculate_load_imbalance() const {
    std::lock_guard<std::mutex> lock(lock_);
    
    std::vector<double> loads;
    
    for (const auto& instance : prefill_instances_) {
        loads.push_back(instance.second.load);
    }
    
    for (const auto& instance : decode_instances_) {
        loads.push_back(instance.second.load);
    }
    
    if (loads.empty()) {
        return 0.0;
    }
    
    double mean_load = std::accumulate(loads.begin(), loads.end(), 0.0) / loads.size();
    double variance = 0.0;
    
    for (double load : loads) {
        variance += (load - mean_load) * (load - mean_load);
    }
    
    return std::sqrt(variance / loads.size());
}

bool SchedulerCore::should_rebalance() const {
    return calculate_load_imbalance() > 0.3;  // Threshold for rebalancing
}

// Private helper methods
void SchedulerCore::cleanup_expired_instances() {
    garbage_collect();
}

std::vector<InstanceInfo> SchedulerCore::get_instance_list(
    const std::unordered_map<std::string, InstanceInfo>& pool) {
    
    std::vector<InstanceInfo> instances;
    instances.reserve(pool.size());
    
    for (const auto& pair : pool) {
        instances.push_back(pair.second);
    }
    
    return instances;
}

std::optional<InstanceInfo> SchedulerCore::select_from_pool(
    const std::unordered_map<std::string, InstanceInfo>& pool, int& rr_counter) {
    
    auto instances = get_instance_list(pool);
    if (instances.empty()) {
        return std::nullopt;
    }
    
    rr_counter = (rr_counter + 1) % instances.size();
    return instances[rr_counter];
}

std::vector<InstanceInfo> SchedulerCore::select_multiple_from_pool(
    const std::unordered_map<std::string, InstanceInfo>& pool, int k, int& rr_counter) {
    
    auto instances = get_instance_list(pool);
    if (instances.empty()) {
        return {};
    }
    
    int actual_k = std::min(k, static_cast<int>(instances.size()));
    std::vector<InstanceInfo> selected;
    selected.reserve(actual_k);
    
    int start = rr_counter;
    for (int i = 0; i < actual_k; ++i) {
        selected.push_back(instances[(start + i) % instances.size()]);
    }
    
    rr_counter = (start + 1) % instances.size();
    return selected;
}

double SchedulerCore::get_current_time() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// LoadBalancerCore implementation
std::string LoadBalancerCore::select_best_instance(
    const std::vector<InstanceInfo>& instances, const Request& request) {
    
    if (instances.empty()) {
        return "";
    }
    
    static int round_robin_counter = 0;
    
    switch (strategy_) {
        case Strategy::ROUND_ROBIN:
            return round_robin_selection(instances, round_robin_counter);
        case Strategy::LEAST_LOADED:
            return least_loaded_selection(instances);
        case Strategy::WEIGHTED_ROUND_ROBIN:
            return weighted_round_robin_selection(instances, round_robin_counter);
        case Strategy::LEAST_CONNECTIONS:
            return least_connections_selection(instances);
        default:
            return least_loaded_selection(instances);
    }
}

std::vector<std::string> LoadBalancerCore::distribute_requests(
    const std::vector<Request>& requests, const std::vector<InstanceInfo>& instances) {
    
    if (instances.empty()) {
        return {};
    }
    
    std::vector<std::string> assignments;
    assignments.reserve(requests.size());
    
    for (const auto& request : requests) {
        std::string selected = select_best_instance(instances, request);
        assignments.push_back(selected);
    }
    
    return assignments;
}

double LoadBalancerCore::calculate_instance_load(
    const InstanceInfo& instance, const Request& request) {
    
    // Simple load calculation based on request size and current load
    double request_load = request.estimate_memory_usage() * 0.1;  // Scale factor
    return instance.load + request_load;
}

double LoadBalancerCore::calculate_system_load(
    const std::vector<InstanceInfo>& instances) {
    
    if (instances.empty()) {
        return 0.0;
    }
    
    double total_load = 0.0;
    for (const auto& instance : instances) {
        total_load += instance.load;
    }
    
    return total_load / instances.size();
}

std::string LoadBalancerCore::round_robin_selection(
    const std::vector<InstanceInfo>& instances, int& counter) {
    
    counter = (counter + 1) % instances.size();
    return instances[counter].http_address;
}

std::string LoadBalancerCore::least_loaded_selection(
    const std::vector<InstanceInfo>& instances) {
    
    auto min_it = std::min_element(instances.begin(), instances.end(),
        [](const InstanceInfo& a, const InstanceInfo& b) {
            return a.load < b.load;
        });
    
    return min_it->http_address;
}

std::string LoadBalancerCore::weighted_round_robin_selection(
    const std::vector<InstanceInfo>& instances, int& counter) {
    
    // Simplified weighted round-robin based on inverse load
    std::vector<double> weights;
    weights.reserve(instances.size());
    
    for (const auto& instance : instances) {
        double weight = instance.load > 0 ? 1.0 / (1.0 + instance.load) : 1.0;
        weights.push_back(weight);
    }
    
    // Select based on cumulative weights
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, total_weight);
    
    double random_value = dis(gen);
    double cumulative = 0.0;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        cumulative += weights[i];
        if (random_value <= cumulative) {
            counter = i;
            return instances[i].http_address;
        }
    }
    
    return instances[0].http_address;  // Fallback
}

std::string LoadBalancerCore::least_connections_selection(
    const std::vector<InstanceInfo>& instances) {
    
    // For simplicity, use load as proxy for connections
    return least_loaded_selection(instances);
}

// RequestSchedulerCore implementation
std::unordered_map<ModalityType, std::vector<Request>> 
RequestSchedulerCore::group_requests_by_modality(const std::vector<Request>& requests) {
    
    std::unordered_map<ModalityType, std::vector<Request>> grouped;
    
    for (const auto& request : requests) {
        grouped[request.modality_type].push_back(request);
    }
    
    return grouped;
}

std::vector<Request> RequestSchedulerCore::prioritize_requests(
    const std::vector<Request>& requests) {
    
    std::vector<Request> prioritized = requests;
    
    // Sort by priority (lower number = higher priority) and timestamp
    std::sort(prioritized.begin(), prioritized.end(),
        [](const Request& a, const Request& b) {
            if (a.priority != b.priority) {
                return a.priority < b.priority;
            }
            return a.timestamp < b.timestamp;
        });
    
    return prioritized;
}

std::vector<Request> RequestSchedulerCore::filter_by_priority(
    const std::vector<Request>& requests, int min_priority) {
    
    std::vector<Request> filtered;
    
    for (const auto& request : requests) {
        if (request.priority <= min_priority) {
            filtered.push_back(request);
        }
    }
    
    return filtered;
}

RequestSchedulerCore::BatchResult RequestSchedulerCore::process_request_batch(
    const std::vector<Request>& requests,
    const std::vector<InstanceInfo>& prefill_instances,
    const std::vector<InstanceInfo>& decode_instances) {
    
    BatchResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prioritize requests
    auto prioritized_requests = prioritize_requests(requests);
    
    // Process prefill stage
    auto prefill_requests = process_prefill_stage(prioritized_requests, prefill_instances);
    
    // Process decode stage
    auto decode_requests = process_decode_stage(prefill_requests, decode_instances);
    
    result.processed_requests = decode_requests;
    result.failed_requests.reserve(requests.size() - decode_requests.size());
    
    // Identify failed requests
    std::unordered_set<std::string> processed_ids;
    for (const auto& req : result.processed_requests) {
        processed_ids.insert(req.request_id);
    }
    
    for (const auto& req : requests) {
        if (processed_ids.find(req.request_id) == processed_ids.end()) {
            result.failed_requests.push_back(req);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_processing_time = std::chrono::duration<double>(end_time - start_time).count();
    result.total_instances_used = prefill_instances.size() + decode_instances.size();
    
    return result;
}

std::vector<Request> RequestSchedulerCore::process_prefill_stage(
    const std::vector<Request>& requests, const std::vector<InstanceInfo>& instances) {
    
    if (instances.empty()) {
        return {};
    }
    
    // Distribute requests to prefill instances
    auto assignments = load_balancer_.distribute_requests(requests, instances);
    
    // For simplicity, assume all requests are processed successfully
    return requests;
}

std::vector<Request> RequestSchedulerCore::process_decode_stage(
    const std::vector<Request>& requests, const std::vector<InstanceInfo>& instances) {
    
    if (instances.empty()) {
        return {};
    }
    
    // Distribute requests to decode instances
    auto assignments = load_balancer_.distribute_requests(requests, instances);
    
    // For simplicity, assume all requests are processed successfully
    return requests;
}

} // namespace core
} // namespace elasticmm



