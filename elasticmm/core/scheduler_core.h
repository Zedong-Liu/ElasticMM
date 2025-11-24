#ifndef SCHEDULER_CORE_H
#define SCHEDULER_CORE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <memory>
#include <optional>

namespace elasticmm {
namespace core {

// Forward declarations
enum class ModalityType {
    TEXT_ONLY = 0,
    MULTIMODAL = 1
};

enum class InferenceStage {
    ENCODE = 0,
    PREFILL = 1,
    DECODE = 2
};

// Core data structures
struct InstanceInfo {
    std::string http_address;
    std::string zmq_address;
    std::string role;  // "P" for prefill, "D" for decode
    double last_heartbeat_deadline;
    double load = 0.0;
    
    InstanceInfo() = default;
    InstanceInfo(const std::string& http_addr, const std::string& zmq_addr, 
                 const std::string& role, double deadline)
        : http_address(http_addr), zmq_address(zmq_addr), role(role), 
          last_heartbeat_deadline(deadline) {}
};

struct Request {
    std::string request_id;
    ModalityType modality_type;
    int input_length;
    int estimated_output_length;
    int priority = 1;
    double timestamp;
    bool has_images = false;
    
    Request() = default;
    Request(const std::string& id, ModalityType modality, int input_len, int output_len)
        : request_id(id), modality_type(modality), input_length(input_len), 
          estimated_output_length(output_len), timestamp(get_current_time()) {}
    
    double estimate_memory_usage() const {
        double base_memory = 1.0;  // Base memory 1GB
        double input_memory = input_length * 0.001;  // ~1MB per token
        double output_memory = estimated_output_length * 0.001;
        return base_memory + input_memory + output_memory;
    }
    
private:
    static double get_current_time() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now.time_since_epoch()).count();
    }
};

// Core scheduler class
class SchedulerCore {
public:
    explicit SchedulerCore(int ping_seconds = 60);
    ~SchedulerCore() = default;
    
    // Instance management
    void heartbeat(const std::string& http_address, const std::string& zmq_address, 
                  const std::string& role);
    void remove_instance(const std::string& http_address);
    void garbage_collect();
    
    // Instance selection algorithms
    std::optional<InstanceInfo> select_prefill();
    std::optional<InstanceInfo> select_decode();
    std::vector<InstanceInfo> select_prefills(int k);
    std::vector<InstanceInfo> select_decodes(int k);
    std::optional<InstanceInfo> get_decode_with_lowest_utilization();
    std::vector<InstanceInfo> get_all_decode_nodes();
    
    // Request processing
    ModalityType route_request(const Request& request);
    std::vector<Request> filter_requests_by_stage(const std::vector<Request>& requests, 
                                                  InferenceStage stage);
    
    // System status
    std::pair<std::vector<InstanceInfo>, std::vector<InstanceInfo>> list_instances();
    int get_total_instances() const;
    double get_system_load() const;
    
    // Load balancing utilities
    void update_instance_load(const std::string& http_address, double load);
    double calculate_load_imbalance() const;
    bool should_rebalance() const;
    
private:
    // Core data structures
    std::unordered_map<std::string, InstanceInfo> prefill_instances_;
    std::unordered_map<std::string, InstanceInfo> decode_instances_;
    
    // Thread safety
    mutable std::mutex lock_;
    
    // Round-robin counters
    int rr_prefill_;
    int rr_decode_;
    
    // Configuration
    int ping_seconds_;
    
    // Helper methods
    void cleanup_expired_instances();
    std::vector<InstanceInfo> get_instance_list(const std::unordered_map<std::string, InstanceInfo>& pool);
    std::optional<InstanceInfo> select_from_pool(const std::unordered_map<std::string, InstanceInfo>& pool, 
                                                 int& rr_counter);
    std::vector<InstanceInfo> select_multiple_from_pool(const std::unordered_map<std::string, InstanceInfo>& pool, 
                                                        int k, int& rr_counter);
    double get_current_time() const;
};

// Load balancer core
class LoadBalancerCore {
public:
    LoadBalancerCore() = default;
    ~LoadBalancerCore() = default;
    
    // Load balancing algorithms
    std::string select_best_instance(const std::vector<InstanceInfo>& instances, 
                                   const Request& request);
    std::vector<std::string> distribute_requests(const std::vector<Request>& requests,
                                                const std::vector<InstanceInfo>& instances);
    
    // Load calculation
    double calculate_instance_load(const InstanceInfo& instance, const Request& request);
    double calculate_system_load(const std::vector<InstanceInfo>& instances);
    
    // Load balancing strategies
    enum class Strategy {
        ROUND_ROBIN,
        LEAST_LOADED,
        WEIGHTED_ROUND_ROBIN,
        LEAST_CONNECTIONS
    };
    
    void set_strategy(Strategy strategy) { strategy_ = strategy; }
    Strategy get_strategy() const { return strategy_; }
    
private:
    Strategy strategy_ = Strategy::LEAST_LOADED;
    
    // Strategy implementations
    std::string round_robin_selection(const std::vector<InstanceInfo>& instances, int& counter);
    std::string least_loaded_selection(const std::vector<InstanceInfo>& instances);
    std::string weighted_round_robin_selection(const std::vector<InstanceInfo>& instances, int& counter);
    std::string least_connections_selection(const std::vector<InstanceInfo>& instances);
};

// Request scheduler core
class RequestSchedulerCore {
public:
    RequestSchedulerCore() = default;
    ~RequestSchedulerCore() = default;
    
    // Request scheduling
    std::unordered_map<ModalityType, std::vector<Request>> group_requests_by_modality(
        const std::vector<Request>& requests);
    
    std::vector<Request> prioritize_requests(const std::vector<Request>& requests);
    std::vector<Request> filter_by_priority(const std::vector<Request>& requests, int min_priority);
    
    // Batch processing
    struct BatchResult {
        std::vector<Request> processed_requests;
        std::vector<Request> failed_requests;
        double total_processing_time;
        int total_instances_used;
    };
    
    BatchResult process_request_batch(const std::vector<Request>& requests,
                                    const std::vector<InstanceInfo>& prefill_instances,
                                    const std::vector<InstanceInfo>& decode_instances);
    
private:
    LoadBalancerCore load_balancer_;
    
    // Processing helpers
    std::vector<Request> process_prefill_stage(const std::vector<Request>& requests,
                                             const std::vector<InstanceInfo>& instances);
    std::vector<Request> process_decode_stage(const std::vector<Request>& requests,
                                            const std::vector<InstanceInfo>& instances);
};

} // namespace core
} // namespace elasticmm

#endif // SCHEDULER_CORE_H



