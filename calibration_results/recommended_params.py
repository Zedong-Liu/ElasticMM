# Gain-Cost Model Parameters (Calibrated on 2025-11-24 14:23:07)
# Hardware: /root/lzd/model/qwen2.5-VL

# Base latencies (ms per token)
ENCODING_LATENCY_MS_PER_TOKEN = 1.021
PREFILL_LATENCY_MS_PER_TOKEN = 1.253
DECODE_LATENCY_MS_PER_TOKEN = 37.331

# Migration cost (seconds per migration operation)
MIGRATION_COST = 0.0101

# Preemption penalty (slowdown factor, e.g., 0.2 = 20% slowdown)
PREEMPTION_PENALTY = 0.004

# Scalability coefficients (parallel efficiency)
SCALABILITY_ENCODE = 0.80
SCALABILITY_PREFILL = 0.90
SCALABILITY_DECODING = 0.75

# Max token budget (total tokens in decode batch at capacity)
MAX_DECODE_TOKEN_BUDGET = 40768
