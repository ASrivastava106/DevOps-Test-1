# devops_test_calculations.py
#
# This script computes simple capacity, TFLOPs and cost estimates
# for the two use cases described in the DevOps test:
#
#   Use Case 1: Scalable Machine Translation System
#   Use Case 2: Scalable LLM-Based Chat System
#
# Assumptions are explicitly stated and can be referenced in the writeup.

# -----------------------------
# Global assumptions
# -----------------------------

GPU_TFLOPS = 90               # Peak FP32 throughput per GPU (tera-FLOPs per second)
GPU_HOURLY_COST = 2.0         # Assumed cost per GPU-hour in USD

# -----------------------------
# Use Case 1: Machine Translation
# -----------------------------

# Given in problem: 1000 words/minute on 90 TFLOP server
WORDS_PER_MIN_TRANSLATION = 1000                  # words per minute per GPU
WORDS_PER_HOUR_TRANSLATION = WORDS_PER_MIN_TRANSLATION * 60  # 60,000 words/hour

def translation_estimates(words_per_day: int):
    """
    Return (gpu_hours, total_tflops_used, cost_usd) for translation load.
    - gpu_hours: how many GPU-hours needed per day
    - total_tflops_used: total tera-FLOPs consumed per day
    - cost_usd: daily GPU cost in USD
    """
    gpu_hours = words_per_day / WORDS_PER_HOUR_TRANSLATION
    # Total TFLOPs used = gpu_hours * 3600 seconds * 90 TFLOP/s
    total_tflops = gpu_hours * 3600 * GPU_TFLOPS
    cost = gpu_hours * GPU_HOURLY_COST
    return gpu_hours, total_tflops, cost

def translation_doc_time(words: int, num_gpus: int = 1) -> float:
    """
    Time (in minutes) to translate a document of size `words` on `num_gpus` GPUs.
    """
    words_per_min_total = WORDS_PER_MIN_TRANSLATION * num_gpus
    minutes = words / words_per_min_total
    return minutes

def print_translation_summary():
    print("=" * 80)
    print("USE CASE 1: MACHINE TRANSLATION - CAPACITY & COST ESTIMATES")
    print("Assumptions:")
    print(f"  - 1 GPU can translate {WORDS_PER_MIN_TRANSLATION} words/minute "
          f"({WORDS_PER_HOUR_TRANSLATION} words/hour)")
    print(f"  - GPU peak performance: {GPU_TFLOPS} TFLOPs/s")
    print(f"  - GPU hourly cost: ${GPU_HOURLY_COST}/hour\n")

    # SLO check for 10k-word document
    target_words = 10_000
    for gpus in [1, 2]:
        minutes = translation_doc_time(target_words, num_gpus=gpus)
        print(f"Time to translate a {target_words:,}-word document on {gpus} GPU(s): "
              f"{minutes:.2f} minutes")
    print()

    scenarios = [10_000, 100_000, 1_000_000]
    print("Daily translation load scenarios:")
    print("Words/day | GPU-hours/day | Total TFLOPs used/day | Cost/day (USD)")
    print("-" * 80)
    for words in scenarios:
        gpu_hours, total_tflops, cost = translation_estimates(words)
        print(f"{words:9,d} | {gpu_hours:13.2f} | {total_tflops:21.0f} | ${cost:10.2f}")
    print()


# -----------------------------
# Use Case 2: LLM Chat System
# -----------------------------

# Assumptions for LLM inference:
LLM_PARAMS_BILLIONS = 13                  # 13B parameter model
FLOPS_PER_TOKEN = 2 * LLM_PARAMS_BILLIONS * 1e9
# ≈ 2 × params FLOPs per token  → 26e9 FLOPs/token here

TOKENS_PER_USER_PER_DAY = 3000            # avg tokens per user per day (prompt + completion)

def llm_chat_estimates(users_per_day: int):
    """
    Return (tokens, gpu_hours, total_tflops_used, cost_usd) for chat load.
    - tokens: total tokens processed per day
    - gpu_hours: GPU-hours needed per day
    - total_tflops_used: total tera-FLOPs used per day
    - cost_usd: daily GPU cost in USD
    """
    tokens = users_per_day * TOKENS_PER_USER_PER_DAY
    flops_day = tokens * FLOPS_PER_TOKEN
    flops_per_second = GPU_TFLOPS * 1e12
    gpu_seconds = flops_day / flops_per_second
    gpu_hours = gpu_seconds / 3600
    total_tflops = gpu_hours * 3600 * GPU_TFLOPS
    cost = gpu_hours * GPU_HOURLY_COST
    return tokens, gpu_hours, total_tflops, cost

def print_llm_chat_summary():
    print("=" * 80)
    print("USE CASE 2: LLM CHAT - CAPACITY & COST ESTIMATES")
    print("Assumptions:")
    print(f"  - Model size: ~{LLM_PARAMS_BILLIONS}B parameters")
    print(f"  - FLOPs per token ≈ 2 × params ≈ {FLOPS_PER_TOKEN/1e9:.0f} GFLOPs/token")
    print(f"  - Avg tokens per user/day: {TOKENS_PER_USER_PER_DAY} tokens")
    print(f"  - GPU peak performance: {GPU_TFLOPS} TFLOPs/s")
    print(f"  - GPU hourly cost: ${GPU_HOURLY_COST}/hour\n")

    scenarios = [100, 10_000, 100_000]
    print("Daily user load scenarios:")
    print("Users/day | Tokens/day     | GPU-hours/day | Total TFLOPs used/day | Cost/day (USD)")
    print("-" * 90)
    for users in scenarios:
        tokens, gpu_hours, total_tflops, cost = llm_chat_estimates(users)
        print(f"{users:9,d} | {tokens:13,d} | {gpu_hours:13.2f} | "
              f"{total_tflops:21.0f} | ${cost:10.2f}")
    print()


# -----------------------------
# Main entry point
# -----------------------------

if __name__ == "__main__":
    print_translation_summary()
    print_llm_chat_summary()
