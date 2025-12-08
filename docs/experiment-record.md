# experiment record

## Test vllm single tenant

yunwei37@lab:~/workspace/gpu/schedcp/workloads/vllm$ python uvm/test_uvm_baselines.py --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts 1000 --dataset-path /home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --seed 42 --request-rate 5"

