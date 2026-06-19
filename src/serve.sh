MODEL_DIR="outputs/sft-granite-merged"   # the merged dir from training
 
vllm serve "$MODEL_DIR" \
  --served-model-name granite-sft \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --port 8000