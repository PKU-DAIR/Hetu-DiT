export LOG_LEVEL="INFO"
python ../examples/api_benchmark_example_with_trace.py \
  --url "http://[::1]:8000" \
  --trace ../data/flux_trace.txt