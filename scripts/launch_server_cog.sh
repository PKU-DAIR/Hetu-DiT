export LOG_LEVEL="INFO"
python ../hetu_dit/entrypoint/api_server.py --host 127.0.0.1 --port 8000 --model-class cogvideox --model /Path/to/your/models/CogVideoX1.5-5B --adjust_strategy cache --machine_nums 1 --search-mode multi_machine_efficient_ilp --scheduler-strategy multi_machine_efficient_ilp --stage_level
