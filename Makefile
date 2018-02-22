MODEL_NAME = baseline
CKPT_DIR = experiments/$(MODEL_NAME)/best_checkpoint
TINY_DEV_DIR = data/tiny-dev.json
DEV_DIR = data/dev-v1.1.json
PREDOUT_DIR = experiments/$(MODEL_NAME)

tiny-dev:
		python code/main.py --mode=official_eval --json_in_path=$(TINY_DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/tiny-dev-predictions.json
		python code/evaluate.py $(TINY_DEV_DIR) $(PREDOUT_DIR)/tiny-dev-predictions.json
dev:
		python code/main.py --mode=official_eval --json_in_path=$(DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/dev-predictions.json
		python code/evaluate.py $(DEV_DIR) $(PREDOUT_DIR)/dev-predictions.json
