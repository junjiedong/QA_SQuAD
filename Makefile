MODEL_NAME = squad_all_0311
CKPT_DIR = experiments/$(MODEL_NAME)/best_checkpoint
EMA_CKPT_DIR = experiments/$(MODEL_NAME)/ema_best_checkpoint
TINY_DEV_DIR = data/tiny-dev.json
DEV_DIR = data/dev-v1.1.json
PREDOUT_DIR = experiments/$(MODEL_NAME)

tiny-dev:
		python code/main.py --mode=official_eval --json_in_path=$(TINY_DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/tiny-dev-predictions.json
		python code/evaluate.py $(TINY_DEV_DIR) $(PREDOUT_DIR)/tiny-dev-predictions.json
dev:
		python code/main.py --mode=official_eval --json_in_path=$(DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/dev-predictions.json
		python code/evaluate.py $(DEV_DIR) $(PREDOUT_DIR)/dev-predictions.json
tiny-dev-ema:
		python code/main.py --mode=official_eval --json_in_path=$(TINY_DEV_DIR) --ckpt_load_dir=$(EMA_CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/ema-tiny-dev-predictions.json
		python code/evaluate.py $(TINY_DEV_DIR) $(PREDOUT_DIR)/ema-tiny-dev-predictions.json
dev-ema:
		python code/main.py --mode=official_eval --json_in_path=$(DEV_DIR) --ckpt_load_dir=$(EMA_CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/ema-dev-predictions.json
		python code/evaluate.py $(DEV_DIR) $(PREDOUT_DIR)/ema-dev-predictions.json
