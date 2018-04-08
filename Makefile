MODEL_NAME = squad_all_0311
CKPT_DIR = checkpoints
EMA_CKPT_DIR = experiments/$(MODEL_NAME)/ema_best_checkpoint
TINY_DEV_DIR = data/tiny-dev.json
DEV_DIR = data/dev-v1.1.json
PREDOUT_DIR = experiments/$(MODEL_NAME)

tiny-dev:
		python code_ensemble/main.py --mode=official_eval --json_in_path=$(TINY_DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/tiny-dev-predictions.json --single_ensemble=single
		python code_ensemble/evaluate.py $(TINY_DEV_DIR) $(PREDOUT_DIR)/tiny-dev-predictions.json
dev:
		python code_ensemble/main.py --mode=official_eval --json_in_path=$(DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/dev-predictions.json --single_ensemble=single
		python code_ensemble/evaluate.py $(DEV_DIR) $(PREDOUT_DIR)/dev-predictions.json
tiny-dev-ensemble:
		python code_ensemble/main.py --mode=official_eval --json_in_path=$(TINY_DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/ensemble-tiny-dev-predictions.json --single_ensemble=ensemble
		python code_ensemble/evaluate.py $(TINY_DEV_DIR) $(PREDOUT_DIR)/ensemble-tiny-dev-predictions.json
dev-ensemble:
		python code_ensemble/main.py --mode=official_eval --json_in_path=$(DEV_DIR) --ckpt_load_dir=$(CKPT_DIR) --json_out_path=$(PREDOUT_DIR)/ensemble-dev-predictions.json --single_ensemble=ensemble
		python code_ensemble/evaluate.py $(DEV_DIR) $(PREDOUT_DIR)/ensemble-dev-predictions.json
