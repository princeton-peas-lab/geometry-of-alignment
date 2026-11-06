# Convenience launchers for Hydra multirun sweeps of the paper's fp32 protocol
# (conf/sweep/paper_fp32.yaml).
#
# Sweeps are plain Hydra multiruns: the grid axes are `-m key=v1,v2,...` and the
# launcher selects local (sequential) vs SLURM (submitit; the default here).
#   make paper-fp32                 # Table 1 finetunes: model x dataset x seed, each model's fp32 LR
#   make paper-fp32 SEEDS=42 LRS=1e-4,5e-5,3e-5,2e-5,1e-5   # the LR search
#   make repr-fim-sweep             # intervention finetunes: model x dataset x seed x lambda
#   make paper-fp32 LAUNCHER=local MODELS=llama32-3b-instruct DATASETS=alpaca SEEDS=42
#
# The 9B does not fit one GPU in fp32 (it trains on 2 with FSDP2) and the paper's safety
# scoring is not part of these sweeps: scripts/reproduce/ runs both. Requires
# `pip install hydra-submitit-launcher` for LAUNCHER=slurm.

PY ?= .venv/bin/python
LAUNCHER ?= slurm
MODELS ?= llama32-3b-instruct,qwen3-4b-instruct
DATASETS ?= pure_bad,alpaca,samsum,gsm8k,risky_financial_advice,bad_medical_advice
LAMBDA_DATASETS ?= alpaca,samsum,gsm8k,risky_financial_advice,bad_medical_advice
SEEDS ?= 42,1,2
LRS ?=
LAMBDAS ?= 0.0,-0.25,-0.5,-0.75,-1.0

# Empty LRS: each model's LR from conf/experiment/default.yaml.
LR_AXIS = $(if $(LRS),train_args.learning_rate=$(LRS))

.PHONY: test paper-fp32 repr-fim-sweep

test:
	OMP_NUM_THREADS=4 $(PY) -m pytest -q tests

# Table 1 finetunes (and, with LRS, the LR search): model x dataset x seed [x lr].
paper-fp32:
	$(PY) cli/train.py -m +sweep=paper_fp32 launcher=$(LAUNCHER) \
	  model=$(MODELS) dataset=$(DATASETS) seed=$(SEEDS) $(LR_AXIS) \
	  hydra.job.chdir=false

# Representation-FIM intervention finetunes (the intervention figure): model x dataset x seed x lambda.
# Needs each model's subspace (cli/build_subspace.py method=repr_fim_subspace).
repr-fim-sweep:
	$(PY) cli/train.py -m +sweep=paper_fp32 launcher=$(LAUNCHER) \
	  experiment=finetune_repr_fim_subspace \
	  subspace_intervention.enabled=true subspace_intervention.method=repr_fim_subspace \
	  subspace_intervention.lambda_scale=$(LAMBDAS) \
	  model=$(MODELS) dataset=$(LAMBDA_DATASETS) seed=$(SEEDS) $(LR_AXIS) \
	  hydra.job.chdir=false
