# Breaking the Tokenizer Barrier: On-Policy Distillation across Model Families

This repository contains an implementation of Cross-Tokenizer On-Policy Distillation
built on top of `verl`.

---

## Overview

The core OPD switches are:

- `algorithm.adv_estimator=opd`
- `actor_rollout_ref.actor.policy_loss.loss_mode=opd`
- `reward_model.reward_manager=opd`
- `actor_rollout_ref.actor.policy_loss.opd_loss_max_clamp`

---

## Project Structure

```text
On-Policy-Distill/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- setup.py
|-- cross_distill.sh
|-- examples/
|   `-- rollout_correction/
|-- recipe/
|   `-- one_step_off_policy/
`-- verl/
    |-- trainer/
    |   |-- main_ppo.py
    |   |-- main_eval.py
    |   |-- main_generation.py
    |   |-- config/
    |   |   `-- algorithm/
    |   |       `-- rollout_correction.yaml
    |   `-- ppo/
    |       |-- core_algos.py
    |       `-- rollout_corr_helper.py
    `-- workers/
        `-- reward_manager/
            `-- opd.py
```

Key files:

- `cross_distill.sh`: main OPD launch script.
- `verl/workers/reward_manager/opd.py`: teacher client, token alignment, and
  OPD reward-manager logic.
- `verl/trainer/ppo/core_algos.py`: OPD advantage estimator and OPD policy
  loss registration.
- `verl/trainer/main_ppo.py`: PPO trainer entry point with teacher environment
  variables forwarded to Ray workers.
- `recipe/gkd/teacher`: OPD teacher server.

---

## Quick Start

### 1. Setup and Environment


```bash
conda activate baseline
pip install -r requirements.txt
pip install .
```

For an existing environment, review dependency changes before installing
`requirements.txt`, since it is a pinned environment file.

### 2. Prepare Assets

Prepare these assets before training:

- **Student model:** set by `MODEL_PATH`.
- **Teacher model:** set by `TEACHER_CKPT_PATH`.
- **Training data:** set by `TRAIN_FILE`.
- **Validation data:** set by `TEST_FILE`.
- **Checkpoint directory:** set by `CKPTS_DIR`.

The training and validation files should be parquet files compatible with
`verl`'s RL dataset loader. The default script expects a prompt column named
`prompt`.

### 3. Configure Teacher Service and Paths

Edit `cross_distill.sh` or export the variables below before launching:

```bash
export RAY_DATA_HOME=/path/to/workspace
export MODEL_PATH=/path/to/student/model
export TEACHER_CKPT_PATH=/path/to/teacher/model
export TRAIN_FILE=/path/to/train.parquet
export TEST_FILE=/path/to/val.parquet
export CKPTS_DIR=/path/to/checkpoints

export TEACHER_SERVER_IP=127.0.0.1
export TEACHER_SERVER_PORT=15555
export TEACHER_N_WORKERS=1
export TEACHER_MAX_SEQ_LEN=30720
```

`OPDRewardManager` communicates with the teacher service over ZeroMQ. The
teacher response must contain:

- `responses`
- `teacher_topk_logprobs`
- `teacher_topk_indices`

Start the teacher service using bash script in recipe/gkd/teacher before running the OPD training job.

### 4. Launch Training

Start or connect to the Ray cluster required by your training setup, then run:

```bash
bash cross_distill.sh
```

The default script is configured for multi-node training:

- `NNODES=2`
- `NGPUS_PER_NODE=8`
- `trainer.total_training_steps=500`
- `trainer.total_epochs=10`

Override these variables or edit `cross_distill.sh` for your hardware and
experiment scale.

---

## Models and Checkpoints

No trained OPD checkpoint is bundled in this package. By default,
`cross_distill.sh` writes checkpoints to:

```text
${RAY_DATA_HOME}/ckpts/ON_POLICY_DISTILL/OPD
```

Set `CKPTS_DIR` to change the output location.

## Citation

If you find this work helpful for your research, please cite our paper:

```bibtex
@misc{niu2026breakingtokenizerbarrieronpolicy,
      title={Breaking the Tokenizer Barrier: On-Policy Distillation across Model Families}, 
      author={Yifan Niu and Han Xiao and Dongyi Liu and Zelong Wang and Dihong Gong and Yasheng Wang and Jia Li},
      year={2026},
      eprint={2606.09456},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.09456}, 
}
```