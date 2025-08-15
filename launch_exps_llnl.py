# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = (
    "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"
)

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

# TIME_LIMIT = 29
# TIME_LIMIT = 59
TIME_LIMIT = 360

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/bd3lms/outputs"

BASE_RUN_NAME = f"debug"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

# NODES = 1
# GPN = 1
# NODES = 1
# GPN = 4
NODES = 4
GPN = 4

BLOCK_SIZE = 4
# BLOCK_SIZE = 8
# BLOCK_SIZE = 16

# PRETRAINED_CHECKPOINT="null"
PRETRAINED_CHECKPOINT="kuleshov-group/bd3lm-owt-block_size1024-pretrain"

# run_name = f"bd3lm-owt-block_size{BLOCK_SIZE}_N{NODES}n{NODES*GPN}"
run_name = f"bd3lm-owt-pretrained-block_size{BLOCK_SIZE}_N{NODES}n{NODES*GPN}"

# Cfgs
exp_list = [
    [f"""\
python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=openwebtext-split \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    data.insert_valid_eos=False \
    model.length=1024 \
    block_size={BLOCK_SIZE} \
    wandb.name={run_name} \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained={PRETRAINED_CHECKPOINT}
""", run_name]
]


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        run_name,
    ) = exp

    # put together the actual "train.py" command
    custom_invocation = f"{script}"

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
        --wandb_offline={WANDB_OFFLINE} \
        --rocm_version={ROCM_VERSION} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --rccl_cfg={RCCL_CFG} \
        --qos={QOS} \
        --bank={BANK} \
        --minutes={TIME_LIMIT} \
        --nodes={NODES} \
        --gpus_per_node={GPN} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        print(command)

print(f"Total launches: {total_launches}")
