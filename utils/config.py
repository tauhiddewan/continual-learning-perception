import ast
import torch
from dotenv import dotenv_values

# If config.py is in utils/, .env is one level up:
env_vars = dotenv_values("../.env")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YCB_CLASSES = {
    0: "background",
    1: "master_chef_can",
    2: "cracker_box",
    3: "sugar_box",
    4: "tomato_soup_can",
    5: "mustard_bottle",
    6: "tuna_fish_can",
    7: "pudding_box",
    8: "gelatin_box",
    9: "potted_meat_can",
    10: "banana",
    11: "pitcher_base",
    12: "bleach_cleanser",
    13: "bowl",
    14: "mug",
    15: "power_drill",
    16: "wood_block",
    17: "scissors",
    18: "large_marker",
    19: "large_clamp",
    20: "extra_large_clamp",
    21: "foam_brick",
}

# Segformer config
NUM_CLASSES = int(env_vars.get("NUM_CLASSES", 22))
IGNORE_INDEX = int(env_vars.get("IGNORE_INDEX", 255))
BASE_CLASS_IDS = ast.literal_eval(
    env_vars.get("BASE_CLASS_IDS", "[1,2,3,4,5,6,7,8,9,10]")
)
NEW_CLASS_IDS = ast.literal_eval(
    env_vars.get("NEW_CLASS_IDS", "[11,12,13,14,15,16,17,18,19,20,21]")
)

# Dataset paths
TRAIN_SPLIT = env_vars.get("TRAIN_SPLIT", "outputs/splits/train_all.txt")
VAL_SPLIT = env_vars.get("VAL_SPLIT", "outputs/splits/val_all.txt")

# Common hyperparameters
NUM_EPOCHS = int(env_vars.get("NUM_EPOCHS", 50))
BATCH_SIZE = int(env_vars.get("BATCH_SIZE", 32))
RESIZE = ast.literal_eval(env_vars.get("RESIZE", "(384,384)"))

# base experiment hyperparameters
LR = float(env_vars.get("LR", "6e-5"))
CHECKPOINT_PATH = env_vars.get(
    "CHECKPOINT_PATH",
    "outputs/checkpoints/segformer_ycb_base.pth",
)
RESULTS_PATH = env_vars.get(
    "RESULTS_PATH",
    "outputs/segformer_ycb_base_metrics.json",
)
LOG_FILE_PATH = env_vars.get(
    "LOG_FILE_PATH",
    "outputs/train.base.log",
)

# finetuning (naive incremental) hyperparameters
NAIVE_LR = float(env_vars.get("NAIVE_LR", "6e-5"))
NAIVE_CHECKPOINT_PATH = env_vars.get(
    "NAIVE_CHECKPOINT_PATH",
    "outputs/checkpoints/segformer_ycb_incremental_naive.pth",
)
NAIVE_RESULTS_PATH = env_vars.get(
    "NAIVE_RESULTS_PATH",
    "outputs/segformer_ycb_incremental_naive_metrics.json",
)
NAIVE_LOG_FILE_PATH = env_vars.get(
    "NAIVE_LOG_FILE_PATH",
    "outputs/train.incremental_naive.log",
)

# LoRA incremental hyperparameters
LORA_NUM_EPOCHS = int(env_vars.get("LORA_NUM_EPOCHS", 25))
LORA_LR = float(env_vars.get("LORA_LR", "1e-4"))
LORA_R = int(env_vars.get("LORA_R", 8))
LORA_ALPHA = int(env_vars.get("LORA_ALPHA", 16))

LORA_CHECKPOINT_PATH = env_vars.get(
    "LORA_CHECKPOINT_PATH",
    "outputs/checkpoints/segformer_ycb_incremental_lora_custom.pth",
)
LORA_RESULTS_PATH = env_vars.get(
    "LORA_RESULTS_PATH",
    "outputs/segformer_ycb_incremental_lora_custom_metrics.json",
)
LORA_LOG_FILE_PATH = env_vars.get(
    "LORA_LOG_FILE_PATH",
    "outputs/train.incremental_lora.log",
)


# Adapter fusion hyperparameters
FUSION_NUM_EPOCHS = int(env_vars.get("FUSION_NUM_EPOCHS", 5))
FUSION_LR = float(env_vars.get("FUSION_LR", "1e-4"))

FUSION_CHECKPOINT_PATH = env_vars.get(
    "FUSION_CHECKPOINT_PATH",
    "outputs/checkpoints/segformer_ycb_adapter_new_lora.pth",
)
FUSION_RESULTS_PATH = env_vars.get(
    "FUSION_RESULTS_PATH",
    "outputs/segformer_ycb_adapter_new_lora_metrics.json",
)
FUSION_LOG_FILE_PATH = env_vars.get(
    "FUSION_LOG_FILE_PATH",
    "outputs/train.adapter_fusion.log",
)


# KD-LoRA incremental hyperparameters
KD_LORA_NUM_EPOCHS = int(env_vars.get("KD_LORA_NUM_EPOCHS", 20))
KD_LORA_LR = float(env_vars.get("KD_LORA_LR", "1e-4"))
KD_LORA_LAMBDA = float(env_vars.get("KD_LORA_LAMBDA", "0.1"))
KD_LORA_T = float(env_vars.get("KD_LORA_T", "2.0"))

KD_LORA_CHECKPOINT_PATH = env_vars.get(
    "KD_LORA_CHECKPOINT_PATH",
    "outputs/checkpoints/segformer_ycb_incremental_lora_kd.pth",
)
KD_LORA_RESULTS_PATH = env_vars.get(
    "KD_LORA_RESULTS_PATH",
    "outputs/segformer_ycb_incremental_lora_kd_metrics.json",
)
KD_LORA_LOG_FILE_PATH = env_vars.get(
    "KD_LORA_LOG_FILE_PATH",
    "outputs/train.incremental_lora_kd.log",
)
