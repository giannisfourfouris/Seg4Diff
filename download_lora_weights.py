from huggingface_hub import hf_hub_download

local_dir = "./lora_weights"

# Download COCO-trained lora weights
# LORA_PATH_COCO = hf_hub_download(
#     repo_id="chyun/seg4diff-coco-lora",
#     filename="lora_weights.pth",
#     local_dir=cache_dir,
# )
# print("Downloaded to: ", LORA_PATH_COCO)

# Download SA1B-trained lora weights
LORA_PATH_SA1B = hf_hub_download(
    repo_id="chyun/seg4diff-sa1b-lora",
    filename="lora_weights.pth",
    local_dir=local_dir,
)
print("Downloaded to: ", LORA_PATH_SA1B)