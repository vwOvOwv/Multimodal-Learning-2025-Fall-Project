# Instruct-CLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning

Reprodcution and Improvement of Instruct-CLIP (Chen *et. al*, CVPR 2025).

## 1. Environment

Clone this repo:

```bash
git clone git@github.com:vwOvOwv/Multimodal-Learning-2025-Fall-Project.git
```

We have updated `requirement.txt` to fit Python 3.12 + PyTorch 2.9.1 (supports the newest [xFormers](https://github.com/facebookresearch/xformers) and [FlashAttention](https://github.com/Dao-AILab/flash-attention)), and removed conflicting packages or packages unused (200+ -> 100-).

Build dependencies:

```bash
conda create -n iclip python=3.12 -y
conda activate iclip
pip install -r requirements.txt
```

Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) manually: 

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## 2. Image Editing Instruction Refinement

### 2.1 Data Preparation

Download [timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) to a new folder `instructclip_datasets`. This is the training data for both LD-DINOv2 and InstructCLIP. Also, download [ip2p_clip_feat.npy](https://www.dropbox.com/scl/fo/id2ow98wqhc38x6csjmxe/AC3SmN-0klY-C6yuZMSd_ic?rlkey=o7mmbh1x60br2y8l3fas2eag2&st=22mqu2si&dl=0) to the same folder. It contains the CLIP text features of all edit instruction in the InstructPix2Pix dataset, which we will use to refine edit instructions later. The dataset folder should look like this:

```
instructclip_datasets
├── instructpix2pix-clip-filtered
│   └── data
└── ip2p_clip_feat.npy
```

### 2.2 LD-DINOv2 Training

To train LD-DINOv2, run the following command, which save checkpoints in `ckpts/lddinov2` by default:

```bash
bash scripts/train_lddinov2.sh
```

Below are reproduced training curves of LD-DINOv2:

![LD-DINOv2](assets/LDDINOv2.png)

### 2.3 Instruct-CLIP Training

To train Instuct-CLIP, run the following command, which load the latest LD-DINOv2 checkpoint from `ckpts/lddinov2/final.ckpt` and save its checkpoints in `ckpts/instructclip` by default:

```bash
bash scripts/train_iclip.sh
```

Below are reproduced training curves of Instruct-CLIP:
![ICLIP](assets/ICLIP.png)

### 2.4 Test Edit Instruction Refinement

After training, to get the edit instruction from an image pair, run:

```bash
python get_edit_instruction.py --input_path <input_path> --output_path <output_path>
```

Below are reproduced refinement examples:

![results](assets/results.png)

## 3. Image Editing

### 3.1 Data Preparation with Refined Instructions

The authors provide over 120K samples with refined editing instructions [here](https://huggingface.co/datasets/SherryXTChen/InstructCLIP-InstructPix2Pix-Data). Download it to `instructclip_datasets` as well. Now the folder should look like this:

```
instructclip_datasets
├── InstructCLIP-InstructPix2Pix-Data
│   ├── dataset_dict.json
│   └── train
├── instructpix2pix-clip-filtered
│   └── data
└── ip2p_clip_feat.npy
```

### 3.2 Training

Instruct-CLIP is needed for fine-tuning our image editing models. To fine-tune InstructPixPix on our dataset, run the following command where the checkpoints are stored in `ckpts/ip2p_finetuned` by default:

```bash
bash scripts/train_instruct_pix2pix.sh
```

Below are reproduced fine-tuning curves:

![finetune](assets/finetune.png)

### 3.3 Test Image Editing

After training, to apply an edit instruction on a certain image, run:

```bash
python inference.py
```

## 4. Our Improvement 

### 4.1 Refined InstructPix2Pix on Removal Instructions

```bash
bash scripts/train_instruct_pix2pix_augmented.sh
```

### 4.2 Evaluation on MagicBrush Dataset

Download the test split of MagicBrush under the guidance of [the official repo](https://github.com/OSU-NLP-Group/MagicBrush?tab=readme-ov-file#dataset-access) (OneDrive). To prevent data leakage, the test set is unavailable on HuggingFace.

Unzip the zip file you have downloaded to `instructclip_datasets/MagicBrush`:

```bash
unzip test.zip -d instructclip_datasets/MagicBrush
```

Now the dataset folder should look like this:

```
instructclip_datasets
├── InstructCLIP-InstructPix2Pix-Data
│   ├── dataset_dict.json
│   └── train
├── instructpix2pix-clip-filtered
│   └── data
├── MagicBrush
│   └── test
│       ├── edit_turns.json
│       ├── global_descriptions.json
│       └── images
└── ip2p_clip_feat.npy
```

Evaluate the finetuned model (`ckpts/ip2p_finetuned_test/pytorch_lora_weights.safetensors` by default):

```bash
bash scripts/eval_magicbrush.sh
```