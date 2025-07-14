# 🧭 Intruduction

This repository provides a batch inference pipeline using Qwen2.5-VL, a large vision-language model, for Multimodal Reasoning Competition Track1 (VG-RS). Given a set of image-question pairs, the model outputs the corresponding bounding box coordinates.

---

## 📦 Environment Requirements

This script is tested with the following setup:

- Python == 3.9.21  
- PyTorch == 2.6.0  
- Transformers == 4.51.3
- qwen-vl-utils == 0.0.8
- modelscope == 1.25.0

---

## 🗂 Directory Structure

```
project_root/
├── test_qwen_lens_batch_inference.py          # Main inference script
├── images/                                    # Folder with images
│   └── *.jpg / *.png
├── VG-RS-question.json                        # Input questions and image paths
└── predict_grounding_full_3b.json             # Output predictions (bounding boxes)
```

---

## 🔧 Model and Processor Setup

### 🧠 Download Model with ModelScope

You can download the Qwen2.5-VL-3B-Instruct model manually or using the following script:

```python
from modelscope import snapshot_download

snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='/root/models')
```

---

## 🧪 How to Run Inference

### ✍️ Prepare Input JSON

The file `VG-RS-question.json` should be a list of entries in this format:

```json
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?"
  },
  ...
]
```

### 🚀 Run Script

```bash
python test_qwen_lens_batch_inference.py \
    --model_dir /root/models/Qwen2.5-VL-3B-Instruct/ \  # path/to/your/local/model
    --json_path ./VG-RS-question.json \
    --json_save_path ./predict_grounding_full_3b.json \
    --batch_size 8
```

---

## 📤 Output Format

The result will be saved as a JSON file containing predicted bounding boxes for each input:

```json
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?",
    "result": [[x1, y1], [x2, y2]]
  },
  ...
]
```

> Note: Bounding boxes are in the format `[[x_min, y_min], [x_max, y_max]]`.

---

## 📝 Reference

If you use this code and our data, please cite:
> @article{yao2025lens,  
> title={LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models},  
> author={Yao, Ruilin and Zhang, Bo and Huang, Jirui and Long, Xinwei and Zhang, Yifang and Zou, Tianyu and Wu, Yufei and Su, Shichao and Xu, Yifan and Zeng, Wenxi and others},  
> journal={arXiv preprint arXiv:2505.15616},  
> year={2025}  
> }  


> @article{Qwen2VL,  
> title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},  
> author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},  
> journal={arXiv preprint arXiv:2409.12191},  
> year={2024}  
> }  

---

## 💬 Contact

If you encounter any issues or have questions, feel free to open an issue on GitHub.