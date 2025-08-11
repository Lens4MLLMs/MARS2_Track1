# Groma (VG-RS)

This repository provides Groma baseline, for Multimodal Reasoning Competition Track1 [(VG-RS)](https://eval.ai/web/challenges/challenge-page/2552/overview).


## 📦 Environment Requirements & Data Preparations

Please refer the installation guide at [[Link]](https://github.com/FoundationVision/Groma).

You will need a model checkpoint: `groma_7b_finetune`. 


## 🚀 Run Script

```bash
python groma/eval/eval_vgrs.py \
    --model-name {path_to_groma_7b_finetune} \
    --img-prefix {path_to_image_root} \
    --ann-file {path_to_VG-RS-question.json} \
    --json-save-path {path_to_save_predictions_json}
```


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
