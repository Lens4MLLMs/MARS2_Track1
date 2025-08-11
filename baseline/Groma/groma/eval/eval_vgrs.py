import os
import sys
root_dir = os.getcwd()
sys.path.insert(0, f'{root_dir}')

import numpy as np
import torch
import random
import json
import argparse
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers.image_transforms import center_to_corners_format
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.pipelines import Compose

from groma.utils import init_distributed_mode
from groma.constants import DEFAULT_TOKENS
from groma.model.groma import GromaModel
from groma.data.datasets.refcoco_rec import RefCOCO, INSTRUCTIONS
from groma.data.datasets.det_data import normalize_box_coordinates
from groma.data.conversation import conv_templates


def read_json_and_extract_fields(file_path='VG-RS-question.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


class RSVSTest(Dataset):
    def __init__(self, ann_file=None, img_prefix=None, tokenizer=None, conv_temp='default'):
        self.tokenizer = tokenizer
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

        self.data_infos = read_json_and_extract_fields(ann_file)
        self.data_root = img_prefix

        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True)

        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img']),
        ]

        self.pipeline = Compose(pipeline)
        
        
    def preprocess(self, data_item):
        image = data_item['img'].data
        label = data_item['gt_labels'][0]
        img_shape = data_item['img_metas'].data['img_shape']
        bboxes = torch.zeros((1,4)).to(image)

        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        refexp = DEFAULT_TOKENS['boe'] + label.strip() + DEFAULT_TOKENS['eoe']
        instruct = INSTRUCTIONS[0].format(refexp) # random.choice(INSTRUCTIONS).format(refexp) 
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], ''))
        prompt = self.conv_temp.get_prompt(conversations)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids

        data_dict = dict(
            input_ids=input_ids,
            image=image,
            bboxes=bboxes,
            label=label.strip(),
            img_name=data_item['img_metas'].data['filename'])

        return data_dict

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data = self.data_infos[idx]
        img_name = data["image_path"].split("\\")[-1]
        text_query = data["question"]

        img = Image.open(os.path.join(self.data_root, img_name))
        width, height = img.size

        data = {
            "img_info": {
                "filename": img_name, 
                "height": height, 
                "width": width, 
            }, 
            "img_prefix": self.data_root
        }
        
        data_item = self.pipeline(data)
        data_item["gt_labels"] = [text_query]

        data_dict = self.preprocess(data_item)
        return data_dict


def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']
    image = batch[0]['image'].unsqueeze(dim=0)
    labels = [batch[0]['label']]
    img_names = [batch[0]['img_name']]
    return input_ids, image, labels, img_names


def eval_model(args):
    # Model
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres

    dataset = RSVSTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        conv_temp='llava'
    )
    distributed_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
        sampler=distributed_sampler, collate_fn=custom_collate_fn)

    content_list = []
    itr, invalid = 0, 0
    for input_ids, image, labels, img_names in dataloader:
        batch_size = len(input_ids)
        input_ids = input_ids.cuda()
        image = image.cuda()
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=3,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )
        output_ids = outputs.sequences
        pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
        input_token_len = input_ids.shape[1]
        predicted_box_tokens = [id for id in output_ids[0, input_token_len:] if id in model.box_idx_token_ids]
        selected_box_inds = [model.box_idx_token_ids.index(id) for id in predicted_box_tokens]
        selected_box_inds = [id for id in selected_box_inds if id < len(pred_boxes)]
        if len(selected_box_inds) == 0:
            print(['[Unable to detect samples ]', f'{img_names[i_batch]}'])
            invalid += 1
            continue
        selected_boxes = pred_boxes[selected_box_inds]

        # Save results
        for i_batch in range(batch_size):
            pil_image = Image.open(img_names[i_batch])
            img_w, img_h = pil_image.size

            bounding_box = selected_boxes[i_batch]
            x_center, y_center, w, h = bounding_box
            xmin = int((x_center - w / 2) * img_w)
            ymin = int((y_center - h / 2) * img_h)
            xmax = int((x_center + w / 2) * img_w)
            ymax = int((y_center + h / 2) * img_h)

            content = {
                "image_path": 'images\\' + img_names[i_batch].split('/')[-1],
                'question': labels[i_batch],
                "result": [[xmin, ymin], [xmax, ymax]]}

            content_list.append(content)

        itr += 1
        print(f"{itr}/{len(dataloader)}")

    with open(args.json_save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(content_list, ensure_ascii=False, indent=2) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--ann-file", type=str, default="./VG-RS-question.json")
    parser.add_argument("--img-prefix", type=str, default="./images")
    parser.add_argument("--json-save-path", type=str, default="./predict_grounding_full_3b.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--box_score_thres", type=float, default=0.15)
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    args = parser.parse_args()

    eval_model(args)
