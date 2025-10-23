from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from PIL import Image
import torch
import tqdm
import os
import random
import json
from typing import List
import re


class UnifiedRewardQwen:
    def __init__(self, model_path='CodeGoat24/UnifiedReward-qwen-7b', device="cpu"):
        self.model_path = model_path
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def score(self, prompts: List[str] | str, images: List[Image.Image] | Image.Image) -> List[float | None] | float | None:
        """
        calculates Unified Reward Qwen for List[prompt], List[image] or prompt, image. None is returned when the error occured while parsing final score
        """
        if isinstance(prompts, list) or isinstance(images, list):
            assert isinstance(prompts, list) and isinstance(images, list)
            assert len(prompts) == len(images)
        else:
            prompts, images = [prompts], [images]

        results = []
        for prompt, image in zip(prompts, images):
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f'You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]'
                        },
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)


            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            match = re.search(r'Final Score:\s*([0-9]+\.?[0-9]*)', output_text)
            if match:
                final_score = float(match.group(1))
            else:
                final_score = None
            
            results.append(final_score)
        return results
