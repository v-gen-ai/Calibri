from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from src.utils.utils import to_pil_list


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        images = to_pil_list(images)
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from src.metrics.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score():
    from src.metrics.clip_scorer import ClipScorer

    scorer = ClipScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from src.metrics.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from src.metrics.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from src.metrics.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from src.metrics.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def ocr_score(device):
    from src.metrics.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from src.metrics.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn


def unified_reward_qwen(device):

    from src.metrics.unified_reward_qwen_scorer import UnifiedRewardQwen

    scorer = UnifiedRewardQwen(device=device)

    def _fn(images, prompts, metadata=None):
        images = to_pil_list(images)
        scores = scorer.score(prompts, images)
        return scores, {}
    
    return _fn


def qalign_score(device):
    """
    In-proc Q-Align scorer.
    Возвращает функцию fn(images, prompts, metadata) -> (scores, {}).
    metadata может содержать:
      - qalign_task: "quality" или "aesthetics" (по умолчанию "quality")
      - qalign_input: "image" или "video" (по умолчанию "image")
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM
    from src.utils.utils import to_pil_list  # уже есть в окружении

    dtype = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "q-future/one-align",
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=str(device),
        ).eval()
    except Exception as e:
        raise ImportError(
            "Q-Align model is not available in this environment. "
            "Consider using qalign_score_remote with a running HTTP service."
        ) from e

    def fn(images, prompts, metadata=None):
        # prompts не требуются, но сохраняем сигнатуру совместимой
        pil_images = to_pil_list(images)
        if len(pil_images) == 0:
            return [], {}
        md = metadata or {}
        task = md.get("qalign_task", "quality")     # "quality" | "aesthetics"
        input_type = md.get("qalign_input", "image")  # "image" | "video"

        import torch
        with torch.inference_mode():
            out = model.score(pil_images, task_=task, input_=input_type)
        # Приводим к списку чисел
        try:
            if isinstance(out, torch.Tensor):
                scores = out.detach().cpu().tolist()
            elif hasattr(out, "tolist"):
                scores = out.tolist()
            else:
                scores = [float(x) for x in out]
        except Exception:
            scores = [float(x) for x in out]

        return scores, {}

    return fn


def qalign_score_remote(device, url=None):
    """
    Remote Q-Align scorer (HTTP).
    URL из env QALIGN_URL либо аргумента.
    Возвращает fn(images, prompts, metadata) -> (scores, {}).
    metadata:
      - qalign_task: "quality" | "aesthetics" (default "quality")
      - qalign_input: "image" | "video" (default "image")
    Серверный ответ: pickle.dumps({"scores": List[float]})
    """
    import os
    import io
    import pickle
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from PIL import Image
    import torch
    import numpy as np

    from src.utils.utils import to_pil_list

    url = url or os.environ.get("QALIGN_URL", "http://127.0.0.1:18088")
    batch_size = int(os.environ.get("QALIGN_BATCH", "8"))

    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def fn(images, prompts, metadata=None):
        torch.cuda.empty_cache()
        pil_images = to_pil_list(images)
        md = metadata or {}
        task = md.get("qalign_task", "quality")
        input_type = md.get("qalign_input", "image")

        all_scores = []
        for i in range(0, len(pil_images), batch_size):
            imgs = pil_images[i:i + batch_size]

            jpeg_images = []
            for img in imgs:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                jpeg_images.append(buf.getvalue())

            payload = {
                "images": jpeg_images,
                "task": task,
                "input": input_type,
                # "prompts": prompts[i:i+batch_size],  # опционально, не используется моделью
            }
            data = pickle.dumps(payload)

            resp = sess.post(url, data=data, timeout=600)
            if resp.status_code != 200:
                err = None
                try:
                    err = pickle.loads(resp.content).get("error")
                except Exception:
                    err = f"raw={resp.content[:200]!r}"
                raise RuntimeError(f"Q-Align remote HTTP {resp.status_code}: {err}")

            result = pickle.loads(resp.content)
            all_scores.extend(result["scores"])
        return all_scores, {}
    return fn


def hpsv3score(device):
    """
    In-proc HPSv3 scorer (использовать только если зависимости HPSv3 стоят в текущем env).
    Возвращает функцию fn(images, prompts, metadata) -> (scores, {})
    """
    try:
        from PIL import Image
        import torch
        from hpsv3 import HPSv3RewardInferencer
    except Exception as e:
        raise ImportError("HPSv3 not available in this environment. Use hpsv3score_remote instead.") from e

    inferencer = HPSv3RewardInferencer(device=str(device))

    def fn(images, prompts, metadata=None):
        pil_images = to_pil_list(images)
        scores = inferencer.reward(prompts=prompts, image_paths=pil_images)
        scores = [reward[0].item() for reward in scores]
        return scores, {}
    return fn


def hpsv3score_remote(device, url=None):
    """
    Remote HPSv3 scorer: общается с отдельным HTTP-сервисом в окружении HPSv3.
    URL можно переопределить через env HPSV3_URL или аргументом.
    Возвращает функцию fn(images, prompts, metadata) -> (scores, {})
    """
    import os
    import io
    import pickle
    import numpy as np
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from PIL import Image
    import torch

    url = url or os.environ.get("HPSV3_URL", "http://127.0.0.1:18067") # 18087
    batch_size = 8

    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    # def _to_pil_list(images):
    #     if isinstance(images, torch.Tensor):
    #         arr = (images * 255.0 if images.dtype.is_floating_point else images).round().clamp(0, 255) \
    #                 .to(torch.uint8).cpu().numpy()
    #         arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    #         return [Image.fromarray(im) for im in arr]
    #     else:
    #         return [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images]

    def fn(images, prompts, metadata=None):
        torch.cuda.empty_cache()
        pil_images = to_pil_list(images)
        all_scores = []

        # батчевка для экономии памяти и стабильности
        for i in range(0, len(pil_images), batch_size):
            imgs = pil_images[i:i+batch_size]
            prms = prompts[i:i+batch_size]

            # JPEG-компрессия
            jpeg_images = []
            for img in imgs:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                jpeg_images.append(buf.getvalue())

            payload = {"images": jpeg_images, "prompts": prms}
            data_bytes = pickle.dumps(payload)

            resp = sess.post(url, data=data_bytes, timeout=600)
            # resp.raise_for_status()
            if resp.status_code != 200:
                # вытащим текст ошибки, который сервер упаковал pickle'ом
                err = None
                try:
                    err = pickle.loads(resp.content).get("error")
                except Exception:
                    err = f"raw={resp.content[:200]!r}"
                raise RuntimeError(f"HPSv3 remote HTTP {resp.status_code}: {err}")
            result = pickle.loads(resp.content)
            all_scores.extend(result["scores"])

        return all_scores, {}
    return fn


def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "unifiedreward_qwen": unified_reward_qwen,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "hpsv3": hpsv3score,
        "hpsv3_remote": hpsv3score_remote,
        "qalign": qalign_score,
        "qalign_remote": qalign_score_remote
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "./data/images/000000.png",
        "./data/images/000001.png",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        "a red boat and a blue book",
        "A dog is playing tug-of-war with its owner and wagging its tail."
    ]
    metadata = {}
    score_dict = {
        # "hpsv3_remote": 1.0,
        "clipscore": 1.0,
        "pickscore": 1.0
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    scores, _ = scoring_fn(images, prompts, metadata)
    print("Scores:", scores)


if __name__ == "__main__":
    main()
