import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline, StableDiffusion3Pipeline

from app.ComfyUI_FluxCustomId import NODE_DISPLAY_NAME_MAPPINGS


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


class AMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["FLUX", "SD3"], {"default": "FLUX"}),
                "model_name": ("STRING", {
                    "default": "black-forest-labs/FLUX.1-dev",
                    "sd3_default": "stabilityai/stable-diffusion-3-medium-diffusers"
                }),
                "load_local_model": ("BOOLEAN", {"default": False}),
                "dtype": (["float16", "bfloat16"], {"default": "float16"}),
                "enable_cpu_offload": ("BOOLEAN", {"default": True}),
            }, "optional": {
                "local_model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
            }
        }

    RETURN_TYPES = ("MODEL_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_pipe"

    def load_pipe(self, model_type, model_name, load_local_model, dtype, enable_cpu_offload, *args, **kwargs):
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        if load_local_model:
            model_name = kwargs.get("local_model_path", "black-forest-labs/FLUX.1-dev")

        if model_type == "FLUX":
            pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
            if hasattr(pipe, 'vae'):
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

        if enable_cpu_offload:
            pipe.enable_model_cpu_offload() if model_type == "SD3" else pipe.enable_sequential_cpu_offload()

        return (pipe,)


class AttentionMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL_PIPE",),
                "positive_prompt": (
                    "STRING", {"multiline": True, "default": "a tiny astronaut hatching from an egg on the moon"}),
                "negative_prompt": ("STRING", {"default": ""}),
                "max_length": ("INT", {"default": 512, "min": 64, "max": 1024}),
            }
        }

    RETURN_TYPES = ("ATTENTION_MASK", "PROMPT_BATCH")
    FUNCTION = "process_prompts"

    def process_prompts(self, pipe, positive_prompt, negative_prompt, max_length):
        # SD3需要特殊处理提示结构
        if isinstance(pipe, StableDiffusion3Pipeline):
            prompts = [
                negative_prompt,
                positive_prompt,
            ]
            tokenizer = pipe.tokenizer_3
            attention_mask = pipe.tokenizer_3(
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).attention_mask
        else:
            prompts = [positive_prompt]
            tokenizer = pipe.tokenizer_2
            attention_mask = pipe.tokenizer_2(
                prompts,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).attention_mask

        # SD3需要截断前两个提示
        final_prompts = prompts[1:] if isinstance(pipe, StableDiffusion3Pipeline) else prompts
        return (attention_mask.to(dtype=torch.float16), final_prompts)


class AMSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL_PIPE",),
                "attention_mask": ("ATTENTION_MASK",),
                "prompt_batch": ("PROMPT_BATCH",),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0}),
                "width": ("INT", {"default": 1360, "min": 256}),
                "height": ("INT", {"default": 768, "min": 256}),
                "steps": ("INT", {"default": 50, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "generate"

    def generate(self, pipe, attention_mask, prompt_batch, **kwargs):
        generator = torch.Generator(device="cuda").manual_seed(kwargs['seed'])

        # 自动处理不同模型的参数差异
        base_params = {
            'prompt': prompt_batch,
            'joint_attention_kwargs': {'attention_mask': attention_mask},
            'generator': generator,
            'num_inference_steps': kwargs['steps'],
            'guidance_scale': kwargs['guidance_scale'],
            'width': kwargs['width'],
            'height': kwargs['height'],
            'batch_size': kwargs['batch_size']
        }

        # 添加SD3专用参数
        if isinstance(pipe, StableDiffusion3Pipeline):
            base_params.update({
                'negative_prompt': [""]  # 与prompt处理器中的结构对应
            })

        outputs = pipe(**base_params).images
        images = convert_preview_image(outputs)
        return (images,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AMModelLoader": AMModelLoader,
    "AttentionMask": AttentionMask,
    "AMSample": AMSample
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AMModelLoader": "AMModelLoader",
    "AttentionMask": "AttentionMask",
    "AMSample": "AMSample"
}
