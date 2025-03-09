from transformers import GPT2Tokenizer, AutoProcessor

from PIL import Image
from typing import List, Union
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput


class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        # Use Qwen2.5-VL’s processor for image preprocessing
        self.vl_processor = AutoProcessor.from_pretrained(
            config.qwen_vl_hf_model,
            trust_remote_code=True  # Required for Qwen2.5-VL
        )
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
        self.tokeniser.pad_token = self.tokeniser.bos_token
        self.tokeniser.add_eos_token = add_eos_token

        # Bind a new method to gpt2_tokeniser
        self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
            self.tokeniser
        )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        text_inputs = self.tokeniser(
            texts, padding=padding, *args, **kwargs
        ) if texts is not None else None

        # Use Qwen2.5-VL processor for images
        if images is not None:
            # Qwen2.5-VL expects a list of images or a single image
            if not isinstance(images, list):
                images = [images]
            # Process images with Qwen2.5-VL’s processor
            image_inputs = self.vl_processor(
                images=images,
                return_tensors='pt',
                size={"height": 448, "width": 448}  # Match Qwen2.5-VL default
            )
            pixel_values = image_inputs['pixel_values']
        else:
            pixel_values = None

        return DTrOCRProcessorOutput(
            pixel_values=pixel_values,
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )


def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    if self.add_eos_token:
        eos_token_ids = [self.eos_token_id]
    else:
        eos_token_ids = []

    output = bos_token_ids + token_ids_0 + eos_token_ids

    if token_ids_1 is None:
        return output

    return output + bos_token_ids + token_ids_1
