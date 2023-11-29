# coding=utf-8

import torch
from typing import Dict
from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from uniem.model import UniemEmbedder


class CustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwags):
        preprocess_kwargs = {}
        if "max_length" in kwags:
            preprocess_kwargs["max_length"] = kwags["max_length"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text: str, max_length=512) -> Dict[str, torch.Tensor]:
        model_input = self.tokenizer(text, return_tensors=self.framework, truncation=True, padding=True,
                                     max_length=max_length)
        return {"input_ids": model_input["input_ids"], "attention_mask": model_input["attention_mask"]}

    def _forward(self, model_inputs: Dict[str, torch.Tensor], **forward_parameters: Dict) -> torch.Tensor:
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs: torch.Tensor) -> torch.Tensor:
        return model_outputs

if __name__ != "__main__":
    PIPELINE_REGISTRY.register_pipeline(
        "embedding_search",
        pipeline_class=CustomPipeline,
        pt_model=UniemEmbedder
    )