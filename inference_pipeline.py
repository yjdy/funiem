# coding=utf-8
from __future__ import annotations

import os, sys

sys.path.append(os.getcwd())
from tqdm import tqdm

from transformers import AutoTokenizer, pipeline

import torch
from torch.utils.data import DataLoader

from uniem.model import create_uniem_embedder, PoolingStrategy
from uniem.utils import create_finetune_datasets, load_all_datasets_json
from uniem.data_structures import RecordType
import deepspeed

def inference_single_gpu_scored_pair(model_name_or_path: str | None,
                                     data_path: str | None, model_class=None, pooling_strategy=None | PoolingStrategy,
                                     metric=None, query_instruction='', record_type=RecordType.SCORED_PAIR, deepspeed=False,
                                     max_length: int = 512, batch_size: int = 32, device: int = 0) -> None:
    tokenizers = AutoTokenizer.from_pretrained(model_name_or_path)
    model = create_uniem_embedder(model_name_or_path=model_name_or_path,
                                  model_class=model_class,
                                  pooling_strategy=pooling_strategy)
    # init deepspeed inference engine
    if deepspeed:
        model = deepspeed.init_inference(
            model=model,  # Transformers models
            mp_size=1,  # Number of GPU
            dtype=torch.half,  # dtype of the weights (fp16)
            # injection_policy={"BertLayer" : HFBertLayerPolicy}, # replace BertLayer with DS HFBertLayerPolicy
            replace_method="auto",  # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=True,  # replace the model with the kernel injector
        )

    # create acclerated pipeline
    ds_clf = pipeline("embedding_search", model=model, tokenizer=tokenizers,
                      device=device, max_length=max_length, batch_size=batch_size)
    all_datasets = load_all_datasets_json(data_path)
    if "validation" in all_datasets:
        train_dataset, valid_dataset = create_finetune_datasets(all_datasets["train"], all_datasets["validation"],
                                                                record_type, query_instruction=query_instruction)
    else:  # 只导入了测试集
        valid_dataset, _ = create_finetune_datasets(all_datasets["train"], None,
                                                    record_type, query_instruction=query_instruction)
    validation_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=lambda x: x, shuffle=False, pin_memory=True, drop_last=False
    )

    predicts = []
    labels = []
    for batch in tqdm(validation_dataloader):
        query = [b.sentence1 for b in batch]
        docs = [b.sentence2 for b in batch]
        batch_labels = [b.label for b in batch]

        query_embeddings = ds_clf(query)
        doc_embeddings = ds_clf(docs)

        batch_output = torch.cosine_similarity(
            torch.stack(query_embeddings, dim=0), torch.stack(doc_embeddings, dim=0), dim=-1
        )
        predicts.extend(batch_output.detach().cpu().numpy())
        labels.extend(batch_labels)
    metric_result = metric(labels, predicts)
    print(metric_result)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    model_name_or_path = ""
    data_path = ""
    inference_single_gpu_scored_pair(model_name_or_path, data_path,
    pooling_strategy = PoolingStrategy.last_mean, metric = roc_auc_score, query_instruction = "为这个句子生成表示以用于检索相关文章：")
