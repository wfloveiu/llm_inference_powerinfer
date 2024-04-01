---
language:
- en
library_name: transformers
license: llama2
---


# ReluLLaMA-7B

- Model creator: [Meta](https://huggingface.co/meta-llama)
- Original model: [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Fine-tuned by: [THUNLP](https://nlp.csai.tsinghua.edu.cn/) and [ModelBest](modelbest.cn)

### Background

Sparse computation is increasingly recognized as an important direction in enhancing the computational efficiency of large language models (LLMs). Among various approaches, the mixture-of-experts (MoE) method, exemplified by models like Mixtral, has shown particular promise. MoE works by selectively activating different model components (experts), thus optimizing resource usage.

Recent studies ([Zhang el al., 2021](https://arxiv.org/abs/2110.01786); [Liu et al., 2023](https://openreview.net/pdf?id=wIPIhHd00i); [Mirzadeh et al., 2023](https://arxiv.org/abs/2310.04564)) reveal that LLMs inherently exhibit properties conducive to sparse computation when employing the ReLU activation function. This insight opens up new avenues for model efficiency, akin to MoE's selective activation. By dynamically choosing model parameters for computation, we can substantially boost efficiency.

However, the widespread adoption of ReLU-based models in the LLM field remains limited. Referring to the transformation methods from existing works ([Zhang el al., 2021](https://arxiv.org/abs/2110.01786); [Mirzadeh et al., 2023](https://arxiv.org/abs/2310.04564)), we convert existing models to ReLU-activated versions through fine-tuning. We hope these open-source ReLU LLMs could promote the development of sparse LLMs.

### Dataset

We finetune the model on about 5 billion tokens, including:

* Wikipedia
* Pile
* StackOverflow

We optimistically believe that by continuing to train with more tokens (covering a wider variety of data), the model will further approach its original performance.

### Training Details

We jointly optimize the model on the conventional language modeling objective and the knowledge distillation objective. The knowledge distillation objective is to minimize the KL divergence between the teacher model and the student model. The teacher model is the original LLM, and the student model is the ReLU-activated version. Since the size of the fine-tuning data is relatively small, we introduce the knowledge distillation objective to avoid overfitting and enhance the generalization ability of the model, which can be also seen as a technique of label smoothing.

| Parameter             | Value       |
|-----------------------|-------------|
| Finetune_Type         | Full FT     |
| Batch_Size            | 2048        |
| GPUs                  | 8xA100(80G) |
| LR_Scheduler          | cosine      |
| LR                    | 3e-5        |


### Evaluation

We evaluate the model on the datasets of [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). The results are shown below:

| Metric                | ReLU Value | [Orig Value](https://huggingface.co/datasets/open-llm-leaderboard/details_meta-llama__Llama-2-7b-hf) |
|-----------------------|-------|------------|
| ARC (25-shot)         | 49.48 | 53.07      |
| HellaSwag (10-shot)   | 74.67 | 78.59     |
| MMLU (5-shot)         | 44.84 | 46.87     |
| TruthfulQA (0-shot)   | 39.04 | 38.76     |
| Winogrande (5-shot)   | 69.37 | 74.03     |
| GSM8K (5-shot)        | 10.61 | 14.48     |
| Average               | 48.00 | 50.97     |

### Inference Tool

We utilize [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) for inference, here we present the inference speeds of pure CPU-based inference with fp16 precision.
The CPU configuration includes an Intel i9-13900K processor (eight performance cores at 5.4GHz) and 192GB of host memory (with a memory bandwidth of 67.2 GB/s).

Dense Inference: 5.17 tokens/s

Sparse Inference: 8.21 tokens/s

### License Disclaimer:

This model is bound by the license & usage restrictions of the original Llama-2 model. And comes with no warranty or gurantees of any kind.

### Limitations & Biases:

Llama 2 and fine-tuned variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Llama 2 and any fine-tuned varient's potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 2 variants, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available at https://ai.meta.com/llama/responsible-use-guide/

### Citiation:

Please kindly cite using the following BibTeX:

```bibtex
@misc{sparsellm,
    title={Sparse Large Language Models with ReLU Activation}, 
    author={SpaseLLM Team},
    year={2023}
}
```

#### Acknowledgments:

The model card is modified from [ORCA_LLaMA_70B_QLoRA](https://huggingface.co/fangloveskari/ORCA_LLaMA_70B_QLoRA).
