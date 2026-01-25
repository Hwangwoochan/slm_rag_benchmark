---
library_name: peft
model_name: qwen2.5_0.5b_rag_lora
tags:
- base_model:adapter:/home/nvidia/Desktop/llama.cpp/Qwen2.5-0.5B-Instruct
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
base_model: /home/nvidia/Desktop/llama.cpp/Qwen2.5-0.5B-Instruct
---

# Model Card for qwen2.5_0.5b_rag_lora

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 0.27.1
- Transformers: 4.57.6
- Pytorch: 2.5.1+cu121
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```