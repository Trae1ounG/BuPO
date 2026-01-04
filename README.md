<h1 align="center">
<img src="assets/bupo_logo.png" width="230" alt="bupo-logo" />
    <br>
  <em>Bottom-up Policy Optimization:</em><br>
  Your Language Model Policy Secretly Contains Internal Policies

<div>

[![arXiv](https://img.shields.io/badge/arXiv-2512.19673-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.19673)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

</div>

## 📰 News
* 2026-1-4: 🔥 We release the visualization code of internal policy entropy flow.
* 2025-12-24: 🎉 Ranked **#2** of the day on <a href="https://huggingface.co/papers/date/2025-12-24">Huggingface Daily Papers</a>.
* 2025-12-23: 🔥 We release the BuPO <a href="https://github.com/Trae1ounG/BuPO">code</a> and the <a href="https://arxiv.org/abs/2512.19673">paper</a>.


## 👁 Overview
***Bottom-up Policy Optimization*** provides a novel framework to decompose LLM policies into internal layer and modular policies, reveals distinct reasoning patterns across different model architectures, and introduces a bottom-up optimization algorithm that leverages these insights to enhance complex reasoning.

<p align="center">
  <img src="./assets/framework.png" alt="BuPO" width="90%">
</p>



## 🤯 Key Findings:
- **Internal Policies**: Decomposes the unified LLM policy into samplable distributions from individual layers and modules (self-attention & FFN).
- **Progressive Reasoning Pattern**: Discovered a human-like "Exploration-Integration-Convergence" (EIC) pattern in Qwen models, contrasting with the abrupt convergence in Llama models.
- **Bottom-up Policy Optimization (BuPO)**: A novel two-phase RL algorithm that first optimizes an internal, lower-layer policy to reconstruct foundational reasoning, then fine-tunes the full model.
- **Enhanced Reasoning Performance**: BuPO significantly outperforms standard RL on complex reasoning benchmarks.



## 🚀 Quick Start
### Installation 

```
conda create -y -n bupo python=3.10.17 && conda activate bupo
pip install -r requirements
python -m pip install flash-attn --no-build-isolation
pip install -e .
```
### Training
BuPO: specify `k` (internal layer policy index), `iterative_steps` (steps of internal policy optimization) in `run_code/BuPO_qwen3.sh` and `run_code/BuPO_llama.sh` to train the model with BuPO.
```
cd BuPO
conda activate bupo
bash run_code/BuPO_qwen3.sh
```
GRPO:
```
bash run_code/GRPO_qwen3.sh
```

### Internal Policy Entropy Plot 🥷🏼
* First, run `run_eval.sh` in `scripts` folder to obtain the dataset used for visualization.
* Then, run `plot_internal_entropy.py` in `visualization` folder to obtain the plot of internal policy entropy flow.

### Implementation Details 🤔
Our mainly design lays in:
* `verl/models/custom_model`: we modify the source file of model forward pass in `transformers ` to get internal hidden states and internal policy effectively/
* `verl/workers/actor/dp_actor.py/_forward_micro_batch_layer_k()`: here to switch to compute the importance ratio of internal layer policy and update it.


## 🙇‍♂️ Acknowledgement
We thank the <a href="https://github.com/volcengine/verl">verl</a> for their valuable contributions to the open-source community.

## 📬 Contact
For questions, discussion, or collaboration opportunities, feel free to contact us!

* Yuqiao Tan: tanyuqiao2025@ia.ac.cn
* Minzheng Wang: wangminzheng2023@ia.ac.cn

## ✍️ Citation
If you find our work helpful, please cite as:

```
@article{tan2025bupo,
      title={Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies}, 
      author={Yuqiao Tan and Minzheng Wang and Shizhu He and Huanxuan Liao and Chengfeng Zhao and Qiunan Lu and Tian Liang and Jun Zhao and Kang Liu},
      year={2025},
      journal={arXiv preprint arXiv:2512.19673},
      url={https://arxiv.org/abs/2512.19673}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Trae1ounG/BuPO&type=Date)](https://star-history.com/#Trae1ounG/BuPO&Date)
