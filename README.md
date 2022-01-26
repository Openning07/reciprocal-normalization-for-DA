# reciprocal-normalization-for-DA [[pdf]](https://arxiv.org/pdf/2112.10474.pdf)
- Batch normalization (BN) is widely used in modern deep neural networks, which has been shown to represent the domain-related knowledge, and thus is ineffective for cross-domain tasks like unsupervised domain adaptation (UDA). Existing BN variant methods aggregate source and target domain knowledge in the same channel in normalization module. However, the misalignment between the features of corresponding channels across domains often leads to a suboptimal transferability. In this paper, we exploit the cross-domain relation and propose a novel normalization method, Reciprocal Normalization (RN). Specifically, RN first presents a Reciprocal Compensation (RC) module to acquire the compensatory for each channel in both domains based on the cross-domain channel-wise correlation. Then RN develops a Reciprocal Aggregation (RA) module to adaptively aggregate the feature with its cross-domain compensatory components. As an alternative to BN, RN is more suitable for UDA problems and can be easily integrated into popular domain adaptation methods. Experiments show that the proposed RN outperforms existing normalization counterparts by a large margin and helps state-of-the-art adaptation approaches achieve better results.

## The problem and the motivation
- In the context of **domain adaptation**, the misalignment of visual features between source domain and target domain can lead to poor adaptation performance.
- TODO.

## Compared with the existing normalization methods
- TODO.

<p align="center">
  <img src="https://github.com/Openning07/reciprocal-normalization-for-DA/blob/main/other_VS_ours.png" alt="The technical differences between our RN and the existing counterparts." width="75%">
</p>

## Reciprocal Normalization
- TODO.

<p align="center">
  <img src="https://github.com/Openning07/reciprocal-normalization-for-DA/blob/main/RN.png" alt="The technical differences between our RN and the existing counterparts." width="85%">
</p>

## About the code
- Requirements
  - python == 3.6.2
  - pytorch == 0.4.0
  - torchvision == 0.2.2
  - numpy == 1.18.1
  - CUDA == 10.1.105

- Data
  - Office-Home.

- The example shell files.
  - xxx.sh


- The expected outputs.

## Citation
Please cite the following paper if you use this repository in your reseach~ Thank you ^ . ^
```
@article{huang2021reciprocal,
  title={Reciprocal Normalization for Domain Adaptation},
  author={Huang, Zhiyong and Sheng, Kekai and Li, Ke and Liang, Jian and Yao, Taiping and Dong, Weiming and Zhou, Dengwen and Sun, Xing},
  journal={arXiv preprint arXiv:2112.10474},
  year={2021}
}
```
