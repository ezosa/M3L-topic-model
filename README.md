Code for our COLING 2022 paper [Multilingual and Multimodal Topic Modelling with Pretrained Embeddings](https://aclanthology.org/2022.coling-1.355)

Our proposed topic model is:
- multilingual 
- multimodal (image-text) 
- multimodal *and* multilingual (M3L)

### Abstract

We present M3L-Contrast--—a novel multimodal multilingual (M3L) neural topic model for comparable data that maps multilingual texts and images into a shared topic space using a contrastive objective. As a multilingual topic model, it produces aligned *language-specific topics* and as multimodal model, it infers textual representations of semantic concepts in images. We also show that our model performs almost as well on unaligned embeddings as it does on aligned embeddings.

Our model is based on the [Contextualized Topic Model](https://github.com/MilaNLProc/contextualized-topic-models)

We use the PyTorch Metric Learning library for the [InfoNCE/NTXent loss](https://github.com/KevinMusgrave/pytorch-metric-learning/)

### Model architecture

![Multilingual and multimodal topic model architecture](https://github.com/ezosa/M3L-topic-model/blob/master/images/multilingual_and_multimodal.png)


### Citation
```
@inproceedings{zosa-pivovarova-2022-multilingual,
    title = "Multilingual and Multimodal Topic Modelling with Pretrained Embeddings",
    author = "Zosa, Elaine  and  Pivovarova, Lidia",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.355",
    pages = "4037--4048",
}
```
