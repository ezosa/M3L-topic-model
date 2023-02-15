Code for our COLING 2022 paper [Multilingual and Multimodal Topic Modelling with Pretrained Embeddings](https://aclanthology.org/2022.coling-1.355)

### Abstract

We present M3L-Contrast--—a novel multimodal multilingual (M3L) neural topic model for comparable data that maps multilingual texts and images into a shared topic space using a contrastive objective. As a multilingual topic model, it produces aligned *language-specific topics* and as multimodal model, it infers textual representations of semantic concepts in images. We also show that our model performs almost as well on unaligned embeddings as it does on aligned embeddings.

Our proposed topic model is:
- multilingual 
- multimodal (image-text) 
- multimodal *and* multilingual (M3L)

Our model is based on the [Contextualized Topic Model](https://github.com/MilaNLProc/contextualized-topic-models) (Bianchi et al., 2021)

We use the PyTorch Metric Learning library for the [InfoNCE/NTXent loss](https://github.com/KevinMusgrave/pytorch-metric-learning/)

### Model architecture

<img src="https://github.com/ezosa/M3L-topic-model/blob/master/images/multilingual_and_multimodal.png" width="290" height="350" />

### Dataset
- Aligned articles from the [Wikipedia Comparable Corpora](https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/)
- Images from the [WIT](https://github.com/google-research-datasets/wit) dataset
- We will release the article titles and image urls in the train and test sets (soon!)

### Talks and slides
- [Slides](https://blogs.helsinki.fi/language-technology/files/2022/11/LT-seminar-Elaine-Zosa-2022-11-10.pdf) and [video](https://unitube.it.helsinki.fi/unitube/embed.html?id=dae2b02d-47e7-46b0-adc3-86da8034ed58) from my talk at the Helsinki Language Technology seminar


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
