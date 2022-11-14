Code for our COLING 2022 paper **Multilingual and Multimodal Topic Modelling with Pretrained Embeddings**

Our proposed topic model is:
- multilingual 
- multimodal (image-text) 
- multimodal *and* multilingual (M3L)

**Abstract**

We present M3L-Contrast--—a novel multimodal multilingual (M3L) neural topic model for comparable data that maps multilingual texts and images into a shared topic space. Our model is trained jointly on text and image embeddings and uses a contrastive objective to map similar examples close to each other in the topic space. As a multilingual topic model, it produces aligned *language-specific topics* and as multimodal model, it infers textual representations of semantic concepts in images. We also show that our model performs almost as well on unaligned embeddings as it does on aligned embeddings.

Paper: <https://aclanthology.org/2022.coling-1.355>

Our model is based on the Contextualized Topic Model: <https://github.com/MilaNLProc/contextualized-topic-models>

We use the PyTorch Metric Learning library for the InfoNCE/NTXent loss: <https://github.com/KevinMusgrave/pytorch-metric-learning/>


### Citation
```
@inproceedings{zosa-pivovarova-2022-multilingual,
    title = "Multilingual and Multimodal Topic Modelling with Pretrained Embeddings",
    author = "Zosa, Elaine  and
      Pivovarova, Lidia",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.355",
    pages = "4037--4048",
    abstract = "This paper presents M3L-Contrast{---}a novel multimodal multilingual (M3L) neural topic model for comparable data that maps texts from multiple languages and images into a shared topic space. Our model is trained jointly on texts and images and takes advantage of pretrained document and image embeddings to abstract the complexities between different languages and modalities. As a multilingual topic model, it produces aligned language-specific topics and as multimodal model, it infers textual representations of semantic concepts in images. We demonstrate that our model is competitive with a zero-shot topic model in predicting topic distributions for comparable multilingual data and significantly outperforms a zero-shot model in predicting topic distributions for comparable texts and images. We also show that our model performs almost as well on unaligned embeddings as it does on aligned embeddings.",
}
```
