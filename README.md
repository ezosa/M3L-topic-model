Code for our COLING 2022 paper **Multilingual and Multimodal Topic Modelling with Pretrained Embeddings**

Our proposed topic model is:
- multilingual 
- multimodal (image-text) 
- multimodal *and* multilingual (M3L)

**Abstract**

We present M3L-Contrast--—a novel multimodal multilingual (M3L) neural topic model for comparable data that maps multilingual texts and images into a shared topic space. Our model is trained jointly on text and image embeddings and uses a contrastive objective to map similar examples close to each other in the topic space. As a multilingual topic model, it produces aligned *language-specific topics* and as multimodal model, it infers textual representations of semantic concepts in images. We also show that our model performs almost as well on unaligned embeddings as it does on aligned embeddings.

Link to paper: *To appear*

Our model is based on the Contextualized Topic Model: https://github.com/MilaNLProc/contextualized-topic-models

We use the PyTorch Metric Learning library for the InfoNCE/NTXent loss: https://github.com/KevinMusgrave/pytorch-metric-learning/


