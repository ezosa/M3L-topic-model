import pandas as pd
import tarfile
from tqdm import tqdm
import gzip
import os
import shutil
from PIL import Image
from sentence_transformers import SentenceTransformer
from io import BytesIO
import base64

# tar file of Wikipedia images from https://www.kaggle.com/c/wikipedia-image-caption/


def unpack_pixels_and_clip_encode(tar_file="image_data_train.tar", outfile="image_data_train_clip.csv", clip_encoder="clip-ViT-B-32",):
    """Unpack pixels of Wikipedia images and encode with CLIP."""
    pixels_path = "image_data_train/image_pixels"
    tar = tarfile.open(tar_file)
    for f in tar.getmembers():
        if os.path.exists(outfile) is False:
            if f.name.endswith("gz"):
                if f.name.startswith(pixels_path):
                    tar.extract(f)
                    print("Extracted tar member:", f.name)
                    pixels_df = pd.read_csv(f.name, compression='gzip', header=None, sep='\t')
                    image_urls = list(pixels_df[0])
                    image_pixels = []
                    valid_urls = []
                    for i, img_url in enumerate(image_urls):
                        if '.svg' not in img_url:
                            img = Image.open(BytesIO(base64.b64decode(pixels_df.iloc[i][1])))
                            img = img.convert("RGB")
                            image_pixels.append(img)
                            valid_urls.append(img_url)
                    os.remove(f.name)
                    img_model = SentenceTransformer(clip_encoder)
                    encoded_images = img_model.encode(image_pixels)
                    encdf = pd.DataFrame(encoded_images)
                    encdf['image_url'] = valid_urls
                    print("encdf shape:", encdf.shape)
                    encdf.to_csv(outfile, index=False)
                    print("Saved encoded file as", outfile)


def unpack_resnet_embeddings(tar_file="image_data_train.tar", outfile="image_data_train_resnet.csv"):
    """Unpack ResNet embeddings of Wikipedia images."""
    emb_path = "image_data_train/resnet_embeddings"
    tar = tarfile.open(tar_file)
    with open(outfile, "w") as out:
        for f in tar.getmembers():
            if f.name.endswith("gz"):
                if f.name.startswith(emb_path):
                    tar.extract(f)
                    print(f.name)
                    with gzip.open(f.name, "rt") as imp:
                        for line in tqdm(imp):
                            line = line.strip()
                            image_url, emb = line.split("\t")
                            print(image_url+","+emb, file=out)
                    os.remove(f.name)
    print("Done unpacking ResNet encodings")



