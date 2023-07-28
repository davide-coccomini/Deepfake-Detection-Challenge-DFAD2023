# ELSA MEDIA ANALYTICS BENCHMARK

# Round 1
The dataset for the first round is composed of 1M fake images and 1M real images. Fake Images are generated from a subset of captions extracted from LAION 400M [1] using a stable diffusion model. The data are generated using two nodes of the Leonardo HPC facility each one equipped with 4x NVIDIA A100 gpus. The real images are extracted directly from LAION 400M.

# Structure
The data are organized using the following structure:

Dataset

      |---  fake-images

      |              |-------------- part-000001

      |              |--------------  ....

      |              |-------------- part-N


Each folder contains nearly 1k images and a JSON file with metadata.

# Real Images:
Real images are available for download in the file 'laion_train_real.csv', comprising approximately 1 million records of images sourced from LAION 400M [1]. To facilitate the download process, a basic script named 'laion_train_download.py' is provided.

The arguments considered by 'laion_train_download.py' are:
- "--output_folder" that represents the folder for the downloaded images, default current working directory
- "--path" that represents the path of the csv file, default "./train_downloaded_0.csv"
- "--images_per_folder" that represents the number of images per folder, default 10000



# The Metadata are:

- ID: Laion image ID
- original_prompt: Laion Prompt
- positive_prompt: positive prompt used for image generation
- negative_prompt: negative prompt used for image generation
- model: model used for the image generation
- nsfw: nsfw tag from Laion
- url_real_image: Url of the real image associated to the same prompt
- filepath: filepath of the fake image
- aspect_ratio: aspect ratio of the generated image
 
# Training Set
(Links will be available soon)

# Test Set
The test set contains 50k examples of both fake and real images [test-set](https://benchmarks.elsa-ai.eu/?com=downloads&amp;action=download&amp;ch=3&amp;f=aHR0cHM6Ly9haWxiLXdlYi5pbmcudW5pbW9yZS5pdC9wdWJsaWNmaWxlcy9kcml2ZS9lbHNhX2RhdGFzZXQvdmVyc2lvbl8xL21lZGlhX2FuYWx5dGljc19jaGFsbGVuZ2UvRWxzYV90ZXN0LnRhci5neg==").

# Acknowledge
[1] Christoph Schuhmann, et al: “LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs”, 2021 - [link to dataset](https://laion.ai/blog/laion-400-open-dataset/);