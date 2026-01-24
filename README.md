# PRISM

This is our Pytorch implementation for "[PRISM: Personalized Recommendation via Information Synergy Module](https://arxiv.org/abs/2601.10944)"


## Abstract

Multimodal sequential recommendation (MSR) leverages diverse item modalities to improve recommendation accuracy, while achieving effective and adaptive fusion remains challenging. Existing MSR models often overlook synergistic information that emerges only through modality combinations. Moreover, they typically assume a fixed importance for different modality interactions across users. To address these limitations, we propose \textbf{P}ersonalized \textbf{R}ecommend-ation via \textbf{I}nformation \textbf{S}ynergy \textbf{M}odule (PRISM), a plug-and-play framework for sequential recommendation (SR). PRISM explicitly decomposes multimodal information into unique, redundant, and synergistic components through an Interaction Expert Layer and dynamically weights them via an Adaptive Fusion Layer guided by user preferences. This information-theoretic design enables fine-grained disentanglement and personalized fusion of multimodal signals. Extensive experiments on four datasets and three SR backbones demonstrate its effectiveness and versatility.

<p>
<img src="./images/Model.png" >
</p>


## Datasets

We use the Amazon Review datasets Home, Beauty, Sports and Yelp. The data split is done in the leave-one-out setting.
Make sure you download the datasets from the [Amazon](https://snap.stanford.edu/data/amazon/productGraph) and [Yelp](https://business.yelp.com/data/resources/open-dataset).


## Settings

```
python = 3.8
pytorch = 2.1.0
transformers = 4.36.2
clip = 1.0
cuda = 12.1 
```


## DataProcessing

Enter the data folder for data processing and make sure you change the DATASET variable value to your dataset name, you run:

```
cd data
python DataProcessing.py
python Yelp_Process.py
```

Then run this command to get image and text about item:

```
python Image_download.py
```

Then run this command to get image and text embedding:

```
python process_clip.py
```


## Train

Please make sure all datas are in corresponding folder location, then run this command to Training and Prediction:

```
python main.py
```


## Citation
If this work is useful for your research, please cite:

```
@article{zhang2026prism,
  title={PRISM: Personalized Recommendation via Information Synergy Module},
  author={Zhang, Xinyi and Li, Yutong and Sun, Peijie and Sha, Letian and Han, Zhongxuan},
  journal={arXiv preprint arXiv:2601.10944},
  year={2026}
}
```