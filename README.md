# PRISM

## Datasets

We use the Amazon Review datasets Home, Beauty, Sports and Yelp. The data split is done in the leave-one-out setting.
Make sure you download the datasets from the Amazon and Yelp.

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
