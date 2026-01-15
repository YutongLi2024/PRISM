from pathlib import Path
import os
import json
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict
import gzip
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


DATANAME = 'Beauty'  
BASE_DIR = Path(__file__).resolve().parent

REVIEW_FILE = BASE_DIR / f'{DATANAME}/reviews_{DATANAME}_5.json.gz'
META_FILE = BASE_DIR / f'{DATANAME}/meta_{DATANAME}.json.gz'
IMAGE_TEXT_DIR = BASE_DIR / f'{DATANAME}' / f'image_text/{DATANAME}_image'
IMAGE_JSON = BASE_DIR / f'{DATANAME}' / f'image_text/{DATANAME}_image.json'
TEXT_JSON = BASE_DIR / f'{DATANAME}' / f'image_text/{DATANAME}_text.json'

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def meta_to_5core(review_file, meta_file, save_path):
    countU = defaultdict(int)
    countP = defaultdict(int)
    for one_interaction in tqdm(parse(review_file)):
        rev = one_interaction['reviewerID']
        asin = one_interaction['asin']
        countU[rev] += 1
        countP[asin] += 1

    meta_dict = {item_meta['asin']: item_meta for item_meta in parse(meta_file)}

    itemmap = dict()
    itemnum = 1
    num = 1
    core5_item_dict = {}

    for one_interaction in parse(review_file):
        rev = one_interaction['reviewerID']
        asin = one_interaction['asin']
        if countU[rev] < 5 or countP[asin] < 5:
            continue
        if asin in itemmap:
            continue
        itemmap[asin] = itemnum
        itemnum += 1
        core5_item_dict[asin] = meta_dict.get(asin, {})

    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(core5_item_dict, f, ensure_ascii=False)


def extract_text(item_path, save_path, fields=None, merge_fields=True):
    if fields is None:
        fields = ['title', 'description', 'brand', 'categories']

    with open(item_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    item_text = {}
    no_text = []

    for one_interaction in tqdm(data.values()):
        asin = one_interaction.get('asin', None)
        if not asin:
            continue

        field_values = []
        missing = True

        for field in fields:
            value = one_interaction.get(field, "")
            if field == 'categories':
                if isinstance(value, list):
                    if value and isinstance(value[0], list):
                        value = ' > '.join(value[0])
                    elif value:
                        value = ' > '.join(map(str, value))
                    else:
                        value = ''
                else:
                    value = str(value)
            if value:
                missing = False
            field_values.append(value)

        if missing:
            no_text.append(one_interaction)
            continue

        if merge_fields:
            combined_text = ' '.join([str(x) for x in field_values if x])
            item_text[asin] = combined_text
        else:
            item_text[asin] = dict(zip(fields, field_values))

    print(f'item does not have any text field：{no_text}')
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(item_text, f, ensure_ascii=False)


def _download_single_image(asin, imUrl, dataset):
    try:
        response = requests.get(imUrl, timeout=5)
        if response.status_code == 200:
            image_data = Image.open(BytesIO(response.content))
            image_data.save(f'{dataset}/{asin}.png')
            return (asin, True)
        else:
            return (asin, False)
    except Exception:
        return (asin, False)

def download_image(dataset, item_path, parent_folder, max_workers=16):
    os.makedirs(dataset, exist_ok=True)
    subfolders = {f.split('.')[0] for f in os.listdir(parent_folder)}
    with open(item_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    download_tasks = []
    for one_interaction in data.values():
        asin = one_interaction.get('asin')
        imUrl = one_interaction.get('imUrl')
        if not imUrl or not asin or asin in subfolders:
            continue
        download_tasks.append((asin, imUrl))

    no_image = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_image, asin, imUrl, dataset): asin
            for asin, imUrl in download_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            asin, success = future.result()
            if not success:
                no_image.append(asin)

    if no_image:
        print(f'Items failed to download image: {no_image}')


# def download_image(dataset, item_path, parent_folder):
#     os.makedirs(dataset, exist_ok=True)
#     subfolders = {f.split('.')[0] for f in os.listdir(parent_folder)}
#     with open(item_path, 'r', encoding='utf-8') as json_file:
#         data = json.load(json_file)
#     count = 0
#     no_image = []
#     for one_interaction in tqdm(data.values()):
#         if 'imUrl' not in one_interaction:
#             print(f'{one_interaction} not image_text')
#             no_image.append(one_interaction)
#             continue
#         count += 1
#         asin = one_interaction['asin']
#         if asin in subfolders:
#             print(f'{count}:file{asin} exist')
#             continue
#         imUrl = one_interaction['imUrl']
#         # print(f' from {imUrl} download')
#         try:
#             response = requests.get(imUrl, timeout=5)
#             if response.status_code == 200:
#                 image_data = Image.open(BytesIO(response.content))
#                 image_data.save(f'{dataset}/{asin}.png')
#                 # print(f"{count}: image {asin} save in {dataset}/{asin}.png")
#             else:
#                 print(f"{count}: failed to download image for {asin}")
#         except Exception as e:
#             print(f"{count}: error downloading {asin}: {e}")
#             no_image.append(one_interaction)
#     print(f' item does not have image：{no_image}')


if __name__ == '__main__':
    meta_to_5core(REVIEW_FILE, META_FILE, IMAGE_JSON)
    extract_text(
        item_path=IMAGE_JSON,
        save_path=TEXT_JSON,
        fields=['title', 'description', 'brand', 'categories'],
        merge_fields=True 
    )

    download_image(str(IMAGE_TEXT_DIR), IMAGE_JSON, str(IMAGE_TEXT_DIR))
