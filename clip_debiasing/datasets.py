import os
import subprocess
from abc import ABC
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from clip_debiasing import Dotdict, FAIRFACE_DATA_PATH, FACET_DATA_PATH, UTKFACE_DATA_PATH
import json
import re
import random

class IATDataset(Dataset, ABC):
    GENDER_ENCODING = {"Female": 1, "Male": 0}
    AGE_ENCODING = {"0-2": 0, "3-9": 0, "10-19": 0, "20-29": 0, "30-39": 1,
                    "40-49": 1, "50-59": 1, "60-69": 2, "more than 70": 2}
    RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}
    RACE_ENCODING_UTK = {"White": 0, "Black": 1, "Asian": 2, "Indian": 3, "Others": 4}
    SKIN_ENCODING = {"Light-skinned": 0, "Dark-skinned": 1}

    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.iat_labels: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None

    def gen_labels(self, iat_type: str, label_encoding: object = None, isUTK = False):
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            if not isUTK:
                label_encoding = IATDataset.RACE_ENCODING if label_encoding is None else label_encoding
            else:
                label_encoding = IATDataset.RACE_ENCODING_UTK if label_encoding is None else label_encoding
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = IATDataset.AGE_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "skin_tone":
            labels_list = self.labels['skin_tone']
            label_encoding = IATDataset.SKIN_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "joint":
            labels_list = self.labels["gender"] # any type will suffice, just placeholder
            label_encoding = IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
        else:
            print(iat_type)
            raise NotImplementedError
        labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)
        return labels_list, len(label_encoding)


class FairFace(IATDataset):
    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, ):
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        partition = 'val' if mode == 'test' else 'train'
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", partition, f"{partition}_labels.csv"))
        self.labels.sort_values("file", inplace=True)

        if mode == "val":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = val_labels
        elif mode == "train":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = self.labels.loc[self.labels.index.difference(val_labels.index)]
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)
        
        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))
        
        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)


class AugmentedDataset(Dataset, ABC):
    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.text_embeddings: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None
        self._tokenizer = None


class FairFaceDebiasing_Gender(AugmentedDataset):
    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "train",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, tokenizer: Callable = None,):
        print("Mode:", mode)
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        partition = 'val' if mode == 'test' else 'train'
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", partition, f"{partition}_labels.csv"))
        self.labels.sort_values("file", inplace=True)

        if mode == "val":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = val_labels
        elif mode == "train":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = self.labels.loc[self.labels.index.difference(val_labels.index)]

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self._tokenizer = tokenizer

    def _load_fairface_sample(self, sample_labels) -> dict:
        opposite_dict = {"Male": "Female", "Female": "Male"}
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        text1 = f"This is the photo of a {res.race} {res.gender.lower()} aged {res.age}."
        text2 = f"This is the photo of a {res.race} {opposite_dict[res.gender].lower()} aged {res.age}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        res.word1 = self._tokenizer(f"The concept of {res.gender.lower()}.")
        res.word2 = self._tokenizer(f"The concept of {opposite_dict[res.gender].lower()}.")
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        return ff_sample

    def __len__(self):
        return len(self.labels)

class FairFaceDebiasing_Age(AugmentedDataset):
    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = 'train',
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, tokenizer: Callable = None,):
        print("Mode:", mode)
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        partition = 'val' if mode == 'test' else 'train'
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", partition, f"{partition}_labels.csv"))
        self.labels.sort_values("file", inplace=True)

        if mode == "val":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = val_labels
        elif mode == "train":
            val_labels = self.labels.sample(frac=0.125, random_state=1)
            self.labels = self.labels.loc[self.labels.index.difference(val_labels.index)]

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self._tokenizer = tokenizer

    def _load_fairface_sample(self, sample_labels) -> dict:
        age_conversion = {"0-2": "young", "3-9": "young", "10-19": "young", "20-29": "young", "30-39": "middle-aged",
                "40-49": "middle-aged", "50-59": "middle-aged", "60-69": "old", "more than 70": "old"}
        opposite_dict = {"young": ["middle-aged", "old"], "middle-aged": ["young", "old"], "old": ["young", "middle-aged"]}
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        age = age_conversion[res.age]
        text1 = f"This is the photo of a {age} {res.race} {res.gender.lower()}."
        
        prob = random.random()
        idx = 0 if prob < 0.5 else 1
        text2 = f"This is the photo of a {opposite_dict[age][idx]} {res.race} {res.gender.lower()}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        return ff_sample

    def __len__(self):
        return len(self.labels)

class FairFaceDebiasing_Race(FairFace):
    def __init__(self, iat_type: str = 'race', lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, tokenizer: Callable = None,):
        super().__init__(iat_type=iat_type, lazy=lazy, mode=mode, _n_samples=_n_samples, transforms=transforms, equal_split=equal_split)
        self._tokenizer=tokenizer

    def _load_fairface_sample(self, sample_labels) -> dict:
        races = ["White", "Southeast Asian", "Middle Eastern", "Black", "Indian", "Latino_Hispanic", "East Asian"]
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        text1 = f"This is the photo of a {res.age} {res.race} {res.gender.lower()}."
        
        randIdx = random.randint(0, 5) # 6 remaining races
        if races[randIdx] == res.race:
            randIdx = (randIdx + 1) % 6 # skip the repeating race
        text2 = f"This is the photo of a {res.age} {races[randIdx]} {res.gender.lower()}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        return ff_sample

class FairFaceDebiasing_Joint(FairFace):
    def __init__(self, iat_type: str = 'gender', lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, tokenizer: Callable = None,):
        super().__init__(iat_type=iat_type, lazy=lazy, mode=mode, _n_samples=_n_samples, transforms=transforms, equal_split=equal_split)
        self._tokenizer=tokenizer

    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        text1 = f"This is the photo of a {res.age} {res.race} {res.gender.lower()} celebrity."
        
        # race
        races = ["White", "Southeast Asian", "Middle Eastern", "Black", "Indian", "Latino_Hispanic", "East Asian"]
        randIdx = random.randint(0, 5) # 6 remaining races
        if races[randIdx] == res.race:
            randIdx = (randIdx + 1) % 6 # skip the repeating race
        raceIdx = randIdx

        # age
        age_conversion = {"0-2": "young", "3-9": "young", "10-19": "young", "20-29": "young", "30-39": "middle-aged",
                "40-49": "middle-aged", "50-59": "middle-aged", "60-69": "old", "more than 70": "old"}
        opposite_dict_age = {"young": ["middle-aged", "old"], "middle-aged": ["young", "old"], "old": ["young", "middle-aged"]}
        age = age_conversion[res.age]
        prob = random.random()
        idx = 0 if prob < 0.5 else 1
        ageIdx = idx

        # gender
        opposite_dict_gender = {"Male": "Female", "Female": "Male"}

        text2 = f"This is the photo of a {opposite_dict_age[age][ageIdx]} {races[raceIdx]} {opposite_dict_gender[res.gender].lower()}."

        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
    
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        return ff_sample
        
class FACET(IATDataset): # for bias evaluation only
    SKIN_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, ):
        self.DATA_PATH = str(FACET_DATA_PATH)
        self._transforms = (lambda x: x) if transforms is None else transforms
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "annotations", "annotations.csv"))

        self.labels['gender'] = self.labels.apply(self._label_gender, axis=1)
        self.labels['age'] = self.labels.apply(self._label_age, axis=1)
        self.labels['skin_tone'] = self.labels.apply(self._label_skin_tone, axis=1)

        self.labels.sort_values("filename", inplace=True)

        if iat_type == 'gender':
            self.labels = self.labels[self.labels['gender'] != 'Non-binary']
        
        elif iat_type == 'age':
            labels_young = self.labels.loc[self.labels['age'] == '3-9']
            labels_middle = self.labels.loc[self.labels['age'] == '30-39']
            labels_old = self.labels.loc[self.labels['age'] == '60-69']
            self.labels = labels_young.append(labels_middle, ignore_index=True)
            self.labels = self.labels.append(labels_old, ignore_index=True)

        elif iat_type == 'skin_tone':
            self.labels = self.labels[self.labels['skin_tone'] != 'No-skintone']

        elif iat_type == 'joint':
            self.labels = self.labels[self.labels['gender'] != 'Non-binary']
            self.labels = self.labels[self.labels['age'] != 'No-age']
            self.labels = self.labels[self.labels['skin_tone'] != 'No-skintone']

        test_labels = self.labels.sample(frac=0.1, random_state=1)
        self.labels = test_labels

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))
        
        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def _label_gender(self, row):
        if row['gender_presentation_masc'] == 1:
            return 'Male'
        elif row['gender_presentation_fem'] == 1:
            return 'Female'
        else:
            return 'Non-binary'

    def _label_age(self, row):
        if row['age_presentation_young'] == 1:
            return '3-9'
        elif row['age_presentation_middle'] == 1:
            return '30-39'
        elif row['age_presentation_older'] == 1:
            return '60-69'
        else:
            return 'No-age'
    
    def _label_skin_tone(self, row):
        if row['skin_tone_na'] > 0:
            return 'No-skintone'
        else:
            count = 0
            sum = 0
            for i in range(1, 11):
                if row[f'skin_tone_{i}'] != 0:
                    sum += i
                    count += 1
            if count == 0:
                return 'No-skintone'
            if sum / count <= 5.5: 
                return "Light-skinned"
            else:
                return "Dark-skinned"

        
    def _load_facet_sample(self, sample_labels) -> dict:
        res = Dotdict({'filename': sample_labels['filename']})

        img_fname = self._search_dir(res.filename)

        assert img_fname != None
        bbox = json.loads(sample_labels['bounding_box'])
        left = int(bbox["x"])
        top = int(bbox["y"])
        right = int(bbox["x"] + bbox["width"])
        bottom = int(bbox["y"] + bbox["height"])
        img = Image.open(img_fname)
        img_cropped = img.crop((left, top, right, bottom))

        res.img = self._transforms(img_cropped)
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_facet_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample
    
    def _search_dir(self, img_name):
        possible_dirs = []
        for idx in range(1, 4):
            possible_dirs.append(os.path.join(self.DATA_PATH, f"imgs_{idx}", f"{img_name}"))

        for possible_dir in possible_dirs:
            if os.path.isfile(possible_dir):
                return possible_dir
        return None
    
    def __len__(self):
        return len(self.labels)

class UTKface(IATDataset):
    SKIN_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, ):
        self.DATA_PATH = str(UTKFACE_DATA_PATH)
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "utk_annotation.csv"))
        # print(len(self.labels))
        self.labels['gender'] = self.labels.apply(self._label_gender, axis=1)
        self.labels['age'] = self.labels.apply(self._label_age, axis=1)
        self.labels['race'] = self.labels.apply(self._label_race, axis=1)

        self.labels.sort_values("filename", inplace=True)

        if iat_type == 'race':
            self.labels = self.labels[self.labels['race'] != 'Others']

        if mode == "test":
            test_labels = self.labels.sample(frac=0.1, random_state=1)
            self.labels = test_labels
        elif mode == "val":
            test_labels = self.labels.sample(frac=0.1, random_state=1)
            train_labels = self.labels.loc[self.labels.index.difference(test_labels.index)]
            train_val_labels = train_labels.sample(frac=1/9, random_state=1)
            self.labels = train_val_labels
        else:
            test_labels = self.labels.sample(frac=0.1, random_state=1)
            train_labels = self.labels.loc[self.labels.index.difference(test_labels.index)]
            train_val_labels = train_labels.sample(frac=1/9, random_state=1)

            train_train_labels = train_labels.loc[train_labels.index.difference(train_val_labels.index)] 
            self.labels = train_train_labels

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))
        
        self.iat_labels = self.gen_labels(iat_type=iat_type, isUTK=True)[0]

    def _label_gender(self, row):
        if row['gender_utk'] == 0:
            return 'Male'
        else:
            return 'Female'

    def _label_age(self, row):
        if row['age_utk'] < 30:
            return '3-9'
        elif row['age_utk'] < 60:
            return '30-39'
        else:
            return '60-69'
    
    def _label_race(self, row):
        if row['race_utk'] == 0:
            return 'White'
        elif row['race_utk'] == 1:
            return 'Black'
        elif row['race_utk'] == 2:
            return 'Asian'
        elif row['race_utk'] == 3:
            return 'Indian'
        else:
            return 'Others'
        
    def _load_utkface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, res.filename)
        res.img = self._transforms(Image.open(img_fname))
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_utkface_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample
    
    def __len__(self):
        return len(self.labels)
    
class UTKfaceDebiasing_Gender(UTKface):
    def __init__(self, tokenizer, iat_type: str = 'gender', lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, ):
        super().__init__(iat_type, lazy, mode, _n_samples, transforms, equal_split)
        self._tokenizer = tokenizer

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_utkface_sample(self.labels.iloc[index])
        return ff_sample
    
    def _load_utkface_sample(self, sample_labels) -> dict:
        opposite_dict = {"Male": "Female", "Female": "Male"}
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, res.filename)
        res.img = self._transforms(Image.open(img_fname))

        text1 = f"This is the photo of a {res.race} {res.gender.lower()} aged {res.age_utk}."
        text2 = f"This is the photo of a {res.race} {opposite_dict[res.gender].lower()} aged {res.age_utk}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        return res
    
class UTKfaceDebiasing_Age(UTKface):
    def __init__(self, tokenizer, iat_type: str = 'age', lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, ):
        super().__init__(iat_type, lazy, mode, _n_samples, transforms, equal_split)
        self._tokenizer = tokenizer

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_utkface_sample(self.labels.iloc[index])
        return ff_sample
    
    def _load_utkface_sample(self, sample_labels) -> dict:
        age_conversion = {"0-2": "young", "3-9": "young", "10-19": "young", "20-29": "young", "30-39": "middle-aged",
                "40-49": "middle-aged", "50-59": "middle-aged", "60-69": "old", "more than 70": "old"}
        opposite_dict = {"young": ["middle-aged", "old"], "middle-aged": ["young", "old"], "old": ["young", "middle-aged"]}
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, res.filename)
        res.img = self._transforms(Image.open(img_fname))
        age = age_conversion[res.age]
        text1 = f"This is the photo of a {age} {res.race} {res.gender.lower()}."
        
        prob = random.random()
        idx = 0 if prob < 0.5 else 1
        text2 = f"This is the photo of a {opposite_dict[age][idx]} {res.race} {res.gender.lower()}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        return res

class UTKfaceDebiasing_Race(UTKface):
    def __init__(self, tokenizer, iat_type: str = 'age', lazy: bool = True, mode: str = None,
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = False, ):
        super().__init__(iat_type, lazy, mode, _n_samples, transforms, equal_split)
        self._tokenizer = tokenizer

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_utkface_sample(self.labels.iloc[index])
        return ff_sample
    
    def _load_utkface_sample(self, sample_labels) -> dict:
        races = ["White", "Black", "Asian", "Indian"]
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, res.filename)
        res.img = self._transforms(Image.open(img_fname))
        text1 = f"This is the photo of a {res.age} {res.race} {res.gender.lower()}."
        
        randIdx = random.randint(0, 2) # 3 remaining races
        if races[randIdx] == res.race:
            randIdx = (randIdx + 1) % 6 # skip the repeating race
        text2 = f"This is the photo of a {res.age} {races[randIdx]} {res.gender.lower()}."
        res.text1 = self._tokenizer(text1)
        res.text2 = self._tokenizer(text2)
        return res

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index