# Joint Vision-Language Social Bias Removal for CLIP [CVPR 2025]

[![arxiv](https://img.shields.io/badge/paper-Arxiv-blue.svg)](https://arxiv.org/abs/2411.12785)
## Usage
### Train for Joint Vision-Language Social Bias Removal
Install all dependencies in the ```requirements.txt```. Specifically, PyTorch 1.7.1 can be installed using the following command, whereas the rest can be installed using ```pip```.

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
```

Then, download FairFace, UTKFace and FACET datasets from their websites. 

To start training of our V-L debiasing model, change settings in ```train.py``` then use the following command:
```
$ python -u train.py --version (any name)
```
where a folder corresponding to this experiment will be automatically created inside the ```exp``` folder containing ```stdout.log```, ```stderr.log``` and ```best.pth```.

### Evaluate Social Bias and V-L Performance
To evaluate the trained model or the original CLIP model, change the settings in ```eval_all.py``` including the path of model weights to be loaded. 

Then, simply run 
```
$ python eval_all.py
``` 
to check the results.


## Citation:
If you found this repo helpful, please kindly consider citing the following paper :+1: :
```ruby
@inproceedings{joint_vl_debiasing,
      author={Haoyu Zhang and Yangyang Guo and Mohan Kankanhalli},
      title={Joint Vision-Language Social Bias Removal for CLIP}, 
      booktitle={CVPR},
      year={2025},
}
```

## Acknowledgements
The code of fairness evaluation and dataset is based on [Berg et al., 2022](https://github.com/oxai/debias-vision-lang).

Our model implementation is inspired by [Li et al., 2021](https://github.com/salesforce/ALBEF)

The CLIP-clip implementation uses code from [Gao et al., 2017](https://github.com/wgao9/mixed_KSG).

The Biased-prompt implementation is based on [Chuang et al., 2023](https://github.com/chingyaoc/debias_vl).
