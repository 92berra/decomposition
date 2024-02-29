<h1>Decompose</h1>
<a href="https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=2acbb96fadd1a450ffe0bdc3ef48d419&keyword=%EC%88%AD%EC%8B%A4%EB%8C%80%20%EC%98%A4%ED%98%84%ED%99%94">Thesis URL</a>

This repository is not completed. Just in progress. 

<br>

<h2>How to run</h2>

<h3>Setting up the environment</h3>

```
# create environment
conda create --name decompose python=3.9.13

# activate environment
conda activate decompose

# install Pytorch according to your gpu
# conda
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch

# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

<h3>Datasets</h3>
<li>This repository provides default *.ttf and *.txt.</li>

```
# change directory to tools

# generate content images
python font2img.py --label_file ../labels/50-common-hangul.txt --font_dir ../fonts/source --output_dir ../images/source

# generate target images
python font2img.py --label_file ../labels/50-common-hangul.txt --font_dir ../fonts/target --output_dir ../images/target --start_idx 1
```

<h3>Separate style components</h3>

```
# change directory to tools/separator

python separator-1type.py
python separator-2type.py
python separator-3type.py
python separator-4type.py
python separator-5type.py
python separator-6type.py
```

<h3>Train</h3>

```
CUDA_VISIBLE_DEVICES=3 python main.py --mode train --output_dir trained_model --max_epochs 100 --checkpoint trained_model
```

<h3>Test</h3>

```
CUDA_VISIBLE_DEVICES=3 python main.py --mode test --output_dir tested_model --checkpoint trained_model_add
```

<h3>Accuracy quantification</h3>

```
CUDA_VISIBLE_DEVICES=1 python testing_codes/image-data-seperation.py 
python testing_codes/computing_ssim.py
CUDA_VISIBLE_DEVICES=2 python testing_codes/L1L2LossWithoutTensorflow.py
```

<h3>Check via user interface</h3>

```
#in the directory tools/ui-test
python draw-text.py
```
