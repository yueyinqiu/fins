Modified from [https://github.com/kyungyunlee/fins](https://github.com/kyungyunlee/fins)

### Setup

Python 3.11 with conda:

```
conda install ffmpeg

pip install -r requirements.txt
```

### Dataset

Integrate your dataset by modifying the code `fins/dataset/process_data.py` and check it by 

```
python fins/data/process_data.py
```

Example is given with the BIRD dataset and DAPS speech dataset.

### Train

```
python fins/main
```
