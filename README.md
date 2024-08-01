# mPrompt

Crowd Counting via Segmenter-Regressor Mutual Prompt Learning

**Testing code of mPrompt is available.**

# Datasets Preparation
Download the datasets `ShanghaiTech A`, `ShanghaiTech B`, `UCF-QNRF` and `NWPU`. 
Then generate the density maps via `gen_den_map.py`.

# Test
Download the pretrained model via Link：https://pan.baidu.com/s/1wv4szaK1Z8UDKHOwi_ZtTA, Extract Password：bked

or

```bash
bash download_models.sh
```

And put the model into folder `./models/`

```bash
bash run_test.sh
```
