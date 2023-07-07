# AGC-net

## Prepare:
```
pip install -r requirements -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```


## train:

```
python main.py --predict_len=12 --cuda --att --data_path=./data/METR-LA-12 --feature_len=3 --wavelets_num=20 --transpose --epochs=1 --best_model_save_path=best_model_12_30w
```
