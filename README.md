# AGC-Net: Adaptive Graph Convolution Networks for Traffic Flow Forecasting



AGC-Net (Adaptive Graph Convolution Networks) is an advanced model designed to predict traffic flow. The paper is available [here](https://arxiv.org/abs/2307.05517).


## Citation

If you find this work useful for your research, please cite our paper:


```bibtex

@article{li2023adaptive,
      title={Adaptive Graph Convolution Networks for Traffic Flow Forecasting}, 
      author={Zhengdao Li and Wei Li and Kai Hwang},
      year={2023},
      eprint={2307.05517},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```


## Installation

Before proceeding with the model training, ensure all necessary packages are installed. To install the requirements, run the following command:



```bash

pip install -r requirements -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

```


## Data Preparation



To prepare the data, please make sure you have the METR-LA dataset placed inside the `./data/` directory in a sub-directory named `METR-LA-12`. If you don't have the dataset, you can download it from the [official repository](#) (Note: replace with the appropriate link). The `feature_len` parameter denotes the feature length of the dataset. Here, we use a `feature_len` of 3.


## Training



Once the data is prepared, you can train the AGC-Net model. The following is a sample command to initiate the training:



```bash

python main.py --predict_len=12 --cuda --att --data_path=./data/METR-LA-12 --feature_len=3 --wavelets_num=20 --transpose --epochs=1 --best_model_save_path=best_model_12_30w

```



Here is a brief explanation of the command-line arguments:



* `--predict_len`: The number of future time steps to be predicted by the model (12 in this case).

* `--cuda`: If present, use GPU for training.

* `--att`: If present, use the attention mechanism in the model.

* `--data_path`: The path to the directory where the dataset is stored.

* `--feature_len`: The feature length of the dataset.

* `--wavelets_num`: The number of wavelet functions to be used (20 in this case).

* `--transpose`: If present, transpose the input data.

* `--epochs`: The number of epochs to train the model.

* `--best_model_save_path`: The path where the model with the best validation performance should be saved.



The `best_model_12_30w` will be saved in the provided path upon successful training of the model.



Feel free to explore and adapt the model to suit your own requirements. We look forward to your contribution and feedback.






## License

[MIT](https://choosealicense.com/licenses/mit/)
