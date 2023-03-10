## 鋰電池壽命預測模型復現

復現的模型為學長發表的論文: [Deep neural network battery life and voltage prediction by using data of one cycle only](https://www.sciencedirect.com/science/article/pii/S0306261921014112?via%3Dihub)

### code說明

* [SeversonDataset_preprocess.py](SeversonDataset_preprocess.py): 處理MIT資料集的原檔
* [data_preprocessing.py](data_preprocessing.py): 完成normalization並編寫pytorch使用的dataset
* [discharge_model.py](discharge_model.py): 編寫放電模型的神經網路架構
* [full_model.py](full_model.py): 編寫完全充放電模型的神經網路架構
* [dis_dimreduct_train.py](dis_dimreduct_train.py): 訓練discharge model的feature selector
* [full_dimreduct_train.py](full_dimreduct_train.py): 訓練full model的feature selector
* [predictor1_train.py](predictor1_train.py): 訓練放電模型(EOL/chargetime)
* [predictor3_train.py](predictor3_train.py): 訓練完全充放電模型(EOL)
* [model_evaluation.py](model_evaluation.py): 模型測試
* [utils.py](utils.py): 訓練神經網路使用的工具
