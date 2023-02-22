## 鋰電池壽命預測模型復現

復現的模型為學長發表的論文: [Deep neural network battery life and voltage prediction by using data of one cycle only](https://www.sciencedirect.com/science/article/pii/S0306261921014112?via%3Dihub)

### code說明

* [SeversonDataset_preprocess.py](SeversonDataset_preprocess.py): 處理MIT資料集的原檔
* [data_preprocessing.py](data_preprocessing.py): 完成normalization並編寫pytorch使用的dataset
* [discharge_model.py](discharge_model.py): 編寫放電模型的神經網路架構
* [full_model.py](full_model.py): 編寫完全充放電模型的神經網路架構
* [utils.py](utils.py): 訓練神經網路使用的工具
