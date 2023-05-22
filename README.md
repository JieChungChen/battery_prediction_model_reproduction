## 鋰電池壽命預測模型復現  

復現的模型為學長發表的論文: [Deep neural network battery life and voltage prediction by using data of one cycle only](https://www.sciencedirect.com/science/article/pii/S0306261921014112?via%3Dihub)

### 基本名詞解釋  
* SoH(state of health): 目前電池的最大容量與出廠容量的比例。ex.一個原廠設定最大容量是1Ah的電池被耗損到只能充進0.8Ah的電量→80%SOH
* EoL(end of life): 電池執行完全充放電至壽命結束所耗費的循環數量，一般來說壽命結束被定義為電池最大容量損耗到僅剩80%SoH
* RUL(remain useful life): 目前電池距離EoL剩餘的循環數量。ex.EoL=1000的電池在第900循環時的RUL為100(因為1000-900)

### 文獻回顧
* [Data-driven prediction of battery cycle life before capacity degradation(Severson資料集原論文)](https://www.nature.com/articles/s41560-019-0356-8)
* [Closed-loop optimization of fast-charging protocols for batteries with machine learning](https://www.nature.com/articles/s41586-020-1994-5#Sec2)

### Severson資料集下載  
* [Data-driven prediction of battery cycle life before capacity degradation](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)
* 由124顆商用LFP電池(APR18650M1A)組成，以快充及4C放電循環至EoL。其額定電容量為1.11Ah；額定電壓為3.3V；資料被分為三個batch

### 研究內容解析  

#### last padding  
<img src="https://hackmd.io/_uploads/HkJI4TeHh.png" width="400" alt="img01"/>  

* 由於一般CNN對輸入特徵的尺寸要求是非常嚴格的，因此一個已經固定結構的CNN無法接受不同矩陣大小的輸入。為了使本模型能夠對不同循環長度的電池資訊做預測，這裡使用了last padding的技巧將不同循環長度的特徵都能一致化矩陣尺寸。
* 首先給定一個最大的循環數(這裡設為100)，因此所有的特徵循環數量都要小於100，對於循環數小於100的樣本，我們則不斷複製其最後一個時間步階的數值直到其滿足矩陣長度的要求。
* 該方法也作為一種資料擴增手段，對於最大輸入長度為100的樣本，可以透過last padding生成出100倍的數據量

#### 模型結構  
<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0306261921014112-gr1_lrg.jpg" width="600" alt="img01"/>  

* discharge model: 以Dim. Reduction 1&2作為特徵篩選器，將每循環內放電部分4x500的資訊量減少到只剩EoL及t^charge^(n=EoL)兩種特徵。之後與循環的整體資訊拼接成8x100的特徵，用於預測EoL及最後一循環的充電時間與Q-V curve

#### 訓練結果  
<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0306261921014112-gr2_lrg.jpg" width="600" alt="img01"/>  

#### 特徵相關性分析  
<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0306261921014112-gr3_lrg.jpg" width="600" alt="img01"/>  

* 以上相關性是由Deep Taylor Decomposition計算得出
* 可以看出Discharge model中，特徵篩選器取出的特徵相關性特別高，整體的溫度資訊次之 
* Full model中，充電部分特徵篩選器取出的特徵重要性特別高


### 重現經驗紀錄  

* code詳見: [實驗室Github](https://github.com/smcmlab-nycu/battery-model-reproduction)(需要登入有權限的帳號)

#### 深度學習API版本  
```
tensorflow==2.3.0
torch==1.11.0
```

#### 資料篩選  
* Discharge model使用115顆電池(其前100cycles品質較佳)
* Full model使用95顆電池(其charge curve較理想)
* RUL model使用81顆電池(要幾乎所有循環的資料都品質好，條件較嚴苛)

以下為discharge model中刪除的電池樣本

| 刪除序號              | 刪除原因                   | 備註 |
|:--------------------- |:-------------------------- |:---- |
| batch1: 0,1,2,3,4     | 和batch2中的資料重複了     |      |
| batch1: 8,10,12,13,22 | 尚未到達80%SOH就中斷       |      |
| batch1: 18            | 前100cycle中有異常Qc值     |      |
| batch1: 5,14,15       | 前100cycle中有異常充電時間 |      |
| batch2: 1             | 壽命過低(EOL=170)          |      |
| batch2: 6,10          | 循環內溫度偏移             |      |
| batch2: 9             | 壽命過低                   |      |
| batch2: 15            | 初始電容量太小(<1Ah)       |      |
| batch2: 21,25         | 熱電偶在實驗過程中掉落     |      |
| batch3: 5,6           | 循環內溫度偏移             |      |
| batch3: 23,32,37      | 資料存在noise              |      |

* 溫度偏移判斷: 左邊是正常的溫度訊號，每循環的溫度都隨時間逐漸上升  
<img src="https://hackmd.io/_uploads/rJrggjeBh.png" width="600" alt="img01"/>

#### EoL處理  
* 透過對EoL的數值取log2來減少過大過小值對模型的影響，可以看到在取log後的數據分布更加集中且較接近常態分佈。  
<img src="https://hackmd.io/_uploads/Sk8zCoeBh.png" width="600" alt="img01"/>
