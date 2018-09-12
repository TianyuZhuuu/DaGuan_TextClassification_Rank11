# DaGuan_TextClassification_Rank11

"达观杯"文本智能处理挑战赛 
http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

最终排名11/2862, F1值0.79810

#### 使用到的模型:

| 模型          | 本地CV         | 线上A榜得分   |
| ------------- |:-------------:| -----:|
| FastText      | 0.758800 | 0.766523 |
| TextCNN       | 0.760200 |   0.773990 |
| High Dropout Pooled BiLSTM | 0.761100      |    0.778444 |
|Pooled BiLSTM | 0.762300 | 0.779099 |
|Pooled BiLSTM 2Layer | 0.764700 | 0.783885 |
|TextGRU Ultimate | 0.766500 | 0.784384 |
|TextGRUCNN | 0.768300 | **0.786284**|

最好的单模型在A榜可以排到38名。上述模型均为单词级模型，为增加模型多样性，除此以外还训练了字符级模型和SVD降维后的模型，本地CV远低于单词集模型，因此未在线上进行提交验证。

#### 模型融合
全部模型用10折交叉验证生成out-of-fold预测，用3个深度不同的LGB模型进行stacking(每个深度10个随机种子，共30个)，所用模型的结果使用HillClimbing集成计算权重，
加权得到预测结果，最终结果本地CV 0.795714，A榜0.79864，B榜0.79810
