# IMDB_sentiment_analysis_torch

## 准确率统计表格 (按准确率排序)
| **file** | **accuracy** |
| :---: | :---: |
| roberta_trainer | 0.95404 |
| bert_trainer | 0.93664 |
| bert_scratch | 0.93352 |
| distilbert_trainer | 0.92828 |
| distilbert_native | 0.91252 |
| bert_native | 0.90088 |
| attention_lstm | 0.86168 |
| cnnlstm | 0.83648 |
| transformer | 0.83208 |
| cnn | 0.77256 |
| gru | 0.76368 |
| capsule_lstm | 0.5 |
| lstm | 0.5 |


## 10.26 进度
- 完成cnn、gru、lstm、attention_lstm和capsule_lstm的运行和准确率统计，统计数据源文件为./result/accuracy_summary.csv
- transformer报错修改还未完成
- bert运行时显卡过热关机，考虑使用云端服务器或去修理
- 其余还未进行运行
## 11.3 进度
- 完成所有方法的运行与准确率统计，统计数据源文件为./result/accuracy_summary.csv
- 完成bert原理视频的学习