# IMDB_sentiment_analysis_torch

## 准确率统计表格
| **file** | **accuracy** |
| :---: | :---: |
| attention_lstm | 0.86168 |
| cnnlstm | 0.83648 |
| cnn | 0.77256 |
| gru | 0.76368 |
| capsule_lstm | 0.5 |
| lstm | 0.5 |

## 10.26 V1.0说明
- 完成cnn、gru、lstm、attention_lstm和capsule_lstm的运行和准确率统计，统计数据源文件为./result/accuracy_summary.csv
- transformer报错修改还未完成
- bert运行时显卡过热关机，考虑使用云端服务器或去修理
- 其余还未进行运行