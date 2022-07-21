# 渐进式深度集成架构搜索算法研究
## 环境安装
进入到系统的目录下，使用pip –r install requirements.txt命令进行安装
## 快速开始
对于论文里的实验，主要是scripts目录下的各个文件运行产生的，该目录下的各个文件夹和文件包含了对集群实验的设置，其主要命令也是通过python –m xxx。下面以benchmarks目录下的bdas_cell.py文件的运行说明为例，具体的运行命令为：
```python
python -m benchamarks.bdas_cell [-d 11] [-s DAG] [-K 8] [-C 8] [--cell_time_limit=120] [--model_time_limit=960] [--strategy=best]

# 其中-d 11指定数据集ID为11，-s DAG指定搜索空间为DAG，-K 8指定集束搜索参数为8，-C 8指定最大Cell数量为8，--cell_timme_limit=120指定每个Cell的最大运行时间为120，-- model_time_limit=960指定每个深度集成架构的运行时间为960秒，--strategy best指定每次保留最优的K个架构。
```
