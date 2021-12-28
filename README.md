### 代码说明

##### 时序建模：

模型在LSTM.py中，

```bash
在linux下运行需要在当前文件夹的终端内输入./run.sh,终止运行需要输入./stop.sh
注意：每次重新运行必须关闭上一次的服务，否则会造成端口冲突
在Windows下运行，需要在当前文件夹终端内输入nnictl create --config config_detailed.yml，终止运行需要输入nnictl stop
```

时序模型的代码运行需要安装nni包，如果无法成功安装nni包，请联系我们。

##### 非时序模型：

主程序在multi-classify.ipynb中，从featrue2.py中获取所需要的特征。
