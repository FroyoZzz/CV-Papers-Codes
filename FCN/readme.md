# Pytorch-FCN

## Requirements
- pytorch >= 1.0.0
- torchvision >= 0.3.0
- visdom
- Pillow
- numpy

## Training
训练集图像: **data/BagImages**

训练集图像标签: **data/BagImagesMasks**

训练默认参数: **--lr=1e-3 --batch-size=16 --workers=1 --num-classes=2 --save-dir="./models"**

训练好的模型会保存在 **models** 文件夹下
```
python train.py
```

## Testing
测试集图像: **data/testImages**

测试集图像标签: **data/testMasks**

测试集图像预测结果: **data/testPreds**

测试时会加载 **models** 下的 **xxx.ckpt** 模型进行预测

```
python train.py --mode test --ckpt "models\xxx.ckpt"
```
## visdom
安装了 visdom 之后可以可视化训练过程，安装:
```
pip install visdom
```
安装完成之后需要启动 visdom 服务:
```
python -m visdom.server
```
就可以访问 **http://localhost:8097/** 看到可视化的训练过程了