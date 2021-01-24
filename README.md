# Deep_Learning_Keras_WithDeeplizard
* 主要来自[Deeplizard](https://deeplizard.com/learn/playlist/PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL)的Keras - Python Deep Learning Neural Network API，更多详情可以去对应官网查看~
* 是Deeplizard的中文+Colab版

##### 1.Keras With TensorFlow Prerequisites - Getting Started With Neural Networks
* 介绍Keras与TensorFlow的关系
##### 2.TensorFlow And Keras GPU Support - CUDA GPU Setup
* Linux、Windows如何配置GPU(这里建议自行百度、Google比较好)
* 验证TensorFlow是否检测到GPU
##### 3.Keras With TensorFlow - Data Processing For Neural Network Training
* fit()期望输入的数据类型
* 自己构建一个小规模的数值数据
* 数据处理
  * 特征缩放
  * reshape!
##### 4.Create An Artificial Neural Network With TensorFlow's Keras API
* 构建一个Sequential模型
##### 5.Train An Artificial Neural Network With TensorFlow's Keras API
* 编译并且训练神经网络模型
  * 理解compile中的loss的binary_crossentropy和categorical_crossentropy的关系
  * 理解epoch=1
    * 所有训练数据都要经过了一次神经网络模型的训练
  * 理解batch_size=10
    * 以每次10张图片为一批进入神经网络模型进行训练
  * 所以一个epoch要training set size / batch_size个批次完成一个所有数据的训练
##### 6.Build A Validation Set With TensorFlow's Keras API
* 理解验证集
* 使用验证集
  * 传入验证集
    * validation_data=valid_set
  * 分割训练集
    * validation_split=0.1
* 查看验证精度
  * verbose=2
##### 7.Neural Network Predictions With TensorFlow's Keras API
* 什么是预测(推理)
* 构建验证集
* 进行预测
  * predictions = model.predict(x=scaled_test_samples,batch_size = 10,verbose=0)
* 查看预测结果
  * for i in predictions:...
##### 8.Create A Confusion Matrix For Neural Network Predictions
* 如何评估预测结果
* 混淆矩阵(Confusion Matrix)
  * 理解
  * 绘制

