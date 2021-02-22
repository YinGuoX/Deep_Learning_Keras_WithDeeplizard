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
##### 9.Save And Load A Model With TensorFlow's Keras API
* 全面保存和加载模型
  * model.save(filepath)
  * load_model(filepath)
* 只保存和加载模型架构
  * json_string=model.to_json() or to_yaml()...
  * model_from_json(json_string)
* 只保存和加载模型权重
  * model.save_weights(filepath)
  * model2.load_weights(filepath)
##### 10.Image Preparation For A Convolutional Neural Network With TensorFlow's Keras API
* 为接下来训练识别猫狗的CNN做数据准备
* 组织数据
* 数据处理
  * 使用Keras的ImageDataGenerator()来创建一批数据
* 可视化数据
##### 11.Code Update For CNN Training With TensorFlow's Keras API
* TensorFlow中的steps_per_epoch，validation_steps参数问题
##### 12.Build And Train A Convolutional Neural Network With TensorFlow's Keras API
* 建立、编译、训练识别猫狗的CNN
##### 13.Convolutional Neural Network Predictions With TensorFlow's Keras API
* 看看过拟合下的测试集的结果
  * 使用predictions=model.predict(x,steps,verbose)
* 使用混淆矩阵来观察测试集预测结果
##### 14.Build A Fine-Tuned Neural Network With TensorFlow's Keras API
* 导入VGG16模型
* 微调VGG16模型
  * 将VGG16模型的层添加到新模型中
  * 并且冻结一些层的权重
  * 给模型添加一些新层
##### 15.Train A Fine-Tuned Neural Network With TensorFlow's Keras API
* 根据14节中微调VGG16后的新模型，喂入数据，进行训练
##### 16.Predict With A Fine-Tuned Neural Network With TensorFlow's Keras API
* 根据14节中微调VGG16后的新模型，喂入测试集，查看预测效果
* 绘制混淆矩阵来观察测试集预测结果
##### 17.MobileNet Image Classification With TensorFlow's Keras API
* 导入MobileNet模型
* 使用MobileNet模型进行预测
 * 采用随机的样本图像
 * 查看预测结果
   * from tensorflow.keras.applications import imagenet_utils
   * results = imagenet_utils.decode_predictions(predictions)
##### 18.Process Images For Fine-Tuned MobileNet With TensorFlow's Keras API
* 为接下来训练识别手势识别的微调MobileNet做数据准备
* 组织数据
* 数据处理
  * 使用Keras的ImageDataGenerator()来创建一批数据
* 可视化数据
##### 19.Fine-Tuning MobileNet On A Custom Data Set With TensorFlow's Keras API
* 导入MobileNet模型
* 微调MobileNet模型
  * 挑选合适的MobileNet模型的层，并添加到新模型中(采用与之前微调VGG16不一样的方式)
  * 并且冻结一些层的权重
  * 给模型添加一些新层(输出层)
* 编译、训练新模型
* 使用混淆矩阵来观察预测结果(不再是二分类，而是多分类了！)
##### 20.Data Augmentation With TensorFlow's Keras API
* 数据扩增
 * 如：水平，垂直翻转，旋转，放大，缩小，裁切等操作
* 为什么要数据扩增
 * 使训练集变大，减少过拟合
* 使用Keras进行数据扩增
 * gen = ImageDataGenerator(...)
 * aug_iter = gen.flow(image)
* 保存数据扩增后的数据
 *  aug_iter = gen.flow(image,save_to_dir='./Dog',save_prefix='aug-image-',save_format='jpeg')
##### 21.Mapping Keras Labels To Image Classes
* 使用Keras ImageDataGenerator时，如何查看Keras分配给相应图像的类的id或标签
* 在ImageDataGenerator上访问一个名为class_indices的属性，它将返回包含从类名到类索引映射的字典。
 * 表明对应的类别在one-hot编码上的第几个位置
  * 如： {'cat': 1, 'dog': 0} 表示：10：狗,01：猫
##### 22. Reproducible Results With Keras
* 如何使用Keras通过人工神经网络获得可重复的结果。
* 也即去除训练过程中发生的随机性
* 设置随机种子
##### 23. Initializing And Accessing Bias In Keras
* 如何用Keras代码初始化和访问神经网络中的偏差。
* 参数：use_bias
* 参数：bias_initializer
##### 24.Trainable Parameters In A Keras Model
* 如何快速访问和计算Keras模型中可学习参数的数量。
* model.summary()
##### 25.Trainable Parameters In A Keras Convolutional Neural Network
* 如何使用Keras代码快速访问和计算卷积神经网络中可学习参数的数量
* 是否带零填充
* 是否带最大池化
* 这些都很影响神经网络中的可训练参数的数量
