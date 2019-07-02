# Style transfer

卷积神经网络的每一层的激活值都可以看做是图像的抽象表示。卷积神经网络中**某层**的每个激活值都可以看做是一个**分类器**，众多的分类结果组成了**抽象表示**。

基于以上思路，使用预训练好的VGGNet进行content、style、result的特征提取。

1. content feature  
    content_features在离输入层近的层会比较接近content原图。计算content和result差的loss。

2. style feature
    每一个卷积核对应一种特征提取手段，生成的特征图不同，两两之间直接内积，得到k*k的相似度矩阵Gram Matrix。对content和result的特征图都计算Gram Matrix。求两者的MSE。

```
def gram_matrix(x):
        """Calulates gram matrix
        Args:
        - x: feaures extracted from VGG Net. shape: [1, width, height, ch]
        """
        b, w, h, ch = x.get_shape().as_list()
        features =  tf.reshape(x, [b, w*h, ch])   # [ch, ch] -> (i, j)
        # 选择两列计算余弦相似度
        # [h*w, ch] matrix -> [ch, h*w] * [h*w, ch] -> [ch, ch]
        # 除以维度，防止(高维造成的)过大
        gram = tf.matmul(features, features, adjoint_a=True) \
               / tf.constant(ch * w * h, tf.float32) 
        return gram
```

3. VGGNet
    在风格转换算法中，并不需要全连接层的数据，并且在构建计算图时在这部分的文件读取和变量赋值操作很耗时（参数量很大），所以不考虑。只计算的特征图抽取部分即可。

## 优化策略
    - 训练对象改变，训练的不是随机初始化的图片，而是将内容图像通过一个转换网络的输出代替之前的随机初始化的输入。而训练的对象就变成了这个转换网络。此时，只需要移除style loss计算部分就可以实现训练高清化图片网络的效果。
        * 注意一：转换网络使用conv代替pooling，获得更多的信息。不使用 pooling层,使用 strided和 fractionally strided卷积来做 downsampling和 upsampling。
        * 注意二：使用五个 residual blocks。residual保证了在这种输入和输出共享的信息较多的时候，可以向前传播更多的信息。
        * 注意三：输出层使用 scaled tanh保证输出值在[0,255]
        * 注意四：第一个和最后一个卷积核使用9X9的核,其他使用3×3

            > ◆先down- sampling再做up- sampling减小了 feature_map的大小,提高性能
            
            > ◆提高结果图像中的视野域
            
            > ◆风格转换会导致物体变形,因而,结果图像中每个图像对应着初始结果中的视野越大越好
            
            > ◆使用 residual connection输入和输出可以共享一部分信息，帮助学到需要变换的部分
    
    - Gram矩阵虽然有效,但是并不直观,无从解释。分割成比如[112，112]的多个小图片，content loss每个图片分别做（对应位置），和V1一样。计算风格loss时，直接将某一个生成的激活值和风格图像的每一个小图片进行余弦相似度计算，找到最match的那一个，计算平方差损失。重新定义风格损失。
