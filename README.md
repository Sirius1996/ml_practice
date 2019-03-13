### 练习说明

#### Task2

实现的random walk部分代码如下。

```python
def build_deepwalk_corpus(G, num_paths, path_length,rand=random.Random(0)):
    walk_seq = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        # Generate random walk sequence for every node
        for node in nodes:
            # Implement random walk.
            path = [node]
            out_path = []

            while len(path) < path_length:
                cur = path[-1]
                if len(G[cur]) > 0:
                    path.append(choice(G[cur]))
                else:
                    break

            for ret in path:
                out_path.append(str(ret))

            walk_seq.append(out_path)

    return walk_seq
```

返回随机游走的序列。

产生的结果，生成在 deepwalk/result 路径下。

#### Task3

首先从给定的数据集中进行随机选择，我在load之前对数据集名称进行随机选择处理。

```python
# Set dataset range
dataset_sources = ['citeseer','cora','pubmed']
dataset_source = choice(dataset_sources)
```

接下来添加8个隐藏层，首先初始化

```python
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 16, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 16, 'Number of units in hidden layer 6.')
flags.DEFINE_integer('hidden7', 16, 'Number of units in hidden layer 7.')
flags.DEFINE_integer('hidden8', 16, 'Number of units in hidden layer 8.')
```

接下来，在model中GCN方法下，增加层数。

```python
    def _build(self):
        # 由于过长，不全部显示
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        
        # ...
```

由于Experiment中添加了一个residual connection，是把上一层的输出添加到本层的输出中。我的处理是，在layer中，将每层的输出增加一个x_shortcut：

```python
output = tf.add_n([output,x_shortcut])
```

在cora伤的训练结果为

```bash
2019-03-13 22:07:12.975205: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Epoch: 0001 train_loss= 1.95461 train_acc= 0.07857 val_loss= 1.95068 val_acc= 0.17600 time= 0.11170
Epoch: 0002 train_loss= 1.94861 train_acc= 0.25714 val_loss= 1.94722 val_acc= 0.32400 time= 0.01022
Epoch: 0003 train_loss= 1.94277 train_acc= 0.50000 val_loss= 1.94340 val_acc= 0.43400 time= 0.01071
Epoch: 0004 train_loss= 1.93624 train_acc= 0.55714 val_loss= 1.93984 val_acc= 0.45000 time= 0.01149
Epoch: 0005 train_loss= 1.92757 train_acc= 0.67857 val_loss= 1.93628 val_acc= 0.44200 time= 0.01333
Epoch: 0006 train_loss= 1.92056 train_acc= 0.66429 val_loss= 1.93239 val_acc= 0.44600 time= 0.01056
Epoch: 0007 train_loss= 1.91186 train_acc= 0.64286 val_loss= 1.92828 val_acc= 0.44400 time= 0.01285
Epoch: 0008 train_loss= 1.90454 train_acc= 0.62857 val_loss= 1.92421 val_acc= 0.44200 time= 0.01059
Epoch: 0009 train_loss= 1.89406 train_acc= 0.70000 val_loss= 1.92011 val_acc= 0.45800 time= 0.01834
Epoch: 0010 train_loss= 1.88493 train_acc= 0.70000 val_loss= 1.91599 val_acc= 0.47800 time= 0.01220
Epoch: 0011 train_loss= 1.87616 train_acc= 0.74286 val_loss= 1.91180 val_acc= 0.51000 time= 0.01323
Epoch: 0012 train_loss= 1.87072 train_acc= 0.73571 val_loss= 1.90748 val_acc= 0.52800 time= 0.01206
Epoch: 0013 train_loss= 1.86251 train_acc= 0.70714 val_loss= 1.90315 val_acc= 0.55200 time= 0.01296
Epoch: 0014 train_loss= 1.84735 train_acc= 0.70714 val_loss= 1.89871 val_acc= 0.56400 time= 0.01123
Epoch: 0015 train_loss= 1.83780 train_acc= 0.72857 val_loss= 1.89422 val_acc= 0.57600 time= 0.01229
Epoch: 0016 train_loss= 1.81915 train_acc= 0.77143 val_loss= 1.88969 val_acc= 0.57800 time= 0.01423
Epoch: 0017 train_loss= 1.81948 train_acc= 0.75714 val_loss= 1.88508 val_acc= 0.59000 time= 0.01232
Epoch: 0018 train_loss= 1.81147 train_acc= 0.77143 val_loss= 1.88036 val_acc= 0.59000 time= 0.01291
Epoch: 0019 train_loss= 1.78988 train_acc= 0.80000 val_loss= 1.87534 val_acc= 0.59800 time= 0.01491
Epoch: 0020 train_loss= 1.78313 train_acc= 0.76429 val_loss= 1.87020 val_acc= 0.60400 time= 0.01248
Epoch: 0021 train_loss= 1.76766 train_acc= 0.80000 val_loss= 1.86487 val_acc= 0.60600 time= 0.01360
Epoch: 0022 train_loss= 1.74921 train_acc= 0.80714 val_loss= 1.85943 val_acc= 0.61000 time= 0.01304
Epoch: 0023 train_loss= 1.74168 train_acc= 0.75000 val_loss= 1.85380 val_acc= 0.61400 time= 0.01334
Epoch: 0024 train_loss= 1.72814 train_acc= 0.77143 val_loss= 1.84797 val_acc= 0.61400 time= 0.01196
Epoch: 0025 train_loss= 1.71721 train_acc= 0.78571 val_loss= 1.84191 val_acc= 0.61400 time= 0.01462
Epoch: 0026 train_loss= 1.70792 train_acc= 0.82143 val_loss= 1.83568 val_acc= 0.62600 time= 0.01241
Epoch: 0027 train_loss= 1.68943 train_acc= 0.81429 val_loss= 1.82938 val_acc= 0.63000 time= 0.01171
Epoch: 0028 train_loss= 1.67868 train_acc= 0.77143 val_loss= 1.82284 val_acc= 0.64400 time= 0.01612
Epoch: 0029 train_loss= 1.66631 train_acc= 0.81429 val_loss= 1.81602 val_acc= 0.65200 time= 0.01085
Epoch: 0030 train_loss= 1.67101 train_acc= 0.80000 val_loss= 1.80893 val_acc= 0.66200 time= 0.01308
Epoch: 0031 train_loss= 1.61723 train_acc= 0.81429 val_loss= 1.80165 val_acc= 0.66600 time= 0.01737
Epoch: 0032 train_loss= 1.64374 train_acc= 0.80000 val_loss= 1.79417 val_acc= 0.66400 time= 0.01109
Epoch: 0033 train_loss= 1.60085 train_acc= 0.85000 val_loss= 1.78656 val_acc= 0.67600 time= 0.01627
Epoch: 0034 train_loss= 1.59783 train_acc= 0.82857 val_loss= 1.77900 val_acc= 0.67600 time= 0.01276
Epoch: 0035 train_loss= 1.56746 train_acc= 0.85000 val_loss= 1.77145 val_acc= 0.68200 time= 0.01199
Epoch: 0036 train_loss= 1.57142 train_acc= 0.85714 val_loss= 1.76377 val_acc= 0.68400 time= 0.01384
Epoch: 0037 train_loss= 1.53495 train_acc= 0.87857 val_loss= 1.75596 val_acc= 0.69000 time= 0.01375
Epoch: 0038 train_loss= 1.52689 train_acc= 0.85000 val_loss= 1.74782 val_acc= 0.69600 time= 0.01254
Epoch: 0039 train_loss= 1.51676 train_acc= 0.85714 val_loss= 1.73937 val_acc= 0.70800 time= 0.01306
Epoch: 0040 train_loss= 1.49229 train_acc= 0.82143 val_loss= 1.73085 val_acc= 0.70800 time= 0.01680
Epoch: 0041 train_loss= 1.47217 train_acc= 0.87143 val_loss= 1.72235 val_acc= 0.71000 time= 0.01577
Epoch: 0042 train_loss= 1.46521 train_acc= 0.88571 val_loss= 1.71376 val_acc= 0.71400 time= 0.01091
Epoch: 0043 train_loss= 1.48915 train_acc= 0.85000 val_loss= 1.70508 val_acc= 0.71800 time= 0.01581
Epoch: 0044 train_loss= 1.42744 train_acc= 0.87857 val_loss= 1.69643 val_acc= 0.72200 time= 0.01191
Epoch: 0045 train_loss= 1.44664 train_acc= 0.85714 val_loss= 1.68788 val_acc= 0.72400 time= 0.01157
Epoch: 0046 train_loss= 1.38420 train_acc= 0.89286 val_loss= 1.67896 val_acc= 0.72800 time= 0.01622
Epoch: 0047 train_loss= 1.41269 train_acc= 0.86429 val_loss= 1.66994 val_acc= 0.73200 time= 0.01319
Epoch: 0048 train_loss= 1.36044 train_acc= 0.89286 val_loss= 1.66085 val_acc= 0.73600 time= 0.01218
Epoch: 0049 train_loss= 1.37246 train_acc= 0.87143 val_loss= 1.65166 val_acc= 0.74400 time= 0.01479
Epoch: 0050 train_loss= 1.34976 train_acc= 0.84286 val_loss= 1.64231 val_acc= 0.74600 time= 0.01389
Epoch: 0051 train_loss= 1.30979 train_acc= 0.88571 val_loss= 1.63305 val_acc= 0.74800 time= 0.01360
Epoch: 0052 train_loss= 1.34911 train_acc= 0.85714 val_loss= 1.62362 val_acc= 0.75000 time= 0.01340
Epoch: 0053 train_loss= 1.32759 train_acc= 0.88571 val_loss= 1.61426 val_acc= 0.75200 time= 0.01261
Epoch: 0054 train_loss= 1.28248 train_acc= 0.90714 val_loss= 1.60485 val_acc= 0.75400 time= 0.01149
Epoch: 0055 train_loss= 1.29519 train_acc= 0.88571 val_loss= 1.59545 val_acc= 0.75600 time= 0.01336
Epoch: 0056 train_loss= 1.28051 train_acc= 0.85000 val_loss= 1.58634 val_acc= 0.76400 time= 0.01176
Epoch: 0057 train_loss= 1.23875 train_acc= 0.93571 val_loss= 1.57727 val_acc= 0.76400 time= 0.01433
Epoch: 0058 train_loss= 1.23108 train_acc= 0.92143 val_loss= 1.56772 val_acc= 0.77000 time= 0.01602
Epoch: 0059 train_loss= 1.22798 train_acc= 0.89286 val_loss= 1.55801 val_acc= 0.77400 time= 0.01132
Epoch: 0060 train_loss= 1.22555 train_acc= 0.92857 val_loss= 1.54826 val_acc= 0.77400 time= 0.01234
Epoch: 0061 train_loss= 1.20502 train_acc= 0.90000 val_loss= 1.53905 val_acc= 0.77400 time= 0.01552
Epoch: 0062 train_loss= 1.20688 train_acc= 0.90000 val_loss= 1.53007 val_acc= 0.77400 time= 0.01267
Epoch: 0063 train_loss= 1.12479 train_acc= 0.92143 val_loss= 1.52103 val_acc= 0.77400 time= 0.01112
Epoch: 0064 train_loss= 1.15719 train_acc= 0.94286 val_loss= 1.51218 val_acc= 0.77600 time= 0.01487
Epoch: 0065 train_loss= 1.11420 train_acc= 0.91429 val_loss= 1.50361 val_acc= 0.77800 time= 0.01138
Epoch: 0066 train_loss= 1.12602 train_acc= 0.92857 val_loss= 1.49483 val_acc= 0.77800 time= 0.01472
Epoch: 0067 train_loss= 1.10880 train_acc= 0.94286 val_loss= 1.48608 val_acc= 0.77800 time= 0.01168
Epoch: 0068 train_loss= 1.14477 train_acc= 0.92143 val_loss= 1.47759 val_acc= 0.77800 time= 0.01424
Epoch: 0069 train_loss= 1.10611 train_acc= 0.92857 val_loss= 1.46927 val_acc= 0.77800 time= 0.01354
Epoch: 0070 train_loss= 1.12255 train_acc= 0.92143 val_loss= 1.46145 val_acc= 0.77600 time= 0.01292
Epoch: 0071 train_loss= 1.07954 train_acc= 0.89286 val_loss= 1.45367 val_acc= 0.77600 time= 0.01302
Epoch: 0072 train_loss= 1.06249 train_acc= 0.93571 val_loss= 1.44593 val_acc= 0.77600 time= 0.01282
Epoch: 0073 train_loss= 1.07659 train_acc= 0.92143 val_loss= 1.43838 val_acc= 0.77400 time= 0.01368
Epoch: 0074 train_loss= 1.07699 train_acc= 0.92857 val_loss= 1.43076 val_acc= 0.77400 time= 0.01696
Epoch: 0075 train_loss= 1.04676 train_acc= 0.91429 val_loss= 1.42319 val_acc= 0.77400 time= 0.01129
Epoch: 0076 train_loss= 1.00799 train_acc= 0.93571 val_loss= 1.41567 val_acc= 0.77600 time= 0.01395
Epoch: 0077 train_loss= 1.01398 train_acc= 0.94286 val_loss= 1.40805 val_acc= 0.77800 time= 0.01785
Epoch: 0078 train_loss= 1.02109 train_acc= 0.93571 val_loss= 1.40074 val_acc= 0.77800 time= 0.01183
Epoch: 0079 train_loss= 1.03909 train_acc= 0.93571 val_loss= 1.39384 val_acc= 0.77800 time= 0.01155
Epoch: 0080 train_loss= 1.00015 train_acc= 0.97143 val_loss= 1.38720 val_acc= 0.77600 time= 0.01722
Epoch: 0081 train_loss= 1.00614 train_acc= 0.90714 val_loss= 1.38084 val_acc= 0.77600 time= 0.01096
Epoch: 0082 train_loss= 0.99936 train_acc= 0.92143 val_loss= 1.37477 val_acc= 0.77600 time= 0.01372
Epoch: 0083 train_loss= 1.00355 train_acc= 0.90000 val_loss= 1.36866 val_acc= 0.77600 time= 0.01565
Epoch: 0084 train_loss= 0.97826 train_acc= 0.92143 val_loss= 1.36263 val_acc= 0.77600 time= 0.01106
Epoch: 0085 train_loss= 0.96908 train_acc= 0.93571 val_loss= 1.35691 val_acc= 0.77600 time= 0.01350
Epoch: 0086 train_loss= 0.96761 train_acc= 0.92143 val_loss= 1.35139 val_acc= 0.77600 time= 0.01649
Epoch: 0087 train_loss= 0.97634 train_acc= 0.93571 val_loss= 1.34582 val_acc= 0.77600 time= 0.01141
Epoch: 0088 train_loss= 0.92449 train_acc= 0.92857 val_loss= 1.33998 val_acc= 0.77600 time= 0.01332
Epoch: 0089 train_loss= 0.93838 train_acc= 0.92857 val_loss= 1.33409 val_acc= 0.77400 time= 0.01473
Epoch: 0090 train_loss= 0.91158 train_acc= 0.92857 val_loss= 1.32840 val_acc= 0.77600 time= 0.01261
Epoch: 0091 train_loss= 0.93719 train_acc= 0.94286 val_loss= 1.32327 val_acc= 0.77600 time= 0.01342
Epoch: 0092 train_loss= 0.94949 train_acc= 0.95714 val_loss= 1.31862 val_acc= 0.77600 time= 0.01405
Epoch: 0093 train_loss= 0.85234 train_acc= 0.95000 val_loss= 1.31410 val_acc= 0.77400 time= 0.01188
Epoch: 0094 train_loss= 0.86402 train_acc= 0.93571 val_loss= 1.30961 val_acc= 0.77400 time= 0.01302
Epoch: 0095 train_loss= 0.89524 train_acc= 0.95000 val_loss= 1.30514 val_acc= 0.77400 time= 0.01198
Epoch: 0096 train_loss= 0.87066 train_acc= 0.93571 val_loss= 1.30005 val_acc= 0.77400 time= 0.01610
Epoch: 0097 train_loss= 0.92863 train_acc= 0.91429 val_loss= 1.29527 val_acc= 0.77400 time= 0.01100
Epoch: 0098 train_loss= 0.92609 train_acc= 0.87857 val_loss= 1.29066 val_acc= 0.77400 time= 0.01226
Epoch: 0099 train_loss= 0.89037 train_acc= 0.91429 val_loss= 1.28547 val_acc= 0.77600 time= 0.01325
Epoch: 0100 train_loss= 0.88579 train_acc= 0.92143 val_loss= 1.28054 val_acc= 0.77800 time= 0.01236
Epoch: 0101 train_loss= 0.86651 train_acc= 0.91429 val_loss= 1.27569 val_acc= 0.77800 time= 0.01652
Epoch: 0102 train_loss= 0.77166 train_acc= 0.96429 val_loss= 1.27073 val_acc= 0.78000 time= 0.01054
Epoch: 0103 train_loss= 0.82397 train_acc= 0.95714 val_loss= 1.26585 val_acc= 0.77800 time= 0.01355
Epoch: 0104 train_loss= 0.86391 train_acc= 0.95714 val_loss= 1.26094 val_acc= 0.77800 time= 0.01317
Epoch: 0105 train_loss= 0.86808 train_acc= 0.94286 val_loss= 1.25597 val_acc= 0.77800 time= 0.01219
Epoch: 0106 train_loss= 0.84248 train_acc= 0.95000 val_loss= 1.25098 val_acc= 0.77800 time= 0.01562
Epoch: 0107 train_loss= 0.84908 train_acc= 0.91429 val_loss= 1.24640 val_acc= 0.77800 time= 0.01146
Epoch: 0108 train_loss= 0.86796 train_acc= 0.92143 val_loss= 1.24264 val_acc= 0.77800 time= 0.01294
Epoch: 0109 train_loss= 0.86809 train_acc= 0.95000 val_loss= 1.23947 val_acc= 0.77800 time= 0.01453
Epoch: 0110 train_loss= 0.77492 train_acc= 0.97143 val_loss= 1.23592 val_acc= 0.77600 time= 0.01112
Epoch: 0111 train_loss= 0.84524 train_acc= 0.95000 val_loss= 1.23276 val_acc= 0.77600 time= 0.01284
Epoch: 0112 train_loss= 0.82436 train_acc= 0.95714 val_loss= 1.22940 val_acc= 0.77800 time= 0.01263
Epoch: 0113 train_loss= 0.83012 train_acc= 0.95000 val_loss= 1.22622 val_acc= 0.78000 time= 0.01345
Epoch: 0114 train_loss= 0.80895 train_acc= 0.92857 val_loss= 1.22306 val_acc= 0.78000 time= 0.01345
Epoch: 0115 train_loss= 0.80211 train_acc= 0.95000 val_loss= 1.21955 val_acc= 0.78400 time= 0.01208
Epoch: 0116 train_loss= 0.78301 train_acc= 0.96429 val_loss= 1.21619 val_acc= 0.78400 time= 0.01528
Epoch: 0117 train_loss= 0.78213 train_acc= 0.97143 val_loss= 1.21326 val_acc= 0.78600 time= 0.01351
Epoch: 0118 train_loss= 0.80806 train_acc= 0.94286 val_loss= 1.21037 val_acc= 0.78200 time= 0.01363
Epoch: 0119 train_loss= 0.76637 train_acc= 0.95714 val_loss= 1.20691 val_acc= 0.78000 time= 0.01556
Epoch: 0120 train_loss= 0.78262 train_acc= 0.94286 val_loss= 1.20380 val_acc= 0.78200 time= 0.01228
Epoch: 0121 train_loss= 0.76158 train_acc= 0.93571 val_loss= 1.20087 val_acc= 0.78200 time= 0.01390
Epoch: 0122 train_loss= 0.77224 train_acc= 0.97143 val_loss= 1.19766 val_acc= 0.78200 time= 0.01603
Epoch: 0123 train_loss= 0.72653 train_acc= 0.99286 val_loss= 1.19419 val_acc= 0.78200 time= 0.01201
Epoch: 0124 train_loss= 0.78579 train_acc= 0.95000 val_loss= 1.19064 val_acc= 0.78000 time= 0.01241
Epoch: 0125 train_loss= 0.75468 train_acc= 0.95714 val_loss= 1.18695 val_acc= 0.78000 time= 0.01597
Epoch: 0126 train_loss= 0.75187 train_acc= 0.95714 val_loss= 1.18305 val_acc= 0.77800 time= 0.01292
Epoch: 0127 train_loss= 0.78744 train_acc= 0.94286 val_loss= 1.17976 val_acc= 0.77600 time= 0.01294
Epoch: 0128 train_loss= 0.75010 train_acc= 0.94286 val_loss= 1.17618 val_acc= 0.77600 time= 0.01595
Epoch: 0129 train_loss= 0.70196 train_acc= 0.97143 val_loss= 1.17235 val_acc= 0.77600 time= 0.01207
Epoch: 0130 train_loss= 0.77875 train_acc= 0.94286 val_loss= 1.16897 val_acc= 0.77600 time= 0.01433
Epoch: 0131 train_loss= 0.73701 train_acc= 0.97143 val_loss= 1.16613 val_acc= 0.77600 time= 0.01466
Epoch: 0132 train_loss= 0.70017 train_acc= 0.95000 val_loss= 1.16354 val_acc= 0.77800 time= 0.01340
Epoch: 0133 train_loss= 0.71598 train_acc= 0.97857 val_loss= 1.16143 val_acc= 0.77800 time= 0.01782
Epoch: 0134 train_loss= 0.79477 train_acc= 0.94286 val_loss= 1.15989 val_acc= 0.77800 time= 0.01716
Epoch: 0135 train_loss= 0.72684 train_acc= 0.95714 val_loss= 1.15848 val_acc= 0.77800 time= 0.01581
Epoch: 0136 train_loss= 0.72221 train_acc= 0.95000 val_loss= 1.15722 val_acc= 0.77800 time= 0.01361
Epoch: 0137 train_loss= 0.76138 train_acc= 0.95000 val_loss= 1.15601 val_acc= 0.77800 time= 0.01258
Epoch: 0138 train_loss= 0.71077 train_acc= 0.95000 val_loss= 1.15485 val_acc= 0.77800 time= 0.01521
Epoch: 0139 train_loss= 0.71587 train_acc= 0.95714 val_loss= 1.15350 val_acc= 0.77800 time= 0.01693
Epoch: 0140 train_loss= 0.70428 train_acc= 0.97857 val_loss= 1.15174 val_acc= 0.78000 time= 0.01243
Epoch: 0141 train_loss= 0.71417 train_acc= 0.95000 val_loss= 1.15027 val_acc= 0.77800 time= 0.01933
Epoch: 0142 train_loss= 0.69698 train_acc= 0.97143 val_loss= 1.14911 val_acc= 0.77800 time= 0.01366
Epoch: 0143 train_loss= 0.74747 train_acc= 0.92143 val_loss= 1.14783 val_acc= 0.77800 time= 0.01171
Epoch: 0144 train_loss= 0.75270 train_acc= 0.95714 val_loss= 1.14594 val_acc= 0.77800 time= 0.01445
Epoch: 0145 train_loss= 0.68286 train_acc= 0.95714 val_loss= 1.14391 val_acc= 0.77800 time= 0.01188
Epoch: 0146 train_loss= 0.73422 train_acc= 0.97857 val_loss= 1.14105 val_acc= 0.77800 time= 0.01219
Epoch: 0147 train_loss= 0.71266 train_acc= 0.93571 val_loss= 1.13823 val_acc= 0.77800 time= 0.01458
Epoch: 0148 train_loss= 0.68726 train_acc= 0.93571 val_loss= 1.13560 val_acc= 0.78200 time= 0.01165
Epoch: 0149 train_loss= 0.70256 train_acc= 0.98571 val_loss= 1.13302 val_acc= 0.78200 time= 0.01528
Epoch: 0150 train_loss= 0.70240 train_acc= 0.97143 val_loss= 1.13023 val_acc= 0.78000 time= 0.01281
Epoch: 0151 train_loss= 0.64483 train_acc= 0.95000 val_loss= 1.12727 val_acc= 0.78000 time= 0.01088
Epoch: 0152 train_loss= 0.64050 train_acc= 0.97143 val_loss= 1.12446 val_acc= 0.78000 time= 0.01499
Epoch: 0153 train_loss= 0.71223 train_acc= 0.95714 val_loss= 1.12179 val_acc= 0.78000 time= 0.01186
Epoch: 0154 train_loss= 0.68286 train_acc= 0.98571 val_loss= 1.11875 val_acc= 0.78000 time= 0.01673
Epoch: 0155 train_loss= 0.70416 train_acc= 0.95714 val_loss= 1.11599 val_acc= 0.78000 time= 0.01568
Epoch: 0156 train_loss= 0.71902 train_acc= 0.97143 val_loss= 1.11415 val_acc= 0.77800 time= 0.01327
Epoch: 0157 train_loss= 0.67448 train_acc= 0.97857 val_loss= 1.11215 val_acc= 0.77800 time= 0.01383
Epoch: 0158 train_loss= 0.66449 train_acc= 0.95714 val_loss= 1.11017 val_acc= 0.78000 time= 0.01621
Epoch: 0159 train_loss= 0.64438 train_acc= 0.96429 val_loss= 1.10892 val_acc= 0.78000 time= 0.01156
Epoch: 0160 train_loss= 0.64329 train_acc= 0.97143 val_loss= 1.10770 val_acc= 0.78200 time= 0.01186
Epoch: 0161 train_loss= 0.68548 train_acc= 0.97857 val_loss= 1.10604 val_acc= 0.78000 time= 0.01584
Epoch: 0162 train_loss= 0.67061 train_acc= 0.95000 val_loss= 1.10408 val_acc= 0.78000 time= 0.01344
Epoch: 0163 train_loss= 0.63683 train_acc= 0.97143 val_loss= 1.10300 val_acc= 0.78200 time= 0.01153
Epoch: 0164 train_loss= 0.66205 train_acc= 0.93571 val_loss= 1.10241 val_acc= 0.78000 time= 0.01586
Epoch: 0165 train_loss= 0.65909 train_acc= 0.97857 val_loss= 1.10189 val_acc= 0.77800 time= 0.01283
Epoch: 0166 train_loss= 0.65230 train_acc= 0.97857 val_loss= 1.10092 val_acc= 0.78000 time= 0.01166
Epoch: 0167 train_loss= 0.65994 train_acc= 0.94286 val_loss= 1.09960 val_acc= 0.78200 time= 0.01653
Epoch: 0168 train_loss= 0.68495 train_acc= 0.97143 val_loss= 1.09867 val_acc= 0.78200 time= 0.01107
Epoch: 0169 train_loss= 0.65944 train_acc= 0.97143 val_loss= 1.09731 val_acc= 0.78000 time= 0.01259
Epoch: 0170 train_loss= 0.66710 train_acc= 0.92857 val_loss= 1.09560 val_acc= 0.78000 time= 0.01468
Epoch: 0171 train_loss= 0.69614 train_acc= 0.95000 val_loss= 1.09331 val_acc= 0.78200 time= 0.01173
Epoch: 0172 train_loss= 0.65031 train_acc= 0.97857 val_loss= 1.09166 val_acc= 0.78200 time= 0.01448
Epoch: 0173 train_loss= 0.64674 train_acc= 0.95714 val_loss= 1.09022 val_acc= 0.78400 time= 0.01230
Epoch: 0174 train_loss= 0.58903 train_acc= 0.97857 val_loss= 1.08896 val_acc= 0.78200 time= 0.01149
Epoch: 0175 train_loss= 0.61592 train_acc= 0.96429 val_loss= 1.08718 val_acc= 0.78200 time= 0.01435
Epoch: 0176 train_loss= 0.67490 train_acc= 0.95714 val_loss= 1.08439 val_acc= 0.78400 time= 0.01173
Epoch: 0177 train_loss= 0.64824 train_acc= 0.98571 val_loss= 1.08142 val_acc= 0.78600 time= 0.01211
Epoch: 0178 train_loss= 0.60770 train_acc= 0.97857 val_loss= 1.07844 val_acc= 0.78400 time= 0.01184
Epoch: 0179 train_loss= 0.64699 train_acc= 0.95000 val_loss= 1.07634 val_acc= 0.78400 time= 0.01387
Epoch: 0180 train_loss= 0.66983 train_acc= 0.95000 val_loss= 1.07500 val_acc= 0.78600 time= 0.01186
Epoch: 0181 train_loss= 0.64909 train_acc= 0.96429 val_loss= 1.07334 val_acc= 0.78600 time= 0.01139
Epoch: 0182 train_loss= 0.60040 train_acc= 0.96429 val_loss= 1.07226 val_acc= 0.78600 time= 0.01640
Epoch: 0183 train_loss= 0.65879 train_acc= 0.95714 val_loss= 1.07135 val_acc= 0.78600 time= 0.01149
Epoch: 0184 train_loss= 0.64051 train_acc= 0.96429 val_loss= 1.07060 val_acc= 0.78600 time= 0.01201
Epoch: 0185 train_loss= 0.60902 train_acc= 0.96429 val_loss= 1.06962 val_acc= 0.78800 time= 0.01753
Epoch: 0186 train_loss= 0.63279 train_acc= 0.94286 val_loss= 1.06878 val_acc= 0.78800 time= 0.01294
Epoch: 0187 train_loss= 0.59645 train_acc= 0.98571 val_loss= 1.06733 val_acc= 0.78600 time= 0.01115
Epoch: 0188 train_loss= 0.63872 train_acc= 0.94286 val_loss= 1.06527 val_acc= 0.78600 time= 0.01618
Epoch: 0189 train_loss= 0.59015 train_acc= 0.97143 val_loss= 1.06310 val_acc= 0.78800 time= 0.01191
Epoch: 0190 train_loss= 0.56642 train_acc= 0.97143 val_loss= 1.06111 val_acc= 0.78800 time= 0.01203
Epoch: 0191 train_loss= 0.62657 train_acc= 0.95714 val_loss= 1.05898 val_acc= 0.78600 time= 0.01428
Epoch: 0192 train_loss= 0.64796 train_acc= 0.95000 val_loss= 1.05727 val_acc= 0.78800 time= 0.01110
Epoch: 0193 train_loss= 0.62399 train_acc= 0.97857 val_loss= 1.05572 val_acc= 0.78800 time= 0.01232
Epoch: 0194 train_loss= 0.61233 train_acc= 0.98571 val_loss= 1.05403 val_acc= 0.78600 time= 0.01224
Epoch: 0195 train_loss= 0.59135 train_acc= 0.97857 val_loss= 1.05217 val_acc= 0.78600 time= 0.01305
Epoch: 0196 train_loss= 0.64115 train_acc= 0.93571 val_loss= 1.05070 val_acc= 0.78800 time= 0.01199
Epoch: 0197 train_loss= 0.58866 train_acc= 0.95714 val_loss= 1.04922 val_acc= 0.78600 time= 0.01105
Epoch: 0198 train_loss= 0.59634 train_acc= 0.95714 val_loss= 1.04746 val_acc= 0.78600 time= 0.01460
Epoch: 0199 train_loss= 0.58434 train_acc= 0.98571 val_loss= 1.04561 val_acc= 0.78800 time= 0.01144
Epoch: 0200 train_loss= 0.61550 train_acc= 0.95714 val_loss= 1.04385 val_acc= 0.79000 time= 0.01234
Optimization Finished!
Test set results: cost= 1.00631 accuracy= 0.80900 time= 0.00636
```



#### 其他

实际上，由于之前没有接触过深度学习，我花了不少时间学习CNN、GCN，Tensorflow训练一个神经网络的基本过程等等很多基本知识。就任务本身来说，第一个作业要求实现一个random walk算法，实际上本身不是太复杂，skip-gram直接调用了Word2Vec。做第二个任务，我花了很多时间学习了Tensorflow的一些基本知识，学习了神经网络、CNN的一些基本的概念、相关知识，相对来说，还是有很多不足。我觉得这是一个过程，在不断的解决问题的同时，学习相关的知识，而非在完全准备好之后在去做些什么。

作业二中，在增加了网络层数之后，似乎accuracy并没有显著提高。很多方面以及在论文中也提到了这一点accuracy并不是一定会随着层数增加而更高，甚至有些时候，层数过多时还会出现退化问题。
