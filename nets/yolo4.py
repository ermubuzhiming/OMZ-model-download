from functools import wraps
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.mobilenetv2 import MobileNetV2
from keras.regularizers import l2
from utils.utils import compose


# --------------------------------------------------#
#   单次卷积
# --------------------------------------------------#
@wraps(Conv2D)
@wraps(SeparableConv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return SeparableConv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def make_five_convs(x, num_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # 生成darknet53的主干模型
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights=None)
    feat3 = mobilenetv2.get_layer('block_16_project_BN').output
    feat2 = mobilenetv2.get_layer('block_12_project_BN').output
    feat1 = mobilenetv2.get_layer('block_5_project_BN').output
    # feat1,feat2,feat3 = darknet_body(inputs)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85)
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    p5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(p5)
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(p5)
    # 使用了SPP结构，即不同尺度的最大池化后堆叠。
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(p5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(p5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(p5)
    p5 = Concatenate()([maxpool1, maxpool2, maxpool3, p5])
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(p5)
    p5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(p5)
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(p5)

    p5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(p5)

    p4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    p4 = Concatenate()([p4, p5_upsample])
    p4 = make_five_convs(p4, 256)

    p4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(p4)

    p3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    p3 = Concatenate()([p3, p4_upsample])
    p3 = make_five_convs(p3, 128)

    p3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(p3)
    p3_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(p3_output)

    # 26,26 output
    p3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(p3)
    p3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(p3_downsample)
    p4 = Concatenate()([p3_downsample, p4])
    p4 = make_five_convs(p4, 256)

    p4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(p4)
    p4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(p4_output)

    # 13,13 output
    p4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(p4)
    p4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(p4_downsample)
    p5 = Concatenate()([p4_downsample, p5])
    p5 = make_five_convs(p5, 512)

    p5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(p5)
    p5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(p5_output)

    return Model(inputs=inputs, outputs=[p5_output, p4_output, p3_output])


# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13,13, 1, 2)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss is True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# ---------------------------------------------------#
#   获取每个box和它的得分
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # 获得得分和box
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# ---------------------------------------------------#
#   图片预测
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    # 获得特征层的数量
    num_layers = len(yolo_outputs)
    # 特征层1对应的anchor是678
    # 特征层2对应的anchor是345
    # 特征层3对应的anchor是012
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # 对每个特征层进行处理
    for e in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[e], anchors[anchor_mask[e]], num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 非极大抑制，去掉box重合程度高的那一些
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果
        # 下列三个分别是
        # 框的位置，得分与种类
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
