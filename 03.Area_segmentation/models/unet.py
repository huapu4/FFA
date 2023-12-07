from models import Network
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class UNet(Network):
    def __init__(self, num_classes, version='UNet', base_model='VGG16', **kwargs):
        """
        The initialization of UNet.
        :param num_classes: the number of predicted classes.
        :param version: 'UNet'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        self.base_model = base_model
        self.base_model = 'VGG16' if self.base_model is None else self.base_model

        assert version == 'UNet'
        assert self.base_model in ['VGG16',
                              'VGG19',
                              'MobileNetV1',
                              'MobileNetV2',
                              'Swin_Transformer']
        super(UNet, self).__init__(num_classes, version, base_model, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        if self.base_model != 'Swin_Transformer':
            return self._unet(inputs)
        else:
            return self._STunet(inputs)

    def _conv_bn_relu(self, x, filters, kernel_size=1, strides=1):
        x = layers.Conv2D(filters, kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    def _STunet(self,inputs):
        num_classes = self.num_classes

        X = self.encoder(inputs)
        if num_classes == 2:
            X = layers.Dense(num_classes, activation='sigmoid', name='last_linear_layer')(X)
        else:
            X = layers.Dense(num_classes, activation='softmax', name='last_linear_layer')(X)
        outputs = layers.BatchNormalization()(X)

        # Model configuration
        return models.Model(inputs, outputs, name=self.version)



    def _unet(self, inputs):
        num_classes = self.num_classes

        c1, c2, c3, c4, c5 = self.encoder(inputs, output_stages=['c1', 'c2', 'c3', 'c4', 'c5'])

        x = layers.Dropout(0.5)(c5)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 512, 2, strides=1)
        x = layers.Concatenate()([x, c4])
        x = self._conv_bn_relu(x, 512, 3, strides=1)
        x = self._conv_bn_relu(x, 512, 3, strides=1)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 256, 2, strides=1)
        x = layers.Concatenate()([x, c3])
        x = self._conv_bn_relu(x, 256, 3, strides=1)
        x = self._conv_bn_relu(x, 256, 3, strides=1)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 128, 2, strides=1)
        x = layers.Concatenate()([x, c2])
        x = self._conv_bn_relu(x, 128, 3, strides=1)
        x = self._conv_bn_relu(x, 128, 3, strides=1)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = self._conv_bn_relu(x, 64, 2, strides=1)
        x = layers.Concatenate()([x, c1])
        x = self._conv_bn_relu(x, 64, 3, strides=1)
        x = self._conv_bn_relu(x, 64, 3, strides=1)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(num_classes, 1, strides=1,
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        outputs = x
        return models.Model(inputs, outputs, name=self.version)
