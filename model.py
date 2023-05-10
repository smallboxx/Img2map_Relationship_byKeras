import tensorflow as  tf

class Image_FeatureExtractor(tf.keras.Sequential):
    def __init__(self):
        super(Image_FeatureExtractor, self).__init__([
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(1, 1),
                                   activation='relu',
                                   padding='same'),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   strides=(2,2)),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(1,1),
                                   activation='relu',
                                   padding='same'),
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),
                                            activation='relu',
                                            padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(1,1),
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),
                                            activation='relu',
                                            padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(1,1),
                                   activation='relu',
                                   padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        ])

class trackmodel(tf.keras.Model):
    def __init__(self):
        super(trackmodel, self).__init__()
        # 400,400,3
        self.feature_extractor = Image_FeatureExtractor()
        # 12,12,64
        self.conv1  = tf.keras.layers.Conv2D(filters=32,kernel_size=(1,1),activation='relu',padding='same') 
        self.F1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')
        self.convtrans1 = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(3,3),padding='valid')

    def call(self, inputs):
        x1 = self.feature_extractor(inputs[0])
        x2 = self.feature_extractor(inputs[1])
        x3 = self.feature_extractor(inputs[2])
        x4 = self.feature_extractor(inputs[3])
        x_img = tf.concat([x1,x2,x3,x4],axis=3)
        x_img = self.conv1(x_img)
        x_imglabel = self.convtrans1(inputs[-1])
        x = tf.concat([x_img,x_imglabel],axis=3)
        x = self.F1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def train_step(self, dataset):
        Map_y = dataset[8]
        for id in range(len(Map_y)):
            y = Map_y[id]
            image_label1 = tf.expand_dims(dataset[4][id],axis=0)
            image_label2 = tf.expand_dims(dataset[5][id],axis=0)
            image_label3 = tf.expand_dims(dataset[6][id],axis=0)
            image_label4 = tf.expand_dims(dataset[7][id],axis=0)
            image_label  = tf.expand_dims(tf.expand_dims(tf.concat([image_label1,image_label2,image_label3,image_label4],axis=0),axis=0),axis=-1)
            inputs = (dataset[0], dataset[1], dataset[2], dataset[3], image_label)
            with tf.GradientTape() as tape:
                y_pred = self(inputs,training=True)
                loss = self.compiled_loss(y, y_pred)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}