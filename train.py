import tensorflow as tf
import numpy as np
import os

class CustomDataset(tf.data.Dataset):
    def __init__(self, datas):
        self.length = len(datas)
        self.datas  = datas

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #img_path,rectlabel_path,coordinate_path=datas[idx]
        # load img
        image1 = self.load_image(self.datas[idx][0])
        image2 = self.load_image(self.datas[idx+1][0])
        image3 = self.load_image(self.datas[idx+2][0])
        image4 = self.load_image(self.datas[idx+3][0])

        # load label
        label1 = self.load_label(self.datas[idx][1])
        label2 = self.load_label(self.datas[idx+1][1])
        label3 = self.load_label(self.datas[idx+2][1])
        label4 = self.load_label(self.datas[idx+3][1])
        label_map = self.load_label(self.datas[idx][2])
        # 将四张图像和标签作为元组返回
        return (image1, image2, image3, image4, label1, label2, label3, label4, label_map)

    def load_image(self,imgpath):
        image = tf.io.read_file(imgpath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image,(225,400))
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    def load_label(self,labelpath):
        labels= []
        with open(labelpath,'r') as f:
            lines=f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                id    = parts[0]
                box   = np.array([float(x) for x in parts[1:]])
                labels.append((id,box))
        return  np.array(labels, dtype=object)
    
    def __iter__(self):
        # 迭代数据集
        for i in range(self.length//4):
            yield self[i*4]
    def _inputs(self):
        # 返回数据集的输入张量，这里返回一个元组，包含四张图像和标签
        return (tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32))

    def element_spec(self):
        return (tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32))
    
def load_data(img_dir, rectlabel_path, coordinate_path):
    # 从TXT文件中读取标签数据
    files = os.listdir(rectlabel_path)
    files.sort()
    datas = []
    for f in files:
        name = os.path.splitext(f)[0]
        imgs_path = os.path.join(img_dir, name + ".jpg")
        rect_path = os.path.join(rectlabel_path, name + ".txt")
        coor_path = os.path.join(coordinate_path, name.split("_")[0] + ".txt")
        datas.append([imgs_path, rect_path, coor_path])
    return datas

if __name__=="__main__":
    img_path="data\\imgs"
    rectlabel_path="data\\annotations"
    coordinate_path="data\\aerlabel"

    datas  = load_data(img_path,rectlabel_path,coordinate_path)
    dataset= CustomDataset(datas)
    # Define input layers
    input1 = tf.keras.layers.Input(shape=(225, 400, 3))
    input2 = tf.keras.layers.Input(shape=(225, 400, 3))
    input3 = tf.keras.layers.Input(shape=(225, 400, 3))
    input4 = tf.keras.layers.Input(shape=(225, 400, 3))
    input5 = tf.keras.layers.Input(shape=(None,))
    input6 = tf.keras.layers.Input(shape=(None,))
    input7 = tf.keras.layers.Input(shape=(None,))
    input8 = tf.keras.layers.Input(shape=(None,))

    # Define convolutional layers
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

    # Define flatten and dense layers
    flatten = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(128, activation='relu')
    dense2 = tf.keras.layers.Dense(3, activation=None)

    # Process input1
    x1 = conv2(conv1(input1))
    x1 = flatten(x1)
    x1 = dense1(x1)

    # Process input2
    x2 = conv2(conv1(input2))
    x2 = flatten(x2)
    x2 = dense1(x2)

    # Process input3
    x3 = conv2(conv1(input3))
    x3 = flatten(x3)
    x3 = dense1(x3)

    # Process input4
    x4 = conv2(conv1(input4))
    x4 = flatten(x4)
    x4 = dense1(x4)

    # Concatenate the four feature vectors
    x = tf.keras.layers.concatenate([x1, x2, x3, x4])

    # Predict the label_map
    label_map = dense2(x)

    # Define the model inputs and outputs
    model_inputs = [input1, input2, input3, input4, input5, input6, input7, input8]
    model_outputs = [label_map]

    # Define the model
    model = tf.keras.models.Model(inputs=model_inputs, outputs=model_outputs)

    # Define the loss function and optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Train the model
    model.fit(dataset, epochs=1)

    # Save the weights
    model.save_weights('my_model_weights.h5')