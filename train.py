import tensorflow as tf
import numpy as np
import os
from model import trackmodel

class trackDataset(tf.data.Dataset):
    def __init__(self, datas):
        self.length = len(datas)
        self.datas  = datas
        self.index = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # load img
        idx = self.index
        image1 = self.load_image(self.datas[idx][0])
        image2 = self.load_image(self.datas[idx+1][0])
        image3 = self.load_image(self.datas[idx+2][0])
        image4 = self.load_image(self.datas[idx+3][0])

        # load label
        label_map = self.load_label(self.datas[idx][2])
        label1 = self.load_imglabel(self.datas[idx][1],label_map)
        label2 = self.load_imglabel(self.datas[idx+1][1],label_map)
        label3 = self.load_imglabel(self.datas[idx+2][1],label_map)
        label4 = self.load_imglabel(self.datas[idx+3][1],label_map)
        return (image1, image2, image3, image4, label1, label2, label3, label4, label_map)

    def load_image(self,imgpath):
        image = tf.io.read_file(imgpath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image,(400,400))
        image = tf.expand_dims(image,axis=0)
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    def load_imglabel(self,labelpath,maptensor):
        num = len(maptensor)
        tensor_label = np.zeros((num,4),dtype=np.float32)
        with open(labelpath,'r') as f:
            lines=f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                id    = int(parts[0])
                box   = np.array([float(x) for x in parts[1:]])
                tensor_label[id][0]=box[0]
                tensor_label[id][1]=box[1]
                tensor_label[id][2]=box[2]
                tensor_label[id][3]=box[3]
        return  tf.constant(tensor_label)

    def load_label(self,labelpath):
        with open(labelpath,'r') as f:
            lines=f.readlines()
            num  = len(lines)
            tensor_label = np.zeros((num,2),dtype=np.float32)
            for line in lines:
                parts = line.strip().split(' ')
                id    = int(parts[0])
                box   = np.array([float(x) for x in parts[1:]])
                tensor_label[id][0]=box[0]
                tensor_label[id][1]=box[1]
        return  tf.constant(tensor_label)
    
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        # load img
        idx = self.index
        image1 = self.load_image(self.datas[idx][0])
        image2 = self.load_image(self.datas[idx+1][0])
        image3 = self.load_image(self.datas[idx+2][0])
        image4 = self.load_image(self.datas[idx+3][0])

        # load label
        label_map = self.load_label(self.datas[idx][2])
        label1 = self.load_imglabel(self.datas[idx][1],label_map)
        label2 = self.load_imglabel(self.datas[idx+1][1],label_map)
        label3 = self.load_imglabel(self.datas[idx+2][1],label_map)
        label4 = self.load_imglabel(self.datas[idx+3][1],label_map)
        return (image1, image2, image3, image4, label1, label2, label3, label4, label_map)

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
    data   = load_data(img_path,rectlabel_path,coordinate_path)
    dataset= trackDataset(data)
    model  = trackmodel()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn   = tf.keras.losses.MeanSquaredError()
    for epoch in range(3):
        for temp_slice in dataset:
            Map_y = temp_slice[8]
            for id in range(len(Map_y)):
                y = Map_y[id]
                image_label1 = tf.expand_dims(temp_slice[4][id],axis=0)
                image_label2 = tf.expand_dims(temp_slice[5][id],axis=0)
                image_label3 = tf.expand_dims(temp_slice[6][id],axis=0)
                image_label4 = tf.expand_dims(temp_slice[7][id],axis=0)
                image_label  = tf.expand_dims(tf.expand_dims(tf.concat([image_label1,image_label2,image_label3,image_label4],axis=0),axis=0),axis=-1)
                inputs = (temp_slice[0], temp_slice[1], temp_slice[2], temp_slice[3], image_label)
                with tf.GradientTape() as tape:
                    y_pred = model(inputs)
                    y_pred = tf.squeeze(y_pred, axis=0)
                    loss   = loss_fn(y, y_pred)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print("Epoch:", epoch,"Loss:", float(loss))

        # # 计算验证损失
        # val_losses = []
        # for temp_slice in val_dataset:
        #     Map_y = temp_slice[8]
        #     for id in range(len(Map_y)):
        #         y = Map_y[id]
        #         image_label1 = tf.expand_dims(temp_slice[4][id],axis=0)
        #         image_label2 = tf.expand_dims(temp_slice[5][id],axis=0)
        #         image_label3 = tf.expand_dims(temp_slice[6][id],axis=0)
        #         image_label4 = tf.expand_dims(temp_slice[7][id],axis=0)
        #         image_label  = tf.expand_dims(tf.expand_dims(tf.concat([image_label1,image_label2,image_label3,image_label4],axis=0),axis=0),axis=-1)
        #         inputs = (temp_slice[0], temp_slice[1], temp_slice[2], temp_slice[3], image_label)
        #         y_pred = model(inputs)
        #         y_pred = tf.squeeze(y_pred, axis=0)
        #         val_loss = loss_fn(y, y_pred)
        #         val_losses.append(val_loss)
        #     mean_val_loss = tf.reduce_mean(val_losses)
        #     print("Epoch:", epoch, "Validation Loss:", float(mean_val_loss))

# 保存模型的权重参数
model.save_weights('my_model_weights.h5')
# 保存整个模型
model.save('my_model.h5')