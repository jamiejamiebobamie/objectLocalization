import csv
import math

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, Reshape, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import epsilon

# 0.35, 0.5, 0.75, 1.0
ALPHA = 0.35

# 96, 128, 160, 192, 224
IMAGE_SIZE = 96

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 50

MULTI_PROCESSING = False
THREADS = 1

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

CLASSES = 2

class_names = {"Car":0,"Pedestrian":1}
image_width = 1242
image_height = 375

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.coords = np.zeros((sum(1 for line in file), 4 + CLASSES))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:5]):
                    r = float(r)
                    row[i+1] = int(r)

                path, x0, y0, x1, y1, class_name = row
                self.coords[index, 0] = x0 * IMAGE_SIZE / image_width
                self.coords[index, 1] = y0 * IMAGE_SIZE / image_height
                self.coords[index, 2] = (x1 - x0) * IMAGE_SIZE / image_width
                self.coords[index, 3] = (y1 - y0) * IMAGE_SIZE / image_height
                self.coords[index, min(4 + class_names[class_name],
                                                self.coords.shape[1]-1)] = 1

                self.paths.append(path)

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3),
                                                            dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img.convert('RGB')

            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return batch_images, [batch_coords[...,:4], batch_coords[...,4:]]

class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        accuracy = 0

        intersections = 0
        unions = 0

        iou = 0

        length_of_dataset = len(self.generator)
        range_of_dataset = range(length_of_dataset)
        for i in range_of_dataset:
            batch_images, (gt, class_id) = self.generator[i]

            pred, pred_class = self.model.predict_on_batch(batch_images)
            # print(gt[0])
            # print(pred[0])
            # break

            # euclidean distance between the ground truth coords
                # and the predicted coords
            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]
            # print(pred.shape[0] == BATCH_SIZE) # True
            #
            # print()
            # print(class_id[0])
            # print(pred_class[0])
            # break
            pred_class = np.argmax(pred_class, axis=1)
            class_id = np.argmax(class_id, axis=1)
            #
            # print()
            # print(class_id[0])
            # print(pred_class[0])

            # max_value = max(my_list)
            # max_index = my_list.index(max_value)

            accuracy += np.sum(class_id == pred_class)
            # print("hey1",len(pred))

            pred = np.maximum(pred, 0)

            # print("hello1",len(gt))
            # print("hey2",len(pred))
            # print(gt, pred)
            iou_per_batch = 0
            for j in range(BATCH_SIZE):
                length_of_gt = len(gt)
                if length_of_gt == BATCH_SIZE:
                    iou_per_batch += IoU(gt[j],pred[j])
                    # print(j, iou_per_batch)
                else:
                    for k in range(length_of_gt):
                        iou_per_batch += IoU(gt[j],pred[j])
                        # print(j, iou_per_batch)
                    break
                # print("gt",gt[j])
                # print("pred",pred[j])

            iou += iou_per_batch / BATCH_SIZE
            # print("IOU", iou)
            #
            # diff_width = (np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2])
            #                                 - np.maximum(gt[:,0], pred[:,0]))
            # diff_height = (np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3])
            #                                 - np.maximum(gt[:,1], pred[:,1]))
            # intersection = (np.maximum(diff_width, 0)
            #                                     * np.maximum(diff_height, 0))
            #
            # area_gt = gt[:,2] * gt[:,3]
            # area_pred = pred[:,2] * pred[:,3]
            # union = np.maximum(area_gt + area_pred - intersection, 0)
            #
            # intersections += np.sum(intersection * (union > 0))
            # unions += np.sum(union)

        # an IoU of 1 means the ground truth box and the predicted box are
            # right on top of one another.
        iou = np.round(iou / length_of_dataset, 4)
        # iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        accuracy = np.round(accuracy / len(self.generator.coords), 4)
        logs["val_acc"] = accuracy

        print(" - val_iou: {} - val_mse: {} - val_acc: {}".format(iou,
                                                                mse, accuracy))

# from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def IoU(boxA, boxB):
    # print("boxA",len(boxA), boxA[0])
    # print("boxB",len(boxB), boxB[0])
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    # print(boxAArea, boxBArea, interArea)
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    # print(iou)
    return iou

def create_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                include_top=False, alpha=ALPHA)

    # to freeze layers
    for layer in model.layers:
        layer.trainable = trainable

    out = model.layers[-1].output

    x = Conv2D(4, kernel_size=3)(out)
    x = Reshape((4,), name="coords")(x)

    y = GlobalAveragePooling2D()(out)
    y = Dense(CLASSES, name="classes", activation="softmax")(y)

    return Model(inputs=model.input, outputs=[x, y])


def log_mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.log1p(tf.math.squared_difference(y_pred,
                                                            y_true)), axis=-1)

def focal_loss(alpha=0.9, gamma=2):
  def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return ((tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits))
                                    * (weight_a + weight_b) + logits * weight_b)

  def loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon(), 1 - epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha,
                                                    gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

  return loss

def main():
    model = create_model()

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))

    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss={"coords" : log_mse, "classes" : focal_loss()},
    loss_weights={"coords" : 1, "classes" : 1}, optimizer=optimizer, metrics=[])
    checkpoint = ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou",
                                verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max")
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=10,
                                        min_lr=epsilon(), verbose=1, mode="max")

    model.summary()

    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        callbacks=[validation_datagen, checkpoint,
                                                    reduce_lr, stop],
                        workers=THREADS,
                        use_multiprocessing=MULTI_PROCESSING,
                        shuffle=True,
                        verbose=1)


if __name__ == "__main__":
    main()
