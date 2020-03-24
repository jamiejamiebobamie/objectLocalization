import glob
import cv2
import numpy as np

from train import create_model, IMAGE_SIZE

WEIGHTS_FILE = "model-0.03.h5"
test_IMAGES = "/Users/jamesmccrory/faster_r-cnn/my_implementation/kitti-object-detection/kitti_single/testing/image_2/*"
train_IMAGES = "/Users/jamesmccrory/faster_r-cnn/my_implementation/kitti-object-detection/kitti_single/training/image_2/*"
sample_image = "/Users/jamesmccrory/faster_r-cnn/my_implementation/kitti-object-detection/kitti_single/training/image_2/000000.png"
image_width = 1242
image_height = 375
IMAGE_SIZE = 96

write_to_filename = "/Users/jamesmccrory/objectLocalization/example.png"

class_names_LOOKUP = {0:"Car",1:"Pedestrian"}

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    sorted_files_iterator = sorted(glob.glob(train_IMAGES))

    for filename in sorted_files_iterator:
        unscaled = cv2.imread(filename)

        image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))

        region, class_id = model.predict(x=np.array([image]))
        region = region[0]

        x0 = int(region[0] * image_width / IMAGE_SIZE)
        y0 = int(region[1]  * image_height / IMAGE_SIZE)

        x1 = int((region[0] + region[2]) * image_width / IMAGE_SIZE)
        y1 = int((region[1] + region[3]) * image_height / IMAGE_SIZE)

        class_id = np.argmax(class_id, axis=1)

        label = class_names_LOOKUP[class_id[0]]

        cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(unscaled, "class: {}".format(label), (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("image", unscaled)

        cv2.waitKey(1000)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
