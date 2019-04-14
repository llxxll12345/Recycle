import glob
import numpy as np
import cv2
import os
import pandas as pd

data = []
class_cnt = 0
class_num = {}

def name_to_lable(class_name):
    if class_name in class_num:
        return class_num[class_name]
    else:
        class_num.update({class_name: class_cnt})
        class_cnt += 1

def get_img():
    for class_path in glob.glob("../images/*"):
        class_name = class_path.split('/')[2]
        print(class_name)
        num = name_to_lable(class_name)
        for img_path in glob.glob(class_path + "/*.jpg"):
            img = cv2.imread(img_path)
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            arr = np.array(resized).flatten().tolist()
            data.append({"type":num, "data":tuple(arr)})
    
    df = pd.DataFrame(data, columns=['type', 'data'])
    print(df.head())
    test_img = np.array(df.iloc[100]['data']).reshape(224, 224, 3)
    print(test_img)
    cv2.imshow("testing", test_img)
    cv2.waitKey(0)

    df.to_hdf("dataset.h5", key="dataset")
    df.to_csv("dataset.csv")

def read_img():
    df = pd.read_hdf("model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128.h5")
    print(df.describe())
    #print(df['type'].describe())
    #dataread = df['data'].values
    #datapro = np.array([np.array(data).reshape(224, 224, 3) for data in dataread])
    #print(datapro.shape)

def main():
    #get_img()
    read_img()

if __name__ == "__main__":
    main()
