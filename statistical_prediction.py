# -*-coding:utf-8 -*-
import os
import numpy as np
import torch
import cv2
import argparse
import pandas as pd
from nets.resunet import resunet
from nets.unet import unet
from nets.fcn import fcn
import matplotlib.pyplot as plt
from dataset.utils import preprocess_input
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties

def main(args):
    cuda = True if torch.cuda.is_available() else False
    model = resunet(in_channels=3,out_channels=1,depth=4,basewidth=32,drop_rate=0,)
    # model = fcn(in_channels=3,out_channels=1,depth=4,basewidth=32,drop_rate=0)
    # model = unet(in_channels=3,out_channels=1,depth=4,basewidth=32,drop_rate=0)

    # print(model)

    # model_filename
    # model_list = ["(30)semi-supervisedResUNet.h5", 
    #            "(100)semi-supervisedResUNet.h5",
    #            "(1100)semi-supervisedResUNet.h5",
    #            "supervisedFCN.h5",
    #            "supervisedUNet.h5",
    #            "supervisedResUNet.h5"]
    # model_filename = model_list[4]
    model_filename = args.model_filename
    training_log_dir = "./logs/"
  
    # load model weights
    model_filename = os.path.join(training_log_dir, model_filename)
    if os.path.exists(model_filename):
        print('Load weights {}.'.format(model_filename))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_filename, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise FileNotFoundError("model path does not exist")

    if cuda:
        model = model.cuda()

    data_path_list = ["./datasets/dataset1","./datasets/dataset2","./datasets/dataset3","./datasets/dataset4"]
    mean_acc_list = []
    mean_precision_list = []
    mean_recall_list = []
    mean_F1_list = []
    for data_path in data_path_list:
        accuracy_list = []
        recall_list = []
        precision_list = []
        F1_list = []
        images_list = []
        labels_list = []
        for i in range(0,15):
            image_path = os.path.join(os.path.join(data_path, "valid_images"), str(i) + '.jpg')
            label_path = os.path.join(os.path.join(data_path, 'valid_labels'), str(i) + '.npy')
            image, label = image_load(image_path, label_path)
            output_image = predict_single_image(model, image, cuda)
            images_list.append(output_image)
            labels_list.append(label)

            groundtruth_flat = label.flatten()
            prediction_flat = output_image.flatten()

            # calculate metrics
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                groundtruth_flat, prediction_flat, average='binary'
            )
            accuracy = accuracy_score(groundtruth_flat, prediction_flat)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            F1_list.append(f1_score)

        mean_acc_list.append(sum(accuracy_list)/len(accuracy_list))
        mean_precision_list.append(sum(precision_list)/len(precision_list))
        mean_recall_list.append(sum(recall_list)/len(recall_list))
        mean_F1_list.append(sum(F1_list)/len(F1_list))

        # roc curve
        fpr, tpr, thresholds = roc_curve(np.array(labels_list).flatten(), np.array(images_list).flatten())
        # auc
        roc_auc = round(auc(fpr, tpr),4)
        # draw roc curve
        plt.plot(fpr, tpr, label=f'{data_path.split("/")[-1]} (AUC = {roc_auc})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16, fontname='Times New Roman')
    plt.ylabel('True Positive Rate', fontsize=16, fontname='Times New Roman')
    # legend settings
    font = FontProperties(family='Times New Roman', style='normal', size=16)
    plt.legend(prop=font)
    plt.legend(loc='lower right')
    save_path = './static_results/' + model_filename.split("/")[-1].split(".")[0] + '_ROC.png'
    plt.savefig(save_path)

    df = pd.DataFrame({
        "data_path": data_path_list,
        "mean_acc": mean_acc_list,
        "mean_precision": mean_precision_list,
        "mean_recall": mean_recall_list,
        "mean_F1": mean_F1_list
    })

    csv_file_path = f'./static_results/{model_filename.split("/")[-1].split(".")[0]}_static_result.csv'
    df.to_csv(csv_file_path, index=False)  


def image_load(image_path, label_path):
    image = np.array(cv2.imread(image_path),dtype=np.float32)
    label = np.array(np.load(label_path),np.float32)
    image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
    image = image[np.newaxis,...]
    return image, label

def predict_single_image(model, image, cuda=False):
    with torch.no_grad():
        image = torch.from_numpy(image)    
        if cuda:
            image = image.cuda()
        output = model(image)
        # sigmoid = torch.nn.Softmax(dim=1)
        # output = sigmoid(output)
    output_image = np.array(output.to("cpu"))[0][0]
    threshold = 0.5
    output_image[output_image>=threshold] = 1
    output_image[output_image<threshold] = 0
    return output_image

def pre(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / (predicted_positives + 1e-7)

def re(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    possible_positives = np.sum(y_true == 1)
    return true_positives / (possible_positives + 1e-7)

def f1(y_true, y_pred):
    p = pre(y_true, y_pred)
    r = re(y_true, y_pred)
    return 2 * ((p * r) / (p + r + 1e-7))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filename', type=str)
    args = parser.parse_args()
    main(args)
