# -*-coding:utf-8 -*-
import os
import numpy as np
import argparse
import torch
import cv2
from nets.resunet import resunet
from dataset.utils import preprocess_input
from PIL import Image, ImageDraw


def main(args):
    # 是否使用GPU训练
    cuda = True if torch.cuda.is_available() else False

    # 模型定义
    model = resunet(in_channels=3,out_channels=1,depth=4,basewidth=32,drop_rate=0)
    # model_list = ["(30)semi-supervisedResUNet.h5", 
    #            "(100)semi-supervisedResUNet.h5",
    #            "(1100)semi-supervisedResUNet.h5",
    #            "supervisedFCN.h5",
    #            "supervisedUNet.h5",
    #            "supervisedResUNet.h5"]
    # model_name = model_list[2]
    model_name = args.model_filename
    training_log_dir = "./logs"

    # 载入模型权重
    model_filename = os.path.join(training_log_dir, model_name)
    if os.path.exists(model_filename):
        print('Load weights {}.'.format(model_filename))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_filename, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        RuntimeError("model path does not exist")

    if cuda:
        model = model.cuda()

    image_dir = './datasets/dataset4/valid_images'
    image_name = '4.jpg'
    image_path = os.path.join(image_dir,image_name)
    label_dir = './datasets/dataset4/valid_labels'
    label_name = '4.npy'
    label = np.load(os.path.join(label_dir,label_name))
    origin_image = Image.open(image_path)
    image = np.transpose(preprocess_input(np.array(origin_image, np.float32)), (2, 0, 1))
    image = image[np.newaxis,...]
    with torch.no_grad():
        image = torch.from_numpy(image)    
        if cuda:
            image = image.cuda()
        output = model(image)
    output_image = np.array(output.to("cpu"))[0][0]
    threshold = 0.5
    output_image[output_image>=threshold] = 1
    output_image[output_image<threshold] = 0

    label_image = binary2rgb(label, origin_image)
    output_image = binary2rgb(output_image, origin_image)
    

    # 保存结果
    save_dir = './image_results'
    Image.fromarray(label_image).save(os.path.join(save_dir,model_name.split('.')[0]+"_dataset4_label_"+image_name.split('.')[0]+".png"))
    Image.fromarray(output_image).save(os.path.join(save_dir,model_name.split('.')[0]+"_dataset4_output_"+image_name.split('.')[0]+".png"))

def binary2rgb(binary_image, origin_image):
    if not isinstance(binary_image, np.ndarray):
        try:
            binary_image = np.array(binary_image)
        except Exception as e:
            raise ValueError("binary_image could not be converted to a numpy array: {}".format(e))
    if binary_image.ndim != 2:
        try:
            binary_image = binary_image.reshape(-1, binary_image.shape[-1])
        except Exception as e:
            raise ValueError("binary_image could not be reshaped to 2D: {}".format(e))
    
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    
    if not isinstance(origin_image, Image.Image):
        try:
            origin_image = Image.fromarray(origin_image)
        except Exception as e:
            raise ValueError("origin_image could not be converted to a PIL Image object: {}".format(e))
    
    # 创建一个与原始图像大小相同的透明图像
    overlay = Image.new('RGBA', origin_image.size, (0, 0, 0, 0))
    
    # 将二值图像转换为PIL图像，并设置透明度为30%
    mask = Image.fromarray(binary_image * 255)
    draw = ImageDraw.Draw(overlay)
    # draw.bitmap((0, 0), mask, fill=(0, 255, 0, int(255*0.3)))  # RGBA中的76大约等于0.3透明度
    
    # 使用OpenCV找到二值图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在叠加层上绘制轮廓
    for contour in contours:
        # 将轮廓坐标从OpenCV格式转换为PIL格式
        pil_contour = [(x[0], x[1]) for x in contour[:, 0]]
        draw.line(pil_contour, fill=(255, 0, 0, 255), width=2)  # 使用红色绘制轮廓
    
    # 将透明图像和原始图像合并
    result_image = Image.alpha_composite(origin_image.convert('RGBA'), overlay)
    
    return np.array(result_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filename', type=str)
    args = parser.parse_args()
    main(args)
