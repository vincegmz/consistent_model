import os
import torch
from PIL import Image

num_imgs_per_digit = 4

root = '/media/minzhe/ckpt/dataset/mnistm'
data_file = os.path.join(root,"MNISTM",'processed',"mnist_m_train.pt")
output_dir = '/home/minzhe_guo/Downloads/datasets/color_digits' 
count_dict = {digit:0 for digit in range(0,10)}
data, targets = torch.load(data_file)
index = 0
number_to_name = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}
while count_dict != {}:
    num = targets[index].item()
    # if self.train:
    #     if num == 0:
    #         index+=1
    #         continue
    if num not in count_dict:
        index+=1
        continue
    img = Image.fromarray(data[index].squeeze().numpy(), mode="RGB")
    os.makedirs(os.path.join(output_dir,number_to_name[num]),exist_ok=True)
    img.save(os.path.join(output_dir,number_to_name[num],f'{number_to_name[num]}_{count_dict[num]}.png'))
    count_dict[num]+=1
    index+=1
    if count_dict[num] == num_imgs_per_digit:
        count_dict.pop(num)