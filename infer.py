from model import embedding
import torch
from torchvision import transforms
import cv2
from PIL import Image
import os
import argparse
from glob import glob

class Inference():
    def __init__(self, check_point="/mnt/28857F714F734EE8/quan_tran/palmline/pytorch-siamese-triplet/results/Custom_v2/best_none.pth", device = "cuda") -> None:
        self.means = (0.485, 0.456, 0.406)
        self.stds = (0.229, 0.224, 0.225)
        self.device = device
        self.transform = transforms.Compose([
                   transforms.ToPILImage(),
                   transforms.ToTensor(),
                   transforms.Normalize(self.means, self.stds)
               ])
        self.model = self.load_model(check_point)

    def load_model(self, path):
        model = embedding.EmbeddingResnet()
        print("=> Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict({k.replace("module.embeddingNet.",""):v for k,v in checkpoint['state_dict'].items()})
        print("=> Loaded checkpoint '{}'".format(path))
        return model.to(self.device)

    def transform_image(self, path_images, size=(228, 228)):
        list_image = []
        for path_image in path_images:
            # img = Image.open(path_image)
            # img = img.resize(size)
            img = cv2.imread(path_image)
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            # print("img", img.shape)
            list_image.append(img)
        # print(len(list_image))
        return torch.stack(list_image, dim=0).to(self.device)
    
    def infer(self,list_path_images):
        image = self.transform_image(list_path_images)
        print("image", image.shape)
        embedding = self.model(image)
        return embedding

def main():
    parser = argparse.ArgumentParser(description='Infer model PyTorch Siamese Example')
    parser.add_argument('--path_image', type=str,
                        help='Directory of image')
    parser.add_argument('--path_checkpoint', type=str, default="/mnt/28857F714F734EE8/quan_tran/palmline/pytorch-siamese-triplet/results/Custom_1024_mg4/best_quarter.pth",
                        help='path to checkpoint')
    args = parser.parse_args()
                  
    prefix_image = ["jpg", "png", "bmp", "JPG", "PNG", "jpeg"]
    if os.path.isdir(args.path_image):
        list_path_file = glob(args.path_image+"/*")
        list_images = [i for i in list_path_file if i.split(".")[-1] in prefix_image]
    elif os.path.isfile(args.path_image):
        if args.path_image.split(".")[-1] in prefix_image:
            list_images = [args.path_image]
        else:
            raise TypeError(f"Only {prefix_image} is allowed")
    else:
        raise TypeError(f"path image had to directory of image or path to image, recheck it!")

    Infer = Inference(args.path_checkpoint)
    # print("list_images", list_images)
    embedding = Infer.infer(list_images)
    resut_distance = torch.cdist(embedding.cpu(), embedding.cpu(), 2)
    embedding = embedding.cpu().detach().tolist()
    print(resut_distance)


if __name__ == '__main__':
    main()
