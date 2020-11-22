import cv2
from torchvision import transforms

def GaussianBlur(input_img, kernel_size=int(0.1 * 32), sigma = (0.1, 2.0)):
    rnd_sigma = (sigma[0] - sigma[1]) * np.random.random_sample() + sigma[0]
    output_img = cv2.GaussianBlur(np.array(input_img), (kernel_size, kernel_size), rnd_sigma)
    return output_img

def get_transform(input_size=32, color_s=0.5, simclr_aug=False, blur=False, 
                  finetune_train=False, finetune_test=False, resnet=False, savingnature=False):
    transform_list = []
    
    if simclr_aug:
        color_jitter = transforms.ColorJitter(0.8*color_s, 0.8*color_s, 0.8*color_s, 0.2*color_s)
        transform_list += [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)]
    
    if blur:
        guassian_blur = transforms.Lambda(lambda x: GaussianBlur(x, kernel_size=int(0.1 * input_size), sigma=(0.1, 2.0)))
        transforms_list += [transforms.RandomApply([guassian_blur], p=0.5)]
        
    if finetune_train:
        transform_list += [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5)]
        
    if finetune_test:
        transform_list += [
            transforms.Resize(input_size+8),
            transforms.CenterCrop(input_size)]
    
    if resnet:
        transform_list += [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(input_size)]
    
    if savingnature:
        transform_list += [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size)]
    
    transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]  #(0.2023, 0.1994, 0.2010)
    return transforms.Compose(transform_list)