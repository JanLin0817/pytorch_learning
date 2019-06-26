import torchvision

def transform_PIL():
    '''
    some test about pytorch transforms
    pytorch only accept PIL image ?
    '''

    # load image by PIL
    img = Image.open('cityscapes2.png')
    print('src image size: {}'.format(img.size))

    # Scale: 依比例將短編縮至 scale
    resize = transforms.Resize(480)
    resizeed_img=resize(img)
    print('Scale image size: {}'.format(resizeed_img.size))
    #resizeed_img.show()

    # CenterCrop
    center_crop = transforms.CenterCrop(100)
    center_crop_img = center_crop(img)

    # RandomCrop
    Random_crop = transforms.RandomCrop(100)
    Random_crop_img = Random_crop(img)

    # RandomHorizontalFlip(flip properpility)
    flip = transforms.RandomHorizontalFlip(1)
    flip_img = flip(img)

    # RandomSizedCrop(size): random szie and position crop and resize to size
    ransize_crop = transforms.RandomResizedCrop(100)
    ransize_crop_img = ransize_crop(img)

    # Pad(padding, fill=0)
    padding = transforms.Pad(padding=10, fill=0)
    padding_img = padding(img)

def transform_Tensor():
    img = Image.open('cityscapes2.png')
    
    # PIL or numpy.ndarray to torch.FloatTensor
    tensor = transforms.ToTensor()
    tensor_img = tensor(img)

    # Normalize(mean, std)
    Normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    Normalize_img = Normalize(tensor_img)
    print(type(Normalize_img))
    
    # Compose
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    output = compose(img)
    print(type(output))

if __name__ == "__main__":
    # PILimage indexing through getpixel putpixel
    pass


