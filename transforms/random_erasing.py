import random
import math
import torch


class RandomErasing:
    """
    Randomly erases a rectangular region in an image during training.

    Args:
        p        : probability that an image gets erased (default: 0.5)
        sl       : minimum fraction of image area to erase (default: 0.02)
        sh       : maximum fraction of image area to erase (default: 0.4)
        r1       : minimum aspect ratio of the rectangle (default: 0.3)
        fill_mode: what to fill the erased region with. Options:
                   'random' - random pixel values (RE-R, paper default)
                   'mean'   - ImageNet mean pixel value (RE-M)
                   'zero'   - all zeros / black (RE-0)
                   'max'    - all 255s / white (RE-255)
    """

    # ImageNet mean pixel values
    IMAGENET_MEAN = [0.4914, 0.4822, 0.4465]

    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, fill_mode='random'):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = 1.0 / r1
        self.fill_mode = fill_mode

    def __call__(self, img):
        """
        Called automatically when this transform is applied to an image.
        img: a PyTorch tensor of shape (3, Height, Width)
        """

        # decide whether to erase this image at all
        if random.random() > self.p:
            return img

        # try to find a valid rectangle
        for _ in range(100):
            image_area = img.size(1) * img.size(2)
            erase_area = random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img.size(1) and w < img.size(2):
                x1 = random.randint(0, img.size(2) - w)
                y1 = random.randint(0, img.size(1) - h)

                # fill the rectangle based on chosen mode
                if self.fill_mode == 'random':
                    # Random value for each channel
                    img[0, y1:y1+h, x1:x1+w] = random.random()
                    img[1, y1:y1+h, x1:x1+w] = random.random()
                    img[2, y1:y1+h, x1:x1+w] = random.random()

                elif self.fill_mode == 'mean':
                    # Fill with ImageNet mean 
                    img[0, y1:y1+h, x1:x1+w] = self.IMAGENET_MEAN[0]
                    img[1, y1:y1+h, x1:x1+w] = self.IMAGENET_MEAN[1]
                    img[2, y1:y1+h, x1:x1+w] = self.IMAGENET_MEAN[2]

                elif self.fill_mode == 'zero':
                    # Fill with a black rectangle
                    img[:, y1:y1+h, x1:x1+w] = 0.0

                elif self.fill_mode == 'max':
                    # Fill with a white rectangle, since images are in [0,1]
                    img[:, y1:y1+h, x1:x1+w] = 1.0

                return img

        return img