import torch
import torchvision.transforms.functional as functional


class Transforms(object):

    def __init__(self,
                 normalized_image_range=[0, 255],
                 random_brightness=[-1],
                 random_contrast=[-1],
                 random_saturation=[-1],
                 random_noise_type='none',
                 random_noise_spread=-1,
                 random_flip_type=['none']):
        '''
        Transforms and augmentation class
        Note: brightness, contrast, gamma, hue, saturation augmentations expect
        either type int in [0, 255] or float in [0, 1]

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            random_brightness : list[float]
                brightness adjustment [0, B], from 0 (black image) to B factor increase
            random_contrast : list[float]
                contrast adjustment [0, C], from 0 (gray image) to C factor increase
            random_saturation : list[float]
                saturation adjustment [0, S], from 0 (black image) to S factor increase
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
            random_flip_type : list[str]
                none, horizontal, vertical
        '''

        # Image normalization
        self.normalized_image_range = normalized_image_range

        # Photometric augmentations
        self.do_random_brightness = True if -1 not in random_brightness else False
        self.random_brightness = random_brightness
        self.do_random_contrast = True if -1 not in random_contrast else False
        self.random_contrast = random_contrast
        self.do_random_saturation = True if -1 not in random_saturation else False
        self.random_saturation = random_saturation

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > -1) else False

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

        # Geometric augmentations
        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

    def transform(self,
                  images_arr,
                  labels_arr=[],
                  points_arr=[],
                  bounding_boxes_arr=[],
                  random_transform_probability=0.00):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            labels_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            points_arr : list[torch.Tensor]
                list of N x 3 tensors
            bounding_boxes_arr : list[torch.Tensor]
                list of N x 4 tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
            list[torch.Tensor] : list of transformed N x c x H x W label tensors
            list[torch.Tensor] : list of transformed N x 3 points tensors
            list[torch.Tensor] : list of transformed N x 4 bounding box tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            torch.rand(n_batch, device=device) <= random_transform_probability

        '''
        Photometric Transformations (applied only to images)
        '''
        for idx, images in enumerate(images_arr):
            # In case user pass in [0, 255] range image as float type
            if torch.max(images) > 1.0:
                images_arr[idx] = images.int()

        if self.do_random_brightness:

            do_brightness = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            brightness_min, brightness_max = self.random_brightness
            factors = (brightness_max - brightness_min) * values + brightness_min

            images_arr = self.adjust_brightness(images_arr, do_brightness, factors)

        if self.do_random_contrast:

            do_contrast = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            contrast_min, contrast_max = self.random_contrast
            factors = (contrast_max - contrast_min) * values + contrast_min

            images_arr = self.adjust_contrast(images_arr, do_contrast, factors)

        if self.do_random_saturation:

            do_saturation = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            saturation_min, saturation_max = self.random_saturation
            factors = (saturation_max - saturation_min) * values + saturation_min

            images_arr = self.adjust_saturation(images_arr, do_saturation, factors)

        '''
        Convert all images to float and normalize
        '''
        images_arr = [
            images.float() for images in images_arr
        ]

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        '''
        Points augmentation
        '''
        if self.do_random_noise:

            do_add_noise = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            points_arr = self.add_noise(
                points_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        '''
        Geometric transformations (applied to both images and labels)
        '''
        if self.do_random_horizontal_flip:

            do_horizontal_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            labels_arr = self.horizontal_flip(
                labels_arr,
                do_horizontal_flip)



            for bounding_boxes in bounding_boxes_arr:
                for bounding_box_idx in range(0,bounding_boxes.shape[0]):
                    do_hflip = do_horizontal_flip[bounding_box_idx]
                    for box_idx in range(0,bounding_boxes.shape[1]):
                        if do_hflip:
                            temp = bounding_boxes[bounding_box_idx, box_idx, 0].clone()
                            bounding_boxes[bounding_box_idx, box_idx, 0] = n_width - bounding_boxes[bounding_box_idx, box_idx, 2]
                            bounding_boxes[bounding_box_idx, box_idx, 2] = n_width - temp


        if self.do_random_vertical_flip:

            do_vertical_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            labels_arr = self.vertical_flip(
                labels_arr,
                do_vertical_flip)

            for bounding_boxes in bounding_boxes_arr:
                for bounding_box_idx in range(0,bounding_boxes.shape[0]):
                    do_vflip = do_vertical_flip[bounding_box_idx]
                    if do_vflip:
                        temp = bounding_boxes[bounding_box_idx, 1].clone()
                        bounding_boxes[bounding_box_idx, 1] = n_height - bounding_boxes[bounding_box_idx, 3]
                        bounding_boxes[bounding_box_idx, 3] = n_height - temp

        # Return the transformed inputs
        outputs = []

        if len(images_arr) > 0:
            outputs.append(images_arr)

        if len(labels_arr) > 0:
            outputs.append(labels_arr)

        if len(points_arr) > 0:
            outputs.append(points_arr)

        if len(bounding_boxes_arr) > 0:
            outputs.append(bounding_boxes_arr)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]
        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    def adjust_brightness(self, images_arr, do_brightness, factors):
        '''
        Adjust brightness on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_brightness : bool
                N booleans to determine if brightness is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_brightness[b]:
                    images[b, ...] = functional.adjust_brightness(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_contrast(self, images_arr, do_contrast, factors):
        '''
        Adjust contrast on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_contrast : bool
                N booleans to determine if contrast is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_contrast[b]:
                    images[b, ...] = functional.adjust_contrast(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_saturation(self, images_arr, do_saturation, factors):
        '''
        Adjust saturation on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_saturation : bool
                N booleans to determine if saturation is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_saturation[b]:
                    images[b, ...] = functional.adjust_saturation(image, factors[b])

            images_arr[i] = images

        return images_arr

    '''
    Geometric transforms
    '''
    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-1])

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-2])

            images_arr[i] = images

        return images_arr

    def add_noise(self, images_arr, do_add_noise, noise_type, noise_spread):
        '''
        Add noise to images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_add_noise : bool
                N booleans to determine if noise will be added
            noise_type : str
                gaussian, uniform
            noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        for i, images in enumerate(images_arr):
            device = images.device

            for b, image in enumerate(images):
                if do_add_noise[b]:

                    shape = image.shape

                    if noise_type == 'gaussian':
                        image = image + noise_spread * torch.randn(*shape, device=device)
                    elif noise_type == 'uniform':
                        image = image + noise_spread * (torch.rand(*shape, device=device) - 0.5)
                    else:
                        raise ValueError('Unsupported noise type: {}'.format(noise_type))

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr
