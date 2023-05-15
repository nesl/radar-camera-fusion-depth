import torch
import torchvision.transforms.functional as functional


class Transforms(object):

    def __init__(self,
                 normalized_image_range=[0, 255],
                 crop_image_to_shape_on_point=[-1, -1],
                 padding_mode='edge',
                 random_brightness=[-1, -1],
                 random_noise_type='none',
                 random_noise_spread=-1,
                 random_flip_type=['none'],):
        '''
        Transforms and augmentation class
        Note: brightness, contrast, gamma, hue, saturation augmentations expect
        either type int in [0, 255] or float in [0, 1]

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            pad_to_shape : list[int]
                pad image and label to shape
            random_brightness : list[float]
                brightness adjustment [0, B], from 0 (black image) to B factor increase
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
            random_flip_type : list[str]
                none, horizontal, vertical
        '''

        # Image normalization
        self.normalized_image_range = normalized_image_range

        self.padding_mode = padding_mode

        self.do_crop_image_to_shape_on_point = \
            True if -1 not in crop_image_to_shape_on_point else False

        self.crop_image_to_shape_on_point_height = crop_image_to_shape_on_point[0]
        self.crop_image_to_shape_on_point_width = crop_image_to_shape_on_point[1]

        # Photometric augmentations
        self.do_random_brightness = True if -1 not in random_brightness else False
        self.random_brightness = random_brightness

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > -1) else False

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

        # Geometric augmentations
        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

    def transform(self, points_arr, images_arr, labels_arr=[], random_transform_probability=0.50):
        '''
        Applies transform to images and ground truth

        Arg(s):
            points_arry : list[torch.Tensor]
                list of N x 3 tensors
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            labels_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
            list[torch.Tensor] : list of transformed N x c x H x W label tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            torch.rand(n_batch, device=device) <= random_transform_probability

        if self.do_crop_image_to_shape_on_point:

            for i, (points, images, labels) in enumerate(zip(points_arr, images_arr, labels_arr)):

                pad_width = self.crop_image_to_shape_on_point_width // 2

                points[:, 0] = points[:, 0] + pad_width * torch.ones_like(points[:, 0])

                images = functional.pad(
                    images,
                    (pad_width, 0, pad_width, 0),
                    padding_mode=self.padding_mode,
                    fill=0)

                labels = functional.pad(
                    labels,
                    (pad_width, 0, pad_width, 0),
                    padding_mode='constant',
                    fill=2)

                image_crops = []
                label_crops = []

                for point, image, label in zip(points, images, labels):

                    height = image.shape[-2]
                    crop_height = height - self.crop_image_to_shape_on_point_height

                    image_crop = image[:, crop_height:, int(point[0])-pad_width:int(point[0])+pad_width]
                    label_crop = label[:, crop_height:, int(point[0])-pad_width:int(point[0])+pad_width]
                    image_crops.append(image_crop)
                    label_crops.append(label_crop)

                points[:, 0] = pad_width * torch.ones_like(points[:, 0])
                image_crops = torch.stack(image_crops, dim=0)
                label_crops = torch.stack(label_crops, dim=0)

                points_arr[i] = points
                images_arr[i] = image_crops
                labels_arr[i] = label_crops

        # Photometric transformations are only applied to images
        if self.do_random_brightness:

            do_brightness = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) >= 0.50)

            values = torch.rand(n_batch, device=device)

            brightness_min, brightness_max = self.random_brightness
            factors = (brightness_max - brightness_min) * values + brightness_min

            images_arr = self.adjust_brightness(images_arr, do_brightness, factors)

        # Convert all images to float
        images_arr = [
            images.float() for images in images_arr
        ]

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        if self.do_random_noise:

            do_add_noise = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.add_noise(
                images_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        # Geometric transformations are applied to both images and labels (ground truths)
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

        # Return the transformed inputs
        if len(labels_arr) == 0:
            return points_arr, images_arr
        else:
            return points_arr, images_arr, labels_arr

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr : list of normalized N x C x H x W tensors
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

    '''
    Geometric transforms
    '''
    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = functional.hflip(image)

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = functional.vflip(image)

            images_arr[i] = images

        return images_arr
