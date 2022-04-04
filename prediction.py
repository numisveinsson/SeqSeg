import os
import numpy as np
import time

import SimpleITK as sitk

class Prediction:
    #This is a class to get 3D volumetric prediction from the UNet model
    def __init__(self, unet, model, modality,image_fn,label_fn, size, out_fn):
        self.unet=unet
        self.model_name=model
        self.modality=modality
        self.image_fn = image_fn
        self.image_vol = sitk.ReadImage(image_fn)
        self.image_vol = resample_spacing(self.image_vol, template_size=size, order=1)[0]
        self.prediction = None
        self.dice_score = None
        self.original_shape = None
        try:
            os.makedirs(os.path.dirname(self.out_fn))
        except:
            pass

    def volume_prediction(self, num_class, threshold):

        img_vol = sitk.GetArrayFromImage(self.image_vol).transpose(2,1,0)
        img_vol = rescale_intensity(img_vol, self.modality, [750, -750])
        self.original_shape = img_vol.shape

        prob = np.zeros((*self.original_shape,num_class))

        start = time.time()
        prediction = self.unet.predict(np.expand_dims(np.expand_dims(img_vol, axis=-1), axis=0))
        end = time.time()
        self.pred_time = end-start

        prediction = np.squeeze(prediction, axis=-1).squeeze()

        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        pred = sigmoid(prediction)

        self.prob_prediction = pred.astype(float)

        prediction[pred >= threshold] = 1
        prediction[pred < threshold] = 0

        self.prediction = prediction.astype(int)

        return self.prediction

    def dice(self):
        #assuming groud truth label has the same origin, spacing and orientation as input image
        label_vol = sitk.GetArrayFromImage(sitk.ReadImage(self.label_fn))
        self.dice_score = dice_score(swapLabelsBack(label_vol, sitk.GetArrayFromImage(self.pred_label)), label_vol)
        return self.dice_score

    def evaluate_dice(self, num_class):
        reference_segmentation = sitk.Cast(sitk.ReadImage(self.label_fn), sitk.sitkUInt16)
        self.pred_label = sitk.Cast(self.pred_label, sitk.sitkUInt8)
        ref_py = sitk.GetArrayFromImage(reference_segmentation)
        ref_py = swap_labels_ori(ref_py)
        pred_py = sitk.GetArrayFromImage(self.pred_label)
        dice_values = dice_score(pred_py, ref_py)
        return dice_values

    def resample_prediction(self, upsample=False):
        #resample prediction so it matches the original image
        im = sitk.GetImageFromArray(self.prediction.transpose(2,1,0))
        im.SetSpacing(self.image_vol.GetSpacing())
        im.SetOrigin(self.image_vol.GetOrigin())
        im.SetDirection(self.image_vol.GetDirection())

        ori_im = sitk.ReadImage(self.image_fn)
        if upsample:
            size = ori_im.GetSize()
            spacing = ori_im.GetSpacing()
            new_size = [max(s,128) for s in size]
            ref_im = sitk.Image(new_size, ori_im.GetPixelIDValue())
            ref_im.SetOrigin(ori_im.GetOrigin())
            ref_im.SetDirection(ori_im.GetDirection())
            ref_im.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, size, spacing)])
            ctr_im = sitk.Resample(ori_im, ref_im, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear)
            self.pred_label = centering(im, ctr_im, order=0)
        else:
            self.pred_label = centering(im, ori_im, order=0)
        del self.prediction
        #del self.image_vol
        return self.pred_label

def resample_spacing(sitkIm, resolution=0.5, dim=3, template_size=(256, 256, 256), order=1):
    if type(sitkIm) is str:
      image = sitk.ReadImage(sitkIm)
    else:
      image = sitkIm
    orig_direction = image.GetDirection()
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size*(orig_spacing/np.array(resolution))
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    new_size = np.abs(np.matmul(np.reshape(orig_direction, (3,3)), np.array(new_size)))
    ref_img = reference_image_build(resolution, new_size, template_size, dim)
    centered = centering(image, ref_img, order)
    transformed = isometric_transform(centered, ref_img, orig_direction, order)
    return transformed, ref_img

def rescale_intensity(slice_im,m,limit):
    if type(slice_im) != np.ndarray:
        raise RuntimeError("Input image is not numpy array")
    #slice_im: numpy array
    #m: modality, ct or mr
    if m =="ct":
        rng = abs(limit[0]-limit[1])
        threshold = rng/2
        slice_im[slice_im>limit[0]] = limit[0]
        slice_im[slice_im<limit[1]] = limit[1]
        #(slice_im-threshold-np.min(slice_im))/threshold
        slice_im = slice_im/threshold
    elif m=="mr":
        #slice_im[slice_im>limit[0]*2] = limit[0]*2
        #rng = np.max(slice_im) - np.min(slice_im)
        pls = np.unique(slice_im)
        #upper = np.percentile(pls, 99)
        #lower = np.percentile(pls, 10)
        upper = np.percentile(slice_im, 99)
        lower = np.percentile(slice_im, 20)
        slice_im[slice_im>upper] = upper
        slice_im[slice_im<lower] = lower
        slice_im -= int(lower)
        rng = upper - lower
        slice_im = slice_im/rng*2
        slice_im -= 1
    return slice_im

def reference_image_build(spacing, size, template_size, dim):
    #template size: image(array) dimension to resize to: a list of three elements
    reference_size = template_size
    reference_spacing = np.array(size)/np.array(template_size)*np.array(spacing)
    reference_spacing = np.mean(reference_spacing)*np.ones(3)
    #reference_size = size
    reference_image = sitk.Image(reference_size, 0)
    reference_image.SetOrigin(np.zeros(3))
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(np.eye(3).ravel())
    return reference_image

def centering(img, ref_img, order=1):
    dimension = img.GetDimension()
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - ref_img.GetOrigin())
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    reference_center = np.array(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    return transform_func(img, ref_img, centered_transform, order)

def isometric_transform(image, ref_img, orig_direction, order=1, target=None):
    # transform image volume to orientation of eye(dim)
    dim = ref_img.GetDimension()
    affine = sitk.AffineTransform(dim)
    if target is None:
      target = np.eye(dim)

    ori = np.reshape(orig_direction, np.eye(dim).shape)
    target = np.reshape(target, np.eye(dim).shape)
    affine.SetMatrix(np.matmul(target,np.linalg.inv(ori)).ravel())
    affine.SetCenter(ref_img.TransformContinuousIndexToPhysicalPoint(np.array(ref_img.GetSize())/2.0))
    #affine.SetMatrix(image.GetDirection())
    return transform_func(image, ref_img, affine, order)

def transform_func(image, reference_image, transform, order=1):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    if order ==1:
      interpolator = sitk.sitkLinear
    elif order == 0:
      interpolator = sitk.sitkNearestNeighbor
    elif order ==3:
      interpolator = sitk.sitkBSpline
    default_value = 0
    try:
      resampled = sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
    except Exception as e: print(e)

    return resampled
