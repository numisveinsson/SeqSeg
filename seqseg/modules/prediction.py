import os
import numpy as np
import time

import SimpleITK as sitk

class Prediction:
    #This is a class to get 3D volumetric prediction from the UNet model
    def __init__(self, unet, model, modality, image, size, out_fn, threshold, seg_volume = None, global_scale = False):
        self.unet=unet
        self.model_name = model
        self.modality = modality
        self.image_vol = image
        self.image_resampled = resample_spacing(self.image_vol, template_size=size, order=1)[0]
        self.threshold = threshold
        self.global_scale = global_scale
        if seg_volume:
            label_vol = sitk.GetArrayFromImage(seg_volume)
            label_vol = (label_vol/np.max(label_vol))
            label_vol[label_vol >= self.threshold] = 1
            label_vol[label_vol < self.threshold] = 0
            label_vol = sitk.GetImageFromArray(label_vol.astype('int'))
            label_vol.SetOrigin(image.GetOrigin())
            label_vol.SetSpacing(image.GetSpacing())
            label_vol.SetDirection(image.GetDirection())
            self.seg_vol = label_vol
        else:
            self.seg_vol = None
        self.prob_prediction = None
        self.prediction = None
        self.dice_score = None
        self.original_shape = None

    def volume_prediction(self, num_class):

        img_vol = sitk.GetArrayFromImage(self.image_resampled).transpose(2,1,0)
        if not self.global_scale:
            img_vol = rescale_intensity(img_vol, self.modality, [750, -750])
        #print(f"Vol Max: {img_vol.max()}, Vol Min: {img_vol.min()}")
        self.original_shape = img_vol.shape
        #print(f"Min: {img_vol.min():.2f}, Max: {img_vol.max():.2f}")

        prob = np.zeros((*self.original_shape,num_class))

        start = time.time()
        prediction = self.unet.predict(np.expand_dims(np.expand_dims(img_vol, axis=-1), axis=0))
        end = time.time()
        self.pred_time = end-start

        prediction = np.squeeze(prediction, axis=-1).squeeze()

        def sigmoid(z):
            return (np.exp(z))/(1 + np.exp(z))

        pred = sigmoid(prediction)

        self.prob_prediction = pred.astype(float)

        prediction[pred >= self.threshold] = 1
        prediction[pred < self.threshold] = 0

        self.prediction = prediction.astype('int')

        return self.prediction

    def dice(self):
        #assuming groud truth label has the same origin, spacing and orientation as input image
        if self.seg_vol:
            self.dice_score = dice_score(sitk.GetArrayFromImage(self.prediction), sitk.GetArrayFromImage(self.seg_vol).astype('int'))[0]

            return self.dice_score
        else:
            return None

    def write_prediction(self, pd_fn):
        sitk.WriteImage(self.prediction, pd_fn)

    def resample_prediction(self, upsample=False):
        #resample prediction so it matches the original image
        resampled = []
        for pred in [self.prediction, self.prob_prediction]:
            im = sitk.GetImageFromArray(pred.transpose(2,1,0))
            im.SetSpacing(self.image_resampled.GetSpacing())
            im.SetOrigin(self.image_resampled.GetOrigin())
            im.SetDirection(self.image_resampled.GetDirection())

            ori_im = self.image_vol
            if upsample:
                size = ori_im.GetSize()
                spacing = ori_im.GetSpacing()
                new_size = [max(s,128) for s in size]
                ref_im = sitk.Image(new_size, ori_im.GetPixelIDValue())
                ref_im.SetOrigin(ori_im.GetOrigin())
                ref_im.SetDirection(ori_im.GetDirection())
                ref_im.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, size, spacing)])
                ctr_im = sitk.Resample(ori_im, ref_im, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear)
                resampled.append(centering(im, ctr_im, order=0))
            else:
                resampled.append(centering(im, ori_im, order=0))

        self.prediction = resampled[0]
        self.prob_prediction = resampled[1]

        return [self.prediction, self.prob_prediction]

    def evaluate_assd(self):
        from vtk import vtkDistancePolyDataFilter
        def _get_assd(p_surf, g_surf):
            dist_fltr = vtkDistancePolyDataFilter()
            dist_fltr.SetInputData(1, p_surf)
            dist_fltr.SetInputData(0, g_surf)
            dist_fltr.SignedDistanceOff()
            dist_fltr.Update()
            distance_poly = v2n(dist_fltr.GetOutput().GetPointData().GetArray(0))
            return np.mean(distance_poly), dist_fltr.GetOutput()

        from seqseg.modules.vtk_functions import exportSitk2VTK, v2n, vtk_marching_cube

        ref_im = resample_spacing(self.seg_vol, template_size=(256,256,256), order=0)[0]
        ref_im, M = exportSitk2VTK(ref_im)
        pred_im = resample_spacing(self.prediction, template_size=(256, 256, 256), order=0)[0]
        pred_im, M = exportSitk2VTK(pred_im)
        #ref_im_py = swap_labels_ori(v2n(ref_im.GetPointData().GetScalars()))
        #ref_im.GetPointData().SetScalars(numpy_to_vtk(ref_im_py))
        ref_im_py = v2n(ref_im.GetPointData().GetScalars())
        ids = np.unique(ref_im_py)
        pred_poly_l = []
        dist_poly_l = []
        ref_poly_l = []
        dist = [0.]*len(ids)
        #evaluate hausdorff
        haus = [0.]*len(ids)
        for index, i in enumerate(ids):
            if i==0:
                continue
            p_s = vtk_marching_cube(pred_im, 0, i)
            r_s = vtk_marching_cube(ref_im, 0, i)
            from seqseg.modules.vtk_functions import write_vtk_polydata
            write_vtk_polydata(p_s, '/Users/numisveinsson/Downloads/surf1.vtp')
            write_vtk_polydata(r_s, '/Users/numisveinsson/Downloads/surf2.vtp')
            #dist_ref2pred, d_ref2pred = _get_assd(p_s, r_s)
            # dist_pred2ref, d_pred2ref = _get_assd(r_s, p_s)
            # dist[index] = (dist_ref2pred+dist_pred2ref)*0.5

            #haus_p2r = directed_hausdorff(v2n(p_s.GetPoints().GetData()), v2n(r_s.GetPoints().GetData()))
            #haus_r2p = directed_hausdorff(v2n(r_s.GetPoints().GetData()), v2n(p_s.GetPoints().GetData()))
            #haus[index] = max(haus_p2r, haus_r2p)
        #     pred_poly_l.append(p_s)
        #     dist_poly_l.append(d_pred2ref)
        #     ref_poly_l.append(r_s)
        # dist_poly = appendPolyData(dist_poly_l)
        # pred_poly = appendPolyData(pred_poly_l)
        # ref_poly = appendPolyData(ref_poly_l)
        #
        # dist_r2p, _ = _get_assd(pred_poly, ref_poly)
        # dist_p2r, _ = _get_assd(ref_poly, pred_poly)
        # dist[0] = 0.5*(dist_r2p+dist_p2r)

        #haus_p2r = directed_hausdorff(v2n(pred_poly.GetPoints().GetData()), v2n(ref_poly.GetPoints().GetData()))
        #haus_r2p = directed_hausdorff(v2n(ref_poly.GetPoints().GetData()), v2n(pred_poly.GetPoints().GetData()))
        #haus[0] = max(haus_p2r, haus_r2p)

        return 0#dist

def dice_score(pred, true):
    pred = pred.astype(np.int8)
    true = true.astype(np.int8)
    num_class = np.unique(true)

    #change to one hot
    dice_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c*true_c)*2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask =( pred > 0 )+ (true > 0)
    dice_out[0] = np.sum((pred==true)[mask]) * 2. / (np.sum(pred>0) + np.sum(true>0))
    return dice_out

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
    #print(f"New size: {new_size}")
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
    # centered_transform = sitk.CompositeTransform([transform, centering_transform])

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
