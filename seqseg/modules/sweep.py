import time
import SimpleITK as sitk
import numpy as np
from .nnunet import initialize_predictor
from .sitk_functions import copy_settings

def run_global_segmentation(dir_image, model_folder, fold, scale=1):
    """ Run global segmentation on a single image

    Parameters
    ----------
    dir_image : str
        Path to the image
    model_folder : str
        Path to the model folder
    fold : int
        Fold number
    scale : float
        Scale factor

    Returns
    -------
    pred_img : sitk.Image
        Segmentation image
    prob_prediction : sitk.Image
        Probability image
    """

    # Load image
    img = sitk.ReadImage(dir_image)

    spacing_im = img.GetSpacing()
    spacing_im = np.array(spacing_im)
    spacing = (spacing_im * scale).tolist()
    spacing = spacing[::-1]
    props = {}
    props['spacing'] = spacing

    img_np = sitk.GetArrayFromImage(img)
    img_np = img_np[None]
    img_np = img_np.astype('float32')

    predictor = initialize_predictor(model_folder, fold)

    start_time_pred = time.time()
    prediction = predictor.predict_single_npy_array(img_np,
                                                    props,
                                                    None,
                                                    None,
                                                    True)
    print(f"""Prediction time:
            {(time.time() - start_time_pred):.3f} s""")

    # Probability prediction
    prob_prediction = sitk.GetImageFromArray(prediction[1][1])
    prob_prediction = copy_settings(prob_prediction, img)

    # Create segmentation prediction (binary)
    predicted_vessel = prediction[0]
    pred_img = sitk.GetImageFromArray(predicted_vessel)
    pred_img = copy_settings(pred_img, img)

    return pred_img, prob_prediction
