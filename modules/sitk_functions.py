## Functions to bind SITK functionality

import SimpleITK as sitk
import numpy as np

def read_image(file_dir_image):
    """
    Read image from file
    Args:
        file_dir_image: image directory
    Returns:
        SITK image reader
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_dir_image)
    file_reader.ReadImageInformation()
    return file_reader

def create_new(file_reader, PixelID = None):
    """
    Create new SITK image with same formating as another
    Args:
        file_reader: reader from another image, or image object
    Returns:
        SITK image
    """
    if PixelID: pix_id = PixelID
    else: pix_id = file_reader.GetPixelID()

    # check if reader or image object
    if type(file_reader) == sitk.ImageFileReader:
        result_img = sitk.Image(file_reader.GetSize(), pix_id,
                            file_reader.GetNumberOfComponents())
    else:
        result_img = sitk.Image(file_reader.GetSize(), pix_id)
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())
    
    return result_img

def copy_settings(img, ref_img):

    img.SetSpacing(ref_img.GetSpacing())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())

    return img

def write_image(image, outputImageFileName):
    """
    Write image to file
    Args:
        SITK image, filename
    Returns:
        image file
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)
    return file_reader

def keep_component_seeds(image, seeds):
    """
    Keep only the component containing the seed point
    Args:
        SITK image, seed point pointing to point in vessel of interest
        seed(s): location of seeds; list of np arrays (one or multiple points)
    """
    # create list of list of index values for each seed
    index_seeds = []
    for i in range(len(seeds)):
        index_seeds.append(image.TransformPhysicalPointToIndex(seeds[i].tolist()))
    print(f"Index of seeds: {index_seeds}")

    # create a new image with only the labels of interest
    labelImage = remove_other_vessels(image, index_seeds)

    return labelImage

def remove_other_vessels(image, seed):
    """
    Remove all labelled vessels except the one of interest
    Args:
        SITK image, seed point pointing to point in vessel of interest
        seed(s): location of seed; either list (single point) or list of lists (multiple points)
    Returns:
        binary image file (either 0 or 1)
    """

    labels, means = connected_comp_info(image, False)
    ccimage = sitk.ConnectedComponent(image)

    # print(f"Seeds to remove around: {seed}")

    if type(seed[0]) == tuple:
        labels_seeds = []
        for i in range(len(seed)):
            labels_seeds.append(ccimage[seed[i]])
        # create a new image with only the labels of interest
        labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=labels_seeds[0], upperThreshold=labels_seeds[0])
        for i in range(1, len(labels_seeds)):
            labelImage = labelImage + sitk.BinaryThreshold(ccimage, lowerThreshold=labels_seeds[i], upperThreshold=labels_seeds[i])
    else:
        label = ccimage[seed]
        #print("The label we use is: " + str(label))

        if label == 0:
            label = 1
        labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label, upperThreshold=label)

    return labelImage

def connected_comp_info(original_seg, print_condition):
    """
    Print info on the component being kept
    """
    removed_seg = sitk.ConnectedComponent(original_seg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(removed_seg, original_seg)
    means = []

    for l in stats.GetLabels():
        if print_condition:
            print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), stats.GetPhysicalSize(l)))
        means.append(stats.GetMean(l))
    return stats.GetLabels(), means

def extract_volume(reader_im, index_extract, size_extract):
    """
    Function to extract a smaller volume from a larger one using sitk
    args:
        reader_im: sitk image reader
        index_extract: the index of the lower corner for extraction
        size_extract: number of voxels to extract in each direction
    return:
        new_img: sitk image volume
    """
    reader_im.SetExtractIndex(index_extract)
    reader_im.SetExtractSize(size_extract)
    new_img = reader_im.Execute()

    return new_img

def map_to_image(point, radius, size_volume, origin_im, spacing_im, size_im, min_resolution_any_dim = 5):
    """
    Function to map a point and radius to volume metrics
    Also checks if sub-volume is within global
    args:
        point: point of volume center
        radius: radius at that point
        size_volume: multiple of radius equal the intended
            volume size
        origin_im: image origin
        spacing_im: image spacing
    return:
        size_extract: number of voxels to extract in each dim
        index_extract: index for sitk volume extraction
        voi_min/max: boundaries of volume for caps constraint
    """
    min_res = 0

    while min_res < min_resolution_any_dim:
        size_extract = np.ceil(size_volume*radius/spacing_im).astype(int)
        index_extract = np.rint((point-origin_im - (size_volume/2)*radius)/spacing_im).astype(int)
        radius *= 1.05
        min_res = size_extract.min()
        print(f"Subvolume resolution: {size_extract}, Radius: {radius}")

    end_bounds = index_extract+size_extract

    for i, ind in enumerate(np.logical_and(end_bounds > size_im,(end_bounds- size_im) < 1/2*size_extract )):
        if ind:
            print('\nsub-volume outside global volume, correcting\n')
            size_extract[i] = size_im[i] - index_extract[i]

    for i, ind in enumerate(np.logical_and(index_extract < np.zeros(3),(np.zeros(3)-index_extract) < 1/2*size_extract )):
        if ind:
            print('\nsub-volume outside global volume, correcting\n')
            index_extract[i] = 0

    return size_extract.tolist(), index_extract.tolist()

def rotate_volume():
    "TODO: write function"
    return rotated_voi

def import_image(image_dir):
    """
    Function to import image via sitk
    args:
        file_dir_image: image directory
    return:
        reader_img: sitk image volume reader
        origin_im: image origin coordinates
        size_im: image size
        spacing_im: image spacing
    """
    reader_im = read_image(image_dir)
    origin_im = np.array(list(reader_im.GetOrigin()))
    size_im = np.array(list(reader_im.GetSize()))
    spacing_im = np.array(list(reader_im.GetSpacing()))

    return reader_im, origin_im, size_im, spacing_im

def sitk_to_numpy(Image):

    np_array = sitk.GetArrayFromImage(Image)
    return np_array

def numpy_to_sitk(numpy, file_reader = None):

    Image = sitk.GetImageFromArray(numpy)

    if file_reader:

        Image.SetSpacing(file_reader.GetSpacing())
        Image.SetOrigin(file_reader.GetOrigin())
        Image.SetDirection(file_reader.GetDirection())

    return Image
