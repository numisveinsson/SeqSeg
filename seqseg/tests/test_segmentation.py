import os

if __name__ == '__main__':
    """
    Test SeqSeg on a case of already segmented data

    This can be run without the need for:
    - pytorch
    - vmtk

    Only requires:
    - numpy
    - vtk
    - sitk
    """

    # Load the data from subfolder 'test_data'
    data_path = os.path.join(os.path.dirname(__file__), 'test_data')
