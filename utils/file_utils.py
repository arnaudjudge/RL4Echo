
def get_img_subpath(row, suffix='', extension='.nii.gz'):
    """
    Format string for partial path of image in file structure
    :param row: dataframe row with all columns filled in
    :param suffix: suffix before file extension
    :param extension: file extension
    :return: string containing path to image file
    """
    return f"{row['study']}/{row['view'].lower()}/{row['dicom_uuid']}" + suffix + extension
