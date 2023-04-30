from .dataset import ImageFolderDataset


def get_dataset(name):
    if name == 'office_a':
        return ImageFolderDataset('office', 'amazon', 31)
    elif name == 'office_d':
        return ImageFolderDataset('office', 'dslr', 31)
    elif name == 'office_w':
        return ImageFolderDataset('office', 'webcam', 31)
    elif name == 'office_home_a':
        return ImageFolderDataset('OfficeHome', 'Art', 65)
    elif name == 'office_home_c':
        return ImageFolderDataset('OfficeHome', 'Clipart', 65)
    elif name == 'office_home_p':
        return ImageFolderDataset('OfficeHome', 'Product', 65)
    elif name == 'office_home_r':
        return ImageFolderDataset('OfficeHome', 'RealWorld', 65)
    elif name == 'pacs_a':
        return ImageFolderDataset('PACS', 'art_painting', 7)
    elif name == 'pacs_c':
        return ImageFolderDataset('PACS', 'cartoon', 7)
    elif name == 'pacs_p':
        return ImageFolderDataset('PACS', 'photo', 7)
    elif name == 'pacs_s':
        return ImageFolderDataset('PACS', 'sketch', 7)
    elif name == 'visda_t':
        return ImageFolderDataset('VisDA', 'train', 12)
    elif name == 'visda_v':
        return ImageFolderDataset('VisDA', 'validation', 12)
    else:
        raise NotImplementedError()
