from .image_event_restoration_ours_model import ImageEventRestorationOurs
def create_model(opt):
    if opt['model_type'] != 'ImageEventRestorationOurs':
        raise ValueError('This release contains ImageEventRestorationOurs only')
    return ImageEventRestorationOurs(opt)
