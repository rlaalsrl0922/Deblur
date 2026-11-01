from .ours_arch import EFNetOurs
def define_network(opt):
    options = dict(opt)
    name = options.pop('type')
    if name != 'EFNetOurs':
        raise ValueError(f'This release contains EFNetOurs only, got {name}')
    return EFNetOurs(**options)
