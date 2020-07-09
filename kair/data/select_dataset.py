

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['l', 'low-quality', 'input-only']:
        from kair.data.dataset_l import DatasetL as D

    # -----------------------------------------
    # denoising
    # -----------------------------------------
    elif dataset_type in ['dncnn', 'denoising']:
        from kair.data.dataset_dncnn import DatasetDnCNN as D

    elif dataset_type in ['dnpatch']:
        from kair.data.dataset_dnpatch import DatasetDnPatch as D

    elif dataset_type in ['ffdnet', 'denoising-noiselevel']:
        from kair.data.dataset_ffdnet import DatasetFFDNet as D

    elif dataset_type in ['fdncnn', 'denoising-noiselevelmap']:
        from kair.data.dataset_fdncnn import DatasetFDnCNN as D

    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    elif dataset_type in ['sr', 'super-resolution']:
        from kair.data.dataset_sr import DatasetSR as D

    elif dataset_type in ['srmd']:
        from kair.data.dataset_srmd import DatasetSRMD as D

    elif dataset_type in ['dpsr', 'dnsr']:
        from kair.data.dataset_dpsr import DatasetDPSR as D

    elif dataset_type in ['usrnet', 'usrgan']:
        from kair.data.dataset_usrnet import DatasetUSRNet as D

    # -----------------------------------------
    # common
    # -----------------------------------------
    elif dataset_type in ['plain']:
        from kair.data import DatasetPlain as D

    elif dataset_type in ['plainpatch']:
        from kair.data.dataset_plainpatch import DatasetPlainPatch as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
