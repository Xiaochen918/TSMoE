class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/chenyao/Mywork/FERMT_TSMoE'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/chenyao/Mywork/FERMT_TSMoE/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/chenyao/Mywork/FERMT_TSMoE/pretrained_networks'
        self.lasot_dir = '/datasets/lasot'
        self.got10k_dir = '/datasets/got10k/train'
        self.got10k_val_dir = '/datasets/got10k/val'
        self.lasot_lmdb_dir = '/home/chenyao/Mywork/FERMT_TSMoE/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/chenyao/Mywork/FERMT_TSMoE/data/got10k_lmdb'
        self.trackingnet_dir = '/datasets/trackingnet'
        self.trackingnet_lmdb_dir = '/home/chenyao/Mywork/FERMT_TSMoE/data/trackingnet_lmdb'
        self.coco_dir = '/datasets/coco'
        self.coco_lmdb_dir = '/home/chenyao/Mywork/FERMT_TSMoE/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/chenyao/Mywork/FERMT_TSMoE/data/vid'
        self.imagenet_lmdb_dir = '/home/chenyao/Mywork/FERMT_TSMoE/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
