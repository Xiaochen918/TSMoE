from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/got10k_lmdb'
    settings.got10k_path = '/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/itb'
    settings.lasot_extension_subset_path = '/datasets/lasot_ext'
    settings.lasot_lmdb_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/lasot_lmdb'
    settings.lasot_path = '/datasets/lasot'
    settings.network_path = '/home/chenyao/Mywork/FERMT_TSMoE/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/datasets/nfs30'
    settings.otb_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/otb'
    settings.prj_dir = '/home/chenyao/Mywork/FERMT_TSMoE'
    settings.result_plot_path = '/home/chenyao/Mywork/FERMT_TSMoE/output/test/result_plots'
    settings.results_path = '/home/chenyao/Mywork/FERMT_TSMoE/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/chenyao/Mywork/FERMT_TSMoE/output'
    settings.segmentation_path = '/home/chenyao/Mywork/FERMT_TSMoE/output/test/segmentation_results'
    settings.tc128_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/datasets/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/datasets/trackingnet'
    settings.uav_path = '/datasets/uav123'
    settings.vot18_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/vot2018'
    settings.vot22_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/vot2022'
    settings.vot_path = '/home/chenyao/Mywork/FERMT_TSMoE/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

