import cProfile
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from lsst.daf.butler import Butler
from lsst.ip.isr.isrTaskLSST import IsrTaskLSST
from lsst.ts.wep.task import GenerateDonutDirectDetectTask
from lsst.ts.wep.task import CutOutDonutsCwfsTask
from lsst.ts.wep.task import CalcZernikesTask, EstimateZernikesDanishTask

collections=[
    'LSSTCam/raw/all',
]
butler = Butler(
    "/sdf/group/rubin/repo/aos_imsim/",
    instrument='LSSTCam',
    collections=collections
)

day_obs = 20240723
seq_num = 4
detector = 191

data_id_1 = {'day_obs': day_obs, 'seq_num': seq_num, 'detector': detector, 'instrument': 'LSSTCam'}
data_id_2 = {'day_obs': day_obs, 'seq_num': seq_num, 'detector': detector+1, 'instrument': 'LSSTCam'}
#####
# linearizer = butler.get('linearizer', dataId=data_id_1, collections='LSSTCam/defaults')
# bias = butler.get('bias', dataId=data_id_1, collections='LSSTCam/defaults')
# crosstalk = butler.get('crosstalk', dataId=data_id_1, collections='LSSTCam/defaults')
# defects = butler.get("defects", dataId=data_id_1, collections='LSSTCam/defaults')
# dark = butler.get('dark', dataId=data_id_1, collections='LSSTCam/defaults')
# flat = butler.get('flat', dataId=data_id_1, collections='LSSTCam/defaults')
# ptc = butler.get('ptc', dataId=data_id_1, collections='LSSTCam/defaults')

# linearizer_2 = butler.get('linearizer', dataId=data_id_2, collections='LSSTCam/defaults')
# bias_2 = butler.get('bias', dataId=data_id_2, collections='LSSTCam/defaults')
# crosstalk_2 = butler.get('crosstalk', dataId=data_id_2, collections='LSSTCam/defaults')
# defects_2 = butler.get("defects", dataId=data_id_2, collections='LSSTCam/defaults')
# dark_2 = butler.get('dark', dataId=data_id_2, collections='LSSTCam/defaults')
# flat_2 = butler.get('flat', dataId=data_id_2, collections='LSSTCam/defaults')
# ptc_2 = butler.get('ptc', dataId=data_id_2, collections='LSSTCam/defaults')

isr_task_config = IsrTaskLSST.ConfigClass()
# From https://github.com/lsst/obs_lsst/blob/main/config/comCam/isrLSST.py#L27
"""
comCam-specific overrides for IsrTaskLSST
"""
# isr_task_config.doSaturation = True
# isr_task_config.crosstalk.doQuadraticCrosstalkCorrection = True
# isr_task_config.crosstalk.doSubtrahendMasking = True
# isr_task_config.crosstalk.minPixelToMask = 1.0
# isr_task_config.doDeferredCharge = False

# # Maintain compatibility with existing calibrations.
# # TODO: Remove these for running new calibs with DM-48520.
# poscan = isr_task_config.overscanCamera.defaultDetectorConfig.defaultAmpConfig.parallelOverscanConfig
# poscan.doAbsoluteMaxDeviation = True
# poscan.doMedianSmoothingOutlierRejection = False

# isr_task_config.doAmpOffset = True
# isr_task_config.ampOffset.doApplyAmpOffset = True
# isr_task_config.ampOffset.ampEdgeMaxOffset = 10.0

# # Although we don't have to apply the amp offset corrections, we do want
# # to compute them for analyzeAmpOffsetMetadata to report on as metrics.
# isr_task_config.doAmpOffset = False
# isr_task_config.ampOffset.doApplyAmpOffset = False
# # Turn off slow steps in ISR
# isr_task_config.doBrighterFatter = False
# # Mask saturated pixels,
# isr_task_config.doSaturation = True
# isr_task_config.crosstalk.doQuadraticCrosstalkCorrection = True

isr_task_config.qa.saveStats = False
isr_task_config.doStandardStatistics = False
isr_task_config.doInterpolate = False
isr_task_config.doVariance = False


isr_task_config.doBootstrap = True
isr_task_config.doLinearize = False
isr_task_config.doApplyGains = False
isr_task_config.doBias = False
isr_task_config.doCrosstalk = False
isr_task_config.doDeferredCharge = False
isr_task_config.doDefect = False
isr_task_config.doDark = False
isr_task_config.doFlat = False
isr_task_config.doSaturation = False
isr_task_config.doBrighterFatter = False
isr_task_config.doAmpOffset = False
isr_task_config.doSuspect = False
isr_task_config.doITLEdgeBleedMask = False

#####
# Aaron's config
# isr_task_config.doCrosstalk = False
# isr_task_config.doLinearize = False
# isr_task_config.doDark = False

####


isr_task = IsrTaskLSST(config=isr_task_config)

donut_cat_config = GenerateDonutDirectDetectTask.ConfigClass()
donut_cat_config.donutSelector.useCustomMagLimit = True
donut_cat_config.donutSelector.sourceLimit = 20
donut_cat_task = GenerateDonutDirectDetectTask(config=donut_cat_config)

cut_out_config = CutOutDonutsCwfsTask.ConfigClass()
cut_out_config.donutStampSize = 200
cut_out_task = CutOutDonutsCwfsTask(config=cut_out_config)

calc_config = CalcZernikesTask.ConfigClass()
calc_config.estimateZernikes.retarget(EstimateZernikesDanishTask)
calc_config.donutStampSelector.maxFracBadPixels = 2.0e-4
calc_config.estimateZernikes.binning = 4
calc_config.estimateZernikes.nollIndices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28]
calc_config.estimateZernikes.saveHistory = False
calc_config.estimateZernikes.lstsqKwargs = {'ftol': 1.0e-3, 'xtol': 1.0e-3, 'gtol': 1.0e-3}
calc_config.donutStampSelector.maxSelect = 5
calc_task = CalcZernikesTask(config=calc_config)

raw_intra = butler.get('raw', dataId=data_id_2)
raw_extra = butler.get('raw', dataId=data_id_1)
# intra = butler.get("postISRCCD", day_obs=day_obs, seq_num=seq_num, detector=detector)
# extra = butler.get("postISRCCD", day_obs=day_obs, seq_num=seq_num+1, detector=detector)
# intra_table = butler.get("donutTable", day_obs=day_obs, seq_num=seq_num, detector=detector)
# extra_table = butler.get("donutTable", day_obs=day_obs, seq_num=seq_num+1, detector=detector)
camera = butler.get("camera", dataId={"instrument": "LSSTCam"}, collections="LSSTCam/calib/unbounded")
# donutStampsExtra = butler.get("donutStampsExtra", day_obs=day_obs, seq_num=seq_num+1, detector=detector)
# donutStampsIntra = butler.get("donutStampsIntra", day_obs=day_obs, seq_num=seq_num+1, detector=detector)

# # Mimic only having 5 donuts
# intra_table = intra_table[:5]
# extra_table = extra_table[:5]

# with cProfile.Profile() as pr:
#     result = task.run(
#         exposures=[intra, extra],
#         donutCatalog=[intra_table, extra_table],
#         camera=camera
#     )
#     pr.dump_stats("profile_cutout.py.prof")


# with cProfile.Profile() as pr:
#     result = isr_task.run(
#         raw_intra,
#         linearizer=linearizer,
#         bias=bias,
#         crosstalk=crosstalk,
#         defects=defects,
#         dark=dark,
#         flat=flat,
#         ptc=ptc
#     )
#     pr.dump_stats("profile_isr_aaron.py.prof")

# with cProfile.Profile(subcalls=True) as pr:
pr = cProfile.Profile()
pr.enable()
isr_result_intra = isr_task.run(
    raw_intra,
    # linearizer=linearizer_2,
    # bias=bias_2,
    # crosstalk=crosstalk_2,
    # defects=defects_2,
    # dark=dark_2,
    # flat=flat_2,
    # ptc=ptc_2
)

isr_result_extra = isr_task.run(
    raw_extra,
    # linearizer=linearizer,
    # bias=bias,
    # crosstalk=crosstalk,
    # defects=defects,
    # dark=dark,
    # flat=flat,
    # ptc=ptc
)
donut_cat_result_intra = donut_cat_task.run(
    isr_result_intra.outputExposure,
    camera
)
donut_cat_result_extra = donut_cat_task.run(
    isr_result_extra.outputExposure,
    camera
)
cut_out_result_extra = cut_out_task.run(
    exposure=isr_result_extra.outputExposure,
    donutCatalog=donut_cat_result_extra.donutCatalog,
    camera=camera
)
cut_out_result_intra = cut_out_task.run(
    exposure=isr_result_intra.outputExposure,
    donutCatalog=donut_cat_result_intra.donutCatalog,
    camera=camera
)
calc_result = calc_task.run(cut_out_result_extra.donutStampsExtra, cut_out_result_intra.donutStampsIntra)
pr.disable()
pr.dump_stats("profile_full_cwfs_pipe.py.prof")