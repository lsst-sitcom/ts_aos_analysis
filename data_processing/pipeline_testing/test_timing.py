import cProfile
from lsst.daf.butler import Butler
from lsst.ip.isr.isrTaskLSST import IsrTaskLSST
from lsst.ts.wep.task import GenerateDonutDirectDetectTask
from lsst.ts.wep.task import CutOutDonutsScienceSensorTask, GroupPairer
from lsst.ts.wep.task import CalcZernikesTask, EstimateZernikesDanishTask

collections=[
    'u/brycek/aosBaseline_danish',
]
butler = Butler(
    "embargo",
    instrument='LSSTComCam',
    collections=collections
)

day_obs = 20241211
seq_num = 564
detector = 0

data_id_1 = {'day_obs': day_obs, 'seq_num': seq_num, 'detector': detector, 'instrument': 'LSSTComCam'}
#####
linearizer = butler.get('linearizer', dataId=data_id_1, collections='LSSTComCam/defaults')
bias = butler.get('bias', dataId=data_id_1, collections='LSSTComCam/defaults')
crosstalk = butler.get('crosstalk', dataId=data_id_1, collections='LSSTComCam/defaults')
defects = butler.get("defects", dataId=data_id_1, collections='LSSTComCam/defaults')
dark = butler.get('dark', dataId=data_id_1, collections='LSSTComCam/defaults')
flat = butler.get('flat', dataId=data_id_1, collections='LSSTComCam/defaults')
ptc = butler.get('ptc', dataId=data_id_1, collections='LSSTComCam/defaults')

isr_task_config = IsrTaskLSST.ConfigClass()
# From https://github.com/lsst/obs_lsst/blob/main/config/comCam/isrLSST.py#L27
"""
comCam-specific overrides for IsrTaskLSST
"""
isr_task_config.doSaturation = True
isr_task_config.crosstalk.doQuadraticCrosstalkCorrection = True
isr_task_config.crosstalk.doSubtrahendMasking = True
isr_task_config.crosstalk.minPixelToMask = 1.0
isr_task_config.doDeferredCharge = False

# Maintain compatibility with existing calibrations.
# TODO: Remove these for running new calibs with DM-48520.
poscan = isr_task_config.overscanCamera.defaultDetectorConfig.defaultAmpConfig.parallelOverscanConfig
poscan.doAbsoluteMaxDeviation = True
poscan.doMedianSmoothingOutlierRejection = False

isr_task_config.doAmpOffset = True
isr_task_config.ampOffset.doApplyAmpOffset = True
isr_task_config.ampOffset.ampEdgeMaxOffset = 10.0

# Although we don't have to apply the amp offset corrections, we do want
# to compute them for analyzeAmpOffsetMetadata to report on as metrics.
isr_task_config.doAmpOffset = False
isr_task_config.ampOffset.doApplyAmpOffset = False
# Turn off slow steps in ISR
isr_task_config.doBrighterFatter = False
# Mask saturated pixels,
isr_task_config.doSaturation = True
isr_task_config.crosstalk.doQuadraticCrosstalkCorrection = True

isr_task_config.qa.saveStats = False
isr_task_config.doStandardStatistics = False
isr_task_config.doInterpolate = False
isr_task_config.doVariance = False

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

cut_out_config = CutOutDonutsScienceSensorTask.ConfigClass()
cut_out_config.pairer.retarget(GroupPairer)
cut_out_config.donutStampSize = 200
cut_out_task = CutOutDonutsScienceSensorTask(config=cut_out_config)

calc_config = CalcZernikesTask.ConfigClass()
calc_config.estimateZernikes.retarget(EstimateZernikesDanishTask)
calc_config.donutStampSelector.maxFracBadPixels = 2.0e-4
calc_config.estimateZernikes.binning = 4
calc_config.estimateZernikes.nollIndices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28]
calc_config.estimateZernikes.saveHistory = False
calc_config.estimateZernikes.lstsqKwargs = {'ftol': 1.0e-3, 'xtol': 1.0e-3, 'gtol': 1.0e-3}
calc_config.donutStampSelector.maxSelect = 5
calc_task = CalcZernikesTask(config=calc_config)

raw_intra = butler.get('raw', day_obs=day_obs, seq_num=seq_num, detector=detector)
raw_extra = butler.get('raw', day_obs=day_obs, seq_num=seq_num+1, detector=detector)
# intra = butler.get("postISRCCD", day_obs=day_obs, seq_num=seq_num, detector=detector)
# extra = butler.get("postISRCCD", day_obs=day_obs, seq_num=seq_num+1, detector=detector)
# intra_table = butler.get("donutTable", day_obs=day_obs, seq_num=seq_num, detector=detector)
# extra_table = butler.get("donutTable", day_obs=day_obs, seq_num=seq_num+1, detector=detector)
camera = butler.get("camera", day_obs=day_obs, seq_num=seq_num+1, detector=detector)
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

with cProfile.Profile() as pr:
    isr_result_intra = isr_task.run(
        raw_intra,
        linearizer=linearizer,
        bias=bias,
        crosstalk=crosstalk,
        defects=defects,
        dark=dark,
        flat=flat,
        ptc=ptc
    )
    isr_result_extra = isr_task.run(
        raw_extra,
        linearizer=linearizer,
        bias=bias,
        crosstalk=crosstalk,
        defects=defects,
        dark=dark,
        flat=flat,
        ptc=ptc
    )
    donut_cat_result_intra = donut_cat_task.run(
        isr_result_intra.outputExposure,
        camera
    )
    donut_cat_result_extra = donut_cat_task.run(
        isr_result_extra.outputExposure,
        camera
    )
    cut_out_result = cut_out_task.run(
        exposures=[isr_result_intra.outputExposure, isr_result_extra.outputExposure],
        donutCatalog=[donut_cat_result_intra.donutCatalog, donut_cat_result_extra.donutCatalog],
        camera=camera
    )
    calc_result = calc_task.run(cut_out_result.donutStampsExtra, cut_out_result.donutStampsIntra)
    pr.dump_stats("profile_full_pipe.py.prof")