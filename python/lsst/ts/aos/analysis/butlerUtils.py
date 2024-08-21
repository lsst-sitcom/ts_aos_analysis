# This file is part of ts_aos_analysis.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from lsst.daf.butler import Butler
from lsst.pipe.base import TaskMetadata
from astropy.time import Time

__all__ = ['get_task_metadata', 'get_timing_from_metadata']

def get_task_metadata(
    butler_path: str,
    collection: str,
    task_list: list[str],
    visit_list: list[int],
    detector_list: list[int],
) -> dict[str, list[TaskMetadata]]:
    """
    Gather the metadata stored in the butler for a list
    of tasks.

    Parameters
    ----------
    butler_path: str
        Path to butler repository.
    collection: str
        Name of the collection in butler where data is stored.
    task_list: list(str)
        Name of the tasks from which to retrieve metadata.
    visit_list: list(int)
        Visit Ids to gather from butler.
    detector_list: list(int)
        Detector Ids.

    Returns
    -------
    dict
        Task metadata from all visits and detectors organized by task.
    """

    butler = Butler(butler_path, collections=collection)

    md = {}

    for task in task_list:
        task_md = []
        for visit in visit_list:
            for detector in detector_list:
                task_md.append(
                    butler.get(
                        f"{task}_metadata",
                        exposure=visit,
                        visit=visit,
                        detector=detector,
                        collections=collection,
                    )
                )
        md[task] = task_md

    return md


def get_timing_from_metadata(
    task_md: dict[str, list[TaskMetadata]]
) -> tuple[list[tuple[str, Time, Time]], list[Time], list[Time], list[float]]:
    """
    Get the timing information from the task metadata stored in the butler.

    Parameters
    ----------
    task_md: dict
        Task Metadata collected from Butler. Keys should be task name
        and values should be a list of the metadata from each iteration
        of the task in the pipeline.

    Returns
    -------
    list
        Each element in the list contains a tuple for each process ran
        with the name of the task, start time, and finish time.
    list
        Earliest start time of all processes for a type of task
    list
        Latest finish times for all processes for a type of task
    list
        Longest duration for a single process of a type of task
    """
    suffix = 'Utc'
    jobs = []
    first_job = []
    last_job = []
    longest_job = []
    for k, vv in task_md.items():
        print(k, len(vv))
        start_list = []
        stop_list = []
        duration = []
        for v in vv:
            subtask = 'quantum'
            arrs = v[subtask].arrays
            # print(arrs)
            if 'startUtc' in arrs:
                start = min(arrs['prep'+suffix])
                stop = max(arrs['end'+suffix])
            elif 'runQuantumStartCpuTime' in arrs:
                start = min(arrs['runQuantumStart'+suffix])
                stop = max(arrs['runQuantumEnd'+suffix])
            else:
                # Might be empty b/c intra is noop
                continue
            jobs.append((list(v.metadata.keys())[0], start, stop))
            start_list.append(start)
            stop_list.append(stop)
            duration.append((Time(stop[:-6]).mjd - Time(start[:-6]).mjd) * 86400)
        first_job.append(min(start_list))
        last_job.append(max(stop_list))
        longest_job.append(max(duration))

    return jobs, first_job, last_job, longest_job
