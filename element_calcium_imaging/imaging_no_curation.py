import importlib
import inspect
import pathlib
from collections.abc import Callable

import datajoint as dj
import numpy as np
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory

from . import imaging_report, scan
from .scan import (
    get_imaging_root_data_dir,
    get_nd2_files,
    get_prairieview_files,
    get_processed_root_data_dir,
    get_scan_box_files,
    get_scan_image_files,
    get_zstack_files,
)


logger = dj.logger
schema = dj.Schema()

_linking_module = None


def activate(
    imaging_schema_name,
    scan_schema_name=None,
    *,
    create_schema=True,
    create_tables=True,
    linking_module=None,
):
    """Activate this schema.

    Args:
        imaging_schema_name (str): Schema name on the database server to activate the
            `imaging` module.
        scan_schema_name (str): Schema name on the database server to activate the
            `scan` module. Omitted, if the `scan` module is already activated.
        create_schema (bool): When True (default), create schema in the database if it
            does not yet exist.
        create_tables (bool): When True (default), create tables in the database if they
            do not yet exist.
        linking_module (str): A module name or a module containing the required
            dependencies to activate the `imaging` module: + all that are required by
            the `scan` module.

    Dependencies:
    Upstream tables:
        + Session: A parent table to Scan, identifying a scanning session.
        + Equipment: A parent table to Scan, identifying a scanning device.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    scan.activate(
        scan_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        linking_module=linking_module,
    )
    schema.activate(
        imaging_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )
    imaging_report.activate(f"{imaging_schema_name}_report", imaging_schema_name)


# -------------- Table declarations --------------


@schema
class ProcessingMethod(dj.Lookup):
    """Method, package, or analysis suite used for processing of calcium imaging data
        (e.g. Suite2p, CaImAn, etc.).

    Attributes:
        processing_method (str): Processing method.
        processing_method_desc (str): Processing method description.
    """

    definition = """# Method for calcium imaging processing
    processing_method: char(8)
    ---
    processing_method_desc: varchar(1000)  # Processing method description
    """

    contents = [
        ("suite2p", "suite2p analysis suite"),
        ("caiman", "caiman analysis suite"),
        ("extract", "extract analysis suite"),
    ]


@schema
class ProcessingParamSet(dj.Lookup):
    """Parameter set used for the processing of the calcium imaging scans,
    including both the analysis suite and its respective input parameters.

    A hash of the parameters of the analysis suite is also stored in order
    to avoid duplicated entries.

    Attributes:
        paramset_idx (int): Unique parameter set ID.
        ProcessingMethod (foreign key): A primary key from ProcessingMethod.
        paramset_desc (str): Parameter set description.
        param_set_hash (uuid): A universally unique identifier for the parameter set.
        params (longblob): Parameter Set, a dictionary of all applicable parameters to
            the analysis suite.
    """

    definition = """# Processing Parameter Set
    paramset_idx: smallint  # Unique parameter set ID.
    ---
    -> ProcessingMethod
    paramset_desc: varchar(1280)  # Parameter-set description
    param_set_hash: uuid  # A universally unique identifier for the parameter set
    unique index (param_set_hash)
    params: longblob  # Parameter Set, a dictionary of all applicable parameters to the analysis suite.
    """

    @classmethod
    def insert_new_params(
        cls,
        processing_method: str,
        paramset_idx: int,
        paramset_desc: str,
        params: dict,
    ):
        """Insert a parameter set into ProcessingParamSet table.

        This function automates the parameter set hashing and avoids insertion of an
            existing parameter set.

        Attributes:
            processing_method (str): Processing method/package used for processing of
                calcium imaging.
            paramset_idx (int): Unique parameter set ID.
            paramset_desc (str): Parameter set description.
            params (dict): Parameter Set, all applicable parameters to the analysis
                suite.
        """
        if processing_method == "extract":
            assert (
                params.get("extract") is not None and params.get("suite2p") is not None
            ), ValueError(
                "Please provide the processing parameters in the {'suite2p': {...}, 'extract': {...}} dictionary format."
            )

            # Force Suite2p to only run motion correction.
            params["suite2p"]["do_registration"] = True
            params["suite2p"]["roidetect"] = False

        param_dict = {
            "processing_method": processing_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        q_param = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if q_param:  # If the specified param-set already exists
            p_name = q_param.fetch1("paramset_idx")
            if p_name == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    "The specified param-set already exists - name: {}".format(p_name)
                )
        else:
            cls.insert1(param_dict)


@schema
class CellCompartment(dj.Lookup):
    """Cell compartments that can be imaged (e.g. 'axon', 'soma', etc.)

    Attributes:
        cell_compartment (str): Cell compartment.
    """

    definition = """# Cell compartments
    cell_compartment: char(16)
    """

    contents = zip(["axon", "soma", "bouton"])


@schema
class MaskType(dj.Lookup):
    """Available labels for segmented masks (e.g. 'soma', 'axon', 'dendrite', 'neuropil').

    Attributes:
        mask_type (str): Mask type.
    """

    definition = """# Possible types of a segmented mask
    mask_type: varchar(16)
    """

    contents = zip(["soma", "axon", "dendrite", "neuropil", "artefact", "unknown"])


# -------------- Trigger a processing routine --------------


@schema
class ProcessingTask(dj.Manual):
    """A pairing of processing params and scans to be loaded or triggered

    This table defines a calcium imaging processing task for a combination of a
    `Scan` and a `ProcessingParamSet` entries, including all the inputs (scan, method,
    method's parameters). The task defined here is then run in the downstream table
    `Processing`. This table supports definitions of both loading of pre-generated results
    and the triggering of new analysis for all supported analysis methods.

    Attributes:
        scan.Scan (foreign key):
        ProcessingParamSet (foreign key):
        processing_output_dir (str):
        task_mode (str): One of 'load' (load computed analysis results) or 'trigger'
            (trigger computation).
    """

    definition = """# Manual table for defining a processing task ready to be run
    -> scan.Scan
    -> ProcessingParamSet
    ---
    processing_output_dir: varchar(255)  #  Output directory of the processed scan relative to root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """Infer an output directory for an entry in ProcessingTask table.

        Args:
            key (dict): Primary key from the ProcessingTask table.
            relative (bool): If True, processing_output_dir is returned relative to
                imaging_root_dir. Default False.
            mkdir (bool): If True, create the processing_output_dir directory.
                Default True.

        Returns:
            dir (str): A default output directory for the processed results (processed_output_dir
                in ProcessingTask) based on the following convention:
                processed_dir / scan_dir / {processing_method}_{paramset_idx}
                e.g.: sub4/sess1/scan0/suite2p_0
        """
        image_locators = {
            "NIS": get_nd2_files,
            "ScanImage": get_scan_image_files,
            "Scanbox": get_scan_box_files,
            "PrairieView": get_prairieview_files,
        }
        image_locator = image_locators[(scan.Scan & key).fetch1("acq_software")]

        scan_dir = find_full_path(
            get_imaging_root_data_dir(), image_locator(key)[0]
        ).parent
        root_dir = find_root_directory(get_imaging_root_data_dir(), scan_dir)

        method = (
            (ProcessingParamSet & key).fetch1("processing_method").replace(".", "-")
        )

        processed_dir = pathlib.Path(get_processed_root_data_dir())
        output_dir = (
            processed_dir
            / scan_dir.relative_to(root_dir)
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir


@schema
class ZDriftMetrics(dj.Computed):
    """Generate z-axis motion report and drop frames with high motion.

    Attributes:
        ProcessingTask (foreign key): Primary key from ProcessingTask.
        ZDriftParamSet (foreign key): Primary key from ZDriftParamSet.
        z_drift (longblob): Amount of drift in microns per frame in Z direction.
        bad_frames_threshold (float): Drift threshold in microns where frames are deemed as bad frames.
        percent_bad_frames (float): Percentage of frames with z-drift exceeding the threshold.
    """

    definition = """
    -> ProcessingTask
    ---
    bad_frames=NULL: longblob  # `True` if any value in z_drift > threshold from drift_params.
    z_drift: longblob   # Amount of drift in microns per frame in Z direction.
    bad_frames_threshold: float  # Drift threshold in microns where frames are deemed as bad frames.
    percent_bad_frames: float  # Percentage of frames with z-drift exceeding the threshold.
    """

    _default_params = {
        "pad_length": 5,
        "slice_interval": 1,
        "num_scans": 5,
        "bad_frames_threshold": 3,
    }

    def make(self, key):
        import nd2

        def _make_taper(size, width):
            m = np.ones(size - width + 1)
            k = np.hanning(width)
            return np.convolve(m, k, mode="full") / k.sum()

        nchannels = (scan.ScanInfo & key).fetch1("nchannels")
        params = (ProcessingTask * ProcessingParamSet & key).fetch1("params")
        drift_params = params.get("ZDRIFT_PARAMS", self._default_params)

        # use the same channel specified in ProcessingParamSet for this task
        drift_params["channel"] = params.get("align_by_chan", 1)
        drift_params["channel"] -= 1  # change to 0-based indexing

        image_files = (scan.ScanInfo.ScanFile & key).fetch("file_path")
        image_files = [
            find_full_path(get_imaging_root_data_dir()[0], image_file)
            for image_file in image_files
        ]

        try:
            movie_file = next(
                file for file in image_files if not file.name.endswith("_Z.nd2")
            )
        except StopIteration:
            raise FileNotFoundError(
                f"No calcium imaging movie file found in {image_files}"
            )

        zstack_files = get_zstack_files(key)
        assert (
            len(zstack_files) == 1
        ), f"Multiple zstack files found at {zstack_files}. Expected only one."

        ca_imaging_movie = nd2.imread(movie_file)
        zstack = nd2.imread(zstack_files[0])

        if not all(
            parameter in drift_params
            for parameter in [
                "pad_length",
                "slice_interval",
                "channel",
                "num_scans",
            ]
        ):
            raise dj.DataJointError(
                "Z-drift parameters must include keys for 'pad_length', 'slice_interval', 'num_scans', and 'channel'."
            )

        ca_imaging_movie = ca_imaging_movie[:, drift_params["channel"], :, :]

        if drift_params["num_scans"] > 1 and nchannels > 1:
            zstack = zstack.mean(axis=0)
            zstack = zstack[:, drift_params["channel"], :, :]
        elif drift_params["num_scans"] == 1 and nchannels > 1:
            zstack = zstack[:, drift_params["channel"], :, :]
        else:
            raise NotImplementedError(
                "Z-drift metrics for scans with only one channel are not yet supported."
            )

        # center zstack
        zstack = zstack - zstack.mean(axis=(1, 2), keepdims=True)

        # taper zstack
        ytaper = _make_taper(zstack.shape[1], drift_params["pad_length"])
        zstack = zstack * ytaper[None, :, None]

        xtaper = _make_taper(zstack.shape[2], drift_params["pad_length"])
        zstack = zstack * xtaper[None, None, :]

        # normalize zstack
        zstack = zstack - zstack.mean(axis=(1, 2), keepdims=True)
        zstack /= np.sqrt((zstack**2).sum(axis=(1, 2), keepdims=True))

        # pad zstack
        zstack = np.pad(
            zstack,
            (
                (0, 0),
                (drift_params["pad_length"], drift_params["pad_length"]),
                (drift_params["pad_length"], drift_params["pad_length"]),
            ),
        )

        # normalize movie
        ca_imaging_movie = ca_imaging_movie - ca_imaging_movie.mean(
            axis=(1, 2), keepdims=True
        )
        ca_imaging_movie /= np.sqrt(
            (ca_imaging_movie**2).sum(axis=(1, 2), keepdims=True)
        )

        # Vectorized implementation
        middle = (zstack.shape[0] - 1) // 2
        _, ny, nx = ca_imaging_movie.shape
        offsets = list(
            (dy, dx)
            for dx in range(2 * drift_params["pad_length"] + 1)
            for dy in range(2 * drift_params["pad_length"] + 1)
        )
        c = list(
            np.einsum(
                "dij, tij -> td",
                zstack[:, dy : dy + ny, dx : dx + nx],
                ca_imaging_movie,
            )
            for dy, dx in offsets
        )

        drift = ((np.argmax(np.stack(c).max(axis=0), axis=1)) - middle) * drift_params[
            "slice_interval"
        ]

        bad_frames_idx = np.where(
            np.abs(drift) >= drift_params["bad_frames_threshold"]
        )[0]

        self.insert1(
            dict(
                **key,
                z_drift=drift,
                bad_frames=bad_frames_idx,
                bad_frames_threshold=drift_params["bad_frames_threshold"],
                percent_bad_frames=len(bad_frames_idx) / len(drift) * 100,
            ),
        )


@schema
class Processing(dj.Computed):
    """Perform the computation of an entry (task) defined in the ProcessingTask table.
    The computation is performed only on the scans with ScanInfo inserted.

    Attributes:
        ProcessingTask (foreign key): Primary key from ProcessingTask.
        processing_time (datetime): Process completion datetime.
        package_version (str, optional): Version of the analysis package used in
            processing the data.
    """

    definition = """
    -> ProcessingTask
    ---
    processing_time     : datetime  # Time of generation of this set of processed, segmented results
    package_version=''  : varchar(16)
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@imaging-processed
        """

    # Run processing only on Scan with ScanInfo inserted
    @property
    def key_source(self):
        """Limit the Processing to Scans that have their metadata ingested to the
        database."""

        return ProcessingTask & scan.ScanInfo & ZDriftMetrics

    def make(self, key):
        """Execute the calcium imaging analysis defined by the ProcessingTask."""

        task_mode, output_dir = (ProcessingTask & key).fetch1(
            "task_mode", "processing_output_dir"
        )

        if not output_dir:
            output_dir = ProcessingTask.infer_output_dir(key, relative=True, mkdir=True)
            # update processing_output_dir
            ProcessingTask.update1(
                {**key, "processing_output_dir": output_dir.as_posix()}
            )

        try:
            output_dir = find_full_path(
                get_imaging_root_data_dir(), output_dir
            ).as_posix()
        except FileNotFoundError as e:
            if task_mode == "trigger":
                processed_dir = pathlib.Path(get_processed_root_data_dir())
                output_dir = processed_dir / output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = output_dir.as_posix()
            else:
                raise e

        if task_mode == "load":
            method, imaging_dataset = get_loader_result(key, ProcessingTask)
            if method == "suite2p":
                if (scan.ScanInfo & key).fetch1("nrois") > 0:
                    raise NotImplementedError(
                        "Suite2p ingestion error - Unable to handle"
                        + " ScanImage multi-ROI scanning mode yet"
                    )
                suite2p_dataset = imaging_dataset
                key = {**key, "processing_time": suite2p_dataset.creation_time}
            elif method == "caiman":
                caiman_dataset = imaging_dataset
                key = {**key, "processing_time": caiman_dataset.creation_time}
            elif method == "extract":
                raise NotImplementedError(
                    "To use EXTRACT with this DataJoint Element please set `task_mode=trigger`"
                )
            else:
                raise NotImplementedError("Unknown/unimplemented method: {}".format(method))
        elif task_mode == "trigger":
            drop_frames = (ZDriftMetrics & key).fetch1("bad_frames")
            if drop_frames.size > 0:
                np.save(pathlib.Path(output_dir) / "bad_frames.npy", drop_frames)
                raw_image_files = (scan.ScanInfo.ScanFile & key).fetch("file_path")
                files_to_link = [
                    find_full_path(get_imaging_root_data_dir(), raw_image_file)
                    for raw_image_file in raw_image_files
                    if not raw_image_file.endswith("_Z.nd2")
                ]
                image_files = []
                for file in files_to_link:
                    if not (pathlib.Path(output_dir) / file.name).is_symlink():
                        (pathlib.Path(output_dir) / file.name).symlink_to(file)
                        image_files.append((pathlib.Path(output_dir) / file.name))
                    else:
                        image_files.append((pathlib.Path(output_dir) / file.name))
            else:
                image_files = (scan.ScanInfo.ScanFile & key).fetch("file_path")
                image_files = [
                    find_full_path(get_imaging_root_data_dir(), image_file)
                    for image_file in image_files
                    if not image_file.endswith("_Z.nd2")
                ]

            method = (ProcessingParamSet * ProcessingTask & key).fetch1(
                "processing_method"
            )

            params = (ProcessingTask * ProcessingParamSet & key).fetch1("params")
            params.pop("ZDRIFT_PARAMS", None)

            if method == "suite2p":
                import suite2p

                suite2p_params = params
                suite2p_params["save_path0"] = output_dir
                (
                    suite2p_params["fs"],
                    suite2p_params["nplanes"],
                    suite2p_params["nchannels"],
                ) = (scan.ScanInfo & key).fetch1("fps", "ndepths", "nchannels")

                input_format = pathlib.Path(image_files[0]).suffix
                suite2p_params["input_format"] = input_format[1:]

                suite2p_paths = {
                    "data_path": [image_files[0].parent.as_posix()],
                    "tiff_list": [f.as_posix() for f in image_files],
                }
                suite2p.run_s2p(ops=suite2p_params, db=suite2p_paths)  # Run suite2p

                _, imaging_dataset = get_loader_result(key, ProcessingTask)
                suite2p_dataset = imaging_dataset
                key = {**key, "processing_time": suite2p_dataset.creation_time}
            else:
                raise NotImplementedError(f"Unknown method: {method}")
        else:
            raise ValueError(f"Unknown task mode: {task_mode}")

        self.insert1(key)

        if task_mode == "trigger":
            self.File.insert(
                [
                    {
                        **key,
                        "file_name": f.relative_to(output_dir).as_posix(),
                        "file": f,
                    }
                    for f in pathlib.Path(output_dir).rglob("*")
                    if f.is_file()
                ],
                ignore_extra_fields=True,
            )


# -------------- Motion Correction --------------


@schema
class MotionCorrection(dj.Imported):
    """Results of motion correction shifts performed on the imaging data.

    Attributes:
        Processing (foreign key): Primary key from Processing.
        scan.Channel.proj(motion_correct_channel='channel') (int): Channel used for
            motion correction in this processing task.
    """

    definition = """# Results of motion correction
    -> Processing
    ---
    -> scan.Channel.proj(motion_correct_channel='channel') # channel used for motion correction in this processing task
    """

    class RigidMotionCorrection(dj.Part):
        """Details of rigid motion correction performed on the imaging data.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            outlier_frames (longblob): Mask with true for frames with outlier shifts
                (already corrected).
            y_shifts (longblob): y motion correction shifts (pixels).
            x_shifts (longblob): x motion correction shifts (pixels).
            z_shifts (longblob, optional): z motion correction shifts (z-drift, pixels).
            y_std (float): standard deviation of y shifts across all frames (pixels).
            x_std (float): standard deviation of x shifts across all frames (pixels).
            z_std (float, optional): standard deviation of z shifts across all frames
                (pixels).
        """

        definition = """# Details of rigid motion correction performed on the imaging data
        -> master
        ---
        outlier_frames=null : longblob  # mask with true for frames with outlier shifts (already corrected)
        y_shifts            : longblob  # (pixels) y motion correction shifts
        x_shifts            : longblob  # (pixels) x motion correction shifts
        z_shifts=null       : longblob  # (pixels) z motion correction shifts (z-drift)
        y_std               : float     # (pixels) standard deviation of y shifts across all frames
        x_std               : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null          : float     # (pixels) standard deviation of z shifts across all frames
        """

    class NonRigidMotionCorrection(dj.Part):
        """Piece-wise rigid motion correction - tile the FOV into multiple 3D
        blocks/patches.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            outlier_frames (longblob, null): Mask with true for frames with outlier
                shifts (already corrected).
            block_height (int): Block height in pixels.
            block_width (int): Block width in pixels.
            block_depth (int): Block depth in pixels.
            block_count_y (int): Number of blocks tiled in the y direction.
            block_count_x (int): Number of blocks tiled in the x direction.
            block_count_z (int): Number of blocks tiled in the z direction.
        """

        definition = """# Details of non-rigid motion correction performed on the imaging data
        -> master
        ---
        outlier_frames=null : longblob # mask with true for frames with outlier shifts (already corrected)
        block_height        : int      # (pixels)
        block_width         : int      # (pixels)
        block_depth         : int      # (pixels)
        block_count_y       : int      # number of blocks tiled in the y direction
        block_count_x       : int      # number of blocks tiled in the x direction
        block_count_z       : int      # number of blocks tiled in the z direction
        """

    class Block(dj.Part):
        """FOV-tiled blocks used for non-rigid motion correction.

        Attributes:
            NonRigidMotionCorrection (foreign key): Primary key from
                NonRigidMotionCorrection.
            block_id (int): Unique block ID.
            block_y (longblob): y_start and y_end in pixels for this block
            block_x (longblob): x_start and x_end in pixels for this block
            block_z (longblob): z_start and z_end in pixels for this block
            y_shifts (longblob): y motion correction shifts for every frame in pixels
            x_shifts (longblob): x motion correction shifts for every frame in pixels
            z_shift=null (longblob, optional): x motion correction shifts for every frame
                in pixels
            y_std (float): standard deviation of y shifts across all frames in pixels
            x_std (float): standard deviation of x shifts across all frames in pixels
            z_std=null (float, optional): standard deviation of z shifts across all frames
                in pixels
        """

        definition = """# FOV-tiled blocks used for non-rigid motion correction
        -> master.NonRigidMotionCorrection
        block_id        : int
        ---
        block_y         : longblob  # (y_start, y_end) in pixel of this block
        block_x         : longblob  # (x_start, x_end) in pixel of this block
        block_z         : longblob  # (z_start, z_end) in pixel of this block
        y_shifts        : longblob  # (pixels) y motion correction shifts for every frame
        x_shifts        : longblob  # (pixels) x motion correction shifts for every frame
        z_shifts=null   : longblob  # (pixels) x motion correction shifts for every frame
        y_std           : float     # (pixels) standard deviation of y shifts across all frames
        x_std           : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null      : float     # (pixels) standard deviation of z shifts across all frames
        """

    class Summary(dj.Part):
        """Summary images for each field and channel after corrections.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            scan.ScanInfo.Field (foreign key): Primary key from scan.ScanInfo.Field.
            ref_image (longblob): Image used as alignment template.
            average_image (longblob): Mean of registered frames.
            correlation_image (longblob, optional): Correlation map (computed during
                cell detection).
            max_proj_image (longblob, optional): Max of registered frames.
        """

        definition = """# Summary images for each field and channel after corrections
        -> master
        -> scan.ScanInfo.Field
        ---
        ref_image               : longblob  # image used as alignment template
        average_image           : longblob  # mean of registered frames
        correlation_image=null  : longblob  # correlation map (computed during cell detection)
        max_proj_image=null     : longblob  # max of registered frames
        """

    def make(self, key):
        """Populate MotionCorrection with results parsed from analysis outputs"""
        method, imaging_dataset = get_loader_result(key, ProcessingTask)

        field_keys, _ = (scan.ScanInfo.Field & key).fetch(
            "KEY", "field_z", order_by="field_z"
        )

        if method == "suite2p":
            suite2p_dataset = imaging_dataset

            motion_correct_channel = suite2p_dataset.planes[0].alignment_channel

            # ---- iterate through all s2p plane outputs ----
            rigid_correction, nonrigid_correction, nonrigid_blocks = {}, {}, {}
            summary_images = []
            for idx, (plane, s2p) in enumerate(suite2p_dataset.planes.items()):
                if not all(k in s2p.ops for k in ["xblock", "yblock", "nblocks"]):
                    logger.warning(
                        f"Unable to load/ingest non-rigid motion correction for plane {plane}."
                        "Non-rigid motion correction data is not saved by Suite2p for versions above 0.10.*."
                    )
                else:
                    # -- rigid motion correction --
                    if idx == 0:
                        rigid_correction = {
                            **key,
                            "y_shifts": s2p.ops["yoff"],
                            "x_shifts": s2p.ops["xoff"],
                            "z_shifts": np.full_like(s2p.ops["xoff"], 0),
                            "y_std": np.nanstd(s2p.ops["yoff"]),
                            "x_std": np.nanstd(s2p.ops["xoff"]),
                            "z_std": np.nan,
                            "outlier_frames": s2p.ops["badframes"],
                        }
                    else:
                        rigid_correction["y_shifts"] = np.vstack(
                            [rigid_correction["y_shifts"], s2p.ops["yoff"]]
                        )
                        rigid_correction["y_std"] = np.nanstd(
                            rigid_correction["y_shifts"].flatten()
                        )
                        rigid_correction["x_shifts"] = np.vstack(
                            [rigid_correction["x_shifts"], s2p.ops["xoff"]]
                        )
                        rigid_correction["x_std"] = np.nanstd(
                            rigid_correction["x_shifts"].flatten()
                        )
                        rigid_correction["outlier_frames"] = np.logical_or(
                            rigid_correction["outlier_frames"], s2p.ops["badframes"]
                        )
                    # -- non-rigid motion correction --
                    if s2p.ops["nonrigid"]:
                        if idx == 0:
                            nonrigid_correction = {
                                **key,
                                "block_height": s2p.ops["block_size"][0],
                                "block_width": s2p.ops["block_size"][1],
                                "block_depth": 1,
                                "block_count_y": s2p.ops["nblocks"][0],
                                "block_count_x": s2p.ops["nblocks"][1],
                                "block_count_z": len(suite2p_dataset.planes),
                                "outlier_frames": s2p.ops["badframes"],
                            }
                        else:
                            nonrigid_correction["outlier_frames"] = np.logical_or(
                                nonrigid_correction["outlier_frames"],
                                s2p.ops["badframes"],
                            )
                        for b_id, (b_y, b_x, bshift_y, bshift_x) in enumerate(
                            zip(
                                s2p.ops["xblock"],
                                s2p.ops["yblock"],
                                s2p.ops["yoff1"].T,
                                s2p.ops["xoff1"].T,
                            )
                        ):
                            if b_id in nonrigid_blocks:
                                nonrigid_blocks[b_id]["y_shifts"] = np.vstack(
                                    [nonrigid_blocks[b_id]["y_shifts"], bshift_y]
                                )
                                nonrigid_blocks[b_id]["y_std"] = np.nanstd(
                                    nonrigid_blocks[b_id]["y_shifts"].flatten()
                                )
                                nonrigid_blocks[b_id]["x_shifts"] = np.vstack(
                                    [nonrigid_blocks[b_id]["x_shifts"], bshift_x]
                                )
                                nonrigid_blocks[b_id]["x_std"] = np.nanstd(
                                    nonrigid_blocks[b_id]["x_shifts"].flatten()
                                )
                            else:
                                nonrigid_blocks[b_id] = {
                                    **key,
                                    "block_id": b_id,
                                    "block_y": b_y,
                                    "block_x": b_x,
                                    "block_z": np.full_like(b_x, plane),
                                    "y_shifts": bshift_y,
                                    "x_shifts": bshift_x,
                                    "z_shifts": np.full(
                                        (
                                            len(suite2p_dataset.planes),
                                            len(bshift_x),
                                        ),
                                        0,
                                    ),
                                    "y_std": np.nanstd(bshift_y),
                                    "x_std": np.nanstd(bshift_x),
                                    "z_std": np.nan,
                                }

                # -- summary images --
                motion_correction_key = (
                    scan.ScanInfo.Field * Processing & key & field_keys[plane]
                ).fetch1("KEY")
                summary_images.append(
                    {
                        **motion_correction_key,
                        "ref_image": s2p.ref_image,
                        "average_image": s2p.mean_image,
                        "correlation_image": s2p.correlation_map,
                        "max_proj_image": s2p.max_proj_image,
                    }
                )

            self.insert1({**key, "motion_correct_channel": motion_correct_channel})
            if rigid_correction:
                self.RigidMotionCorrection.insert1(rigid_correction)
            if nonrigid_correction:
                self.NonRigidMotionCorrection.insert1(nonrigid_correction)
                self.Block.insert(nonrigid_blocks.values())
            self.Summary.insert(summary_images)
        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


# -------------- Segmentation --------------


@schema
class Segmentation(dj.Computed):
    """Result of the Segmentation process.

    Attributes:
        Processing (foreign key): Primary key from Processing.
    """

    definition = """# Different mask segmentations.
    -> Processing
    """

    class Mask(dj.Part):
        """Details of the masks identified from the Segmentation procedure.

        Attributes:
            Segmentation (foreign key): Primary key from Segmentation.
            mask (int): Unique mask ID.
            scan.Channel.proj(segmentation_channel='channel') (foreign key): Channel
                used for segmentation.
            mask_npix (int): Number of pixels in ROIs.
            mask_center_x (int): Center x coordinate in pixel.
            mask_center_y (int): Center y coordinate in pixel.
            mask_center_z (int): Center z coordinate in pixel.
            mask_xpix (longblob): X coordinates in pixels.
            mask_ypix (longblob): Y coordinates in pixels.
            mask_zpix (longblob): Z coordinates in pixels.
            mask_weights (longblob): Weights of the mask at the indices above.
        """

        definition = """ # A mask produced by segmentation.
        -> master
        mask            : smallint
        ---
        -> scan.Channel.proj(segmentation_channel='channel')  # channel used for segmentation
        mask_npix       : int       # number of pixels in ROIs
        mask_center_x   : int       # center x coordinate in pixel
        mask_center_y   : int       # center y coordinate in pixel
        mask_center_z   : int       # center z coordinate in pixel
        mask_xpix       : longblob  # x coordinates in pixels
        mask_ypix       : longblob  # y coordinates in pixels
        mask_zpix       : longblob  # z coordinates in pixels
        mask_weights    : longblob  # weights of the mask at the indices above
        """

    def make(self, key):
        """Populate the Segmentation with the results parsed from analysis outputs."""

        method, imaging_dataset = get_loader_result(key, ProcessingTask)

        if method == "suite2p":
            suite2p_dataset = imaging_dataset

            # ---- iterate through all s2p plane outputs ----
            masks, cells = [], []
            for plane, s2p in suite2p_dataset.planes.items():
                mask_count = len(masks)  # increment mask id from all "plane"
                for mask_idx, (is_cell, cell_prob, mask_stat) in enumerate(
                    zip(s2p.iscell, s2p.cell_prob, s2p.stat)
                ):
                    masks.append(
                        {
                            **key,
                            "mask": mask_idx + mask_count,
                            "segmentation_channel": s2p.segmentation_channel,
                            "mask_npix": mask_stat["npix"],
                            "mask_center_x": mask_stat["med"][1],
                            "mask_center_y": mask_stat["med"][0],
                            "mask_center_z": mask_stat.get("iplane", plane),
                            "mask_xpix": mask_stat["xpix"],
                            "mask_ypix": mask_stat["ypix"],
                            "mask_zpix": np.full(
                                mask_stat["npix"],
                                mask_stat.get("iplane", plane),
                            ),
                            "mask_weights": mask_stat["lam"],
                        }
                    )
                    if is_cell:
                        cells.append(
                            {
                                **key,
                                "mask_classification_method": "suite2p_default_classifier",
                                "mask": mask_idx + mask_count,
                                "mask_type": "soma",
                                "confidence": cell_prob,
                            }
                        )

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1(
                    {
                        **key,
                        "mask_classification_method": "suite2p_default_classifier",
                    },
                    allow_direct_insert=True,
                )
                MaskClassification.MaskType.insert(
                    cells, ignore_extra_fields=True, allow_direct_insert=True
                )
        else:
            raise NotImplementedError(f"Unknown/unimplemented method: {method}")


@schema
class MaskClassificationMethod(dj.Lookup):
    """Available mask classification methods.

    Attributes:
        mask_classification_method (str): Mask classification method.
    """

    definition = """
    mask_classification_method: varchar(48)
    """

    contents = zip(["suite2p_default_classifier", "caiman_default_classifier"])


@schema
class MaskClassification(dj.Computed):
    """Classes assigned to each mask.

    Attributes:
        Segmentation (foreign key): Primary key from Segmentation.
        MaskClassificationMethod (foreign key): Primary key from
            MaskClassificationMethod.
    """

    definition = """
    -> Segmentation
    -> MaskClassificationMethod
    """

    class MaskType(dj.Part):
        """Type assigned to each mask.

        Attributes:
            MaskClassification (foreign key): Primary key from MaskClassification.
            Segmentation.Mask (foreign key): Primary key from Segmentation.Mask.
            MaskType: Primary key from MaskType.
            confidence (float, optional): Confidence level of the mask classification.
        """

        definition = """
        -> master
        -> Segmentation.Mask
        ---
        -> MaskType
        confidence=null: float
        """

    def make(self, key):
        pass


# -------------- Activity Trace --------------


@schema
class Fluorescence(dj.Computed):
    """Fluorescence traces.

    Attributes:
        Segmentation (foreign key): Primary key from Segmentation.
    """

    definition = """# Fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
        """Traces obtained from segmented region of interests.

        Attributes:
            Fluorescence (foreign key): Primary key from Fluorescence.
            Segmentation.Mask (foreign key): Primary key from Segmentation.Mask.
            scan.Channel.proj(fluo_channel='channel') (int): The channel that this trace
                comes from.
            fluorescence (longblob): Fluorescence trace associated with this mask.
            neuropil_fluorescence (longblob, optional): Neuropil fluorescence trace.
        """

        definition = """
        -> master
        -> Segmentation.Mask
        -> scan.Channel.proj(fluo_channel='channel')  # The channel that this trace comes from
        ---
        fluorescence                : longblob  # Fluorescence trace associated with this mask
        neuropil_fluorescence=null  : longblob  # Neuropil fluorescence trace
        """

    def make(self, key):
        """Populate the Fluorescence with the results parsed from analysis outputs."""

        method, imaging_dataset = get_loader_result(key, ProcessingTask)

        if method == "suite2p":
            suite2p_dataset = imaging_dataset

            # ---- iterate through all s2p plane outputs ----
            fluo_traces, fluo_chn2_traces = [], []
            for s2p in suite2p_dataset.planes.values():
                mask_count = len(fluo_traces)  # increment mask id from all "plane"
                for mask_idx, (f, fneu) in enumerate(zip(s2p.F, s2p.Fneu)):
                    fluo_traces.append(
                        {
                            **key,
                            "mask": mask_idx + mask_count,
                            "fluo_channel": 0,
                            "fluorescence": f,
                            "neuropil_fluorescence": fneu,
                        }
                    )
                if len(s2p.F_chan2):
                    mask_chn2_count = len(
                        fluo_chn2_traces
                    )  # increment mask id from all planes
                    for mask_idx, (f2, fneu2) in enumerate(
                        zip(s2p.F_chan2, s2p.Fneu_chan2)
                    ):
                        fluo_chn2_traces.append(
                            {
                                **key,
                                "mask": mask_idx + mask_chn2_count,
                                "fluo_channel": 1,
                                "fluorescence": f2,
                                "neuropil_fluorescence": fneu2,
                            }
                        )

            self.insert1(key)
            self.Trace.insert(fluo_traces + fluo_chn2_traces)
        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    """Available activity extraction methods.

    Attributes:
        extraction_method (str): Extraction method.
    """

    definition = """
    extraction_method: varchar(32)
    """

    contents = zip(["suite2p", "caiman", "FISSA"])


@schema
class ActivityExtractionParamSet(dj.Lookup):
    definition = """  #  Activity extraction parameter set used for the analysis/extraction of calcium events
    activity_extraction_paramset_idx:  smallint
    ---
    -> ActivityExtractionMethod
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(
        cls,
        extraction_method: str,
        activity_extraction_paramset_idx: int,
        paramset_desc: str,
        params: dict,
    ):
        param_dict = {
            "extraction_method": extraction_method,
            "activity_extraction_paramset_idx": activity_extraction_paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        q_param = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1("activity_extraction_paramset_idx")
            if (
                pname == activity_extraction_paramset_idx
            ):  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    "The specified param-set already exists - name: {}".format(pname)
                )
        else:
            cls.insert1(param_dict)


@schema
class Activity(dj.Computed):
    """Inferred neural activity from fluorescence trace.

    Attributes:
        Fluorescence (foreign key): Primary key from Fluorescence.
        ActivityExtractionParamSet (foreign key): Primary key from
            ActivityExtractionParamSet.
    """

    definition = """# Neural Activity
    -> Fluorescence
    -> ActivityExtractionParamSet 
    """

    class Trace(dj.Part):
        definition = """
        -> master
        -> Fluorescence.Trace
        activity_type: varchar(16)  # e.g. z_score, Fcorrected
        ---
        activity_trace: longblob  # z score, neuropil corrected Fluorescence
        """

    def make(self, key):
        # This code estimates the following quantities for each cell:
        # 1) neuropil corrected fluorescence,
        # 2) dff (only in old Fissa where F >= 0 constraint did not exist),
        # 3) zscore.

        import fissa
        from collections import Counter
        from .utils import calculate_dff, calculate_zscore, detect_events, combine_trials

        fissa_params = (ActivityExtractionParamSet & key).fetch1("params")
        sampling_rate = (scan.ScanInfo & key).fetch1("fps")

        task_mode, output_dir = (ProcessingTask & key).fetch1(
            "task_mode", "processing_output_dir"
        )
        # processing_output_dir contains the paramset_id. The upload & ingest should be done accordingly.
        output_dir = find_full_path(
            get_imaging_root_data_dir(),
            output_dir,
        )

        reg_img_dir = output_dir / "suite2p/plane0/reg_tif"
        fissa_output_dir = output_dir / "FISSA_Suite2P"

        # Required even though the FISSA is not triggered
        cell_ids, mask_xpixs, mask_ypixs = (
            Segmentation.Mask & MaskClassification.MaskType & key
        ).fetch("mask", "mask_xpix", "mask_ypix")

        # Trigger Condition
        if not any(
            (fissa_output_dir / p).exists() for p in ["separated.npy", "separated.npz"]
        ):
            if fissa_params.get("task_mode", task_mode) == "load":
                raise FileNotFoundError(f"No FISSA results found in {fissa_output_dir}")

            fissa_output_dir.mkdir(parents=True, exist_ok=True)

            Ly, Lx = (
                (MotionCorrection.Summary & key)
                .fetch("average_image", limit=1)[0]
                .shape
            )
            rois = [np.zeros((Ly, Lx), dtype=bool) for n in range(len(cell_ids))]

            # Find overlapping pixels to remove
            pixel_counts = Counter(
                list(zip(np.hstack(mask_xpixs), np.hstack(mask_ypixs)))
            )
            overlapping_pixels = [k for k, v in pixel_counts.items() if v > 1]

            for i, (mask_xpix, mask_ypix) in enumerate(zip(mask_xpixs, mask_ypixs)):
                is_pixel_overlapping = np.array(
                    [
                        True if (x, y) in overlapping_pixels else False
                        for x, y in zip(mask_xpix, mask_ypix)
                    ]
                )
                rois[i][
                    mask_ypix[~is_pixel_overlapping], mask_xpix[~is_pixel_overlapping]
                ] = 1

            experiment = fissa.Experiment(
                reg_img_dir.as_posix(),
                [rois],
                fissa_output_dir.as_posix(),
                **fissa_params["init"],
            )
            experiment.separate(**fissa_params["exec"])

        # Load the results
        fissa_output_file = list(fissa_output_dir.glob("separated.np*"))[0]
        fissa_output = np.load(fissa_output_file, allow_pickle=True)

        # Old and new FISSA outputs are stored differently
        # Two versions can be distinguised with the output file suffix.
        # The new version infers non-zero traces; therefore no need for dff calculation.

        info, mixmat, sep, results = fissa_output 
        trace_list = []
        if fissa_output_file.suffix == ".npy":
            for cell_id, result in zip(
                cell_ids, results
            ):  
                trace = result[0][0, :]  # take the 1st `signal` (always 1st `trial`)
                trace_list.append(
                    dict(
                        **key,
                        mask=cell_id,
                        fluo_channel=0,
                        activity_type="f_corrected",
                        activity_trace=trace,
                    )
                )
                dff = calculate_dff(trace)
                trace_list.append(
                    dict(
                        **key,
                        mask=cell_id,
                        fluo_channel=0,
                        activity_type="dff",
                        activity_trace=dff,
                    )
                )
                zscore = calculate_zscore(dff, sampling_rate)
                trace_list.append(
                    dict(
                        **key,
                        mask=cell_id,
                        fluo_channel=0,
                        activity_type="z_score",
                        activity_trace=zscore,
                    )
                )
                ca_events = detect_events(dff, sampling_rate)
                trace_list.append(
                    dict(
                        **key,
                        mask=cell_id,
                        fluo_channel=0,
                        activity_type="ca_events",
                        activity_trace=ca_events,
                    )
                )
        else:
            traces = combine_trials(fissa_output)
            for cell_id, trace in zip(cell_ids, traces):
                trace_list.append(
                    dict(
                        **key,
                        mask=cell_id,
                        fluo_channel=0,
                        activity_type="f_corrected",
                        activity_trace=trace,
                    )
                )
                trace_list.append(
                    dict(
                        **key,
                        mask=cell_id,
                        fluo_channel=0,
                        activity_type="z_score",
                        activity_trace=calculate_zscore(trace),
                    )
                )

        self.insert1(key)
        self.Trace.insert(trace_list)


@schema
class SpikeStat(dj.Computed):
    """Spike Statistics

    Attributes:
        Activity (foreign key): Primary key from Activity.
    """

    definition = """
    -> Activity
    """

    class Trace(dj.Part):
        """Deconvolve the neuropil corrected calcium traces with OASIS to infer the spikes,
        and calculate inter-event interval and area under the spike.

        Attributes:
            SpikeStat (Primary key): Parent key from SpikeStat
            Activity.Trace (Foreign key): Primary key from Activity.Trace.
            spikes (longblob): Discretized deconvolved neural activity.
            inferred_trace (longblob): Inferred fluorescence trace.
            baseline (float): Inferred baseline calcium strength.
            lambda (float): Optimal Lagrange multiplier for noise constraint (sparsity parameter).
            g (float): Parameters of the autoregressive model, cardinality equivalent to p.
            interevent_interval (longblob): Interevent interval calculated from the spikes.
            area_under_spike (longblob): Area under the spike.
        """

        definition = """
        -> master
        -> Activity.Trace
        ---
        spikes: longblob                # Discretized deconvolved neural activity.
        inferred_trace: longblob        # Inferred fluorescence trace.
        baseline: float                 # Inferred baseline calcium strength.
        lambda: float                   # Optimal Lagrange multiplier for noise constraint (sparsity parameter).
        g: float                        # Parameters of the autoregressive model, cardinality equivalent to p.
        interevent_interval: longblob   # Interevent interval calculated from the spikes.
        area_under_spike: longblob      # Area under the spike.
        """

    def make(self, key):
        from oasis.functions import deconvolve

        trace_keys, Fcorrecteds = np.stack(
            (Activity.Trace & key & "activity_type='Fcorrected'").fetch(
                "KEY", "activity_trace"
            )
        )
        fps = (scan.ScanInfo & key).fetch1("fps")

        entry = []
        for trace_key, Fcorrected in zip(trace_keys, Fcorrecteds):
            c, s, b, g, lam = deconvolve(Fcorrected, penalty=0, optimize_g=5)

            interevent_interval = np.diff(np.where(s > 1e-3)[0])
            interevent_interval = interevent_interval[interevent_interval > 1] / fps

            spike_times = np.where(s > 1e-3)[0]
            grps = np.split(spike_times, np.where(np.diff(spike_times) != 1)[0] + 1)
            area_under_spike = np.array([s[grp].sum() for grp in grps]) / fps

            entry.append(
                {
                    **trace_key,
                    "spikes": s,
                    "inferred_trace": c,
                    "baseline": b,
                    "g": g,
                    "lambda": lam,
                    "interevent_interval": interevent_interval,
                    "area_under_spike": area_under_spike,
                }
            )

        self.insert1(key)
        self.Trace.insert(entry)


# ---------------- HELPER FUNCTIONS ----------------


_table_attribute_mapper = {
    "ProcessingTask": "processing_output_dir",
    "Curation": "curation_output_dir",
}


def get_loader_result(key: dict, table: dj.Table) -> Callable:
    """Retrieve the processed imaging results from a suite2p or caiman loader.

    Args:
        key (dict): The `key` to one entry of ProcessingTask or Curation
        table (dj.Table): A datajoint table to retrieve the loaded results from (e.g.
            ProcessingTask, Curation)

    Raises:
        NotImplementedError: If the processing_method is different than 'suite2p' or
            'caiman'.

    Returns:
        A loader object of the loaded results (e.g. suite2p.Suite2p or caiman.CaImAn,
        see element-interface for more information on the loaders.)
    """
    method, output_dir = (ProcessingParamSet * table & key).fetch1(
        "processing_method", _table_attribute_mapper[table.__name__]
    )

    output_path = find_full_path(get_imaging_root_data_dir(), output_dir)

    if method == "suite2p" or (
        method == "extract" and table.__name__ == "MotionCorrection"
    ):
        from element_interface import suite2p_loader

        loaded_dataset = suite2p_loader.Suite2p(output_path)
    elif method == "caiman":
        from element_interface import caiman_loader

        loaded_dataset = caiman_loader.CaImAn(output_path)
    elif method == "extract":
        from element_interface import extract_loader

        loaded_dataset = extract_loader.EXTRACT(output_path)
    else:
        raise NotImplementedError("Unknown/unimplemented method: {}".format(method))

    return method, loaded_dataset
