import argparse

# import json
import yaml
import sys
from astropy.table import Table
from pathlib import Path
from fractions import Fraction

sys.path.append("../code_src/")
# Lazy-load all other imports to avoid depending on modules that will not actually be used.

KWARG_DEFAULTS_YAML = "../bulk_light_curve_generators/helpers_kwargs_defaults.yml"
KWARG_OPTIONS_YAML = "../bulk_light_curve_generators/_helpers_kwargs_options.yml"


def _build_sample(
    *,
    literature_names,
    consolidate_nearby_objects,
    get_sample_kwargs,
    base_dir,
    sample_filename,
    overwrite_existing_sample,
    **extra_kwargs,
):
    """Build an AGN sample using coordinates from different papers.

    Parameters
    ----------
    sample_filename : str
        Name of the file to write the sample objects to.
    literature_names : list[str]
        Names of papers to get sample objects from. Case-sensitive.
        This will call the function `get_{name}_sample` for every name in literature_names.)
    kwargs : dict[str: dict[str: any]]
        Dict key should be one of literature_names, value should be a dict of keyword arguments
        for the get-sample function.
    base_dir : str
        Base directory for the sample file.
    sample_filename : str
        Name of file to write the sample to.
    overwrite_existing_sample : bool
        Whether to overwrite a preexisting sample file (True) or skip fetching a new sample and return
        the sample on file (False). Has no effect if there is no preexisting file at
        `{base_dir}/{sample_filename}`.

    Returns
    -------
    sample_table : `~astropy.table.Table`
        Coordinates and labels for objects in the sample.
        This function also writes the sample to an ascii.ecsv file.
    """

    import sample_selection

    sample_filepath = Path(base_dir + "/" + sample_filename)
    sample_filepath.parent.mkdir(parents=True, exist_ok=True)

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if sample_filepath.is_file() and not overwrite_existing_sample:
        print(f"Using existing object sample at: {sample_filepath}", flush=True)
        return Table.read(sample_filepath, format="ascii.ecsv")

    # else fetch the sample
    if literature_names == "core":
        literature_names = _load_yaml(KWARG_OPTIONS_YAML)["literature_names_core"]
    print(f"Building object sample from literature: {literature_names}", flush=True)

    # create list of tuples, (get-sample function, kwargs dict)
    get_sample_functions = [
        (getattr(sample_selection, f"get_{name}_sample"), get_sample_kwargs.get(name, {}))
        for name in literature_names
    ]

    # iterate over the functions and get the samples
    coords, labels = [], []
    for get_sample_fnc, kwargs in get_sample_functions:
        get_sample_fnc(coords, labels, **kwargs)

    # create an astropy Table of objects
    sample_table = sample_selection.clean_sample(
        coords, labels, consolidate_nearby_objects=consolidate_nearby_objects
    )

    # save and return the Table
    sample_table.write(sample_filepath, format="ascii.ecsv", overwrite=True)
    print(f"object sample saved to: {sample_filepath}", flush=True)
    return sample_table


def _build_lightcurves(
    *,
    mission,
    mission_kwargs,
    base_dir,
    sample_filename,
    parquet_dataset_name,
    overwrite_existing_data,
    **extra_kwargs,
):
    """Fetch data from the mission's archive and build light curves for objects in sample_filename.

    Parameters
    ----------
    mission : str
        Name of the mission to query for light curves. Case-insensitive.
        (This will call the function `{mission}_get_lightcurve`.)
    base_dir : str
        Base directory for the sample file and parquet dataset.
    sample_filename : str
        Name of the file containing the sample objects.
    parquet_dataset_name : str
        Name of the parquet dataset to write light curves to.
    overwrite_existing_data : bool
        Whether to overwrite an existing data file (True) or skip building light curves if a file
        exists (False). Has no effect if there is no preexisting data file for this mission.

    Returns
    -------
    lightcurve_df : MultiIndexDFObject
        Light curves. This function also writes the light curves to a Parquet file.
    """

    import astropy.units as u

    print(f"Building lightcurves from mission: {mission}.", flush=True)

    sample_filepath = base_dir + "/" + sample_filename
    parquet_filepath = Path(
        f"{base_dir}/{parquet_dataset_name}/mission={mission}/part0.snappy.parquet"
    )
    parquet_filepath.parent.mkdir(parents=True, exist_ok=True)

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if parquet_filepath.is_file() and not overwrite_existing_data:
        import pandas as pd
        from data_structures import MultiIndexDFObject

        print(f"Using existing light curve data at: {parquet_filepath}", flush=True)
        return MultiIndexDFObject(data=pd.read_parquet(parquet_filepath))

    # else load the sample and fetch the light curves
    sample_table = Table.read(sample_filepath, format="ascii.ecsv")
    # arcsec = 1.0 * u.arcsec  # search radius

    # get dict of keyword arguments
    mission_low = mission.lower()
    mission_kwargs_tmp = _load_yaml(KWARG_DEFAULTS_YAML)["missions_kwargs"].get(mission_low, {})
    mission_kwargs_tmp.update(**mission_kwargs)
    # convert radius to float
    my_mission_kwargs = {
        key: (float(Fraction(val)) if key.endswith("radius") else val)
        for key, val in mission_kwargs_tmp.items()
    }

    # Query the mission's archive and load light curves.
    # We need to know which mission we're doing so that we can send the correct args.
    if mission_low == "gaia":
        from gaia_functions import Gaia_get_lightcurve

        # my_mission_kwargs["search_radius"] = float(my_mission_kwargs["search_radius"])
        lightcurve_df = Gaia_get_lightcurve(sample_table, **my_mission_kwargs)

    elif mission_low == "heasarc":
        from heasarc_functions import HEASARC_get_lightcurves

        # heasarc_catalogs = {"FERMIGTRIG": "1.0", "SAXGRBMGRB": "3.0"}
        lightcurve_df = HEASARC_get_lightcurves(sample_table, **my_mission_kwargs)

    elif mission_low == "hcv":
        from HCV_functions import HCV_get_lightcurves

        # my_mission_kwargs["radius"] = float(my_mission_kwargs["radius"])
        # lightcurve_df = HCV_get_lightcurves(sample_table, arcsec.to_value("deg"))
        lightcurve_df = HCV_get_lightcurves(sample_table, **my_mission_kwargs)

    elif mission_low == "icecube":
        from icecube_functions import Icecube_get_lightcurve

        # icecube_select_topN = 3
        lightcurve_df = Icecube_get_lightcurve(sample_table, **my_mission_kwargs)

    elif mission_low == "panstarrs":
        from panstarrs import Panstarrs_get_lightcurves

        # lightcurve_df = Panstarrs_get_lightcurves(sample_table, arcsec.to_value("deg"))
        lightcurve_df = Panstarrs_get_lightcurves(sample_table, **my_mission_kwargs)

    elif mission_low == "tess_kepler":
        from TESS_Kepler_functions import TESS_Kepler_get_lightcurves

        # lightcurve_df = TESS_Kepler_get_lightcurves(sample_table, arcsec.value)
        lightcurve_df = TESS_Kepler_get_lightcurves(sample_table, **my_mission_kwargs)

    elif mission_low == "wise":
        from WISE_functions import WISE_get_lightcurves

        my_mission_kwargs["radius"] = my_mission_kwargs["radius"] * u.arcsec
        # bandlist = ["W1", "W2"]
        # lightcurve_df = WISE_get_lightcurves(sample_table, arcsec, bandlist)
        lightcurve_df = WISE_get_lightcurves(sample_table, **my_mission_kwargs)

    elif mission_low == "ztf":
        from ztf_functions import ZTF_get_lightcurve

        my_mission_kwargs["match_radius"] = my_mission_kwargs["match_radius"] * u.deg
        # ztf_nworkers = 8
        # lightcurve_df = ZTF_get_lightcurve(sample_table, ztf_nworkers)
        lightcurve_df = ZTF_get_lightcurve(sample_table, **my_mission_kwargs)

    else:
        raise ValueError(f"Unknown mission '{mission}'")

    lightcurve_df.data.to_parquet(parquet_filepath)
    print(f"Light curves saved to: {parquet_filepath}", flush=True)
    return lightcurve_df


def _build_other(keyword, kwargs_yaml):
    kwarg_options = _load_yaml(kwargs_yaml) if kwargs_yaml else _load_yaml(KWARG_OPTIONS_YAML)
    kwarg_options.update(_load_yaml(KWARG_DEFAULTS_YAML))

    # if this was called from the command line, we need to print the value so it can be captured by the script
    # this is indicated by a "+" flag appended to the keyword
    print_scalar, print_list = keyword.endswith("+"), keyword.endswith("+l")
    keyword = keyword.strip("+l").strip("+")

    if keyword == "base_dir":
        basedir = kwarg_options.get("base_dir")
        # sample_file = basedir + "/" + kwarg_options.get('sample_filename')
        # parquet_dir = basedir + "/" + kwarg_options.get('parquet_dataset_name')
        # other = [basedir, sample_file, parquet_dir]
        other = basedir
    else:
        other = kwarg_options[keyword]

    if print_scalar:
        print(other)
    if print_list:
        print(" ".join(other))

    return other


def run(build, *, kwargs_yaml=None, **kwargs_dict):
    if build not in ["sample", "lightcurves"]:
        return _build_other(keyword=build, kwargs_yaml=kwargs_yaml)

    my_kwargs = _load_yaml(KWARG_DEFAULTS_YAML)
    if kwargs_yaml:
        kwargs_dict.update(_load_yaml(kwargs_yaml))
    my_kwargs.update(kwargs_dict)

    if build == "sample":
        return _build_sample(**my_kwargs)

    if build == "lightcurves":
        mission_kwargs = my_kwargs["missions_kwargs"].get(my_kwargs["mission"], {})
        return _build_lightcurves(**my_kwargs, mission_kwargs=mission_kwargs)


def _load_yaml(fyaml):
    with open(fyaml, "r") as fin:
        kwargs = yaml.safe_load(fin)
    return kwargs


def _run_main(args_list):
    """Run the function to build either the object sample or the light curves.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments submitted from the command line.
    """
    args = _parse_args(args_list)
    # print('EXTRA ', args.extra_kwargs)
    # parse extra kwargs into a dict, then convert true/false to bool
    kwargs_dict_tmp = {kwarg.split("=")[0]: kwarg.split("=")[1] for kwarg in args.extra_kwargs}
    kwargs_dict = {
        key: (bool(val) if val.lower() in ["true", "false"] else val)
        for (key, val) in kwargs_dict_tmp.items()
    }

    # print('DICT ', kwargs_dict)
    # run(args.build, kwargs_yaml=args.kwargs_yaml, mission=args.mission)
    run(args.build, kwargs_yaml=args.kwargs_yaml, kwargs_dict=kwargs_dict)


def _parse_args(args_list):
    # define the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        type=str,
        default="sample",
        help="Either 'sample', 'lightcurves', or a key in _helpers_kwargs_options.yml",
    )
    parser.add_argument(
        "--kwargs_yaml",
        type=str,
        default="../bulk_light_curve_generators/helpers_kwargs_defaults.yml",
        help="Path to a yaml file containing the function keyword arguments to be used.",
    )
    # parser.add_argument(
    #     "--mission", type=str, default=None, help="Mission name to query for light curves."
    # )
    parser.add_argument(
        # "--extra_kwargs", type=str, default=list(), help="."
        "--extra_kwargs",
        type=str,
        default=list(),
        nargs="*",
        help=".",
    )

    # parse the script arguments
    return parser.parse_args(args_list)


if __name__ == "__main__":
    _run_main(sys.argv[1:])
