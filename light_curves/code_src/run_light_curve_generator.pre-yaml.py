import argparse
import json
import sys

sys.path.append("code_src/")
# Lazy-load all other imports to avoid depending on modules that will not actually be used.


def _build_sample(
    *,
    literature_names=["yang", "SDSS"],
    get_sample_kwargs={"SDSS": {"num": 10}},
    consolidate_nearby_objects=True,
    sample_filename="output/object_sample.ecsv",
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
    sample_filename : str
        Name of file to write the sample to.

    Returns
    -------
    sample_table : `~astropy.table.Table`
        Coordinates and labels for objects in the sample.
        This function also writes the sample to an ascii.ecsv file.
    """

    import sample_selection

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
    sample_table = sample_selection.clean_sample(coords, labels, consolidate_nearby_objects=consolidate_nearby_objects)

    # save and return the Table
    sample_table.write(sample_filename, format="ascii.ecsv", overwrite=True)
    print(f"object sample saved to: {sample_filename}", flush=True)
    return sample_table


def _build_lightcurves(
    *,
    mission="gaia",
    sample_filename="output/object_sample.ecsv",
    parquet_dataset_name="output/lightcurves.parquet",
):
    """Fetch data from the mission's archive and build light curves for objects in sample_filename.

    Parameters
    ----------
    sample_filename : str
        Name of the file containing the sample objects.
    parquet_dataset_name : str
        Name of the parquet dataset to write light curves to.
    mission : str
        Name of the mission to query for light curves. Case-insensitive.
        (This will call the function `{mission}_get_lightcurve`.)

    Returns
    -------
    lightcurve_df : MultiIndexDFObject
        Light curves. This function also writes the light curves to a Parquet file.
    """

    import astropy.units as u
    from astropy.table import Table
    from pathlib import Path

    print(f"Building lightcurves from {mission} mission.", flush=True)

    sample_table = Table.read(sample_filename, format="ascii.ecsv")
    arcsec = 1.0 * u.arcsec  # search radius

    # Query the mission's archive and load light curves.
    # We need to know which mission we're doing so that we can send the correct args.
    if mission.lower() == "gaia":
        from gaia_functions import Gaia_get_lightcurve

        verbose = 0
        lightcurve_df = Gaia_get_lightcurve(sample_table, arcsec.to_value("deg"), verbose)

    elif mission.lower() == "heasarc":
        from heasarc_functions import HEASARC_get_lightcurves

        heasarc_catalogs = {"FERMIGTRIG": "1.0", "SAXGRBMGRB": "3.0"}
        lightcurve_df = HEASARC_get_lightcurves(sample_table, heasarc_catalogs)

    elif mission.lower() == "hcv":
        from HCV_functions import HCV_get_lightcurves

        lightcurve_df = HCV_get_lightcurves(sample_table, arcsec.to_value("deg"))

    elif mission.lower() == "icecube":
        from icecube_functions import Icecube_get_lightcurve

        icecube_select_topN = 3
        lightcurve_df = Icecube_get_lightcurve(sample_table, icecube_select_topN)

    elif mission.lower() == "panstarrs":
        from panstarrs import Panstarrs_get_lightcurves

        lightcurve_df = Panstarrs_get_lightcurves(sample_table, arcsec.to_value("deg"))

    elif mission.lower() == "tess_kepler":
        from TESS_Kepler_functions import TESS_Kepler_get_lightcurves

        lightcurve_df = TESS_Kepler_get_lightcurves(sample_table, arcsec.value)

    elif mission.lower() == "wise":
        from WISE_functions import WISE_get_lightcurves

        bandlist = ["W1", "W2"]
        lightcurve_df = WISE_get_lightcurves(sample_table, arcsec, bandlist)

    elif mission.lower() == "ztf":
        from ztf_functions import ZTF_get_lightcurve

        ztf_nworkers = 8
        lightcurve_df = ZTF_get_lightcurve(sample_table, ztf_nworkers)

    else:
        print(f"cannot build light curves. unknown mission '{mission}'")
        return

    filename = Path(f"{parquet_dataset_name}/mission={mission}/part0.snappy.parquet")
    filename.parent.mkdir(parents=True, exist_ok=True)
    lightcurve_df.data.to_parquet(filename)
    print(f"Light curves saved to: {filename}", flush=True)
    return lightcurve_df


def _parse_args(args_list):
    # define the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build", type=str, default="sample", help="Either 'sample' or 'lightcurves'"
    )
    parser.add_argument(
        "--literature_names",
        type=str,
        nargs="+",
        default=["yang", "SDSS"],
        help="Space-separated list of paper names to get sample objects from.",
    )
    parser.add_argument(
        "--get_sample_kwargs",
        type=json.loads,
        default=r'{"SDSS": {"num": 10}}',
        help="json string representing dicts with keyword arguments.",
    )
    parser.add_argument(
        "--consolidate_nearby_objects",
        type=bool,
        default=True,
        help="Produce a unique sample by consolidating nearby objects.",
    )
    parser.add_argument(
        "--mission", type=str, default="gaia", help="Mission name to query for light curves."
    )
    parser.add_argument(
        "--sample_filename",
        type=str,
        default="output/object_sample.ecsv",
        help="Name of the file containing the object sample.",
    )
    parser.add_argument(
        "--parquet_dataset_name",
        type=str,
        default="output/lightcurves.parquet",
        help="Name of the parquet dataset containing the light curves.",
    )

    # parse the script arguments
    return parser.parse_args(args_list)


def run(args_list):
    """Run the function to build either the object sample or the light curves.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments submitted from the command line.
    """
    args = _parse_args(args_list)

    # build and save the sample, if requested
    if args.build == "sample":
        _ = _build_sample(
            literature_names=args.literature_names,
            get_sample_kwargs=args.get_sample_kwargs,
            consolidate_nearby_objects=args.consolidate_nearby_objects,
            sample_filename=args.sample_filename,
        )

    # build and save light curves, if requested
    if args.build == "lightcurves":
        _ = _build_lightcurves(
            mission=args.mission,
            sample_filename=args.sample_filename,
            parquet_dataset_name=args.parquet_dataset_name,
        )


if __name__ == "__main__":
    run(sys.argv[1:])
