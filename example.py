from daymet_retriver import DayMetRetriever, DayMetManager

if __name__ == "__main__":
    ### Define the bounding box of interest

    bbox = [-127.199707,44.710922,-116.938477,50.553891] # PNW

    timescale = "annual" # "daily", "monthly", or "annual"


    ### Retrieve the DayMet data from the planetary computer
    # Initialize the DayMetRetriever
    retriever = DayMetRetriever(timescale=timescale)

    print("Retrieving DayMet data...")
    ds_raw = retriever.get_daymet_in_bbox(bbox=bbox)

    print(f"DayMet retireved:\n{ds_raw}")

    # take only the prcp for this example,
    # exporting the full dataset is very slow
    ds_simplified = ds_raw[["prcp"]]

    print(f"Data was simplified to annual avg:\n{ds_simplified}")


    ### Save the raw dataset to netCDF
    print("Saving dataset to netCDF...")

    manager = DayMetManager(base_dir="./data")

    output_filename = f"daymet_{timescale}_prcp.nc"
    manager.save_netcdf(ds_simplified, output_filename)

    print("Done!")