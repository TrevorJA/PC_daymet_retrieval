import planetary_computer
import pystac_client
import pyproj
import pystac
import xarray as xr
import requests

from pathlib import Path
from typing import Union, List, Tuple
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import pandas as pd


# Define Daymet projection for raw data
daymet_proj = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


class DayMetRetriever:
    def __init__(self, 
                 timescale: str = "annual"):
        """
        Initializes the DayMetRetriever with a specified timescale.

        Parameters
        ----------
        timescale : str, optional
            The timescale of the Daymet dataset to retrieve. Defaults to "annual".
        """
        self.collection_name = f"daymet-{timescale}-na"
        self.daymet_proj = daymet_proj
        self.ds = None
        
        # Configure logging to suppress Azure storage messages
        import logging
        logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

    def transform_bbox_to_daymet(self, bbox):
        """
        Transforms a WGS84 bounding box (lon/lat) to Daymet's Lambert Conformal Conic (x/y).
        
        Parameters
        ----------
        bbox : list of float
            Bounding box as [west, south, east, north] in WGS84 (lon/lat).
        
        Returns
        -------
        list of float
            Bounding box transformed to Daymet’s coordinate system: [xmin, ymin, xmax, ymax].
        """
        wgs84 = pyproj.CRS("EPSG:4326")  # WGS84
        daymet_crs = pyproj.CRS(daymet_proj)  # Daymet LCC

        transformer = pyproj.Transformer.from_crs(wgs84, daymet_crs, always_xy=True)

        xmin, ymin = transformer.transform(bbox[0], bbox[1])
        xmax, ymax = transformer.transform(bbox[2], bbox[3])

        return [xmin, ymin, xmax, ymax]


    def get_daymet_in_bbox(self, 
                           bbox: List[float]) -> xr.Dataset:
        """
        Retrieves and returns Daymet data for the specified bounding box.

        Parameters
        ----------
        bbox : list of float
            Bounding box defined as [west, south, east, north] coordinates.

        Returns
        -------
        xr.Dataset
            The retrieved Daymet dataset as an xarray Dataset, containing 3 dimensional (x, y, time)
            data for multiple climate variables.
        """
        if len(bbox) != 4:
            raise ValueError("bbox must contain [west, south, east, north] coordinates")
            
        req = requests.get(
            f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{self.collection_name}"
            )
        
        collection = pystac.Collection.from_dict(req.json())
        
        signed = planetary_computer.sign(collection.assets["zarr-abfs"])
        storage_options = signed.to_dict()['xarray:storage_options']
        
        # get the dataset
        self.ds = xr.open_zarr(signed.href, 
                               consolidated=True,
                               storage_options=storage_options)
        
        
        # subset to the bounding box
        bbox_xy = self.transform_bbox_to_daymet(bbox)
        self.ds = self.ds.sel(x=slice(bbox_xy[0], bbox_xy[2]), 
                              y=slice(bbox_xy[3], bbox_xy[1]))
        
        return self.ds
    
    

class DayMetManager:
    """
    Handles data saving and loadign of Daymet dataset in NetCDF.
    """
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initializes the DayMetManager with a specified base directory.

        Parameters
        ----------
        base_dir : str or Path
            The directory where NetCDF files will be stored.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_netcdf(self, 
                    ds: xr.Dataset, 
                    filename: str,
                    complevel = 2,
                    parallel=False) -> None:
        """
        Saves an xarray Dataset as a NetCDF file.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to be saved.
        filename : str
            The name of the NetCDF file.
        complevel : int, optional
            The compression level to use when saving the NetCDF file. Defaults to 2.
        """
        filepath = self.base_dir / filename
        
        # Define valid encoding parameters for netCDF4
        valid_encoding = {
            'dtype', 'zlib', 'complevel', 'chunksizes', 
            'shuffle', '_FillValue', 'contiguous'
        }
        
        encoding = {}
        for var in ds.variables:
            # Extract original encoding
            original_encoding = ds[var].encoding
            
            # Create new encoding with only valid parameters
            encoding[var] = {
                k: v for k, v in original_encoding.items() 
                if k in valid_encoding
            }
            
            encoding[var]['zlib'] = True
            encoding[var]['complevel'] = complevel # lower compression = faster write speed             
            
            if var == 'time':
                encoding[var]['units'] = original_encoding['units']
            
            # Ensure chunking is properly specified if present
            if 'chunks' in original_encoding:
                # Get the actual dimension sizes
                dim_sizes = ds[var].shape
                
                # Adjust chunksizes to not exceed dimension size
                encoding[var]['chunksizes'] = tuple(
                    min(dim_sizes[i], chunk) for i, chunk in enumerate(original_encoding['chunks'])
                )
        
        ds.to_netcdf(
            filepath,
            encoding=encoding,
            engine='netcdf4' if not parallel else 'h5netcdf',
            format='NETCDF4',
        )
        
    def load_netcdf(self, filename: str) -> xr.Dataset:
        """
        Loads a NetCDF file into an xarray Dataset.

        Parameters
        ----------
        filename : str
            The name of the NetCDF file to be loaded.

        Returns
        -------
        xr.Dataset
            The loaded xarray Dataset.
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at {filepath}")
        return xr.open_dataset(filepath, engine='netcdf4', chunks='auto')





class DayMetProcessor:
    """Handles aggregation of Daymet data across dimensions."""

    daymet_proj = daymet_proj
    

    def subset_daymet_using_catchment(self, 
                                      ds: xr.Dataset, 
                                      catchment_geom) -> xr.Dataset:
        """Subsets Daymet data using a catchment geometry."""
            
        # Transform catchment to Daymet projection
        bounds = catchment_geom.bounds
        
        # Subset and clip dataset
        ds_subset = ds.sel(
            x=slice(bounds[0], bounds[2]),
            y=slice(bounds[3], bounds[1])
        )
        
        ds_subset = ds_subset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        ds_subset.rio.write_crs(self.daymet_proj, inplace=True)
        
        return ds_subset.rio.clip([catchment_geom], drop=True)
    
    @staticmethod
    def aggregate_spatial(ds: xr.Dataset, 
                          method: str = 'mean',
                          variable: str = None) -> pd.DataFrame:
        """Aggregates data across spatial dimensions."""
        if method not in ['mean', 'sum']:
            raise ValueError("Method must be either 'mean' or 'sum'")
            
        
        if method == 'mean':
            val = ds[variable].mean(dim=['x', 'y']).to_numpy()
        else:
            val = ds[variable].sum(dim=['x', 'y']).to_numpy()
            
        return val


    def plot_catchment_daymet_data(self, 
                                   ds_clipped: xr.Dataset, 
                                   catchment_geom: gpd.GeoDataFrame,
                                   catchment_id: str,
                                   variable: str = 'prcp', 
                                   time_agg: str = 'mean',
                                   figsize: Tuple[int, int] = (10, 8),
                                   figdir: str = ".") -> None:
        """Plots Daymet data with catchment overlay."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if time_agg == 'mean':
            ds_clipped[variable].mean(dim=['time']).plot(ax=ax)
        elif time_agg == 'sum':
            ds_clipped[variable].sum(dim=['time']).plot(ax=ax)
            
        catchment_gdf = gpd.GeoDataFrame(geometry=[catchment_geom])
        catchment_gdf.boundary.plot(ax=ax, color='k', linewidth=2)
        
        plt.title(f'{time_agg.capitalize()} {variable} with Catchment Boundary')
        plt.tight_layout()
        plt.savefig(Path(figdir) / f'{catchment_id}_{variable}_{time_agg}_catchment.png')
        plt.close()
        return 


