#%%
from daymet_retriver import DayMetRetriever, DayMetManager
import matplotlib.pyplot as plt

manager = DayMetManager(base_dir="./data")

ds = manager.load_netcdf(f"daymet_annual_prcp.nc")

ds['prcp'].mean(dim=['x', 'y']).plot()
plt.savefig("annual_prcp_spatial_avg.png")
plt.show()

ds['prcp'].mean(dim=['time']).plot()
plt.savefig("annual_prcp_temporal_avg.png")
plt.show()

# %%
