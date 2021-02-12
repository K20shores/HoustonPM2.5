import xarray as xr

ds = xr.open_mfdataset('*.nc4')
ds.to_netcdf('merged.nc4')
