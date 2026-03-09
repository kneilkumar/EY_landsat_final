import warnings
warnings.filterwarnings("ignore")

# Standard libraries
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Multi-dimensional data
import xarray as xr

# Dask for parallelism
from dask.distributed import Client, LocalCluster
from dask import delayed, compute

# Planetary Computer STAC access
import pystac_client
import planetary_computer as pc


def load_terraclimate_dataset():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields.get("xarray:open_kwargs", {})
        )
    return ds

def tc_feats(tc_data):
    # Rolling means and std
    mean_3m = tc_data.rolling(time=3, center=False).mean()
    mean_12m = tc_data.rolling(time=12, center=False).mean()
    std_3m = tc_data.rolling(time=3, center=False).std()
    std_12m = tc_data.rolling(time=12, center=False).std()

    # Lags
    lag1 = tc_data.shift(time=1)
    lag2 = tc_data.shift(time=2)
    lag3 = tc_data.shift(time=3)

    # Other transformations
    seasonal_contrast = mean_3m - mean_12m
    z_scored = (tc_data - mean_12m) / std_12m
    first_diff = lag1 - lag2
    second_diff = lag1 - 2*lag2 + lag3

    # Rename vars
    def rename_vars(ds, prefix): return ds.rename({v: f"{prefix}_{v}" for v in ds.data_vars})

    merged = xr.merge([
        rename_vars(mean_3m, '3m_mean'),
        rename_vars(mean_12m, '12m_mean'),
        rename_vars(std_3m, '3m_std'),
        rename_vars(std_12m, '12m_std'),
        rename_vars(lag1, '1m_lag'),
        rename_vars(lag2, '2m_lag'),
        rename_vars(lag3, '3m_lag'),
        rename_vars(seasonal_contrast, 'seasonal_contrast'),
        rename_vars(z_scored, 'z_score'),
        rename_vars(first_diff, '1st_diff'),
        rename_vars(second_diff, '2nd_diff')
    ])
    return merged

@delayed
def terraclimate_mapping(feat_ds, lat_i, lon_i, dates, lat_val, lon_val):
    prior_month = dates - relativedelta(months=1)
    lat_slice = slice(max(0, lat_i-1), min(lat_i+2, feat_ds.sizes['lat']))
    lon_slice = slice(max(0, lon_i-1), min(lon_i+2, feat_ds.sizes['lon']))
    aoi = feat_ds.isel(lat=lat_slice, lon=lon_slice)
    # Select the prior month
    aoi = aoi.sel(time=prior_month)
    # Aggregate spatially
    aoi_agg = aoi.mean(dim=['lat', 'lon'], skipna=True)
    # Convert to pandas series
    tc_row = aoi_agg.to_array().to_series()
    tc_row['Sample Date'] = dates
    tc_row['Latitude'] = lat_val
    tc_row['Longitude'] = lon_val
    return tc_row

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # Load training stations and dates
    trainset = pd.read_csv('water_quality_training_dataset.csv')
    trainset['Sample Date dt'] = pd.to_datetime(trainset['Sample Date'], dayfirst=True)
    station_ids = trainset[['Latitude', 'Longitude']].drop_duplicates()

    # Get all sample dates per station
    datearrays = trainset.groupby(['Latitude','Longitude'], as_index=False)['Sample Date dt'].unique()

    # Load TerraClimate subset for South Africa
    terraclimate = load_terraclimate_dataset()
    print("TerraClimate dataset loaded.")

    tc_vars = ['aet','ppt','pet','def','q','soil','ws','tmax','tmin','pdsi','vpd']
    terraclimate_raw_vars = terraclimate[tc_vars].sel(
        time=slice('2010-01-31','2015-12-31'),
        lat=slice(-21.72,-35.18),
        lon=slice(14.97,32.79)
    )

    # -----------------------------
    # Map stations to closest grid indices
    # -----------------------------
    grid_lat = terraclimate_raw_vars.lat.values
    grid_lon = terraclimate_raw_vars.lon.values

    lat_index = [np.argmin(np.abs(grid_lat - lat)) for lat in station_ids['Latitude']]
    lon_index = [np.argmin(np.abs(grid_lon - lon)) for lon in station_ids['Longitude']]

    tc_matched_index = station_ids.copy()
    tc_matched_index['lat_index'] = lat_index
    tc_matched_index['lon_index'] = lon_index
    tc_matched_index = tc_matched_index.merge(datearrays, on=['Latitude','Longitude'])

    # -----------------------------
    # Compute all features
    # -----------------------------
    entire_tc_feats = tc_feats(terraclimate_raw_vars)

    # -----------------------------
    # Dask cluster
    # -----------------------------
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)

    # -----------------------------
    # Build delayed tasks
    # -----------------------------
    tasks = []
    for i, row in tc_matched_index.iterrows():
        lati = row['lat_index']
        loti = row['lon_index']
        lat_val = row['Latitude']
        lon_val = row['Longitude']
        for dates in row['Sample Date dt']:
            tasks.append(delayed(terraclimate_mapping)(entire_tc_feats, lati, loti, dates, lat_val, lon_val))

    # Compute all tasks in parallel
    computed_samples = compute(*tasks)
    stations_tc_df = pd.DataFrame(computed_samples)
    stations_tc_df.to_csv('tc_test.csv', index=False)
    print("TerraClimate features saved to tc_test.csv")
