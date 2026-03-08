import warnings
import datetime

from cffi.backend_ctypes import long

warnings.filterwarnings("ignore")

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets (e.g., NetCDF, Zarr)
import xarray as xr

from dask.distributed import Client, LocalCluster
from dask import delayed, compute

from scipy.spatial import cKDTree

import pystac_client
import planetary_computer as pc

from datetime import date

def load_terraclimate_dataset():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )

    return ds


def tc_feats(tc_data):
    mean_3m = tc_data.rolling(time=3, center=False).mean()
    mean_12m = tc_data.rolling(time=12, center=False).mean()
    std_3m = tc_data.rolling(time=3, center=False).std()
    std_12m = tc_data.rolling(time=12, center=False).std()
    lag_one_m = tc_data.shift(time=1)
    lag_two_m = tc_data.shift(time=2)
    lag_three_m = tc_data.shift(time=3)
    seasonal_contrast = mean_3m - mean_12m
    z_scored = (tc_data - mean_12m)/std_12m
    first_diff = lag_one_m - lag_two_m
    second_diff = lag_one_m - 2*lag_two_m + lag_three_m

    mean_12m = mean_12m.rename({var:f'12m_mean_{var}' for var in mean_12m.data_vars})
    mean_3m = mean_3m.rename({var:f'3m_mean_{var}' for var in mean_3m.data_vars})

    std_3m = std_3m.rename({var: f'3m_std_{var}' for var in std_3m.data_vars})
    std_12m = std_12m.rename({var: f'12m_std_{var}' for var in std_12m.data_vars})

    lag_one_m = lag_one_m.rename({var: f'1m_lag_{var}' for var in lag_one_m.data_vars})
    lag_two_m = lag_two_m.rename({var: f'2m_lag_{var}' for var in lag_two_m.data_vars})
    lag_three_m = lag_three_m.rename({var: f'3m_lag_{var}' for var in lag_three_m.data_vars})

    seasonal_contrast = seasonal_contrast.rename(
        {var: f'seasonal_contrast_{var}' for var in seasonal_contrast.data_vars})
    z_scored = z_scored.rename({var: f'z_score_{var}' for var in z_scored.data_vars})
    first_diff = first_diff.rename({var: f'1st_diff_{var}' for var in first_diff.data_vars})
    second_diff = second_diff.rename({var: f'2nd_diff_{var}' for var in second_diff.data_vars})

    return xr.merge([mean_3m, mean_12m, std_3m, std_12m, z_scored, first_diff, second_diff, lag_one_m, lag_two_m, lag_three_m, seasonal_contrast])


@delayed
def terraclimate_mapping(feat_ds, lat_i, lon_i, dates, i):
    prior_month = dates - datetime.timedelta(months=1)
    aoi = feat_ds.isel(lat=slice(max(0,lat_i-1), min(lat_i+2, feat_ds.sizes['lat'])), lon=slice(max(0,lon_i-1), min(lon_i+2, feat_ds.sizes['lon'])))
    aoi = aoi.sel(time=prior_month)
    aoi_agg = aoi.mean(dim=['lat', 'lon'],skipna=True)
    tc_row = aoi_agg.to_array().to_series()
    tc_row['Sample Date'] = dates
    tc_row['Latitude'] = tc_matched_index['Latitude'].iloc[i]
    tc_row['Longitude'] = tc_matched_index['Longitude'].iloc[i]
    return tc_row


if __name__ == '__main__':
    trainset = pd.read_csv('water_quality_training_dataset.csv')
    trainset['Sample Date dt'] = pd.to_datetime(trainset['Sample Date'], dayfirst=True)

    station_ids = trainset[['Latitude', 'Longitude']].drop_duplicates()
    datearrays = trainset.groupby(['Latitude', 'Longitude'],as_index=False)['Sample Date dt'].unique()
    terraclimate = load_terraclimate_dataset()
    print('tc loaded')

    terraclimate_raw_vars = terraclimate[['aet', 'ppt', 'pet', 'def', 'q', 'soil', 'ws', 'tmax', 'tmin', 'pdsi', 'vpd']].sel(
        time=slice('2010-01-31', '2015-12-31'), lat=slice(-21.72, -35.18), lon=slice(14.97, 32.79))

    lat_index = []
    lon_index = []

    grid_lat = terraclimate_raw_vars.lat.values
    grid_lon = terraclimate_raw_vars.lon.values

    station_lats = station_ids['Latitude'].to_numpy()
    station_lons = station_ids['Longitude'].to_numpy()

    lat_index = []
    lon_index = []

    for i in range(len(station_lats)):
        lat_index.append(np.argmin(np.abs(grid_lat - station_lats[i])))
        lon_index.append(np.argmin(np.abs(grid_lon - station_lons[i])))

    tc_matched_index = pd.DataFrame({
        "lat_index": lat_index,
        "lon_index": lon_index
    })

    tc_matched_index = pd.concat([tc_matched_index, datearrays, station_ids],axis=1)
    tc_matched_index = tc_matched_index.rename(columns={'0': 'datearrays'})

    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)

    entire_tc_feats = tc_feats(terraclimate_raw_vars)
    stations_list = []

    for i in range(0, station_ids.shape[0]):
        station_samples = []
        lati = tc_matched_index['lat_index'].iloc[i]
        loti = tc_matched_index['lon_index'].iloc[i]
        datearray = datearrays.iloc[i]
        for dates in datearray:
            sample = delayed(terraclimate_mapping)(entire_tc_feats, lati, loti, dates, i)
            station_samples.append(sample)
        stations_list.append(station_samples)

    flattened = np.array(stations_list).flatten().tolist()
    computed_station_samples = compute(*flattened)
    stations_tc_df = pd.DataFrame(computed_station_samples)
    stations_tc_df.to_csv('tc_test.csv')

