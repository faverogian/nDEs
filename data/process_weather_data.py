# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:35:18 2024

@author: Tanaka Akiyama
"""

import netCDF4 as nc
import os

'''
Prints netcdf file metadata.

Parameters:
file_path - string path to netcdf file
'''

import os
import netCDF4 as nc
import pandas as pd

def calculate_biweekly_average(data):
    # Reshape data to have one row per time step
    reshaped_data = data.reshape(data.shape[0], -1)
    # Calculate bi-weekly averages
    biweekly_average = pd.DataFrame(reshaped_data).rolling(window=336).mean().dropna(how='any')
    return biweekly_average

def save_to_csv(biweekly_average, file_path):
    biweekly_average.to_csv(file_path, header=["Temperature", "Humidity", "Time"], index=False)

def process_netcdf(file_path):
    try:
        dataset = nc.Dataset(file_path)
        temperature = dataset.variables['t'][:]
        humidity = dataset.variables['r'][:]
        time = dataset.variables['time'][:]
        biweekly_avg_temperature = calculate_biweekly_average(temperature)
        biweekly_avg_humidity = calculate_biweekly_average(humidity)
        biweekly_avg_time = calculate_biweekly_average(time)
        biweekly_avg_time.index = pd.to_datetime(biweekly_avg_time.index, unit='h', origin='1900-01-01')
        biweekly_average = pd.concat([biweekly_avg_temperature, biweekly_avg_humidity, biweekly_avg_time], axis=1)
        return biweekly_average
    except Exception as e:
        print("Error:", e)


def print_netcdf_metadata(file_path):
    try:
        dataset = nc.Dataset(file_path)
        print("NetCDF file metadata:")
        print("Variables:")
        for var_name in dataset.variables:
            var = dataset.variables[var_name]
            print("\tVariable name:", var_name)
            print("\tDimensions:", var.dimensions)
            print("\tShape:", var.shape)
            print("\tUnits:", var.units)
            print("\tAttributes:")
            for attr_name in var.ncattrs():
                print("\t\t", attr_name, ":", getattr(var, attr_name))
            print("\n")
        print("Global attributes:")
        for attr_name in dataset.ncattrs():
            print("\t", attr_name, ":", getattr(dataset, attr_name))
    except Exception as e:
        print("Error:", e)
        


# Replace 'your_netcdf_file.nc' with the path to your NetCDF file
netcdf_file_path = "adaptor.mars.internal-1711483951.73343-1890-12-9368dd45-a5b4-44e1-8eea-b823672a0f77.nc"

# Get the directory of the current script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Specify the relative path to the NetCDF file within the folder
netcdf_file_name = "adaptor.mars.internal-1711483951.73343-1890-12-9368dd45-a5b4-44e1-8eea-b823672a0f77.nc"
netcdf_file_path = os.path.join(current_directory, "raw", netcdf_file_name)

# print_netcdf_metadata(netcdf_file_path)


# Replace 'your_netcdf_file.nc' with the path to your NetCDF file
#netcdf_file_path = 'your_netcdf_file.nc'
processed_folder = 'processed'
processed_file_name = 'processed_weather_data.csv'

# Ensure the processed folder exists, if not, create it
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

# Construct the file path for saving processed data
processed_file_path = os.path.join(processed_folder, processed_file_name)

# Process NetCDF file and save to CSV
biweekly_average_data = process_netcdf(netcdf_file_path)
if biweekly_average_data is not None:
    save_to_csv(biweekly_average_data, processed_file_path)
    print("Data saved to:", processed_file_path)
else:
    print("Error processing NetCDF file.")
