# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:35:18 2024

@author: Tanaka Akiyama
"""

import os
import numpy as np
import netCDF4 as nc
import pandas as pd

# Function to calculate bi-weekly averages
def calculate_biweekly_averages(data):
    # Number of hours in two weeks
    hours_per_biweek = 24 * 14

    # Determine the number of complete bi-weekly segments
    num_biweeks = len(data) // hours_per_biweek

    # Adjust the data to include only the necessary number of hours
    data = data[:num_biweeks * hours_per_biweek]

    # Reshape data to represent bi-weekly segments
    data = data.reshape(-1, hours_per_biweek)

    # Calculate bi-weekly averages
    biweekly_averages = np.mean(data, axis=1)
    
    return biweekly_averages

# Load NetCDF file
def load_netcdf(file_path):
    dataset = nc.Dataset(file_path)
    return dataset

# Main function
def process_netcdf(file_path):
    dataset = load_netcdf(file_path)
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    time = dataset.variables['time'][:]
    # Extract the values corresponding to the third index (middle value)
    middle_lat_index = len(latitude) // 2
    middle_lon_index = len(longitude) // 2
    temperature = dataset.variables['t'][:, middle_lat_index, middle_lon_index]
    humidity = dataset.variables['r'][:, middle_lat_index, middle_lon_index]
    dataset.close()
    
    # Apply scale and offset to temperature and humidity data
    temperature = temperature * 0.0009400897727278146 + 277.5637541982777
    humidity = humidity * 0.0015225149751930131 + 64.0700190197646

    # Calculate bi-weekly averages for temperature and humidity
    temperature_biweekly = calculate_biweekly_averages(temperature)
    humidity_biweekly = calculate_biweekly_averages(humidity)

    # Create DataFrame with temperature, humidity, and index columns
    df = pd.DataFrame({
        'temperature': temperature_biweekly,
        'humidity': humidity_biweekly,
        'index': range(1, len(temperature_biweekly) + 1)  # 1 to 24
    })

    # Save DataFrame to CSV file
    output_folder = 'processed'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'processed_weather_data.csv')
    df.to_csv(output_file, index=False)
    print("Processed data saved to:", output_file)

'''
Prints netcdf file metadata.

Parameters:
file_path - string path to netcdf file
'''
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
netcdf_file_path = os.path.join(current_directory, "raw/weather", netcdf_file_name)

# print_netcdf_metadata(netcdf_file_path)


# Example usage
process_netcdf(netcdf_file_path)

