from pandas import DataFrame, read_csv
from enum import Enum

import pandas as pd
import numpy as np
import geopandas
import itertools
import pathlib
class PMDataManager:
    def __init__(self):
        # data frames
        self.pmdata = None
        self.parameters = None
        self.units = None
        self.sites = None
        self.roads = None

        # metadata
        self.parameter_code_to_unit_code = None
        self.parameter_codes = None
        # a list that holds the codes that are only PM
        self.pm_codes = []

    def get_pm_data(self):
        if self.pmdata is None:
            self.__read_pm_data()
        return self.pmdata

    def get_parameter(self):
        if self.parameters is None:
            self.__read_parameter_codes()
        return self.parameters

    def get_units(self):
        if self.units is None:
            self.__read_unit_codes()
        return self.units

    def get_sites(self):
        if self.sites is None:
            self.__read_site_codes()
        return self.sites

    def get_roads(self):
        if self.roads is None:
            path = pathlib.Path(__file__) \
                .parent.absolute() \
                .joinpath('../data/houston_roads.zip')
            zip_path = f'zip://{path.as_posix()}!houston_roads/TRANSPORTATION_COMPLETE_STREETS.shp'
            print(zip_path)

            self.roads = geopandas.read_file(zip_path)
        return self.roads

    def parameter_code_to_name(self, code):
        """ Given a paramter code, return the parameter name
        """
        if self.parameters is None:
            self.__read_parameter_codes()
        if isinstance(code, str):
            return code
        return self.parameters[self.parameters.ParameterCode == code].Name.values[0]

    def parameter_name_to_code(self, name):
        """ Given a paramter name, return the parameter code
        """
        if self.parameters is None:
            self.__read_parameter_codes()
        if isinstance(name, int):
            return name
        return self.parameters[self.parameters.Name == name].ParameterCode.values[0]

    def parameter_code_to_unit(self, parameter_code):
        """ Given a parameter code, return the name of the unit for that parameter """
        if self.parameter_code_to_unit_code is None:
            self.__read_pm_data()
        if self.units is None:
            self.__read_unit_codes()
        unit_code = next((p[1] for p in self.parameter_code_to_unit_code if p[0] == parameter_code), None)
        if unit_code is None:
            raise ValueError(f"Could not find a mapping for parameter code {parameter_code} to a unit code")
        return self.units[self.units.UnitCode == unit_code].Units.values[0]
        
    def site_id_to_name(self, site_id):
        """ Given an AQS site id, return the name
        """
        if self.sites is None:
            self.__read_site_codes()
        if isinstance(site_id, str) and site_id in self.sites.SITE_NAME.unique():
            return site_id
        return self.sites[self.sites.AQS_SITE_CD == site_id].SITE_NAME.values[0]

    def site_name_to_id(self, site_id):
        """ Given an AQS site name, return the id
        """
        if self.sites is None:
            self.__read_site_codes()
        return self.sites[self.sites.SITE_NAME == site_id].AQS_SITE_CD.values[0]
    
    def get_site_lat_long(self, site_id):
        """ Given an AQS site id, return the latitude/longitude pair"""
        if self.sites is None:
            self.__read_site_codes()
        site = self.sites[self.sites.AQS_SITE_CD == site_id]
        return site.LAT_DD.values[0], site.LONG_DD.values[0]
    
    def unit_code_to_unit(self, unit_code):
        """Given a unit code, return the unit"""
        if self.units is None:
            self.__read_unit_codes()
        return self.units[self.units.UnitCode == unit_code].Unit.values[0]
    
    def unit_to_code(self, unit):
        """Given a unit name, return the unit code"""
        if self.units is None:
            self.__read_unit_codes()
        return self.units[self.units.Unit == unit].UnitCode.values[0]

    def make_cyclical(self, df, features):
        """ For each feature, convert to a cyclical feature.
        
        This is done by mapping the values to a unit circle.
        The entire range of a feature is shifted down to zero. This is done
        by finding the minimum and subtracting that value from each value in the
        feature column. Then, two columns are added for each feature will be named `feature_cos`, and `feature_sin.
        The feature column will be dropped.
        """
        for feature in features:
            if feature in df.columns.values:
                feature_max = max(df[feature].unique())
                feature_min = min(df[feature].unique())
                df[f'{feature}_cos'] = np.cos(df[feature] - feature_min * 2 * np.pi / feature_max)
                df[f'{feature}_sin'] = np.sin(df[feature] - feature_min * 2 * np.pi / feature_max)
                df = df.drop([feature], axis=1)
            
        return df

    def add_previous_timesteps(self, df, n_last = 5, exclude=['Hour', 'Month', 'DayOfWeek', 'DayOfMonth', 'Date']):
        to_shift = [i for i in df.columns.values if i not in exclude and self.parameter_name_to_code(i) not in self.pm_codes]
        names = list(itertools.product(to_shift, range(1,n_last + 1)))
        for column, number in names:
            df[f'{column}_{number}'] = df[column].shift(periods=number)
        return df

    def rename_columns_to_parameter_names(self, df, exclude=['Hour', 'Month', 'DayOfWeek', 'DayOfMonth', 'Date']):
        mapping = {column: self.parameter_code_to_name(column) for column in df.columns.to_list() if column not in exclude}
        df = df.rename(columns=mapping)
        return df

    def __read_pm_data(self):
        dtypes = {
            'Transaction Type' : 'string',
            'Action' : 'string',
            'State Cd' : 'string',
            'County Cd' : 'string',
            'Site ID' : 'string',
            'Parameter Cd' : 'int64',
            'POC' : 'int64',
            'Dur Cd' : 'int64',
            'Unit Cd' : 'int64',
            'Meth Cd' : 'int64',
            'Date' : 'string',
            'Time' : 'string',
            'Value' :'float64',
            'Null Data Cd' : 'object',
            'Col Freq' :'float64',
            'Mon Protocol ID' :'float64',
            'Qual Cd 1' : 'object',
            'Qual Cd 2' :'object',
            'Qual Cd 3' :'object',
            'Qual Cd 4' :'object',
            'Qual Cd 5' :'object',
            'Qual Cd 6' :'object',
            'Qual Cd 7' :'object',
            'Qual Cd 8' :'object',
            'Qual Cd 9' :'object',
            'Qual Cd 10' : 'object',
            'Alternate MDL' : 'object',
            'Uncertainty Value' : 'object'
            }
        df = read_csv('data/data.zip', dtype=dtypes)
        self.__transform_pm_data(df)   

    def __read_parameter_codes(self):
        dtypes = {
            'Parameter Code' : 'int64',
            'Parameter' : 'string',
            }
        df = read_csv('data/parameters.zip', dtype=dtypes)
        self.parameters = df.rename(columns=
            {
                'Parameter Code' : 'ParameterCode',
                'Parameter' : 'Name'
            })

    def __read_unit_codes(self):
        df = read_csv('data/units.zip')
        self.units = df.rename(columns=
            {
                'Unit Code' : 'UnitCode',
            })

    def __read_site_codes(self):
        dtypes = {
            'AQS_SITE_CD' : 'int64',
            }
        self.sites = read_csv('data/sites.zip', dtype=dtypes).sort_values(by='SITE_ID')

    def __extract_metadata(self, df):
        if self.parameters is None:
            self.__read_parameter_codes()
        self.parameter_code_to_unit_code = list(df.groupby(['Parameter Cd', 'Unit Cd']).groups.keys())
        self.parameter_codes = df['Parameter Cd'].unique()
        for code in self.parameter_codes:
            name = self.parameters[self.parameters.ParameterCode == code].Name.values[0]
            if name.lower().find("pm2.5") >= 0:
                self.pm_codes.append(code)

    def __transform_pm_data(self, df):
        """  Transform the dictionary

            Changes the pm data to be a dictionary whose key is the site id and whose value is a dataframe where the columns
            represent the value of each parameter. The column names will identify the parameter.
            If a site does not have all of the paramters, the site's key is None
        """
        # convert the dates to datetime
        df.Date = pd.to_datetime(df.Date + ' ' + df.Time, format='%Y%m%d %H:%M')
        df['Site ID'] = df['State Cd'] + df['County Cd'] + df['Site ID']
        df['Site ID'] = pd.to_numeric(df['Site ID'])
        df = df.set_index('Date')

        self.__extract_metadata(df)

        df = df.drop(
            [
            'Transaction Type',
            'Action',
            'State Cd',
            'County Cd',
            'Dur Cd',
            'Meth Cd',
            'Mon Protocol ID',
            'Null Data Cd',
            'Col Freq',
            'Qual Cd 1',
            'Qual Cd 2',
            'Qual Cd 3',
            'Qual Cd 4',
            'Qual Cd 5',
            'Qual Cd 6',
            'Qual Cd 7',
            'Qual Cd 8',
            'Qual Cd 9',
            'Qual Cd 10',
            'Alternate MDL',
            'Uncertainty Value',
            'Time',
            'Unit Cd',
            'POC'
            ], 
            axis = 1
        )
        df = df.rename(columns=
            {
                'Parameter Cd' : 'ParameterCode',
                'Site ID' : 'SiteID'
            })

        data_by_site = [
            (
                df[df.SiteID == site],
                site
            ) 
            for site in df.SiteID.unique()
        ]

        transformed_data_by_site = {}
        for data, site in data_by_site:
            data_by_parameter = [
                    data[data.ParameterCode == code]
                    .sort_values(by='Date')
                    .rename(columns=
                        {
                            'Value': code
                        }
                    )
                    .drop(
                        [
                            'SiteID',
                            'ParameterCode'
                        ],
                        axis = 1
                    )
                    for code in data.ParameterCode.unique()
                ]
            data_by_parameter = [frame for frame in data_by_parameter if not frame.empty]

            all_data = None
            for parameter_data in data_by_parameter:
                if all_data is None:
                    all_data = parameter_data
                else:
                    all_data = all_data.merge(
                        parameter_data,
                        how='outer', left_index=True, right_index=True)
            all_data.reset_index(inplace=True)
            all_data['Hour'] = all_data.Date.dt.hour
            all_data['Month'] = all_data.Date.dt.month
            all_data['DayOfWeek'] = all_data.Date.dt.dayofweek
            all_data['DayOfMonth'] = all_data.Date.dt.day
            transformed_data_by_site[site] = all_data
        self.pmdata = transformed_data_by_site

