import matplotlib.pyplot as plt
import numpy as np
import geopandas
import os

from . models import ModelTypes

class GroupedPlots():
    """ A class that can plot all of the features from a TCEQ site
    """

    def __init__(self, exclude = ['Month', 'Hour', 'Date', 'DayOfWeek', 'DayOfMonth', 'DateOnly']):
        """ initalize the group plot

        param: exclude, a list of column names that should not be plotted
        """
        self.exclude = exclude

    def plot(self, data, pm_data_manager, save = False, show = False):
        """ This will make a gridded plot which displays each feature in the graph

        param: data, a dictionary mapping AQS site ids to a pandas data frame
        param: pm_data_manager, a data manager
        param: save, if true, save the data
        param: show, if true, call plt.show() 
        """
        with plt.style.context('seaborn'):
            for site_id, df in data.items():

                columns = [i for i in df.columns.to_list() if i not in self.exclude and str(i).find('_') == -1]

                # plus one for the axes instance holding the cartopy axes
                n_features = len(columns) + 1
                n_rows = int(np.ceil(np.sqrt(n_features)))
                n_cols = n_rows

                fig = plt.figure(figsize = [6,6], dpi=800)
                for idx, column in enumerate(columns):
                    code = pm_data_manager.parameter_name_to_code(column)
                    unit = pm_data_manager.parameter_code_to_unit(code)
                    name = pm_data_manager.parameter_code_to_name(code)
                    ax = plt.subplot(n_rows, n_cols, idx+1)
                    ax.plot(df.Date, df[column], lw=.1)
                    ax.set_ylabel(unit, fontsize=4)
                    ax.tick_params(axis='x', labelsize=2)
                    ax.tick_params(axis='y', labelsize=2)
                    ax.set_title(name, fontdict={'fontsize':4})
                
                # plot the location
                ax = plt.subplot(n_rows, n_cols, len(columns)+1)
                self.add_site(ax, pm_data_manager, site_id)
                for idx in range(len(columns)+2, n_rows * n_cols + 1):
                    ax = plt.subplot(n_rows, n_cols, idx)
                    ax.set_visible(False)

                fig.suptitle(pm_data_manager.site_id_to_name(site_id))
                fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                if save:
                    folder = f'graphs/{pm_data_manager.site_id_to_name(site_id)}/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    plt.savefig(f"{folder}/graphs.png")
                if show:
                    plt.show()
        plt.close()
    

    def add_site(self, ax, pm_data_manager, site_id):
        """ Given a TCEQ site, plot the location on ax

        param: ax, a maplotlib Axes instance
        param: pm_data_manager, a data manager
        param: site_id, an AQS site id
        """
        # the bounds of houston we are using
        lats = -95.385, -94.99
        lons = 29.550, 29.815
        ax.set_xlim(lats)
        ax.set_ylim(lons)
        pm_data_manager.get_roads().plot(ax=ax, lw=.05)
        lat, lon = pm_data_manager.get_site_lat_long(site_id)
        ax.plot(lon, lat, 'r*', ms=3)
        ax.text(lon, lat+.030, pm_data_manager.site_id_to_name(site_id), ha='center', fontsize=6)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

class GroupedScatterPlots:
    """ A class that plots each feature from a TCEQ site individually
    """

    def __init__(self, exclude = ['Month', 'Hour', 'Date', 'DayOfWeek', 'DayOfMonth', 'DateOnly']):
        """ initalize the group plot

        param: exclude, a list of column names that should not be plotted
        """
        self.exclude = exclude

    def plot(self, data, pm_data_manager, save = False, show = False):
        """ make scatter plots of the data of each feature versus each PM column

        param: data, a dictionary mapping AQS site ids to a pandas data frame
        param: pm_data_manager, a data manager
        param: save, if true, save the data
        param: show, if true, call plt.show() 
        """
        with plt.style.context('seaborn'):
            for site_id, df in data.items():
                pm_codes = pm_data_manager.pm_codes
                columns = [i for i in df.columns.to_list() if i not in pm_codes]

                n_features = len(columns)
                n_rows = int(np.ceil(np.sqrt(n_features)))
                n_cols = n_rows

                for pm_code in pm_codes:
                    if pm_code not in df.columns:
                        continue
                    fig = plt.figure(dpi=800)
                    y_name = pm_data_manager.parameter_code_to_name(pm_code)
                    for idx, column in enumerate(columns):
                        code = pm_data_manager.parameter_name_to_code(column) if column not in self.exclude else column
                        x_name = pm_data_manager.parameter_code_to_name(code) if code not in self.exclude else code
                        x = df[column]
                        y = df[pm_code]
                        ax = plt.subplot(n_rows, n_cols, idx+1)
                        ax.scatter(x, y, s=5)
                        ax.set_xlabel(x_name, fontsize=4)
                        ax.set_ylabel(y_name, fontsize=4)
                        ax.tick_params(axis='x', labelsize=2)
                        ax.tick_params(axis='y', labelsize=2)
                
                    for idx in range(len(columns)+1, n_rows * n_cols + 1):
                        ax = plt.subplot(n_rows, n_cols, idx)
                        ax.set_visible(False)

                    fig.suptitle(pm_data_manager.site_id_to_name(site_id))
                    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                    if save:
                        folder = f'graphs/{pm_data_manager.site_id_to_name(site_id)}/'
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f"{folder}/{y_name}-scatter.png")
                    if show:
                        plt.show()

class IndividualPlots:
    """ A class that plots each feature from a TCEQ site individually
    """

    def __init__(self, exclude = ['Month', 'Hour', 'Date', 'DayOfWeek', 'DayOfMonth', 'DateOnly']):
        """ initalize the group plot

        param: exclude, a list of column names that should not be plotted
        """
        self.exclude = exclude

    def plot(self, data, pm_data_manager, save = False, show = False):
        """ plot the data, but make a separate graph for each column

        param: data, a dictionary mapping AQS site ids to a pandas data frame
        param: pm_data_manager, a data manager
        param: save, if true, save the data
        param: show, if true, call plt.show() 
        """
        with plt.style.context('seaborn'):
            for site_id, df in data.items():
                columns = [i for i in df.columns.to_list() if i not in self.exclude and str(i).find('_') == -1]
                for column in columns:
                    code = pm_data_manager.parameter_name_to_code(column)
                    unit = pm_data_manager.parameter_code_to_unit(code)
                    name = pm_data_manager.parameter_code_to_name(code)
                    _, ax = plt.subplots(1, 1, dpi=600)
                    ax.plot(df.Date, df[column], lw=1)
                    ax.set_ylabel(unit)
                    ax.set_title(name)
                    if save:
                        folder = f'graphs/{site_id}/'
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plt.savefig(f"{folder}/{name.replace('/', '-').replace(' ', '')}.png")
                    if show:
                        plt.show()
        plt.close()

class FeatureImportancePlots:
    """ A class that plots each feature from a TCEQ site individually
    """

    def __init__(self):
        """ initalize the group plot
        """

    def plot(self, models, pm_data_manager, save = False, show = False):
        """ plot the data, but make a separate graph for each column

        param: data, a dictionary mapping AQS site ids to a pandas data frame
        param: pm_data_manager, a data manager
        param: save, if true, save the data
        param: show, if true, call plt.show() 
        """
        with plt.style.context('seaborn'):
            for site_id, site_models in models.items():
                site_name = pm_data_manager.site_id_to_name(site_id)
                for model in site_models:
                    for model_type in [ModelTypes.RandomForestRegressor, ModelTypes.ExtraTreesRegressor, ModelTypes.AdaBoostRegressor]:
                        features = model.x_test.columns
                        importances = model.models[model_type].feature_importances_
                        indices = np.argsort(importances)
                        _ = plt.figure(dpi=600)
                        plt.suptitle(f'Feature Importances for {model.predicted_variable} at {site_name} for model {model_type.name}')
                        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                        plt.yticks(range(len(indices)), features[indices])
                        plt.gca().tick_params(axis='y', labelsize=7)
                        plt.xlabel('Relative Importance')
                        if save:
                            folder = f'graphs/{site_id}/'
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            plt.savefig(f"{folder}/feature-{model.predicted_variable.replace('/', '-').replace(' ', '')}.png")
                        if show:
                            plt.show()
                plt.close()
