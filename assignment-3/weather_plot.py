import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('Weather Prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import geoviews as gv
import geoviews.feature as gf
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# define schema
tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

# function to read data using spark and return pandas dataframe
def get_data(input_path):
    df_base = spark.read.csv(input_path, schema=tmax_schema)
    return df_base.toPandas()

# get data for a specific year span, group by station and aggregate (max)
def get_data_for_year(df_input, year, span):
    df_input['year'] = df_input['date'].apply(lambda date: str(date).split('-')[0])
    df_filtered = df_input.loc[(df_input['year'] >= str(year - span)) & (df_input['year'] <= str(year + span))]
    df_grouped = df_filtered.groupby(['station'], as_index=False).max()
    df_grouped['label'] = df_grouped['station'] + '<br>' + df_grouped['tmax'].map(str)
    return df_grouped

# function to plot temperature distribution
def plot_map(df, map_title, marker_color_col, use_default_marker = True, default_marker_size = 5, marker_size_col = ''):
    
    if (use_default_marker):
        df['markersize'] = default_marker_size
    else:
        df['markersize'] = df[marker_size_col]
        
    color_scale = [0,"blue"],[0.5, "white"],[1,"red"]

    fig = go.Figure(data=go.Scattergeo(
        lat = df['latitude'],
        lon = df['longitude'],
        text = df['label'],
        marker = dict(
            color = df[marker_color_col],
            colorscale = 'icefire',
            #reversescale = True,
            opacity = 0.5,
            size = df['markersize'],
            colorbar = dict(
                titleside = "right",
                outlinecolor = "rgba(68, 68, 68, 0)",
                ticks = "outside",
                showticksuffix = "last",
                #dtick = 0.1,
            ),
            cmin=-50,
            cmax=+50
        )
    ))

    fig.update_layout(
        geo = dict(
            scope = 'world',
            showland = True, landcolor = "LightGray",
            subunitcolor = "rgb(255, 255, 255)",
            showlakes = True, lakecolor = "Blue",
            showocean = True, oceancolor='LightBlue',
            showsubunits = True,
            showcountries = True, countrycolor='RebeccaPurple',
            resolution = 110,
            projection = dict(
                type = 'miller'
            ),
            fitbounds = 'geojson',
            lonaxis = dict(
                showgrid = True,
                gridwidth = 0.35,
                range= [ -140.0, -55.0 ],
                dtick = 2
            ),
            lataxis = dict (
                showgrid = True,
                gridwidth = 0.35,
                range= [ 20.0, 60.0 ],
                dtick = 2
            )
        ),
        title='<b>' + map_title + '</b>',
    )
    fig.show()

# function to test model
def test_model(model_file, inputs, break_data):
    # get the data
    data = spark.read.csv(inputs, schema=tmax_schema)
    
    # break data to train and test set
    train, test = data.randomSplit([0.7, 0.3]) #Needed for 733 A3
    
    # load the model
    test_model = PipelineModel.load(model_file)
    
    # use the model to make predictions
    if (break_data):
        predictions = test_model.transform(test)
    else:
        predictions = test_model.transform(data)
    
    # return predictions
    return predictions.toPandas()

# function to change model to normal schema
def normal_df_schema(df):
    df['tmax'] = df['prediction']
    df = df.drop(columns=['features', 'prediction'])
    
    return df

# function to calculate prediction error
def calc_errors(df_predictions):

    df_predictions['error'] = abs(df_predictions['prediction'] - df_predictions['tmax'])

    df_predictions['label'] = 'Station: ' + df_predictions['station'] + \
                            '<br>Error: ' + df_predictions['error'].apply(str) + \
                            '<br>Real: ' + df_predictions['tmax'].apply(str) + \
                            '<br>Predicted: ' + df_predictions['prediction'].apply(str) 
    
    return df_predictions

def plot_jet(predictions):
    gv.extension('matplotlib')

    gdp = gv.Dataset(predictions, kdims=['prediction'])
    points = gdp.to(gv.Points, ['longitude', 'latitude'], ['prediction'])

    tiles = gv.tile_sources.Wikipedia

    plt.figure(figsize=(12,6))
    gv.output(
    points.opts(color='prediction', colorbar=True, cmap='bwr', global_extent=True, backend='matplotlib', fig_size=500) * gf.coastline, backend='matplotlib')


if (__name__ == "__main__"):
    # variables
    input_path = 'tmax-2'
    weather_model_name = 'weather_model'
    year1 = 2000
    year2 = 2015

    # read tmax data
    df_base = get_data(input_path)
    print("# of records: %d" % len(df_base))

    # read data for year 2000
    df_year1 = get_data_for_year(df_base, year1, 1)
    print("# of records in year %d: %d" % (year1, len(df_year1)))

    # read data for year 2015
    df_year2 = get_data_for_year(df_base, year2, 1)
    print("# of records in year %d: %d" % (year2, len(df_year2)))

    # plot temperature distributions
    print("now plotting real temperatures over map")
    plot_map(df_year1, 'Real Temperature Distribution, Year=' + str(year1), marker_color_col='tmax', default_marker_size=10)
    plot_map(df_year2, 'Real Temperature Distribution, Year=' + str(year2), marker_color_col='tmax', default_marker_size=10)

    # predict temperature using pre-trained model
    print("using pre-trained model to make predictions")
    predictions = test_model(weather_model_name, input_path, False)

    # normal dataframe
    df_predictions = normal_df_schema(predictions)

    # get predicted data for year 2010
    df_pred_year1 = get_data_for_year(df_predictions, year1, 2)
    print("# of records in year %d: %d" % (year1, len(df_year1)))

    # get predicted data for year 2015
    df_pred_year2 = get_data_for_year(df_predictions, year2, 2)
    print("# of records in year %d: %d" % (year2, len(df_year2)))

    # plot predicted values
    print("now plotting predictions")
    plot_map(df_pred_year1, 'Predicted Temperature Distribution, Year = ' + str(year1), marker_color_col='tmax', default_marker_size=10)
    plot_map(df_pred_year2, 'Predicted Temperature Distribution, Year = ' + str(year2), marker_color_col='tmax', default_marker_size=10)

    # test model on 30% of data
    print("use pre-trained model on 30% of data to calculate error")
    new_predictions = test_model(weather_model_name, input_path, True)

    # get data for year 2010    
    print("now plotting error")
    df_year1 = get_data_for_year(new_predictions, year1, 2)

    # calculate error
    df_errors = calc_errors(df_year1)

    # plot error
    plot_map(df_errors, 'Temperature Prediction Error - Year =' + str(year1), marker_color_col='error', use_default_marker=False, marker_size_col='error')

    # plot density
    plot_jet(predictions)

