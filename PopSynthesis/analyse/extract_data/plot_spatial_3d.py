import geopandas as gpd
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Tuple, Optional

def create_mock_data(gdf: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates mockup data for bars and a heatmap based on an input GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to generate data for. The index
                                must match the intended bar/heatmap data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - A DataFrame for the bar data (e.g., 'Residential', 'Commercial').
            - A Series for the heatmap data (e.g., 'Land_Value').
    """
    print("Generating mockup data...")
    np.random.seed(42)
    
    # Create a new DataFrame for the bar data, using the same index as the GeoDataFrame
    bar_df = pd.DataFrame(index=gdf.index)
    bar_df['Residential'] = np.random.exponential(40, size=len(gdf)) + 10
    bar_df['Commercial'] = np.random.exponential(30, size=len(gdf)) + 5
    bar_df['Industrial'] = np.random.exponential(20, size=len(gdf)) + 5

    # Create a new Series for the heatmap data, also with the same index
    heatmap_series = pd.Series(
        np.random.uniform(500, 2500, size=len(gdf)),
        index=gdf.index,
        name='Land_Value'
    )
    
    return bar_df, heatmap_series

def plot_3d_map(
    map_gdf: gpd.GeoDataFrame, 
    bar_df: pd.DataFrame, 
    heatmap_series: pd.Series, 
    simplify_tolerance: Optional[float] = 1e-4,
    save_to_html: bool = False
) -> pdk.Deck:
    """
    Generates a generic, interactive 3D map with a heatmap and multi-bar layers.

    Args:
        map_gdf (gpd.GeoDataFrame): A GeoDataFrame with polygon geometries for the base map.
        bar_df (pd.DataFrame): A DataFrame where the index matches map_gdf. Each column
                               represents a set of bars to be plotted.
        heatmap_series (pd.Series): A Series where the index matches map_gdf. Its values
                                    determine the color of the heatmap polygons.
        simplify_tolerance (Optional[float]): The tolerance for simplifying geometries. 
                                              Higher values mean more simplification and better performance.
                                              Set to None to disable. Defaults to 1e-4.
        save_to_html (bool, optional): If True, saves the map as an HTML file. Defaults to False.

    Returns:
        pdk.Deck: The pydeck map object, ready for display in a Jupyter Notebook.
    """
    print("Configuring pydeck visualization...")
    
    # --- Data Preparation ---
    bar_df_processed = bar_df.copy()
    if isinstance(bar_df_processed.columns, pd.MultiIndex):
        print("MultiIndex detected in bar_df columns. Flattening automatically.")
        bar_df_processed.columns = ['_'.join(map(str, col)).strip() for col in bar_df_processed.columns.values]

    # OPTIMIZATION: Simplify geometries for performance
    map_gdf_processed = map_gdf.copy()
    if simplify_tolerance:
        print(f"Simplifying geometries with tolerance: {simplify_tolerance}")
        map_gdf_processed.geometry = map_gdf_processed.geometry.simplify(simplify_tolerance)

    full_gdf = map_gdf_processed.join(bar_df_processed).join(heatmap_series)
    
    # Calculate centroids and restructure data for the bars
    full_gdf['lon_centroid'] = full_gdf.geometry.centroid.x
    full_gdf['lat_centroid'] = full_gdf.geometry.centroid.y
    
    long_df = full_gdf.melt(
        id_vars=['lon_centroid', 'lat_centroid', heatmap_series.name],
        value_vars=bar_df_processed.columns,
        var_name='metric',
        value_name='height'
    )

    # --- Colors, Offsets, and Opacity ---
    num_bars = len(bar_df_processed.columns)
    longitude_offset_total = 0.002 * (num_bars - 1) # Increased spacing for thicker bars
    bar_colors = [cm.get_cmap('Set1', num_bars)(i) for i in range(num_bars)]

    metric_props = {}
    for i, col_name in enumerate(bar_df_processed.columns):
        offset = -longitude_offset_total / 2 + i * 0.002
        color = [int(c*255) for c in bar_colors[i][:3]]
        metric_props[col_name] = {'offset': offset, 'color': color}

    long_df['lon'] = long_df['lon_centroid'] + long_df['metric'].apply(lambda m: metric_props[m]['offset'])
    
    # IMPROVEMENT: Calculate variable opacity based on bar height
    min_height, max_height = long_df['height'].min(), long_df['height'].max()
    long_df['alpha'] = 100 + ((long_df['height'] - min_height) / (max_height - min_height) * 155)
    long_df['color_with_alpha'] = long_df.apply(
        lambda row: metric_props[row['metric']]['color'] + [int(row['alpha'])], axis=1
    )

    # Heatmap colors
    norm = mcolors.Normalize(vmin=full_gdf[heatmap_series.name].min(), vmax=full_gdf[heatmap_series.name].max())
    cmap = cm.viridis
    full_gdf['heatmap_color'] = full_gdf[heatmap_series.name].apply(lambda val: [int(c*255) for c in cmap(norm(val))[:3]] + [150])

    # --- Pydeck Layers ---
    view_state = pdk.ViewState(latitude=-37.8136, longitude=144.9631, zoom=9, pitch=50)

    polygon_layer = pdk.Layer('GeoJsonLayer', data=full_gdf, filled=True, stroked=True,
                              get_fill_color='heatmap_color', get_line_color=[150, 150, 150, 200],
                              pickable=True)
                              
    column_layer = pdk.Layer('ColumnLayer', data=long_df, get_position='[lon, lat_centroid]',
                             get_elevation='height', elevation_scale=100, 
                             radius=120, # IMPROVEMENT: Increased bar thickness
                             get_fill_color='color_with_alpha', # Use the new RGBA color
                             pickable=True, auto_highlight=True)
    
    tooltip = {"html": f"<b>Postcode:</b> {{properties.POA_CODE21}}<br/><b>{heatmap_series.name}:</b> {{{heatmap_series.name}:.0f}}"}

    r = pdk.Deck(layers=[polygon_layer, column_layer], initial_view_state=view_state,
                 map_style='mapbox://styles/mapbox/dark-v9', tooltip=tooltip)

    # --- Legend and Output ---
    if save_to_html:
        legend_html = f"""
         <div style="position: fixed; top: 10px; right: 10px; background-color: rgba(34,34,34,0.9);
                     border-radius: 6px; padding: 10px; color: white; font-family: sans-serif;
                     font-size: 12px; z-index: 10; max-width: 200px;">
          <h3 style="margin-top: 0; margin-bottom: 10px; font-weight: bold;">Bar Metrics</h3>"""
        for metric, props in metric_props.items():
            legend_html += f"""<div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 15px; height: 15px; background-color: rgb({props['color'][0]}, {props['color'][1]}, {props['color'][2]}); margin-right: 8px;"></div>
                <span>{metric}</span></div>"""
        
        min_val, max_val = full_gdf[heatmap_series.name].min(), full_gdf[heatmap_series.name].max()
        gradient_colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, 10)]
        legend_html += f"""
          <h3 style="margin-top: 15px; margin-bottom: 10px; font-weight: bold;">{heatmap_series.name}</h3>
          <div style="width: 100%; height: 20px; background: linear-gradient(to right, {', '.join(gradient_colors)}); border-radius: 3px;"></div>
          <div style="display: flex; justify-content: space-between; font-size: 10px; margin-top: 5px;">
              <span>{min_val:,.0f}</span><span>{max_val:,.0f}</span></div></div>"""

        output_filename = 'melbourne_refactored_map.html'
        print(f"Saving interactive map to '{output_filename}'...")
        final_html = r.to_html(as_string=True).replace('</body>', f'{legend_html}</body>')
        with open(output_filename, 'w') as f:
            f.write(final_html)
        print("Save complete.")

    return r

if __name__ == '__main__':
    # --- This block demonstrates how to use the new functions ---
    
    # 1. Load the base map data
    print("Loading base map shapefile...")
    folder_loc = r"C:\\Users\\dlaa0001\\Documents\\PhD\\PopSyn_Monash\\PopSynthesis\\analyse\\data\\synpop_related"
    shapefile_path = os.path.join(folder_loc, "POA_2021_AUST_GDA2020_SHP.zip")
    
    try:
        master_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        master_gdf = master_gdf.set_index('POA_CODE21') # Set a meaningful index
    except Exception as e:
        print(f"Fatal error loading shapefile: {e}")
        exit()

    # Filter for Melbourne metro postal codes
    ls_metro_mel_raw = [
        (3000, 3211), (3335, 3336), (3338, 3338), (3427, 3429),
        (3750, 3752), (3754, 3755), (3759, 3761), (3765, 3775),
        (3781, 3787), (3788, 3815), (3910, 3920), (3926, 3944),
        (3975, 3978), (3980, 3980)
    ]
    ls_metro_poa_all = [str(i) for r in ls_metro_mel_raw for i in range(r[0], r[1] + 1)]
    melbourne_gdf = master_gdf[master_gdf.index.isin(ls_metro_poa_all)].copy()

    # 2. Create mockup data using the dedicated function
    mock_bar_data, mock_heatmap_data = create_mock_data(melbourne_gdf)
    
    # 3. Call the generic plotting function
    # In a real Jupyter notebook, this would display the map directly.
    # Here, we set save_to_html=True to generate the output file.
    plot_3d_map(melbourne_gdf, mock_bar_data, mock_heatmap_data, save_to_html=True)
    
    print("\n--- Main script finished ---")
