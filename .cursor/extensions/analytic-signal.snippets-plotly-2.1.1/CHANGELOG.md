# Change Log

All notable changes to the "snippets-plotly" extension will be documented in this file.

## [2.1.1] 2025-06-18

- Rename `px-update-scene-aspect` to `px-update-scene-aspectratio` to better reflect what it does.  

- Add `px-update-scene-aspectmode` to control the method used to determine a 3d scene's aspect ratios.

- Add `px-update-scattermatrix-visible` to control the visibility of scattermatrix parts.

## [2.1.0] 2025-06-10

- Add an overview snippet tree hierarchy to the reference section.

## [2.0.8] 2025-05-13

- Add `px-update-scene-aspect` to modify the aspect ratio of 3d scenes.

## [2.0.7] 2024-09-12 

- Remove node_modules folder from .vsix

## [2.0.6] 2024-09-11  

- Fix typo in `px-args-map_style`

## [2.0.5] 2024-09-11  

Changes concerning Plotly's move from Mapbox to Maplibre-based map rendering announced on 30th August, 2024. Mapbox-based figures will be marked as deprecated from Plotly.py version 5.24 and completely retired by the end of 2024.  

- Add `px-fig-scatter_map` snippet for a Plotly Express `scatter_map` figure. This will ultimately replace `px-fig-scatter_mapbox`.
- Add `px-fig-line_map` snippet for a Plotly Express `line_map` figure. This will ultimately replace `px-fig-line_mapbox`.
- Add `px-fig-choropleth_map` snippet for a Plotly Express `choropleth_map` figure. This will ultimately replace `px-fig-choropleth_mapbox`.
- Add `px-fig-density_map` snippet for a Plotly Express `density_map` figure. This will ultimately replace `px-fig-density_mapbox`.
- Add `px-args-map_style` snippet to present map style options. This will ultimately replace `px-args-mapbox_style`.  


## [2.0.4] 2024-02-27  

- Add `px-read-parquet` snippet to read tabular data stored in a Parquet format (`.parquet`) file.  
- Add `px-read-feather` snippet to read tabular data stored in a Feather format (`.feather`) file.  
- Modify formatting of `px-update-margin` snippet so that it runs across several lines.

## [2.0.3] 2023-12-24 

- Add complete set of map projections to the `px-args-projections` snippet.  

## [2.0.2] 2023-12-24  

- Add new marker symbols (`arrow` and `arrow-wide`) to `px-args-marker_symbol` and `px-update-marker` snippets.  

## [2.0.1] 2023-04-02

- Fix README.md

## [2.0.0] 2023-04-02

- Increment major version number.  

- Change display name to "Plotly Express Snippets".  

- Add `px-read-plotly` snippet to read and show a Plotly figure from a definition stored in a JSON (`.json`) file. For use when a figure has been previously stored using eg. `fig.write_json("myfile.json")`.

## [1.0.7] 2023-02-19

- Update copyright notices and LICENSE.txt

## [1.0.6] 2023-02-19

- Add `px-args-trendline_options-ols` snippet to choose options for fitting an ordinary least-squares (ols) trendline.
- Add `px-args-trendline_options-lowess` snippet to choose options for fitting a locally weighted regression (lowess) trendline.
- Add `px-args-trendline_scope` snippet to choose a trendline (regression) function scope from list of options.
- Add `px-args-trendline_color_override` snippet to choose a trendline color from list of named CSS options.

## [1.0.5] 2023-01-01

- Add `px-update-rangeselector` snippet to add rangeselector buttons to x-xaxis layout for time series displays.
- Add `px-args-rangeselector-button` snippet to add individual rangeselector buttons to x-axis layout for time series displays.  

## [1.0.4] 2022-01-27  

- Correct typo in README.md

## [1.0.3] 2022-01-26  

- Change trigger naming convention in response to updates to IntelliSense. Convert all separators to `-` instead of `.`
- Add `px-update-autorange` snippet to reverse the direction of the axes of an existing figure.

## [1.0.2] 2021-10-24

- Add `px.update.scaleratio` snippet.

## [1.0.1] 2021-10-20

- Add animated sample gallery and screenshots to README.md.

## [1.0.0] 2021-10-19

- Restructure trigger prefixes as follows:  

    - `px.setup` adds snippets to import modules and set renderer and template defaults.  
    - `px.read` adds snippets to read input data from common file formats.  
    - `px.fig` adds snippets to create instances of all Plotly Express figure types.  
    - `px.update` adds snippets to update styling of an existing figure.  
    - `px.args` adds snippets for function arguments chosen from option lists e.g., colors, marker symbols. For use when creating a new figure or updating an existing one.  

## [0.0.6] 2021-08-25

- Update README.

## [0.0.5] 2021-08-25

- Add snippet for empirical cumulative distribution function (`ecdf`) plot.

## [0.0.4] 2021-07-18

- Fix error in extension name.

## [0.0.3] 2021-07-17

- Update description.

## [0.0.2] 2021-07-17

- Update display name.

## [0.0.1] 2021-07-17

- Initial release.