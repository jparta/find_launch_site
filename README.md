## find_launch_site

Find a good launch site for small-scale stratospheric balloon launches based on OpenStreetMap data.
The suggested sites are away from tall objects such as buildings, trees, and power lines. 
Currently set to work in the Helsinki area.

See https://github.com/jparta/balloon-flights-planner for an example of usage.

Uses https://github.com/jparta/astra_simulator as trajectory prediction backend.


Here's a visual of the data processing path: 

![find_launch_site_geometry_transformation](https://github.com/jparta/find_launch_site/blob/master/images/find_launch_site_geometry_transformation.png?raw=true)


Here's the launch site suggestion finding algorithm overview:

![find_launch_site_suggestion_algorithm](https://github.com/jparta/find_launch_site/blob/master/images/find_launch_site_suggestion_algorithm.png?raw=true)
