# GRID BREAKER
Grid Breaker add-on for Blender 2.93 for generating grid-like meshes with controllable randomizied parameters

# WARNING
This is readme for the PRE-ALPHA version of the add-on. To be updated soon.
USE IT AT YOUR OWN RISK! The author of this program assumes no responsibility for any issues caused by the use of this add-on, in any way. It hasn't been tested much, it may currently work pretty slow and it may be strongly modified in the future.

# INSTALL
* download zip-file (don't unpack)
* Open Blender
* Edit > Preferences > Add-ons > Install >
* Specify the path to the zip file on your computer
* Press Install Add-on button
* Enable Chekbox near its name

# CONTROLS
# Main
* +,- *(buttons)*: add and remove grid systems. New mesh is generated immediately. All other cobtrols appears after new Grid Breaker mesh is created
* live update *(checkbox)*: update mesh in real-time when changing properties
* update *(button)*: appears only wnen live update is turned off. Updates geometry according to the given parameters 
* name *(text)*: Grid Breaker object name
* collection : Blender Collection to store Grid Breaker object to
# Geometry
* Top, Bottom and Sides checkboxes: determine which parts of the mesh will be generated. Currently Sides can be generated only if both Top and Bottom parts of the mesh are generated
# Grid Settings
* size : object initial size on x and y axises (z axis is determined by the height value down below)
* cells : a number of initial cells
* seed : subdivision distribution random seed
* margin : faces margin factor. 0 - no faces, 1 - no margin
* cuts : sundivision distribution recursion depth
* distribution : how many faces on each recursion level will be subdivided
* height_seed : random seed for height randomness
* height : the maximum height of the Grid Breaker object
* random_height : amount of height randomness
# Materials
Add materials which will be randomly used for each grid cell
