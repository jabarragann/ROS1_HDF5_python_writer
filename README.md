# Ros-hdf5 python saver

Package to save data from rostopics into HDF5 format

## Getting started

### Installation



## Visualizing HDF5 files

The package [h5web][1] can be used from a web browser or as a visual studio code extension to visualize the HDF5 files.

[1]: https://github.com/silx-kit/h5web

## Related projects:

- [Rosbag2HDF5](https://github.com/strawlab/bag2hdf5/tree/master)

## Limitations:

- Writer works well with image of 640x480 pixels or smaller. Higher resolutions take to long to write making the data queue to grow.

## TODO:

- [ ] Create YAML configuration file. 
