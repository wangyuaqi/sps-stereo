#!/bin/bash
cmake ..
make 
mkdir result_bound result_disparity result_label result_plane result_segment sps-result
./spsstereo 2.png 32.png
