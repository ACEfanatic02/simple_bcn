# Simple BCn Compressor

This repository is a companion to [this introduction to BCn compression algorithms](https://acefanatic02.github.io/posts/intro_bcn_part1/).  This project is _explicitly not_ a shipping-quality compression tool.  (Or even a low-quality compressor:  compressed results are discarded after evaluating the final error.)

## Test Assets

In order to actually do anything, the `<project_root>/data/` folder must contain raw textures to compress.  Paths to each texture are hardcoded in the `k_test_<type>_file_names[]` arrays.  The test data set used for the results published on the blog use textures from [Intel's version of the Sponza test scene](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html), which are not included in this repository.
