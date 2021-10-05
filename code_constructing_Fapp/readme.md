# Code of constructing Fapp 
### inference

 1. Download the projected facial points (`mengnalisa_source_points.npy`,`mengnalisa_drive_points.npy`,`taile_source_points.npy`,`taile_drive_points.npy`) in [google drive](https://drive.google.com/drive/folders/1OM3AE6rjZKY1v6PVDnv-YwlmkBZOhw1L?usp=sharing).
 2. Put all files into **./test_data**
 4. run 
> python inference.py--reference_projected_mesh_points_path=./test_data/taile_source_points.npy --drive_projected_mesh_points_path=./test_data/taile_drive_points.npy

or
> python inference.py--reference_projected_mesh_points_path=./test_data/mengnalisa_source_points.npy --drive_projected_mesh_points_path=./test_data/mengnalisa_drive_points.npy

to compute Fapp.