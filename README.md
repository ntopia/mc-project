# CNN 기반 이미지 분류 가속화

Final Project (4190.414A-001, 2016 Fall)

[Description](project.pdf)

## Performance results of final version

|   |10  |50  |100 |300 |500 |1024|
|---|---:|---:|---:|---:|---:|---:|
|1 node 1 CPU, OpenCL|10.162|50.068|   |   |   |   |
|1 node 1 GPU + 1 CPU, OpenCL|1.862|5.713|10.651|30.367|   |   |
|1 node 4 GPU + 1 CPU, OpenCL|1.334|2.542|4.066|10.257|16.131|31.822|
|4 node 16 GPU + 4 CPU, OpenCL + MPI|1.133|1.474|1.869|3.391|4.904|9.033|
|4 node 16 GPU + 4 CPU, SnuCL|2.258|3.608|4.873|9.816|14.750|27.958|
