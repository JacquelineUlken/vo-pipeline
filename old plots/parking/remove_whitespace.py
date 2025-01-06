input_file = "/home/gaurav07/mini-project-visualize/evo/parking/poses_parking_gt.txt"
output_file = "/home/gaurav07/mini-project-visualize/evo/parking/poses_parking_kitti_gt.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        outfile.write(line.rstrip() + "\n")

