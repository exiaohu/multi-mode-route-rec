#!/bin/zsh

ubodt_path="ubodt_bj_small_sample.bin"
network_path="bj_small_roads_sample/bj_small_roads_sample.shp"
gps_folder="trajs"
output_folder="mr"
#gps_path="trajs_gcj02/trajs.shp"
#output_path="trajs_gcj02_mr.csv"

for gps_path in $(ls ${gps_folder}/*.shp); do
  #  echo ${gps_path}
  #  echo ${output_folder}/mr-${gps_path:12:8}.csv
  if [ -f ${output_folder}/mr-${gps_path:12:8}.csv ]; then
    echo "${output_folder}/mr-${gps_path:12:8}.csv exists, skip"
  else
    fmm_omp --ubodt $ubodt_path --network $network_path --gps ${gps_path} -k 4 -r 0.4 -e 0.5 --output ${output_folder}/mr-${gps_path:12:8}.csv --output_fields all --source snodeid --target enodeid
  fi
done
