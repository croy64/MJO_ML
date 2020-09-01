#!/bin/bash
file='/home/cccr/roxy/panini/JGRJD_PHD_PART_I/DATA/MSE_calc/20CRV3/monthly/MSE_vert_integrated_20CRV3.nc'
cdo selname,MSE_vint ${file} hadisst.nc
cdo ymonmean -selyear,1905/2015 hadisst.nc clim_sst.nc
cdo sub hadisst.nc clim_sst.nc skt_mon_anomalies_1870-2019.nc

rm hadisst.nc clim_sst.nc  had.nc


for i in $(seq 1905 1 2014)
do
l=$((i + 1)) 
cdo selyear,$i skt_mon_anomalies_1870-2019.nc tempx1.nc
cdo selmon,11,12 tempx1.nc tempx11.nc
cdo selyear,$l skt_mon_anomalies_1870-2019.nc tempy1.nc
cdo selmon,1,2,3,4 tempy1.nc tempy11.nc    

cdo mergetime tempx11.nc tempy11.nc jgr1.nc

cdo timmean jgr1.nc mean1_$l.nc

# cdo mergetime mean*.nc mjo_mean$l.nc

rm temp*.nc
rm jgr*.nc

echo $NUM  

done

cdo mergetime mean1*.nc MSE_raw_nov_apr_1905-2015_20CRV3.nc

rm mean*.nc
rm skt_mon_anomalies_1870-2019.nc
