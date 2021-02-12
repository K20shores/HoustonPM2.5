
vars="WIND TEMP PRESS RH_DP"
for var in $vars
do
    for year in {2010..2020}
    do
        wget "https://aqs.epa.gov/aqsweb/airdata/daily_${var}_${year}.zip" & 
    done
done
