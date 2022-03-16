for file in $(ls /home/rupnik/macocu/task8/data);
do
    ffmpeg -i "/home/rupnik/macocu/task8/data/$file" -ar 16000 "data_16000/$file";
done