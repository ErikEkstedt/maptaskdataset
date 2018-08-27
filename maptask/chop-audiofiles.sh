#!/bin/sh

# Based on
# https://github.com/deepsound-project/samplernn-pytorch/blob/master/datasets/download-from-youtube.sh

# Example:
# chop-audiofiles.sh $DATA_PATH $OUT_PATH 8
# reads data in $DATA_PATH and chops it into 8
# second clips which are stored in $OUT_PATH


dataset_path=$1
out_path=$2
chunk_size=$3

mkdir -p $out_path
num=0
for file in $dataset_path/*.wav
do
	converted=".converted.wav"
	rm -f $converted

	# ffmpeg -i $file -ac 1 -ab 16k -ar 16000 $converted
	ffmpeg -i $file -ac 1 -ar 16000 $converted

	length=$(ffprobe -i $converted -show_entries format=duration -v quiet -of csv="p=0")
	end=$(echo "$length / $chunk_size - 1" | bc)
	echo "splitting..."
	for i in $(seq 0 $end); do
		ffmpeg -hide_banner -loglevel error -ss $(($i * $chunk_size)) -t $chunk_size -i $converted "$out_path/$num.wav"
		num=$(( num+1 ))
	done
	echo "done"
	rm -f $converted
done
