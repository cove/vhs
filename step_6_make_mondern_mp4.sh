#!/bin/bash
# Modern, high-quality version that uses modern color correction and deinterlacing and audio notch for VHS camera mechanical noise
# Usage: ./step_6_make_modern_mp4.zsh input_file.mkv

if [[ -z "$1" ]]; then
  echo "Usage: $0 input_file.mkv"
  exit 1
fi

IN="$1"
BASE="${IN:t:r}"  # remove path and extension

echo "Processing \"$IN\"..."
ffmpeg -i "$IN" \
  -vf "setfield=bff,yadif=1,colorspace=all=5:iall=5,eq=saturation=0.9:gamma=1.0,crop=in_w-2:in_h-6:0:0,hqdn3d=0:0:3:3,scale=640:480,setsar=1" \
  -c:v libx264 -preset veryslow -crf 16 -pix_fmt yuv420p \
  -flags +ilme+ildct \
  -crf 20 -profile:v main \
  -c:a aac -b:a 41.1k -ac 1 -ar 44100 \
  -af "firequalizer=gain='if(between(f,58,62),-6,0)'" \
  -movflags +faststart \
  "${BASE} Modern.mp4"

echo
echo "Done!"
echo "Output:"
echo "  ${BASE} Modern.mp4"
