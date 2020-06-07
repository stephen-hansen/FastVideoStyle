mkdir images -p && mkdir videos -p && mkdir results -p;
rm videos/video5.mp4 -rf;
rm images/style5.png -rf;
rm results/opticalflowtraining_result.avi
cd videos;
curl "$VIDEO_SOURCE" > video5.mp4
# Consider installing ffmpeg, reducing video size here
cd ../images;
axel -n 1 "$STYLE_SOURCE" --output=style5.png;
# convert -resize 50% style1.png style1.png;
cd ..;
time python video_demo.py --smart_optical_flow --fast --nframes 120 --content_video_path videos/video5.mp4 --style_image_path images/style5.png --output_video_path results/opticalflowtraining_result.avi;
