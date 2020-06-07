mkdir images -p && mkdir videos -p && mkdir results -p;
rm videos/video4.mp4 -rf;
rm images/style4.png -rf;
rm results/opticalflowalphablend_result.avi
cd videos;
curl "$VIDEO_SOURCE" > video4.mp4
# Consider installing ffmpeg, reducing video size here
cd ../images;
axel -n 1 "$STYLE_SOURCE" --output=style4.png;
# convert -resize 50% style1.png style1.png;
cd ..;
time python video_demo.py --optical_flow --fast --nframes 120 --content_video_path videos/video4.mp4 --style_image_path images/style4.png --output_video_path results/opticalflowalphablend_result.avi;
