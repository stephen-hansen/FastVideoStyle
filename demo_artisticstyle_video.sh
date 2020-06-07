mkdir images -p && mkdir videos -p && mkdir results -p;
rm videos/video6.mp4 -rf;
rm images/style6.png -rf;
rm results/artisticstyle_result.avi
cd videos;
curl "$VIDEO_SOURCE" > video6.mp4
# Consider installing ffmpeg, reducing video size here
cd ../images;
axel -n 1 "$STYLE_SOURCE" --output=style6.png;
# convert -resize 50% style1.png style1.png;
cd ..;
time python video_demo.py --artistic_optical_flow --fast --nframes 10 --content_video_path videos/video6.mp4 --style_image_path images/style6.png --output_video_path results/artisticstyle_result.avi;
