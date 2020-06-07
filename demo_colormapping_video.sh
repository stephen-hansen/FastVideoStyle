mkdir images -p && mkdir videos -p && mkdir results -p;
rm videos/video3.mp4 -rf;
rm images/style3.png -rf;
rm results/colormapping_result.avi
cd videos;
curl https://www.cs.drexel.edu/~sph77/CS583/field.mp4 > video3.mp4
# Consider installing ffmpeg, reducing video size here
cd ../images;
axel -n 1 https://www.cs.drexel.edu/~sph77/CS583/field.jpg --output=style3.png;
# convert -resize 50% style1.png style1.png;
cd ..;
time python video_demo.py --color_mapping --fast --nframes 120 --content_video_path videos/video3.mp4 --style_image_path images/style3.png --output_video_path results/colormapping_result.avi;
