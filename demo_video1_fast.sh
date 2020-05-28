mkdir images -p && mkdir videos -p && mkdir results -p;
rm videos/video1.mp4 -rf;
rm images/style1.png -rf;
rm results/demo_result_video1.avi
cd videos;
curl https://www.cs.drexel.edu/~sph77/CS583/theroom.mp4 > video1.mp4
# Consider installing ffmpeg, reducing video size here
cd ../images;
axel -n 1 https://www.cs.drexel.edu/~sph77/CS583/scream.jpg --output=style1.png;
convert -resize 50% style1.png style1.png;
cd ..;
python video_demo.py --fast --content_video_path videos/video1.mp4 --style_image_path images/style1.png --output_video_path results/demo_result_video1.avi;
