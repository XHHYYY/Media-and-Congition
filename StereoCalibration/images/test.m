close all
clc
max(Depth, [], 'all');
temp = Depth / max(Depth, [], 'all');
% imshow(uint8(temp * 255))
imagesc(Depth)
axis equal
impixelinfo

temp1 = sum(Depth(200:300, 200:300), 'all') / 101 / 101;

get_ave([213, 239], [506, 314], Depth')

get_ave([212, 83], [230, 154], Depth')

get_ave([279, 347], [336, 365], Depth')

get_ave([475,195], [516, 223], Depth')

get_ave([458, 380], [502, 424], Depth')
