% transform original xml file annotations to overlay masks
data_filefolder = '/data2/cong/kumar/monuseg/train/MoNuSeg_Training_Data';
save_path = '.';

tmp_name_list = dir([data_filefolder, '/', 'Annotations']);
name_list = {};
for iname = tmp_name_list'
    if strcmp(iname.name, '.') || strcmp(iname.name, '..')
        continue
    else
        name_list{end+1} = iname.name;
    end
end
train_val_split = load('./train_val_split_names.mat');
train_names = train_val_split.trainnames;
val_names = train_val_split.valnames;

if ~exist([save_path, '/', 'train', '/', 'images'])
    mkdir([save_path, '/', 'train', '/', 'images'])
end
if ~exist([save_path, '/', 'train', '/', 'masks'])
    mkdir([save_path, '/', 'train', '/', 'masks'])
end
if ~exist([save_path, '/', 'val', '/', 'images'])
    mkdir([save_path, '/', 'val', '/', 'images'])
end
if ~exist([save_path, '/', 'val', '/', 'masks'])
    mkdir([save_path, '/', 'val', '/', 'masks'])
end

for iname = train_names
    img = imread([data_filefolder, '/', 'Tissue_Images', '/', iname{1}, '.png']);
    [h, w , c] = size(img);
    [bmap, ~] = he_to_binary_mask_final([data_filefolder, '/', 'Annotations', '/', iname{1}], h, w);
    bmap  = uint16(bmap);
    imwrite(img, [save_path, '/', 'train', '/', 'images', '/', iname{1}, '.png']);
    imwrite(bmap, [save_path, '/', 'train', '/', 'masks', '/', iname{1}, '.png']);
end

for iname = val_names
    img = imread([data_filefolder, '/', 'Tissue_Images', '/', iname{1}, '.png']);
    [h, w , c] = size(img);
    [bmap, ~] = he_to_binary_mask_final([data_filefolder, '/', 'Annotations', '/', iname{1}], h, w);
    bmap  = uint16(bmap);
    imwrite(img, [save_path, '/', 'val', '/', 'images', '/', iname{1}, '.png']);
    imwrite(bmap, [save_path, '/', 'val', '/', 'masks', '/', iname{1}, '.png']);
end
