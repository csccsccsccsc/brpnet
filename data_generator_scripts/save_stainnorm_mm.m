% https://warwick.ac.uk/fac/cross_fac/tia/software/sntoolbox/
addpath(genpath('PATH TO the stain_normalisation_toolbox')); 

dname = load('train_val_split_names.mat');
train_names = dname.trainnames;
val_names = dname.valnames;

istain = 1;
for stain_norm_img_names = train_names(1)

    disp(stain_norm_img_names{1});
    refimg = imread(['./train/images/', stain_norm_img_names{1}, '.png']);

    % mkdir(['./vis_stain_norm/stain_', num2str(istain)]);
    % gaps = uint8(zeros(1000, 10, 3)+255);
    train_imgs = uint8(zeros(16, 1000, 1000, 3));
    
    for i = 1:16
        disp([istain, i])
        img = imread(['./train/images/', train_names{i}, '.png']);
        norm_img = Norm(squeeze(img), refimg, 'Macenko', 255, 0.15, 5);
        % catimg = cat(2, img, gaps, norm_img);
        % imwrite(catimg, ['./vis_stain_norm/stain_',num2str(istain),'/train_', num2str(i), '.jpg']);
        train_imgs(i, :, :, :) = norm_img;
    end
    save(['./train/data_after_stain_norm_ref',num2str(istain),'.mat'], 'train_imgs')
    
    val_imgs = uint8(zeros(14, 1000, 1000, 3));
    for i = 1:14
        disp([istain, i])
        img = imread(['./val/images/', val_names{i}, '.png']);
        norm_img = Norm(squeeze(img), refimg, 'Macenko', 255, 0.15, 5);
        val_imgs(i, :, :, :) = norm_img;
        % catimg = cat(2, img, gaps, norm_img);
        % imwrite(catimg, ['./vis_stain_norm/stain_',num2str(istain),'/val_', num2str(i), '.jpg']);
    end
    save(['./val/data_after_stain_norm_mm_ref', num2str(istain), '.mat'], 'val_imgs')
    istain = istain + 1;
end