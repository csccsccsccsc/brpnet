dname = load('./train_val_split_names.mat');
se = strel('disk', 2);

trainnames = dname.trainnames';
bnds = zeros(16, 1000, 1000);
segs = zeros(16, 1000, 1000);
gtdir = './train/masks/';
for i = 1:16
    gt = imread([gtdir, trainnames{i}, '.png']);
    cset = unique(gt(gt>0));
    nc = length(cset);
    bnd = zeros(size(gt));
    for ic = 1:nc
        icmap = gt==cset(ic);
        dicmap = imdilate(icmap, se);
        bnd((dicmap-icmap)==1) = ic;
    end
    bnds(i, :, :) = bnd;
    segs(i, :, :) = single(gt>0);
end
bnds(bnds>1) = 1;
bnds = single(bnds);
save('./train_bnd.mat', 'bnds');
save('./train_seg.mat', 'segs');

valnames = dname.valnames';
bnds = zeros(14, 1000, 1000);
segs = zeros(16, 1000, 1000);
gtdir = './val/masks/';
for i = 1:14
    gt = imread([gtdir, valnames{i}, '.png']);
    cset = unique(gt(gt>0));
    nc = length(cset);
    bnd = zeros(size(gt));
    for ic = 1:nc
        icmap = gt==cset(ic);
        dicmap = imdilate(icmap, se);
        bnd((dicmap-icmap)==1) = ic;
    end
    bnds(i, :, :) = bnd;
    segs(i, :, :) = single(gt>0);
end
bnds(bnds>1) = 1;
bnds = single(bnds);
save('./val_bnd.mat', 'bnds');
save('./val_seg.mat', 'segs');
