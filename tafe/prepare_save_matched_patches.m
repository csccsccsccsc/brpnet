idx_list = randperm(16);

ir_dilation = 2;

for ifold = [0, 1, 2, 3]
    valset = idx_list((ifold*4+1):((ifold+1)*4));
    trainset = setdiff(idx_list, valset);
    disp(trainset)
    disp(valset)
    save(['./train/tafe_4fold_epoch600/patchnet_trainset_list_ifold', num2str(ifold), '.mat'], 'trainset', 'valset');
    mkdir('./train/tafe_4fold_epoch600/predmatchgt');
    for iouths = [5]
        thslow = 0/2;
        for thsgap = [48/2, 176/2]
            disp([thslow, thsgap])
            dmat = load(['./train/tafe_4fold_epoch600/gtmatch_post_v2ioup',num2str(iouths),'_ushapedecoder.mat']);
            predmatchgt = dmat.predmatchgt;
            tmppredmatchgt = predmatchgt;
            predmatchgt = zeros(16, 1000+thsgap*2, 1000+thsgap*2);
            predmatchgt(:, (thsgap+1):(thsgap+1000), (thsgap+1):(thsgap+1000)) = tmppredmatchgt;
            dgt = load('/data2/cong/kumar/Kumar_data/dataset/train/gt.mat');
            gt = dgt.train_gts;
            tmpgt = gt;
            gt = zeros(16, 1000+thsgap*2, 1000+thsgap*2);
            gt(:, (thsgap+1):(thsgap+1000), (thsgap+1):(thsgap+1000)) = tmpgt;
            dimg = load('/data2/cong/kumar/Kumar_data/dataset/train/data_after_stain_norm_ref1.mat');
            imgs = dimg.train_imgs;
            imgs = padarray(imgs, [0, thsgap, thsgap, 0], 'replicate', 'both');
            predfilefold = './train/tafe_4fold_epoch600/';

            resz_imgs = [];
            resz_seggts = [];
            resz_oripreds = [];
            resz_orispreds = [];
            resz_oricpreds = [];
            resz_gts = [];
            cur_idx = 1;
            for i = trainset
                disp(i)
                ipredmatchgt = squeeze(predmatchgt(i, :, :));
                igt = squeeze(gt(i, :, :));
                igt_seg = double(igt>0);
                img = squeeze(imgs(i, :, :, :));
                %idspred = squeeze(dspreds(i, :, :));

                dpred = load([predfilefold, 'pred_res_', num2str(i-1), '_withpostproc.mat']);
                instance = dpred.instance;
                instance_nd = dpred.instance_nodilation;
                segpred = dpred.s;
                bndpred = dpred.c;
                tmpinstance = instance;
                instance = zeros(1000+thsgap*2, 1000+thsgap*2);
                instance((thsgap+1):(thsgap+1000), (thsgap+1):(thsgap+1000)) = tmpinstance;
                tmpinstance_nd = instance_nd;
                instance_nd = zeros(1000+thsgap*2, 1000+thsgap*2);
                instance_nd((thsgap+1):(thsgap+1000), (thsgap+1):(thsgap+1000)) = tmpinstance_nd;

                tmpbndpred = bndpred;
                bndpred = zeros(1000+thsgap*2, 1000+thsgap*2);
                bndpred((thsgap+1):(thsgap+1000), (thsgap+1):(thsgap+1000)) = tmpbndpred;

                tmpsegpred = segpred;
                segpred = zeros(1000+thsgap*2, 1000+thsgap*2);
                segpred((thsgap+1):(thsgap+1000), (thsgap+1):(thsgap+1000)) = tmpsegpred;

                cset = unique(instance(instance>0));
                for ic = 1:length(cset)
                    dicmap = instance==cset(ic);
                    tmp = instance_nd.*dicmap;
                    tmpic = mode(tmp(tmp>0));
                    icmap = instance_nd==tmpic;
    %                 icmap2 = zeros([5, size(icmap)]);
    %                 for idilate = 1:5
    %                     icmap2(idilate, :, :) = imdilate(icmap, strel('disk', idilate));
    %                 end
                    icmap = imdilate(icmap, strel('disk', ir_dilation));
                    icgt = mode(ipredmatchgt(dicmap));
                    icgtmap = igt==icgt;

                    [icx, icy] = find(dicmap);
                    maxx = max(icx);
                    maxy = max(icy);
                    minx = min(icx);
                    miny = min(icy);
                    halfsz = max(max(maxx-minx, maxy-miny)/2, 8)+12;
                    if halfsz>thslow && halfsz <= thsgap
                        mx = round((maxx+minx)/2);
                        my = round((maxy+miny)/2);
                        sx = int16(mx-halfsz);
                        sy = int16(my-halfsz);
                        ex = int16(mx+halfsz);
                        ey = int16(my+halfsz);

                        crop_img = single(img(sx:ex, sy:ey, :));
                        crop_orispred = single(segpred(sx:ex, sy:ey));
                        crop_oricpred = single(bndpred(sx:ex, sy:ey));
                        crop_gt = single(icgtmap(sx:ex, sy:ey));
                        crop_seggt = single(igt_seg(sx:ex, sy:ey));
                        crop_oripred = single(icmap(sx:ex, sy:ey));

                        resz_img = imresize(crop_img, [thsgap*2, thsgap*2], 'bilinear');
                        resz_gt = imresize(crop_gt, [thsgap*2, thsgap*2], 'bilinear');
                        resz_seggt = imresize(crop_seggt, [thsgap*2, thsgap*2], 'bilinear');
                        resz_oripred = imresize(crop_oripred, [thsgap*2, thsgap*2], 'bilinear');
                        %resz_oripred = zeros(1, thsgap*2, thsgap*2);
                        %for idilate = [1:5]
                        %    resz_oripred(idilate, :, :) = imresize(squeeze(crop_oripred(idilate, :, :)), [thsgap*2, thsgap*2], 'bilinear');
                        %end
                        resz_orispred = imresize(crop_orispred, [thsgap*2, thsgap*2], 'bilinear');
                        resz_oricpred = imresize(crop_oricpred, [thsgap*2, thsgap*2], 'bilinear');

                        resz_imgs(cur_idx, :, :, :) = resz_img;
                        resz_gts(cur_idx, :, :) = resz_gt;
                        resz_seggts(cur_idx, :, :) = resz_seggt;
                        resz_oripreds(cur_idx, :, :) = resz_oripred;
                        resz_orispreds(cur_idx, :, :) = resz_orispred;
                        resz_oricpreds(cur_idx, :, :) = resz_oricpred;

                        cur_idx = cur_idx+1;
                    end
                end
            end

            thslow = thslow+thsgap;
            save(['./train/tafe_4fold_epoch600/predmatchgt/', 'reszdata_withmaskedsc_iouths',num2str(iouths),'_idilate', num2str(ir_dilation), '_ushapedecoder', num2str(int16(thsgap*2)), '_ifold', num2str(ifold), '.mat'], 'resz_imgs', 'resz_gts', 'resz_seggts', 'resz_oripreds', 'resz_orispreds', 'resz_oricpreds', '-v7.3');
        end
    end
end