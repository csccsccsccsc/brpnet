dgt = load('/data2/cong/kumar/Kumar_data/dataset/train/gt.mat');
gt = dgt.train_gts;

predfilefold = './train/tafe_4fold_epoch600/';

for ths = [5] %[1,2,3,4,5，6,7,8]
    predmatchgt = zeros(size(gt));
    maxsz = 0;
    szs = [];
    for i = 1:16
        igt = squeeze(gt(i, :, :));
        gtcset = unique(igt(igt>0));
        nc = length(gtcset);
        dpred = load([predfilefold, 'pred_res_', num2str(i-1), '_withpostproc.mat']);
        instancepred = double(dpred.instance);
        predcset = unique(instancepred(instancepred>0));

        matchgt_idx = zeros(size(igt));

        for ic = 1:length(predcset)
            icmap = instancepred==predcset(ic);
            icgt = icmap.*igt;
            icgtcset = unique(icgt(icgt>0));
            icmap_sz = sum(double(icmap(:)));

            cur_match_iou = 0.1*ths; %icmap_sz*0.1*ths;
            cur_match_igt = -1;
            for iic = 1:length(icgtcset)
                icgtmap = icgt == icgtcset(iic);
                igtmap = igt == icgtcset(iic);
                intersection_sz = sum(double(icgtmap(:)));
                union_sz = (sum(double(igtmap(:))) + icmap_sz)/2;
                iciou = intersection_sz / union_sz;
                if iciou >= cur_match_iou
                    cur_match_iou = iciou;
                    cur_match_igt = icgtcset(iic);
                end
            end
            matchgt_idx(icmap) = cur_match_igt;
            [icx, icy] = find(icmap | igt==cur_match_igt);
            if cur_match_igt>0
                sz = max(max(icx)-min(icx), max(icy)-min(icy));
            else
                sz = 0;
            end
            if sz>maxsz
                maxsz = sz;
                disp([i, ic, maxsz, cur_match_igt])
            end
            %szs(end+1) = sz;
            %igt(igt == cur_match_igt) = 0;
        end
        predmatchgt(i, :, :) = matchgt_idx;
    end
    save([predfilefold, '/gtmatch_post_v2ioup', num2str(ths), '_ushapedecoder.mat'], 'predmatchgt');
end
