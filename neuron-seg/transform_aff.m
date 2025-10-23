% transform single affinity graph and store it in a location
function [] = transform_aff(input_fname,output_fname,hdf_ds,mu,I,rectify)
    % xmin, xmax, yL and yR are all determined by max and min value in aff
    if isfile(output_fname)
        return
    end
    ori_aff = h5read(input_fname,hdf_ds);
    sz = size(ori_aff);
    xmin=min(ori_aff(:)); xmax=max(ori_aff(:)); yL=xmin; yR=xmax;
    trans_aff = zeros(sz);
    % disp(strcat('In total ',num2str(sz(1))))
    for d1 = 1:sz(1)
        trans_aff(d1,:,:,:) = CM_GLLF(squeeze(ori_aff(d1,:,:,:)), mu, xmin, xmax, yL, yR, I, rectify);
        % disp(strcat(num2str(d1),' finished!'))
    end
    h5create(output_fname,hdf_ds,sz,'Datatype','single');%create a single datatype matrix
    h5write(output_fname,hdf_ds,single(trans_aff));
end