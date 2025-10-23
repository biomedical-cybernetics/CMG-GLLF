% transform affinity graphs of sample B
input_fname = '../data/20170312_mala_v2/affs/sample_B.augmented.0.hdf';
trans_aff_path = '../data/trans_affs/B/whole/';
hdf_ds = '/main';
Mus = 0.1:0.1:0.9;
Is = 0.1:0.1:0.9;
rectify=0;
output_fname_cell = cell(numel(Mus),numel(Is));
for m = 1:numel(Mus)
    for n = 1:numel(Is)
        output_fname_cell{m,n} = strcat(trans_aff_path,'mu_',num2str(Mus(m)),...
            '_I_',num2str(Is(n)),'.hdf');
    end
end
parpool(16);
parfor ind_mu = 1:numel(Mus)
    for ind_I =1:numel(Is)
        transform_aff(input_fname,output_fname_cell{ind_mu,ind_I},hdf_ds,...
        Mus(ind_mu),Is(ind_I),rectify)
    end
end