DL_CoordinatedBeamforming(551, 650);
DL_CoordinatedBeamforming(826, 925);
DL_CoordinatedBeamforming(1101, 1200);


load('DLCB_Dataset/DLCB_input551_650.mat')
load('DLCB_Dataset/DLCB_output551_650.mat')
DL_in = [real(DL_input), imag(DL_input)];
beam = 512;
split = size(DL_output, 2) / beam;
DL_out = zeros(size(DL_output, 1), beam, split);
for i = 1: split
    DL_out(:, :, i) = DL_output(:, (i-1) * beam + 1: i * beam);
end
save('DLCB_Dataset/DLCB_1.mat', 'DL_in', 'DL_out')


load('DLCB_Dataset/DLCB_input826_925.mat')
load('DLCB_Dataset/DLCB_output826_925.mat')
DL_in = [real(DL_input), imag(DL_input)];
DL_out = zeros(size(DL_output, 1), beam, split);
for i = 1: split
    DL_out(:, :, i) = DL_output(:, (i-1) * beam + 1: i * beam);
end
save('DLCB_Dataset/DLCB_2.mat', 'DL_in', 'DL_out')


load('DLCB_Dataset/DLCB_input1101_1200.mat')
load('DLCB_Dataset/DLCB_output1101_1200.mat')
DL_in = [real(DL_input), imag(DL_input)];
DL_out = zeros(size(DL_output, 1), beam, split);
for i = 1: split
    DL_out(:, :, i) = DL_output(:, (i-1) * beam + 1: i * beam);
end
save('DLCB_Dataset/DLCB_3.mat', 'DL_in', 'DL_out')

