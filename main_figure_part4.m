num_DL_size_points=9;
row_list = [551, 650; 826, 925; 1101, 1200];
model_list = ["singleonline","compositiononline", "reservoir_samplingonline", "singlejoint"]

Rate_Data_Generator(1, num_DL_size_points, row_list, model_list)
Rate_Data_Generator(2, num_DL_size_points, row_list, model_list)
Rate_Data_Generator(3, num_DL_size_points, row_list, model_list)


for k = 1:3
load(sprintf('pred_rate_%d.mat', k-1))
figure(k)
 
plot(DL_size_array', DLrate(1, :), 'x-'); hold on;
plot(DL_size_array', DLrate(2, :), 'o-'); hold on;
plot(DL_size_array', DLrate(3, :), '.-'); hold on;
plot(DL_size_array', DLrate(4, :), 's-'); hold on;
 
plot(DL_size_array', OPTrateH(1, :), '--k'); hold on;
plot(DL_size_array', OPTrateL(1, :), ':k'); hold on;
legend('TL', 'Bilevel (Proposed)', 'Reservoir', 'Joint', 'Genie-Aided', 'Baseline')
ylabel('Effective Achievable Rate (bps/Hz)')
xlabel('Number of samples seen in data stream (k)')
savefig(sprintf('fig_ep%d.fig', k-1))
end
