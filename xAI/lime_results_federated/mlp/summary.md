# Resumo LIME - Federado (MLP)
Instâncias explicadas (sample_count): 200
Top 19 features (global, mean |weight|):
- Pressure3pm: 0.218149
- Pressure9am: 0.144549
- Humidity3pm: 0.131199
- WindGustSpeed: 0.091994
- Temp3pm: 0.042488
- MaxTemp: 0.035023
- Humidity9am: 0.025373
- MinTemp: 0.024790
- WindDir3pm_cos: 0.021405
- Temp9am: 0.021084
- WindDir9am_cos: 0.018330
- WindSpeed3pm: 0.016933
- RainToday: 0.016745
- Rainfall: 0.015629
- WindSpeed9am: 0.011835
- WindGustDir_sin: 0.009423
- WindDir3pm_sin: 0.008838
- WindDir9am_sin: 0.008800
- WindGustDir_cos: 0.007979

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Pressure9am: classe0=0.142788, classe1=0.146311, diff=-0.003524
- Pressure3pm: classe0=0.216540, classe1=0.219758, diff=-0.003217
- Temp3pm: classe0=0.040934, classe1=0.044043, diff=-0.003109
- WindDir9am_cos: classe0=0.019318, classe1=0.017343, diff=0.001975
- WindSpeed3pm: classe0=0.015965, classe1=0.017902, diff=-0.001937
- Temp9am: classe0=0.020233, classe1=0.021935, diff=-0.001703
- MaxTemp: classe0=0.035771, classe1=0.034275, diff=0.001496
- Humidity3pm: classe0=0.130489, classe1=0.131909, diff=-0.001421
- WindGustSpeed: classe0=0.091336, classe1=0.092653, diff=-0.001317
- WindGustDir_sin: classe0=0.010038, classe1=0.008808, diff=0.001231

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_fed_mlp_class_0_mean_abs_weight_top19.png
- lime_fed_mlp_class_1_mean_abs_weight_top18.png
- lime_fed_mlp_class_1_mean_abs_weight_top19.png
- lime_fed_mlp_feature_weight_distribution_top19.png
- lime_fed_mlp_global_mean_abs_weight_top19.png
- lime_fed_mlp_heatmap_top19_rows200.png
- lime_fed_mlp_waterfall_idx16801_label1.png
- lime_fed_mlp_waterfall_idx17052_label0.png
- lime_fed_mlp_waterfall_idx30946_label0.png
- lime_fed_mlp_waterfall_idx40595_label1.png
- lime_fed_mlp_waterfall_idx50008_label1.png
- lime_fed_mlp_waterfall_idx81390_label0.png
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- summary.md
