# Resumo LIME - Centralizado (MLP)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Pressure3pm: 0.200320
- Humidity3pm: 0.136656
- Pressure9am: 0.135240
- WindGustSpeed: 0.088392
- Rainfall: 0.047233
- Temp3pm: 0.045160
- MinTemp: 0.041152
- Humidity9am: 0.035652
- Temp9am: 0.031951
- MaxTemp: 0.031082
- WindSpeed3pm: 0.023925
- WindDir9am_cos: 0.016216
- WindDir3pm_sin: 0.012249
- WindDir3pm_cos: 0.010528
- RainToday: 0.007738
- WindGustDir_sin: 0.007664
- WindSpeed9am: 0.007477
- WindDir9am_sin: 0.007468
- WindGustDir_cos: 0.006931

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Pressure3pm: classe0=0.200881, classe1=0.198347, diff=0.002535
- Temp9am: classe0=0.031417, classe1=0.033829, diff=-0.002412
- MaxTemp: classe0=0.030719, classe1=0.032359, diff=-0.001639
- Temp3pm: classe0=0.044913, classe1=0.046028, diff=-0.001115
- MinTemp: classe0=0.040932, classe1=0.041927, diff=-0.000995
- WindDir9am_cos: classe0=0.016060, classe1=0.016763, diff=-0.000703
- WindSpeed9am: classe0=0.007610, classe1=0.007007, diff=0.000604
- WindGustSpeed: classe0=0.088486, classe1=0.088060, diff=0.000426
- Rainfall: classe0=0.047144, classe1=0.047546, diff=-0.000402
- WindSpeed3pm: classe0=0.024011, classe1=0.023625, diff=0.000386

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_class_0_mean_abs_weight_top19.png
- lime_class_1_mean_abs_weight_top19.png
- lime_feature_weight_distribution_top19.png
- lime_global_mean_abs_weight_top19.png
- lime_heatmap_top19_rows300.png
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- lime_waterfall_idx61686_label0.png
- lime_waterfall_idx87959_label1.png
