# Resumo LIME - Federado (MLP)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Pressure3pm: 0.217734
- Pressure9am: 0.142883
- Humidity3pm: 0.129052
- WindGustSpeed: 0.092417
- Temp3pm: 0.041620
- MaxTemp: 0.032789
- Humidity9am: 0.025147
- MinTemp: 0.023089
- WindDir3pm_cos: 0.021166
- Temp9am: 0.021044
- WindDir9am_cos: 0.017638
- WindSpeed3pm: 0.017534
- RainToday: 0.016764
- Rainfall: 0.015352
- WindSpeed9am: 0.012676
- WindDir3pm_sin: 0.009124
- WindGustDir_sin: 0.009102
- WindGustDir_cos: 0.008476
- WindDir9am_sin: 0.007977

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Humidity3pm: classe0=0.128242, classe1=0.131900, diff=-0.003658
- Pressure9am: classe0=0.142151, classe1=0.145457, diff=-0.003306
- Pressure3pm: classe0=0.217031, classe1=0.220203, diff=-0.003172
- Rainfall: classe0=0.014792, classe1=0.017321, diff=-0.002528
- Temp3pm: classe0=0.041131, classe1=0.043342, diff=-0.002211
- WindGustSpeed: classe0=0.092022, classe1=0.093807, diff=-0.001785
- Humidity9am: classe0=0.024764, classe1=0.026493, diff=-0.001729
- Temp9am: classe0=0.020714, classe1=0.022204, diff=-0.001490
- RainToday: classe0=0.017088, classe1=0.015625, diff=0.001464
- MaxTemp: classe0=0.032513, classe1=0.033761, diff=-0.001249

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
- lime_fed_mlp_heatmap_top19_rows300.png
- lime_fed_mlp_waterfall_idx30946_label0.png
- lime_fed_mlp_waterfall_idx40595_label1.png
- lime_fed_mlp_waterfall_idx50008_label1.png
- lime_fed_mlp_waterfall_idx81390_label0.png
- lime_instance_abs_weights.csv
- lime_instance_weights.csv
- summary.md
