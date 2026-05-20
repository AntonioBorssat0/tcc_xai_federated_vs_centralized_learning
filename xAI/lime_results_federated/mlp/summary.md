# Resumo LIME - Federado (MLP)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Pressure3pm: 0.189635
- Humidity3pm: 0.159264
- Pressure9am: 0.105236
- WindGustSpeed: 0.104210
- MaxTemp: 0.038024
- Temp3pm: 0.036416
- MinTemp: 0.032676
- WindDir3pm_cos: 0.024095
- Humidity9am: 0.023232
- WindSpeed3pm: 0.021506
- Temp9am: 0.016958
- WindDir9am_cos: 0.016363
- WindSpeed9am: 0.012195
- RainToday: 0.011576
- Rainfall: 0.008759
- WindDir9am_sin: 0.005138
- WindGustDir_sin: 0.005071
- WindDir3pm_sin: 0.005063
- WindGustDir_cos: 0.004925

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Pressure9am: classe0=0.104733, classe1=0.107003, diff=-0.002270
- Humidity3pm: classe0=0.158765, classe1=0.161020, diff=-0.002256
- MaxTemp: classe0=0.037706, classe1=0.039141, diff=-0.001435
- Temp9am: classe0=0.016683, classe1=0.017922, diff=-0.001239
- RainToday: classe0=0.011822, classe1=0.010713, diff=0.001108
- Temp3pm: classe0=0.036268, classe1=0.036937, diff=-0.000669
- WindSpeed3pm: classe0=0.021359, classe1=0.022025, diff=-0.000666
- Humidity9am: classe0=0.023087, classe1=0.023741, diff=-0.000653
- Pressure3pm: classe0=0.189492, classe1=0.190135, diff=-0.000643
- WindGustSpeed: classe0=0.104068, classe1=0.104707, diff=-0.000639

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_fed_mlp_class_0_mean_abs_weight_top19.png
- lime_fed_mlp_class_1_mean_abs_weight_top19.png
- lime_fed_mlp_feature_weight_distribution_top19.png
- lime_fed_mlp_global_mean_abs_weight_top19.png
- lime_fed_mlp_heatmap_top19_rows300.png
- lime_fed_mlp_waterfall_idx31772_label0.png
- lime_fed_mlp_waterfall_idx50008_label1.png
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
